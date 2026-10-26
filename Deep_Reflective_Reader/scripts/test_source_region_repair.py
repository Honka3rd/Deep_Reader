"""SRREPAIR-01 exact targets and pure eligibility; not mutation acceptance."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.reading_content import (
    ReadingContentConflict, original_blocks, source_token_for,
)
from document_structure.source_region_repair import region_repair_targets
from document_structure.source_review_blocks import region_block_mapping, persist_region_mapping
from scripts.test_source_review_mapping import mapped_document, exact_region, with_regions


def sample(start=2, end=8, text=None):
    document = mapped_document() if text is None else mapped_document(text)
    return with_regions(document, exact_region(document, "r", start, end))


def with_state(document, **state):
    return replace(document, parse_provenance={**document.parse_provenance, "reading_content": state})


class SourceRegionRepairTests(unittest.TestCase):
    def target(self, document):
        before = deepcopy(document.to_dict())
        result = region_repair_targets(document)
        self.assertEqual(document.to_dict(), before)
        return result["regions"]["r"]

    def assert_denied(self, document, reason):
        row = self.target(document)
        self.assertFalse(row["can_apply"])
        self.assertFalse(row["can_delete"])
        self.assertEqual(row["blocked_reason"], reason)
        return row

    def test_partial_block_is_exact_not_whole_block_ownership(self):
        document = sample()
        row = self.target(document)
        self.assertTrue(row["can_apply"])
        self.assertTrue(row["can_delete"])
        self.assertIsNone(row["blocked_reason"])
        chapter, section, unit, block = original_blocks(document)[0]
        self.assertEqual(row["block_spans"], [{
            "chapter_id": chapter.chapter_id, "section_id": section.section_id,
            "task_unit_id": unit.unit_id, "block_id": block.block_id,
            "source_start": 2, "source_end": 8, "block_source_start": 0,
            "block_source_end": 16, "char_from": 2, "char_to": 8}])
        self.assertEqual(row["canonical_span"]["char_start"], 2)
        self.assertEqual(row["canonical_span"]["char_end"], 8)

    def test_cross_block_with_whitespace_gap_has_two_exact_intersections(self):
        row = self.target(sample(10, 25))
        self.assertTrue(row["can_apply"])
        self.assertEqual([(b["source_start"], b["source_end"], b["char_from"], b["char_to"])
                          for b in row["block_spans"]], [(10, 16, 10, 16), (18, 25, 0, 7)])

    def test_cjk_normalization_does_not_change_canonical_offsets(self):
        document = sample(2, 5, "\u4ed6 \u5fc3 \u91cc \u60f3\n\nOther paragraph.")
        block = original_blocks(document)[0][3]
        self.assertEqual(block.content, "\u4ed6\u5fc3\u91cc\u60f3")
        row = self.target(document)
        self.assertTrue(row["can_apply"])
        link, = row["block_spans"]
        self.assertEqual((link["char_from"], link["char_to"]), (2, 5))
        self.assertEqual(document.raw_text[link["source_start"]:link["source_end"]], "\u5fc3 \u91cc")
        self.assertGreater(link["char_to"], len(block.content))

    def test_unicode_offsets_are_code_points_not_utf16(self):
        document = sample(1, 3, "A\U0001f600B cafe\u0301.\n\nOther.")
        result = region_repair_targets(document)
        self.assertEqual(result["offset_basis"], "canonical_unicode_code_points")
        row = result["regions"]["r"]
        self.assertTrue(row["can_apply"])
        self.assertEqual(row["block_spans"][0]["char_to"], 3)
        self.assertEqual(document.raw_text[1:3], "\U0001f600B")

    def test_manual_flag_and_legacy_edit_both_protect_whole_related_block(self):
        document = sample()
        key = original_blocks(document)[0][3].block_id
        for state in ({"block_edit_flags": {key: True}},
                      {"audit": [{"action": "edit", "block_id": key}, {"action": "undo"}]},
                      {"blocks": {key: {"content": "Edited elsewhere in this block"}}}):
            with self.subTest(state=state):
                self.assert_denied(with_state(document, **state), "block_manually_edited")

    def test_manual_delete_and_undone_delete_are_protected(self):
        document = sample()
        key = original_blocks(document)[0][3].block_id
        for state in ({"blocks": {key: {"deleted": True}}},
                      {"audit": [{"action": "delete", "block_id": key}, {"action": "undo"}]},
                      {"audit": [{"action": "delete", "affected_block_ids": [key]}]}):
            with self.subTest(state=state):
                self.assert_denied(with_state(document, **state), "block_manually_deleted")

    def test_one_protected_block_denies_entire_cross_block_region(self):
        document = sample(10, 25)
        key = original_blocks(document)[1][3].block_id
        row = self.assert_denied(with_state(document, block_edit_flags={key: True}), "block_manually_edited")
        self.assertEqual(len(row["block_spans"]), 2)
        self.assertEqual(region_block_mapping(document)["r"]["mapping_status"], "complete")

    def test_unrelated_body_changes_and_history_draft_saves_do_not_protect(self):
        document = sample()
        key = original_blocks(document)[1][3].block_id
        state = {"block_edit_flags": {key: True}, "audit": [
            {"action": "cleanup_edit", "block_id": original_blocks(document)[0][3].block_id}]}
        self.assertTrue(self.target(with_state(document, **state))["can_apply"])

    def test_partial_coverage_denies_without_removing_navigation(self):
        document = sample(0, 25)
        original = original_blocks(document)
        with patch("document_structure.source_review_blocks.original_blocks", return_value=original[:1]):
            row = self.assert_denied(document, "partial_source_coverage")
        self.assertEqual(row["block_ids"], [original[0][3].block_id])

    def test_missing_units_never_invents_targets(self):
        document = mapped_document(units=False)
        document = with_regions(document, exact_region(document, "r", 0, 5))
        row = self.assert_denied(document, "blocks_not_materialized")
        self.assertEqual(row["block_spans"], [])

    def test_overlapping_block_ranges_are_not_writable(self):
        document = sample()
        original = original_blocks(document)
        duplicate = (*original[0][:3], replace(original[0][3], block_id="another-block"))
        with patch("document_structure.source_review_blocks.original_blocks", return_value=original + [duplicate]):
            self.assert_denied(document, "ambiguous_block_coverage")

    def test_duplicate_block_identity_across_spans_is_not_writable(self):
        document = sample(10, 25)
        original = original_blocks(document)
        duplicate_id = (*original[1][:3], replace(original[1][3], block_id=original[0][3].block_id))
        with patch("document_structure.source_review_blocks.original_blocks", return_value=[original[0], duplicate_id]):
            self.assert_denied(document, "ambiguous_block_coverage")

    def test_source_drift_and_ambiguous_legacy_quote_fail_closed(self):
        document = sample()
        for owner in ("source_review", "extraction"):
            broken = deepcopy(document)
            key = "source_sha256" if owner == "source_review" else "text_sha256"
            broken.parse_provenance[owner][key] = "0" * 64
            row = self.assert_denied(broken, "source_mapping_unavailable")
            self.assertIsNone(row["canonical_span"])
        repeated = with_regions(mapped_document("Repeated.\n\nRepeated."),
                                {"id": "r", "page_index": 0, "raw_candidate": "Repeated."})
        self.assert_denied(repeated, "source_mapping_unavailable")

    def test_exact_legacy_unique_page_quote_is_usable(self):
        document = with_regions(mapped_document(), {"id": "r", "page_index": 0, "raw_candidate": "Alpha"})
        row = self.target(document)
        self.assertTrue(row["can_apply"])
        self.assertEqual(row["canonical_span"]["char_end"], 5)

    def test_foreign_cleanup_and_replacement_overlap_are_denied(self):
        document = sample()
        for field in ("source_edits", "cleanup_spans"):
            operation = {"char_start": 3, "char_end": 5, "quote": document.raw_text[3:5]}
            if field == "source_edits":
                operation.update(replacement="Fix", cleanup_record_id="foreign")
            state = {field: [operation], "cleanup_source_sha256": hashlib.sha256(document.raw_text.encode()).hexdigest()}
            self.assert_denied(with_state(document, **state), "reading_range_changed")

    def test_disjoint_cleanup_is_allowed_but_invalid_overlay_proof_is_not(self):
        document = sample()
        operation = {"char_start": 18, "char_end": 22, "quote": document.raw_text[18:22]}
        state = {"cleanup_spans": [operation], "cleanup_source_sha256": hashlib.sha256(document.raw_text.encode()).hexdigest()}
        self.assertTrue(self.target(with_state(document, **state))["can_apply"])
        for corruption in ("hash", "quote", "offset"):
            broken = deepcopy(state)
            if corruption == "hash":
                broken["cleanup_source_sha256"] = "0" * 64
            elif corruption == "quote":
                broken["cleanup_spans"][0]["quote"] = "wrong"
            else:
                broken["cleanup_spans"][0]["char_start"] = False
            self.assert_denied(with_state(document, **broken), "reading_source_changed")

    def test_ignored_or_resolved_region_is_not_writable(self):
        for decision in (True, "cleanup_resolved"):
            self.assert_denied(with_state(sample(), regions={"r": decision}), "region_not_active")

    def test_snapshot_binding_and_pure_reads_do_not_trust_cached_ids(self):
        document = persist_region_mapping(sample())
        document.parse_provenance["storage_structure_version"] = 7
        document = with_state(document, revision=9)
        before = deepcopy(document.to_dict())
        result = region_repair_targets(document)
        self.assertEqual(result["source_token"], source_token_for(document))
        self.assertEqual((result["structure_version"], result["revision"]), ("7", 9))
        self.assertEqual(document.to_dict(), before)
        document.parse_provenance["source_review_mapping"]["regions"]["r"]["block_ids"] = ["forged"]
        self.assertNotIn("forged", self.target(document)["block_ids"])
        self.assertEqual(self.target(document), self.target(document))

    def test_projected_document_cannot_supply_canonical_write_targets(self):
        document = sample()
        document.parse_provenance["reading_projected"] = True
        with self.assertRaisesRegex(ReadingContentConflict, "requires_canonical"):
            region_repair_targets(document)

    def test_canonical_text_can_come_from_saved_sections_without_raw_mirror(self):
        document = replace(sample(), raw_text="")
        row = self.target(document)
        self.assertTrue(row["can_apply"])
        self.assertEqual(row["canonical_span"]["text_sha256"],
                         hashlib.sha256(canonical_text(document).encode()).hexdigest())

    @unittest.skipUnless(os.environ.get("SOURCE_REGION_REPAIR_REAL"), "opt-in private backup read")
    def test_real_backup_natural_evidence_is_read_only(self):
        from scripts.accept_source_review_mapping import load_cases
        cases, hashes = load_cases()
        count = 0
        for label, _, document in cases:
            with self.subTest(case=label):
                before = deepcopy(document.to_dict())
                result = region_repair_targets(document)
                self.assertEqual(document.to_dict(), before)
                for row in result["regions"].values():
                    count += 1
                    if row["can_apply"]:
                        self.assertEqual(row["mapping_status"], "complete")
                        self.assertTrue(row["block_spans"])
                    else:
                        self.assertIsNotNone(row["blocked_reason"])
        self.assertGreater(count, 0)
        for path, expected in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        print(f"Private backup reads: {len(cases)} books, {count} regions, {len(hashes)} hashes unchanged")

    @unittest.skipUnless(os.environ.get("SOURCE_REGION_REPAIR_BACKUP"), "opt-in saved SQL snapshot")
    def test_saved_incident_snapshot_retains_exact_natural_targets(self):
        from document_structure.structured_document import StructuredDocument
        directory = Path(os.environ["SOURCE_REGION_REPAIR_BACKUP"]).resolve()
        corpus = Path(__file__).resolve().parents[1] / "data" / "test-books"
        self.assertTrue(directory.is_relative_to(corpus.resolve()))
        manifest_bytes = (directory / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)
        rows = json.loads((directory / "rows.json").read_bytes())
        snapshot = max(rows["structured_document_snapshots"], key=lambda row: row["source_structure_version"])
        document = StructuredDocument.from_dict(snapshot["structured_document_payload"])
        before = deepcopy(document.to_dict())
        result = region_repair_targets(document)
        self.assertEqual(document.to_dict(), before)
        self.assertTrue(result["regions"])
        linked = [row for row in result["regions"].values() if row["block_spans"]]
        self.assertTrue(linked)
        text = canonical_text(document)
        for row in linked:
            for link in row["block_spans"]:
                self.assertEqual(link["source_start"], link["block_source_start"] + link["char_from"])
                self.assertEqual(link["source_end"], link["block_source_start"] + link["char_to"])
                self.assertTrue(text[link["source_start"]:link["source_end"]])
        for name, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)
        self.assertEqual((directory / "manifest.json").read_bytes(), manifest_bytes)
        print(f"Saved incident: {len(result['regions'])} regions, {len(linked)} mapped, backup hashes unchanged")


if __name__ == "__main__":
    unittest.main()
