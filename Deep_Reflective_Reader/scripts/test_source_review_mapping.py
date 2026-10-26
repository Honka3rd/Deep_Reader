"""Exact SRMAP contracts. Synthetic evidence is not real-corpus acceptance."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import unittest

from document_structure import source_review_blocks as mapping
from document_structure.reading_content import original_blocks, needs_review
from scripts.test_ocr_cleanup_validation import fixture
from shared.task_unit_model import TaskUnit, TaskUnitContentBlock


def mapped_document(text="Alpha paragraph.\n\nBeta paragraph.", *, units=True):
    source, _, document = fixture(text)
    section = document.chapters[0].sections[0]
    unit = TaskUnit(unit_id="u1", title=None, container_title=None, content=text,
                    source_section_ids=[section.section_id], is_fallback_generated=False,
                    parent_section_id=section.section_id)
    return replace(document, chapters=[replace(document.chapters[0], sections=[
        replace(section, task_units=[unit] if units else [])])], parse_provenance={
            **document.parse_provenance, "extraction": source.identity.record(text),
            "source_page_spans": [{"page_index": 0, "char_start": 0, "char_end": len(text)}],
            "source_review": {"source_sha256": source.identity.source_sha256, "regions": []}})


def with_regions(document, *regions):
    provenance = deepcopy(document.parse_provenance)
    provenance.pop("source_review_mapping", None)
    provenance["source_review"]["regions"] = list(regions)
    return replace(document, parse_provenance=provenance)


def exact_region(document, key, start, end):
    text = document.raw_text
    return {"id": key, "page_index": 0, "raw_candidate": text[start:end],
            "canonical_span": {"char_start": start, "char_end": end,
                               "text_sha256": hashlib.sha256(text.encode()).hexdigest()}}


def blank_document():
    document = with_regions(mapped_document(), {"id": "r", "page_index": 1, "raw_candidate": "",
                                               "blank_render_confirmed": True})
    document.parse_provenance["source_page_spans"].append({"page_index": 1,
        "char_start": len(document.raw_text), "char_end": len(document.raw_text)})
    return document


class SourceReviewMappingTests(unittest.TestCase):
    def test_exact_many_to_many_and_partial_block_intersection(self):
        document = mapped_document()
        first, second = [b.block_id for _, _, _, b in original_blocks(document)]
        document = with_regions(document,
            exact_region(document, "first", 0, 16),
            exact_region(document, "inside", 2, 8),
            exact_region(document, "across", 10, 25))
        result = mapping.region_block_mapping(document)
        self.assertEqual(result["first"]["block_ids"], [first])
        self.assertEqual(result["inside"]["block_ids"], [first])
        self.assertEqual(result["across"]["block_ids"], [first, second])
        for entry in result.values():
            self.assertEqual(entry["mapping_status"], "complete")
            self.assertEqual(entry["body_status"], "linked")
            self.assertTrue(entry["mapping_reason"])

    def test_verified_region_with_missing_block_coverage_is_partial(self):
        document = mapped_document()
        section = document.chapters[0].sections[0]
        unit = section.task_units[0]
        first = replace(section, content=document.raw_text[:18], char_end=18,
                        task_units=[replace(unit, content=document.raw_text[:18])])
        second = replace(section, section_id="s1", content=document.raw_text[18:],
                         char_start=18, task_units=[])
        document = replace(document, chapters=[replace(document.chapters[0], sections=[
            first, second])])
        block = original_blocks(document)[0][3]
        document = with_regions(document, exact_region(document, "r", 0, len(document.raw_text)))
        result = mapping.region_block_mapping(document)["r"]
        self.assertEqual(result["mapping_status"], "partial")
        self.assertEqual(result["body_status"], "linked")
        self.assertEqual(result["block_ids"], [block.block_id])

    def test_verified_quote_before_units_is_pending_not_excluded(self):
        document = mapped_document(units=False)
        document = with_regions(document, {"id": "r", "page_index": 0, "raw_candidate": "Alpha"})
        result = mapping.region_block_mapping(document)["r"]
        self.assertEqual(result["body_status"], "pending")
        self.assertEqual(result["block_ids"], [])
        self.assertTrue(needs_review(document))

    def test_duplicate_quote_never_guesses_but_verified_page_disambiguates(self):
        document = mapped_document("Repeated.\n\nRepeated.")
        document = with_regions(document, {"id": "r", "page_index": 0, "raw_candidate": "Repeated."})
        result = mapping.region_block_mapping(document)["r"]
        self.assertEqual(result["body_status"], "unknown")
        self.assertEqual(result["block_ids"], [])
        document.parse_provenance["source_page_spans"] = [
            {"page_index": 0, "char_start": 0, "char_end": 9},
            {"page_index": 1, "char_start": 11, "char_end": 20}]
        self.assertEqual(mapping.region_block_mapping(document)["r"]["block_ids"],
                         [original_blocks(document)[0][3].block_id])

    def test_quotes_are_exact_without_normalization_or_cross_page_search(self):
        document = mapped_document("Cafe\u0301\r\nAlpha\n\nBeta")
        for quote in ("Caf\u00e9", "cafe\u0301", "Alpha\nBeta", "Alpha\n\nBeta"):
            with self.subTest(quote=quote):
                candidate = with_regions(document, {"id": "r", "page_index": 0, "raw_candidate": quote})
                candidate.parse_provenance["source_page_spans"] = [
                    {"page_index": 0, "char_start": 0, "char_end": 12},
                    {"page_index": 1, "char_start": 14, "char_end": len(document.raw_text)}]
                self.assertEqual(mapping.region_block_mapping(candidate)["r"]["body_status"], "unknown")

    def test_exclusion_requires_blank_evidence_and_matching_identity_hash(self):
        document = blank_document()
        for quote in ("", "Rejected OCR candidate"):
            with self.subTest(quote=quote):
                candidate = with_regions(document, {"id": "r", "page_index": 1, "raw_candidate": quote,
                                                     "blank_render_confirmed": True})
                before = deepcopy(candidate.to_dict())
                self.assertEqual(mapping.region_block_mapping(candidate)["r"]["body_status"], "unknown")
                self.assertTrue(mapping.needs_review_from_provenance(candidate.parse_provenance))
                self.assertEqual(candidate.to_dict(), before)
                candidate = mapping.persist_region_mapping(candidate)
                result = mapping.region_block_mapping(candidate)["r"]
                self.assertEqual(result["body_status"], "unknown" if quote else "excluded")
                self.assertEqual(needs_review(candidate), bool(quote))
                self.assertEqual(mapping.needs_review_from_provenance(candidate.parse_provenance), bool(quote))
                self.assertFalse(candidate.parse_provenance.get("reading_content"))
                for fault in ("source", "canonical", "missing_identity", "missing_page", "duplicate_page"):
                    with self.subTest(fault=fault):
                        broken = deepcopy(candidate)
                        provenance = broken.parse_provenance
                        if fault == "source":
                            provenance["source_review"]["source_sha256"] = "0" * 64
                        elif fault == "canonical":
                            provenance["extraction"]["text_sha256"] = "0" * 64
                        elif fault == "missing_identity":
                            provenance["extraction"].pop("identity")
                        elif fault == "missing_page":
                            provenance["source_page_spans"] = []
                        else:
                            provenance["source_page_spans"].append(provenance["source_page_spans"][-1])
                        self.assertEqual(mapping.region_block_mapping(broken)["r"]["body_status"], "unknown")
                        self.assertTrue(needs_review(broken))

    def test_empty_or_absent_quote_on_nonempty_page_is_unknown(self):
        for quote in ("", "absent candidate"):
            document = with_regions(mapped_document(), {"id": "r", "page_index": 0, "raw_candidate": quote})
            self.assertEqual(mapping.region_block_mapping(document)["r"]["body_status"], "unknown")
            self.assertTrue(needs_review(document))

    def test_blank_exclusion_requires_valid_empty_page_proof(self):
        document = blank_document()
        self.assertTrue(mapping.needs_review_from_provenance(document.parse_provenance))
        document = mapping.persist_region_mapping(document)
        self.assertEqual(mapping.region_block_mapping(document)["r"]["body_status"], "excluded")
        self.assertFalse(mapping.needs_review_from_provenance(document.parse_provenance))
        for fault in ("source", "incomplete", "missing_length", "out_of_range", "reversed",
                      "negative", "boolean_offset", "canonical_span", "nonempty_page"):
            with self.subTest(fault=fault):
                broken = deepcopy(document)
                provenance = broken.parse_provenance
                page = provenance["source_page_spans"][-1]
                if fault == "source":
                    provenance["source_review"]["source_sha256"] = "0" * 64
                elif fault == "incomplete":
                    provenance["extraction"]["completion"] = "partial"
                elif fault == "missing_length":
                    provenance["extraction"].pop("text_length")
                elif fault == "out_of_range":
                    page["char_end"] = len(document.raw_text) + 1
                elif fault == "reversed":
                    page["char_start"], page["char_end"] = 20, 5
                elif fault == "negative":
                    page["char_start"] = -1
                elif fault == "boolean_offset":
                    page["char_start"] = False
                elif fault == "nonempty_page":
                    page["char_start"] = 0
                else:
                    provenance["source_review"]["regions"][0]["canonical_span"] = {
                        "char_start": 0, "char_end": 5, "text_sha256": "0" * 64}
                self.assertEqual(mapping.region_block_mapping(broken)["r"]["body_status"], "unknown")
                self.assertTrue(mapping.needs_review_from_provenance(provenance))
                rebuilt = mapping.persist_region_mapping(broken)
                self.assertEqual(mapping.region_block_mapping(rebuilt)["r"]["body_status"], "unknown")
                self.assertTrue(mapping.needs_review_from_provenance(rebuilt.parse_provenance))

    def test_missing_or_unverified_blank_quality_never_excludes(self):
        for quality in (None, {}, [], False, 1, "true"):
            with self.subTest(quality=quality):
                document = mapping.persist_region_mapping(blank_document())
                region = document.parse_provenance["source_review"]["regions"][0]
                if quality is None:
                    region.pop("blank_render_confirmed")
                else:
                    region["blank_render_confirmed"] = quality
                before = deepcopy(document.to_dict())
                self.assertTrue(needs_review(document))
                self.assertEqual(mapping.region_block_mapping(document)["r"]["body_status"], "unknown")
                self.assertEqual(document.to_dict(), before)
                rebuilt = mapping.persist_region_mapping(document)
                self.assertTrue(needs_review(rebuilt))

    def test_omission_gaps_remain_visible_without_guessed_links(self):
        for reason in (None, "page_quality_rejected", "recognition_uncertain", "recognition_empty",
                       "no_text", "execution_failed"):
            for quote in ("", "Important missing text", "Alpha"):
                for empty_page in (False, True):
                    with self.subTest(reason=reason, quote=quote, empty_page=empty_page):
                        document = blank_document()
                        region = document.parse_provenance["source_review"]["regions"][0]
                        region.pop("blank_render_confirmed")
                        region.update(reason=reason, raw_candidate=quote, body_inclusion="excluded",
                                      page_index=1 if empty_page else 0)
                        for candidate in (document, mapping.persist_region_mapping(document)):
                            before = deepcopy(candidate.to_dict())
                            entry = mapping.region_block_mapping(candidate)["r"]
                            self.assertEqual(entry["body_status"], "unknown")
                            self.assertEqual(entry["mapping_status"], "unavailable")
                            self.assertEqual(entry["block_ids"], [])
                            self.assertTrue(needs_review(candidate))
                            self.assertEqual(candidate.to_dict(), before)

    def test_included_uncertain_region_uses_supplied_span_even_with_duplicate_text(self):
        from doc_loaders.ocr_review import bind_review_spans, page_review_regions
        from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
        for units in (False, True):
            with self.subTest(units=units):
                document = mapped_document("Repeated.\n\nRepeated.", units=units)
                text = document.raw_text
                records = page_review_regions(0, text, {"selected_passed": True, "regions": [
                    {"accepted": False, "char_start": 11, "char_end": 20, "bbox": [0, 0, 20, 20]}
                ]}, [0, 0, 100, 100], 300)
                region, = bind_review_spans(records, text, [PdfPageTextBoundary(
                    page_index=0, char_start=0, char_end=len(text), text=text)])
                document = with_regions(document, region)
                before = deepcopy(document.to_dict())
                entry = mapping.region_block_mapping(document)[region["id"]]
                self.assertEqual(entry["body_status"], "linked" if units else "pending")
                expected = [original_blocks(document)[1][3].block_id] if units else []
                self.assertEqual(entry["block_ids"], expected)
                self.assertTrue(needs_review(document))
                self.assertEqual(document.to_dict(), before)
                staged = mapping.persist_region_mapping(document)
                self.assertEqual(mapping.region_block_mapping(staged)[region["id"]], entry)

    def test_natural_empty_recognition_and_unmarked_legacy_gaps_remain_visible(self):
        from doc_loaders.ocr_review import page_review_regions
        natural, = page_review_regions(1, "", {"selected_passed": False}, [0, 0, 100, 100], 300)
        self.assertEqual(natural["reason"], "recognition_empty")
        self.assertNotIn("canonical_span", natural)
        regions = [natural, {"id": "legacy", "page_index": 1, "raw_candidate": ""},
                   {"id": "quality", "page_index": 1, "raw_candidate": "Missing body",
                    "reason": "page_quality_rejected"}]
        for region in regions:
            with self.subTest(region=region["id"]):
                document = mapping.persist_region_mapping(with_regions(blank_document(), region))
                entry = mapping.region_block_mapping(document)[region["id"]]
                self.assertEqual(entry["body_status"], "unknown")
                self.assertEqual(entry["block_ids"], [])
                self.assertTrue(needs_review(document))

    def test_v1_cached_exclusion_is_recomputed_without_writes(self):
        document = blank_document()
        document.parse_provenance["source_review"]["regions"][0].pop("blank_render_confirmed")
        document.parse_provenance["source_review"]["regions"][0].update(
            raw_candidate="Missing important body", reason="page_quality_rejected", body_inclusion="excluded")
        document = mapping.persist_region_mapping(document)
        cached = document.parse_provenance["source_review_mapping"]
        cached["policy"] = "source-review-blocks-v1"
        cached["regions"]["r"] = {"block_ids": [], "mapping_status": "unavailable",
            "body_status": "excluded", "mapping_reason": "never_in_body"}
        before = deepcopy(document.to_dict())
        for _ in range(2):
            self.assertTrue(mapping.needs_review_from_provenance(document.parse_provenance))
            entry = mapping.region_block_mapping(document)["r"]
            self.assertEqual(entry["body_status"], "unknown")
            self.assertEqual(entry["block_ids"], [])
        self.assertEqual(document.to_dict(), before)
        staged = mapping.persist_region_mapping(document)
        self.assertEqual(staged.parse_provenance["source_review_mapping"]["policy"], "source-review-blocks-v2")
        self.assertTrue(needs_review(staged))

    def test_canonical_span_cannot_override_source_or_text_mismatch(self):
        document = mapped_document()
        document = with_regions(document, exact_region(document, "r", 0, 5))
        for field in ("source_sha256", "text_sha256"):
            broken = deepcopy(document)
            owner = broken.parse_provenance["source_review"] if field == "source_sha256" else broken.parse_provenance["extraction"]
            owner[field] = "0" * 64
            result = mapping.region_block_mapping(broken)["r"]
            self.assertEqual(result["body_status"], "unknown")
            self.assertEqual(result["block_ids"], [])

    def test_read_projection_is_pure_and_persistence_is_explicit(self):
        document = mapped_document()
        document = with_regions(document, exact_region(document, "r", 0, 5))
        before = deepcopy(document.to_dict())
        projected = mapping.region_block_mapping(document)
        self.assertTrue(mapping.needs_review_from_provenance(document.parse_provenance))
        self.assertEqual(document.to_dict(), before)
        persisted = mapping.persist_region_mapping(document)
        self.assertIn("source_review_mapping", persisted.parse_provenance)
        self.assertEqual(document.to_dict(), before)
        self.assertEqual(mapping.region_block_mapping(persisted), projected)
        self.assertEqual(mapping.persist_region_mapping(persisted), persisted)

    def test_persisted_links_rebind_after_unit_identity_replacement(self):
        document = mapped_document()
        document = mapping.persist_region_mapping(with_regions(document, exact_region(document, "r", 0, 5)))
        old = mapping.region_block_mapping(document)["r"]["block_ids"]
        section = document.chapters[0].sections[0]
        updated = replace(document, chapters=[replace(document.chapters[0], sections=[replace(section,
            task_units=[section.task_units[0].with_unit_id("new-epoch-unit")])])])
        before = deepcopy(updated.to_dict())
        new = mapping.region_block_mapping(updated)["r"]["block_ids"]
        self.assertTrue(new)
        self.assertTrue(set(old).isdisjoint(new))
        self.assertEqual(updated.to_dict(), before)
        persisted = mapping.persist_region_mapping(updated)
        self.assertEqual(mapping.region_block_mapping(persisted)["r"]["block_ids"], new)

    def test_stale_cached_links_cannot_override_changed_source_evidence(self):
        document = mapped_document()
        document = mapping.persist_region_mapping(with_regions(document, exact_region(document, "r", 0, 5)))
        document.parse_provenance["source_review"]["source_sha256"] = "0" * 64
        self.assertEqual(mapping.region_block_mapping(document)["r"]["body_status"], "unknown")
        self.assertEqual(mapping.region_block_mapping(document)["r"]["block_ids"], [])

    def test_omitted_candidate_keeps_all_target_and_ancestor_guards(self):
        from section_tasks.reading_target_resolver import ReadingTargetResolver
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
        from section_tasks.artifact_validity import ReadingInteractionValidityPolicy
        candidate = "Excluded margin OCR MUST NOT enter generation"
        document = with_regions(mapped_document(), {"id": "r", "page_index": 1, "raw_candidate": candidate})
        document.parse_provenance["source_page_spans"].append({"page_index": 1,
            "char_start": len(document.raw_text), "char_end": len(document.raw_text)})
        document = mapping.persist_region_mapping(document)
        chapter, section, unit, block = original_blocks(document)[0]
        scopes = [{"target_level": "document"}, {"target_level": "chapter", "chapter_id": chapter.chapter_id},
            {"target_level": "section", "section_id": section.section_id},
            {"target_level": "task_unit", "task_unit_id": unit.unit_id},
            {"target_level": "content_block", "task_unit_id": unit.unit_id, "content_block_id": block.block_id}]
        policy = ReadingInteractionValidityPolicy(interaction_label="test", min_content_chars=0, min_alnum_chars=0)
        for scope in scopes:
            target = ReadingTargetResolver().resolve(document=document, **scope)
            self.assertNotIn(candidate, target.content)
            self.assertTrue(target.source_needs_review)
            for kind in ("analysis", "quiz", "critical_thinking_session", "summary"):
                decision = policy.preflight(ReadingInteractionRequest(target=target, interaction_type=kind))
                self.assertEqual(decision.reason, "source_needs_review")
        document.parse_provenance["source_review"]["source_sha256"] = "0" * 64
        for scope in scopes:
            target = ReadingTargetResolver().resolve(document=document, **scope)
            self.assertTrue(target.source_needs_review)
            self.assertEqual(policy.preflight(ReadingInteractionRequest(
                target=target, interaction_type="analysis")).reason, "source_needs_review")

    def test_corrupt_exclusion_proof_does_not_weaken_guards(self):
        document = mapping.persist_region_mapping(blank_document())
        self.assertFalse(mapping.needs_review_from_provenance(document.parse_provenance))
        for key in ("canonical_sha256", "source_key"):
            broken = deepcopy(document)
            broken.parse_provenance["source_review_mapping"][key] = "0" * 64
            before = deepcopy(broken.to_dict())
            self.assertTrue(mapping.needs_review_from_provenance(broken.parse_provenance))
            self.assertEqual(mapping.region_block_mapping(broken)["r"]["body_status"], "unknown")
            self.assertEqual(broken.to_dict(), before)

    def test_custom_block_coordinates_cannot_link_different_text_or_source(self):
        document = mapped_document()
        section = document.chapters[0].sections[0]
        unit = section.task_units[0]
        source_hash = hashlib.sha256(unit.content.encode()).hexdigest()
        for content, metadata in (
            ("Wrong", {"quote_span_start": 0, "quote_span_end": 5, "source_hash": source_hash}),
            ("Alpha", {"quote_span_start": 6, "quote_span_end": 11, "source_hash": source_hash}),
            ("Alpha", {"quote_span_start": 0, "quote_span_end": 5, "source_hash": "0" * 64}),
        ):
            with self.subTest(content=content, metadata=metadata):
                bad = TaskUnitContentBlock(block_id="untrusted-custom", content=content, metadata=metadata)
                candidate = replace(document, chapters=[replace(document.chapters[0], sections=[
                    replace(section, task_units=[replace(unit, content_blocks=[bad])])])])
                candidate = with_regions(candidate, exact_region(candidate, "r", 0, 5))
                result = mapping.region_block_mapping(candidate)["r"]
                self.assertTrue(result["block_ids"])
                self.assertNotIn("untrusted-custom", result["block_ids"])

    def test_title_heading_suppression_invalidates_cached_block_links(self):
        document = mapped_document("Alpha heading\nAlpha paragraph.\n\nBeta paragraph.")
        document = mapping.persist_region_mapping(with_regions(document, exact_region(document, "r", 0, 5)))
        section = document.chapters[0].sections[0]
        changed = replace(document, chapters=[replace(document.chapters[0], sections=[
            replace(section, title="Alpha heading")])])
        self.assertEqual(mapping.region_block_mapping(changed)["r"]["block_ids"], [])


if __name__ == "__main__":
    unittest.main()
