"""Negative harness guards with labelled synthetic fixtures, not OCR acceptance."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.accept_ocr_preservation import docker_command, private_json, verify_body, paragraph_evidence
from scripts.test_source_review_mapping import mapped_document, exact_region, with_regions


def fixture(text="Alpha paragraph.\n\nBeta paragraph."):
    document = mapped_document(text)
    region = exact_region(document, "warning", 0, len(text))
    region.update(body_inclusion="included", reason="recognition_uncertain")
    document = with_regions(document, region)
    identity = SimpleNamespace(**document.parse_provenance["extraction"]["identity"])
    page = SimpleNamespace(page_index=0, char_start=0, char_end=len(text), text=text,
                           source_sha256=identity.source_sha256)
    artifact = SimpleNamespace(result=SimpleNamespace(canonical_text=text, identity=identity),
                               page_boundaries=[page])
    return document, artifact


class HarnessTests(unittest.TestCase):
    def test_exact_mapping_and_read_purity(self):
        document, artifact = fixture()
        before = deepcopy(document.to_dict())
        result = verify_body(document, artifact, "Alpha paragraph")
        self.assertTrue(result["natural_positive_mapping"])
        self.assertEqual(result["review_regions"][0]["mapping_status"], "complete")
        self.assertEqual(document.to_dict(), before)

    def test_existing_nfkc_display_normalization_keeps_exact_source(self):
        document, artifact = fixture("Alpha\uff0c paragraph.")
        result = verify_body(document, artifact, "Alpha")
        self.assertTrue(result["natural_positive_mapping"])
        change, = result["display_normalizations"][0]["nonspace_changes"]
        self.assertEqual(change["source_codepoints"], ["U+FF0C"])
        self.assertEqual(change["display_codepoints"], ["U+002C"])
        self.assertTrue(result["complete_nonspace_source_coverage"])

    def test_missing_passage_is_failure(self):
        document, artifact = fixture()
        observations = {}
        with self.assertRaisesRegex(AssertionError, "expected_phrase_missing"):
            verify_body(document, artifact, "Missing passage", observations)
        self.assertFalse(any(observations["phrase_checks"].values()))
        self.assertTrue(observations["natural_positive_mapping"])

    def test_wrong_canonical_span_is_failure(self):
        document, artifact = fixture()
        document.parse_provenance["source_review"]["regions"][0]["canonical_span"]["char_start"] = 1
        with self.assertRaisesRegex(AssertionError, "review_span_mismatch"):
            verify_body(document, artifact, "Alpha")

    def test_forged_extraction_identity_cannot_link(self):
        document, artifact = fixture()
        document.parse_provenance["source_review"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(AssertionError, "review_wrong_body_blocks"):
            verify_body(document, artifact, "Alpha")

    def test_missing_unit_text_is_failure(self):
        document, artifact = fixture()
        section = document.chapters[0].sections[0]
        document = replace(document, chapters=[replace(document.chapters[0], sections=[
            replace(section, task_units=[replace(section.task_units[0], content="Alpha paragraph.")])])])
        with self.assertRaisesRegex(AssertionError, "task_units_lost_text"):
            verify_body(document, artifact, "Alpha")

    def test_omitted_leading_body_or_heading_is_failure(self):
        from document_structure.reading_content import original_blocks

        document, artifact = fixture()
        blocks = original_blocks(document)
        with patch("document_structure.reading_content.original_blocks", return_value=blocks[1:]):
            with self.assertRaisesRegex(AssertionError, "blocks_omit_source_before_range"):
                verify_body(document, artifact, "Beta")

    def test_overlapping_source_ranges_are_failure(self):
        from document_structure.reading_content import original_blocks

        document, artifact = fixture()
        blocks = original_blocks(document)
        with patch("document_structure.reading_content.original_blocks", return_value=[blocks[0], *blocks]):
            with self.assertRaisesRegex(AssertionError, "block_source_ranges_overlap_or_reorder"):
                verify_body(document, artifact, "Alpha")

    def test_no_warning_is_explicit_coverage_gap(self):
        document, artifact = fixture()
        document = with_regions(document)
        result = verify_body(document, artifact, "Alpha")
        self.assertEqual(result["natural_review_count"], 0)
        self.assertFalse(result["natural_positive_mapping"])

    def test_paragraph_requires_middle_and_end_in_every_layer(self):
        layers = {name: "Start middle ending." for name in
                  ("canonical", "common_sections", "task_units", "reading_blocks")}
        anchors = (("middle", ("middle",)), ("ending", ("ending", "variant")))
        self.assertTrue(paragraph_evidence(layers, "Start", anchors)["complete"])
        layers["reading_blocks"] = "Start ending."
        self.assertFalse(paragraph_evidence(layers, "Start", anchors)["complete"])

    def test_paragraph_cannot_take_end_anchor_from_next_paragraph(self):
        layers = {name: "Start middle.\n\nUnrelated ending." for name in
                  ("canonical", "common_sections", "task_units", "reading_blocks")}
        anchors = (("middle", ("middle",)), ("ending", ("ending",)))
        result = paragraph_evidence(layers, "Start", anchors)
        self.assertFalse(result["complete"])
        self.assertFalse(result["anchors_in_same_paragraph"])

    def test_whole_paragraph_checks_text_between_anchors(self):
        layers = {name: "Start retained middle tail ending." for name in
                  ("canonical", "common_sections", "task_units", "reading_blocks")}
        layers["task_units"] = "Start middle ending."
        anchors = (("middle", ("middle",)), ("ending", ("ending",)))
        result = paragraph_evidence(layers, "Start", anchors)
        self.assertTrue(result["layers"]["task_units"]["all_anchors_found"])
        self.assertFalse(result["complete"])

    def test_report_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            private_json(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                private_json(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_container_is_offline_with_readonly_source(self):
        command = docker_command("sha256:image", "/temporary/output", "test-worker")
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        writable = [m for m in mounts if not m.endswith(",readonly")]
        self.assertEqual(writable, ["type=bind,src=/temporary/output,dst=/output"])
        self.assertTrue(any("dst=/app/Deep_Reflective_Reader,readonly" in m for m in mounts))
        self.assertFalse(any("DSN" in value or "env-file" in value for value in command))


if __name__ == "__main__":
    unittest.main()
