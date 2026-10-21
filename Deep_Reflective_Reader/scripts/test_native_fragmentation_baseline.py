"""NATIVE-01: known-defect contracts plus independent safety controls.

Expected failures express desired behavior, not approval of existing bugs.
Remove each decorator when its owning follow-up implements the contract.
"""

import unittest
import json
from types import SimpleNamespace

from diagnose_native_fragmentation import coverage_report, inspect_text
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.section_split_plan import SplitBoundaryInstruction
from document_structure.section_splitter import CommonSectionSplitter
from document_structure.toc_detector import TableOfContentsDetector
from language.language_code import LanguageCode
from section_tasks.artifact_validity import _semantic_content_insufficient_reason


# Authored synthetic prose, not copied from either private PDF.
PROSE = ("\u6751\u6c11\u8ba8\u8bba\u516c\u5171\u79e9\u5e8f\u5982\u4f55\u5f62\u6210\u3002"
         "\u5927\u5bb6\u8c03\u6574\u7a0b\u5e8f\u5e76\u8bb0\u5f55\u53d8\u5316\u3002")
FRAGMENTED = "\n".join(PROSE)


class NativeFragmentationBaselineTests(unittest.TestCase):
    def setUp(self):
        self.sections = CommonSectionSplitter().split(FRAGMENTED, LanguageCode.ZH)

    def test_word_internal_character_is_not_heading_native03(self):
        self.assertFalse(any(section.title == "\u5e8f" for section in self.sections))

    def test_source_prefix_is_preserved_native03(self):
        self.assertEqual(coverage_report(FRAGMENTED, self.sections)["missing_nonspace_chars"], 0)

    def test_fragmented_prose_is_not_toc_native03(self):
        self.assertFalse(TableOfContentsDetector().detect(FRAGMENTED).detected)

    def test_fragmented_readable_prose_passes_guard_interact01(self):
        self.assertIsNone(_semantic_content_insufficient_reason(
            FRAGMENTED, interaction_label="analysis"))

    def test_cross_line_anchor_maps_to_source_native05(self):
        from document_structure.structure_plan_contract import (
            source_preview, validate_candidate, resolve_candidate_positions, EnhancementIntent)
        preview = source_preview(FRAGMENTED, 1000)
        payload = {"schema_version": "structure_plan_v2", "intent": "restore_source",
            "outcome": "single_body", "boundaries": [{"title": None, "level": 1,
                "origin": "implicit", "role": "main_body", "occurrence": 1,
                "segment_id": preview["segments"][0]["segment_id"], "anchor": PROSE[:6]}]}
        plan = validate_candidate(json.dumps(payload), preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        self.assertEqual(resolve_candidate_positions(plan, preview), [0])

    def test_coverage_oracle_detects_gap_overlap_and_invalid_range(self):
        spans = [SimpleNamespace(char_start=2, char_end=5),
                 SimpleNamespace(char_start=3, char_end=5),
                 SimpleNamespace(char_start=-1, char_end=1)]
        report = coverage_report("a bcd", spans)
        self.assertEqual(report["missing_nonspace_chars"], 1)
        self.assertEqual(report["overlapping_nonspace_chars"], 2)
        self.assertEqual(report["invalid_ranges"], 1)

    def test_normal_prose_is_covered_and_readable(self):
        report = inspect_text(PROSE, LanguageCode.ZH)
        self.assertEqual(report["coverage"]["missing_nonspace_chars"], 0)
        self.assertIsNone(report["analysis_guard_reason"])

    def test_rejected_toc_candidate_is_not_usable(self):
        self.assertFalse(TableOfContentsDetector().detect(FRAGMENTED).usable)

    def test_genuine_standalone_preface_remains_a_heading(self):
        splitter = CommonSectionSplitter()
        text = "\u5e8f\n" + PROSE
        headings = splitter._detect_strong_headings(
            splitter._build_line_infos(text), LanguageCode.ZH)
        self.assertEqual(headings[0].title, "\u5e8f")

    def test_repeated_junk_remains_rejected(self):
        self.assertIsNotNone(_semantic_content_insufficient_reason(
            "noise " * 30, interaction_label="analysis"))


if __name__ == "__main__":
    unittest.main()
