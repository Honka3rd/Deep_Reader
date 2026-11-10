#!/usr/bin/env python3
"""OCR candidate quality must not infer page language from installed packs."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PdfReader
from doc_loaders.pdf_document_loader import PdfDocumentLoader, OCR_QUALITY_VERSION
from document_preparation.pdf_extraction_identity import identity_for_loader


ENGLISH = "Scientific inquiry requires evidence, careful reasoning, and independent examination of competing explanations. " * 3
CHINESE = "\u570b\u5bcc\u8ad6\u5206\u5de5\u5e02\u5834\u50f9\u683c\u81ea\u7136\u50f9\u683c\u8ca8\u5e63\u8cc7\u672c\u52de\u52d5\u571f\u5730\u79df\u7a05"
GARBAGE = "UAE RG Rl Qo JOGA ol 8 TAN ts SS NS RS ot HEME RRKSERRRERM"


class OcrScriptQualityTests(unittest.TestCase):
    def setUp(self):
        self.loader = PdfDocumentLoader(ocr_language="eng+chi_sim+chi_tra")

    def score(self, text):
        return self.loader._score_ocr_candidate(psm="6", text=text)

    def test_quality_is_independent_of_language_pack(self):
        for text in (ENGLISH, CHINESE * 8, ENGLISH + CHINESE * 8,
                     "2020 1234 5678 2021 2345 6789 2022 3456 7890", GARBAGE):
            scores = [PdfDocumentLoader(ocr_language=language)._score_ocr_candidate(psm="6", text=text)
                      for language in ("eng", "chi_sim", "eng+chi_sim+chi_tra")]
            self.assertEqual(scores[0], scores[1])
            self.assertEqual(scores[1], scores[2])

    def test_readable_english_chinese_mixed_numeric_pass(self):
        for text in (ENGLISH, CHINESE, ENGLISH + CHINESE * 8, "Memory OCR text",
                     "2020 1234 5678 2021 2345 6789 2022 3456 7890"):
            with self.subTest(text=text[:15]):
                candidate = self.score(text)
                self.assertTrue(candidate.passed, candidate.reasons)
                self.assertNotIn("latin_heavy", candidate.reasons)

    def test_empty_short_fragmented_symbol_and_encoding_noise_rejected(self):
        for text in ("", "ABC", GARBAGE, "A B C D E F G H I J " * 5,
                     "!@#$%^&*() " * 20, ENGLISH + "\ufffd", ENGLISH + "\x00"):
            with self.subTest(text=text[:15]):
                self.assertFalse(self.score(text).passed)

    def test_candidate_selection_and_summary_prefer_passed_over_rejected_score(self):
        good = self.score(ENGLISH)
        bad = replace(self.score(GARBAGE), score=good.score + 1)
        candidates = [bad, good, bad, bad]
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                patch.object(self.loader, "_score_ocr_candidate", side_effect=candidates), \
                patch.object(self.loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, "output", "")):
            text, scored = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="fixture", page_index=0,
            )
        self.assertEqual(text, ENGLISH.strip())
        summary = self.loader._summarize_page_ocr_quality(page_index=0, candidates=scored)
        self.assertTrue(summary["selected_passed"])
        self.assertEqual(summary["selected_score"], good.score)
        self.assertEqual(summary["quality_version"], OCR_QUALITY_VERSION)

    def test_cjk_word_spacing_is_not_fragmentation(self):
        for width in (1, 2):
            spaced = " ".join(CHINESE[i:i + width] for i in range(0, len(CHINESE), width))
            for text in ((spaced + " , ") * 8, (spaced + " \u3002 ") * 8,
                         (spaced + " ") * 8 + ENGLISH):
                with self.subTest(width=width, text=text[:20]):
                    for language in ("eng", "chi_sim", "eng+chi_sim+chi_tra"):
                        candidate = PdfDocumentLoader(ocr_language=language)._score_ocr_candidate(psm="6", text=text)
                        self.assertTrue(candidate.passed, candidate)

    def test_cjk_cannot_mask_fragmented_latin_numeric_or_noise(self):
        spaced = " ".join(CHINESE) * 8
        for noise in ("A B C D E F G H I J " * 4, GARBAGE * 3,
                      "1 2 3 4 5 6 7 8 9 0 " * 4,
                      "!@#$%^&*() " * 100, "\ufffd", "\x00"):
            with self.subTest(noise=noise[:20]):
                self.assertFalse(self.score(spaced + " " + noise).passed)

    def test_quality_version_invalidates_previous_policy(self):
        self.assertEqual(OCR_QUALITY_VERSION, "observed-script-v7-unicode-signal")
        with patch("doc_loaders.pdf_extraction_identity.shutil.which", return_value=None):
            for mode in ("fast", "ocr", "auto"):
                current = identity_for_loader(self.loader, source_sha256="a" * 64, mode=mode)
                with patch("doc_loaders.pdf_document_loader.OCR_QUALITY_VERSION", "observed-script-v5-candidate-consensus"):
                    previous = identity_for_loader(self.loader, source_sha256="a" * 64, mode=mode)
                self.assertEqual(current == previous, mode == "fast")

    def test_page_80_all_failed_consensus_retains_requested_phrase(self):
        phrase = "\u88d5\u4e4b\u5e76\u4e0d\u5341\u5206\u5728\u4e4e\u8fd9\u4e9b"
        # Reported incident metrics: PSM 5's Latin-branch score exceeds all CJK scores.
        candidates = [replace(self.score(phrase + CHINESE * 35), psm=psm,
                              passed=False, score=score, cjk_ratio=ratio)
                      for psm, score, ratio in (("default", .617, .880), ("6", .634, .886), ("11", .616, .880))]
        outlier = replace(self.score(ENGLISH), psm="5", passed=False, score=.666,
                          cjk_ratio=.124, latin_ratio=.820)
        candidates.insert(1, outlier)
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                patch.object(self.loader, "_score_ocr_candidate", side_effect=candidates), \
                patch.object(self.loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, "output", "")):
            text, scored = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="page80", page_index=79,
            )
        self.assertIn(phrase, text)
        summary = self.loader._summarize_page_ocr_quality(page_index=79, candidates=scored)
        self.assertEqual(summary["selected_psm"], "6")
        self.assertEqual(summary["selected_score"], .634)
        self.assertFalse(summary["selected_passed"])
        self.assertEqual(summary["candidate_count"], 4)

    def test_consensus_requires_two_supporters_and_two_thirds_of_nonempty(self):
        cjk = replace(self.score(CHINESE * 3), score=.3)
        english = replace(self.score(ENGLISH), score=.9)
        empty = replace(self.score(""), passed=True, score=100)
        for candidates, expected in (([cjk, english, empty, empty], english),
                                     ([cjk, cjk, english, english], english),
                                     ([cjk, cjk, english, empty], cjk),
                                     ([cjk, cjk, cjk, english], cjk)):
            with self.subTest(count=len(candidates)):
                self.assertIs(self.loader._select_ocr_candidate(candidates), expected)

    def test_short_cjk_labels_do_not_override_english(self):
        short = replace(self.score(CHINESE[:4]), passed=True, score=.3)
        english = replace(self.score(ENGLISH), score=.9)
        self.assertIs(self.loader._select_ocr_candidate([short, short, short, english]), english)

    def test_consensus_filter_log_contains_counts_and_rule_without_source_text(self):
        cjk = self.score(CHINESE * 3 + " private-source-sentinel")
        english = self.score(ENGLISH)
        with self.assertLogs("doc_loaders.chinese_ocr_strategy", level="INFO") as captured:
            self.loader._select_ocr_candidate([cjk, cjk, english, self.score("")])
        self.assertEqual(len(captured.output), 1)
        message = captured.output[0]
        self.assertIn("candidate_count=4 nonempty_candidate_count=3 consensus_pool_count=2", message)
        self.assertIn("rule=strong_cjk_consensus", message)
        self.assertNotIn("private-source-sentinel", message)
        self.assertNotIn(CHINESE, message)
        self.assertNotIn(ENGLISH.strip(), message)
        with self.assertNoLogs("doc_loaders.chinese_ocr_strategy", level="INFO"):
            self.loader._select_ocr_candidate([cjk, english])
            self.loader._select_ocr_candidate([cjk, cjk])

    def test_bilingual_consensus_retains_both_scripts_and_ignores_installed_packs(self):
        bilingual = replace(self.score(CHINESE * 8 + ENGLISH), score=.3)
        outlier = replace(self.score(ENGLISH), score=.9)
        self.assertGreaterEqual(bilingual.cjk_ratio, .35)
        candidates = [bilingual, bilingual, bilingual, outlier]
        for language in ("eng", "chi_sim", "eng+chi_sim+chi_tra"):
            selected = PdfDocumentLoader(ocr_language=language)._select_ocr_candidate(candidates)
            self.assertIs(selected, bilingual)
            self.assertIn(CHINESE, selected.text)
            self.assertIn(ENGLISH.strip(), selected.text)

    def test_consensus_keeps_pass_first_ranking_within_observed_script(self):
        passed = replace(self.score(CHINESE * 3), score=.3)
        failed = replace(passed, passed=False, score=.9)
        outlier = replace(self.score(ENGLISH), score=1)
        self.assertIs(self.loader._select_ocr_candidate([failed, passed, failed, outlier]), passed)

    def test_best_nonempty_rejected_candidate_beats_empty_higher_scores(self):
        best = replace(self.score("!"), psm="6", score=-0.5)
        worse = replace(self.score("?"), psm="5", score=-0.8)
        for empty in (self.score(""), replace(self.score(" "), passed=True, score=10)):
            candidates = [empty, worse, best, empty]
            with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                    patch.object(self.loader, "_score_ocr_candidate", side_effect=candidates), \
                    patch.object(self.loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, "output", "")):
                text, scored = self.loader._ocr_image_text_with_quality(
                    image_path=Path("unused.png"), doc_name="fixture", page_index=0,
                )
            self.assertEqual(text, "!")
            summary = self.loader._summarize_page_ocr_quality(page_index=0, candidates=scored)
            self.assertEqual(summary["selected_psm"], "6")
            self.assertEqual(summary["selected_score"], -0.5)
            self.assertFalse(summary["selected_passed"])

    def test_empty_candidates_have_no_selected_text(self):
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                patch.object(self.loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, " \n", "")):
            text, candidates = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="fixture", page_index=0,
            )
        self.assertEqual(text, "")
        self.assertIsNone(self.loader._summarize_page_ocr_quality(page_index=0, candidates=candidates)["selected_psm"])

    def test_directed_nonempty_low_quality_text_keeps_region_diagnostics(self):
        regions = ({"accepted": False, "quality_flags": ["recognition_uncertain"]},)
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=SimpleNamespace(text="!", regions=regions)):
            text, candidates = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="fixture", page_index=0,
            )
        self.assertEqual(text, "!")
        self.assertFalse(candidates[0].passed)
        self.assertEqual(candidates[0].region_evidence, regions)

    def test_typed_geometry_fallback_marks_every_nonempty_generic_candidate(self):
        from doc_loaders.chinese_ocr_strategy import ChineseOcrFallback
        from doc_loaders.ocr_review import page_review_regions
        fallback = ChineseOcrFallback(reason="chinese_layout_coverage_uncertain", bbox=(0, 0, 80, 120))
        outputs = [ENGLISH, "", "!", ENGLISH + "Additional readable words."]
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=fallback), \
                patch.object(self.loader, "_run_process", side_effect=[
                    subprocess.CompletedProcess([], 0, output, "") for output in outputs
                ]) as process:
            text, candidates = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="fixture", page_index=0,
            )
        self.assertEqual(process.call_count, 4)
        for candidate in candidates:
            if not candidate.text:
                self.assertEqual(candidate.region_evidence, ())
                continue
            region, = candidate.region_evidence
            self.assertEqual(region["bbox"], [0, 0, 80, 120])
            self.assertEqual((region["char_start"], region["char_end"]), (0, len(candidate.text)))
            self.assertEqual(region["raw_candidate"], candidate.text)
            self.assertFalse(region["accepted"])
            self.assertIn(fallback.reason, region["quality_flags"])
        summary = self.loader._summarize_page_ocr_quality(page_index=0, candidates=candidates)
        self.assertTrue(summary["selected_passed"])
        review, = page_review_regions(0, text, summary, [0, 0, 80, 120], 200)
        self.assertEqual(review["raw_candidate"], text)
        self.assertEqual(review["body_inclusion"], "included")

    def test_ordinary_horizontal_none_does_not_invent_geometry_warning(self):
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=None), \
                patch.object(self.loader, "_run_process", return_value=subprocess.CompletedProcess([], 0, ENGLISH, "")):
            text, candidates = self.loader._ocr_image_text_with_quality(
                image_path=Path("unused.png"), doc_name="fixture", page_index=0,
            )
        self.assertEqual(text, ENGLISH.strip())
        self.assertTrue(all(not candidate.region_evidence for candidate in candidates))

    def test_reported_english_page_ten_when_local_fixture_available(self):
        fixtures = list((Path(__file__).resolve().parents[1] / "data/raw").glob("20250515-*.pdf"))
        if len(fixtures) != 1:
            self.skipTest("private 20250515 fixture not available uniquely")
        text = PdfReader(fixtures[0]).pages[9].extract_text()
        self.assertGreater(len(text), 1000)
        quality = self.score(text)
        self.assertGreater(quality.latin_ratio, 0.8)
        self.assertTrue(quality.passed, quality)


if __name__ == "__main__":
    unittest.main()
