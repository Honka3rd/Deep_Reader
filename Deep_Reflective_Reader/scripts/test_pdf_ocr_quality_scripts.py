#!/usr/bin/env python3
"""OCR candidate quality must not infer page language from installed packs."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import unittest
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
        with patch.object(self.loader, "_score_ocr_candidate", side_effect=candidates), \
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
        self.assertEqual(OCR_QUALITY_VERSION, "observed-script-v3")
        with patch("doc_loaders.pdf_extraction_identity.shutil.which", return_value=None):
            for mode in ("fast", "ocr", "auto"):
                current = identity_for_loader(self.loader, source_sha256="a" * 64, mode=mode)
                with patch("doc_loaders.pdf_document_loader.OCR_QUALITY_VERSION", "observed-script-v2"):
                    previous = identity_for_loader(self.loader, source_sha256="a" * 64, mode=mode)
                self.assertEqual(current == previous, mode == "fast")

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
