"""Typed geometry uncertainty survives generic OCR and canonical assembly."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.chinese_ocr_strategy import ChineseOcrFallback
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_contract import PdfExtractionError
import test_pdf_full_ocr_extraction as full_tests
import test_pdf_raw_recovery as raw_tests


TEXT = "Recognized text retains a complete paragraph despite uncertain page geometry."
FALLBACK = ChineseOcrFallback(reason="chinese_layout_coverage_uncertain", bbox=(0, 0, 30, 40))


class GeometryFallbackTests(unittest.TestCase):
    def assert_review(self, result, page_indexes):
        self.assertIn("source_needs_review", result.warnings)
        regions = result.provenance["source_review"]["regions"]
        self.assertEqual([region["page_index"] for region in regions], page_indexes)
        for region in regions:
            self.assertEqual(region["raw_candidate"], TEXT)
            self.assertEqual(region["body_inclusion"], "included")
            self.assertEqual(region["bbox"], list(FALLBACK.bbox))
            self.assertNotIn("page_span", region)
            span = region["canonical_span"]
            self.assertEqual(result.text[span["char_start"]:span["char_end"]], TEXT)
            self.assertEqual(span["text_sha256"], hashlib.sha256(result.text.encode()).hexdigest())
        for summary in result.page_quality:
            self.assertTrue(summary["selected_passed"])
            self.assertIn(FALLBACK.reason, summary["regions"][0]["quality_flags"])

    def test_auto_binds_fallback_warning_after_native_prefix(self):
        fixture = raw_tests.PdfRawRecoveryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.write_pages("native", "scan")
        with ExitStack() as stack:
            _, recognize = fixture.fake_ocr(stack)
            recognize.side_effect = lambda **kw: PdfDocumentLoader._ocr_image_text_with_quality(fixture.loader, **kw)
            stack.enter_context(patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=FALLBACK))
            process = stack.enter_context(patch.object(fixture.loader, "_run_process",
                return_value=subprocess.CompletedProcess([], 0, TEXT, "")))
            result = fixture.loader.extract("sample", mode="auto")
        self.assertEqual(process.call_count, 4)
        self.assertGreater(result.provenance["source_review"]["regions"][0]["canonical_span"]["char_start"], 0)
        self.assert_review(result, [1])

    def test_full_ocr_binds_passing_generic_candidates_as_uncertain(self):
        fixture = full_tests.FullOcrTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.recognize.side_effect = lambda **kw: PdfDocumentLoader._ocr_image_text_with_quality(fixture.loader, **kw)
        fixture.process.side_effect = lambda command, **kw: subprocess.CompletedProcess(
            command, 0, "eng\n" if "--list-langs" in command else TEXT, "")
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=FALLBACK):
            result = fixture.extract()
        self.assertEqual(fixture.process.call_count, 13)
        self.assert_review(result, [0, 1, 2])

    def test_fallback_does_not_convert_process_failure_to_review(self):
        fixture = full_tests.FullOcrTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.recognize.side_effect = lambda **kw: PdfDocumentLoader._ocr_image_text_with_quality(fixture.loader, **kw)
        fixture.process.side_effect = [subprocess.CompletedProcess([], 0, "eng\n", ""),
                                       subprocess.CompletedProcess([], 1, "", "private-engine-error")]
        with patch("doc_loaders.chinese_ocr_strategy.recognize", return_value=FALLBACK):
            with self.assertRaises(PdfExtractionError) as caught:
                fixture.extract()
        self.assertEqual(caught.exception.reason, "extraction_failed")
        self.assertNotIn("private-engine-error", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
