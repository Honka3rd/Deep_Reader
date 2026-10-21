#!/usr/bin/env python3
"""Full OCR isolation, atomicity and explicit failures without real engines."""
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pypdf import PageObject, PdfWriter
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_execution_pass import PdfBudgetExceeded
from doc_loaders.pdf_extraction_contract import PdfExtractionError
from doc_loaders.document_load_errors import RawTextOcrFailedError


class FullOcrTests(unittest.TestCase):
    def setUp(self):
        strict = patch.dict(os.environ,{"DEEP_READER_OCR_REVIEW_ENABLED":"0"})
        strict.start()
        self.addCleanup(strict.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.loader = PdfDocumentLoader(base_dir=self.temp.name, ocr_enabled=False, ocr_language="eng")
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=100, height=100)
        self.path = Path(self.temp.name) / "sample.pdf"
        writer.write(self.path)
        self.source = self.path.read_bytes()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.forbidden = []
        for name in ("_inspect_page", "_assess_page", "_load_canonical_pages", "_load_pages_with_ocr",
                     "load_page_layout_evidence"):
            self.forbidden.append(self.stack.enter_context(patch.object(
                self.loader, name, side_effect=AssertionError(name),
            )))
        self.forbidden.append(self.stack.enter_context(patch.object(
            PageObject, "extract_text", side_effect=AssertionError("native extraction"),
        )))
        self.addCleanup(lambda: [mock.assert_not_called() for mock in self.forbidden])
        self.stack.enter_context(patch("doc_loaders.pdf_full_ocr_extraction.shutil.which", return_value="engine"))
        self.process = self.stack.enter_context(patch.object(self.loader, "_run_process",
            return_value=subprocess.CompletedProcess([], 0, "List of languages:\neng\n", "")))
        self.render = self.stack.enter_context(patch.object(self.loader, "_render_page_images", side_effect=self.render_page))
        self.blank = self.stack.enter_context(patch.object(self.loader, "_render_is_uniform", return_value=False))
        self.recognize = self.stack.enter_context(patch.object(self.loader, "_ocr_image_text_with_quality", side_effect=self.recognize_page))

    def render_page(self, *, file_path, page_index, output_dir):
        self.assertEqual(file_path.read_bytes(), self.source)
        image = output_dir / f"page{page_index}.png"
        image.write_bytes(b"fixture")
        return [image]

    def recognize_page(self, *, page_index, **kwargs):
        text = f"Recognized page {page_index} contains useful independent evidence and reasoning."
        return text, [self.loader._score_ocr_candidate(psm="6", text=text)]

    def extract(self):
        return self.loader.extract("sample", mode="ocr")

    def test_all_pages_ocr_even_when_legacy_ocr_disabled(self):
        result = self.extract()
        self.assertEqual(self.render.call_count, 3)
        self.assertEqual(self.recognize.call_count, 3)
        self.assertEqual(result.effective_mode, "ocr")
        for span in result.boundary_evidence.boundaries:
            self.assertEqual(result.text[span.char_start:span.char_end], span.text)
        self.assertEqual(len(result.layout_provider()), 3)
        self.assertFalse(result.outline_provider().usable)
        self.assertEqual(self.render.call_count, 3)

    def test_blank_page_preserved_and_all_blank_rejected(self):
        self.blank.side_effect = [False, True, False]
        result = self.extract()
        self.assertEqual(result.empty_page_indexes, (1,))
        self.assertEqual(self.recognize.call_count, 2)
        self.blank.side_effect = None
        self.blank.return_value = True
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason, "no_text")

    def test_missing_engine_or_language_is_explicit(self):
        with patch("doc_loaders.pdf_full_ocr_extraction.shutil.which", return_value=None):
            with self.assertRaises(PdfExtractionError) as caught:
                self.extract()
            self.assertEqual(caught.exception.reason, "engine_unavailable")
        self.process.return_value = subprocess.CompletedProcess([], 0, "fra\n", "")
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason, "engine_unavailable")
        self.render.assert_not_called()

    def test_page_limit_fails_before_render(self):
        self.loader.ocr_page_limit = 2
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason, "budget_exceeded")
        self.render.assert_not_called()

    def test_mid_document_quality_failure_never_publishes_partial_result(self):
        self.recognize.side_effect = [self.recognize_page(page_index=0), ("", [])]
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason, "quality_failed")
        self.assertEqual(self.render.call_count, 2)
        self.assertFalse(self.loader.last_ocr_pages)

    def test_budget_failure_is_typed_without_fallback(self):
        self.render.side_effect = PdfBudgetExceeded("render_attempts")
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason, "budget_exceeded")
        self.recognize.assert_not_called()

    def test_snapshot_survives_source_change(self):
        def recognize(**kwargs):
            self.path.write_bytes(b"changed source")
            return self.recognize_page(**kwargs)
        self.recognize.side_effect = recognize
        result = self.extract()
        self.assertEqual(len(result.pages), 3)
        self.assertFalse(result.outline_provider().usable)

    def test_chinese_rejection_exposes_page_region_without_source(self):
        self.recognize.side_effect = [self.recognize_page(page_index=0),
            RawTextOcrFailedError("private name", "chinese_vertical_region_quality_failed:page=2:region=18:private source")]
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason,"quality_failed")
        self.assertIn("page 2",str(caught.exception))
        self.assertIn("region=18",str(caught.exception))
        self.assertNotIn("private",str(caught.exception))

    def test_unknown_execution_error_is_not_a_quality_warning(self):
        self.recognize.side_effect = RawTextOcrFailedError("private name","private process output")
        with self.assertRaises(PdfExtractionError) as caught:
            self.extract()
        self.assertEqual(caught.exception.reason,"extraction_failed")
        self.assertIn("page 1",str(caught.exception))
        self.assertNotIn("private",str(caught.exception))

    def test_partial_review_preserves_later_pages_and_unresolved_source(self):
        from PIL import Image
        def render(**kwargs):
            path = kwargs["output_dir"]/f"page-{kwargs['page_index']}.png"
            Image.new("RGB",(100,100),"white").save(path)
            return [path]
        self.render.side_effect = render
        self.recognize.side_effect = [self.recognize_page(page_index=0),("",[]),self.recognize_page(page_index=2)]
        with patch.dict(os.environ,{"DEEP_READER_OCR_REVIEW_ENABLED":"1"}):
            result = self.extract()
        self.assertEqual(len(result.pages),3)
        self.assertTrue(result.pages[2])
        review = result.provenance["source_review"]
        self.assertEqual(review["disposition"],"needs_review")
        self.assertEqual(review["regions"][0]["page_index"],1)
        self.assertEqual(review["regions"][0]["bbox"],[0,0,100,100])
        self.assertEqual(result.pages[1],"")

    def test_partial_review_all_bad_still_fails(self):
        from PIL import Image
        def render(**kwargs):
            path = kwargs["output_dir"]/"page.png"
            Image.new("RGB",(100,100),"white").save(path)
            return [path]
        self.render.side_effect = render
        self.recognize.side_effect = lambda **kwargs: ("",[])
        with patch.dict(os.environ,{"DEEP_READER_OCR_REVIEW_ENABLED":"1"}):
            with self.assertRaises(PdfExtractionError) as caught:
                self.extract()
        self.assertEqual(caught.exception.reason,"quality_failed")
        self.assertEqual(self.render.call_count,3)


if __name__ == "__main__":
    unittest.main()
