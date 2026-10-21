#!/usr/bin/env python3
"""Fast extraction uses native text only, including deferred evidence."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PageObject, PdfWriter
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_contract import PdfExtractionError, PdfExtractionFailureReason
from test_pdf_page_inspection import add_image, page_fixture


NATIVE = b"BT /F1 12 Tf 10 70 Td (A readable native paragraph.) Tj ET"


class FastExtractionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "fast.pdf"
        self.loader = PdfDocumentLoader(base_dir=temp.name, ocr_enabled=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.poisoned = []
        for method in ("_inspect_page", "_assess_page", "_load_canonical_pages",
                       "load_page_layout_evidence", "_render_page_images", "_load_pages_with_ocr",
                       "_ocr_image_text_with_quality", "_build_page_layout_evidence"):
            self.poisoned.append(self.stack.enter_context(patch.object(
                self.loader, method, side_effect=AssertionError(f"fast called {method}"),
            )))
        for target in ("doc_loaders.pdf_document_loader.subprocess.run",
                       "doc_loaders.pdf_document_loader.inspect_pdf_page"):
            self.poisoned.append(self.stack.enter_context(patch(
                target, side_effect=AssertionError("fast called forbidden dependency"),
            )))
        self.poisoned.append(self.stack.enter_context(patch.object(
            PageObject, "images", new_callable=PropertyMock,
            side_effect=AssertionError("fast decoded images"),
        )))
        self.addCleanup(self.assert_no_forbidden_calls)

    def assert_no_forbidden_calls(self):
        for mock in self.poisoned:
            mock.assert_not_called()

    def write(self, *kinds, outline=False, encrypted=False):
        writer = PdfWriter()
        for kind in kinds:
            content = NATIVE if kind in {"native", "mixed", "watermarked"} else b""
            if kind == "watermarked":
                content = b"q /GS1 gs " + content + b" Q"
            if kind in {"scan", "mixed"}:
                content += b" q 100 0 0 100 0 0 cm /Im Do Q"
            _, page = page_fixture(content, with_font=bool(content and kind != "scan"))
            if kind in {"scan", "mixed"}:
                add_image(page)
            writer.add_page(page)
        if outline:
            writer.add_outline_item("Chapter One", 0)
        if encrypted:
            writer.encrypt("password")
        writer.write(self.path)

    def test_default_fast_preserves_order_spans_and_empty_pages(self):
        self.write("blank", "native", "blank", "native", "blank")
        original = PageObject.extract_text
        with patch.object(PageObject, "extract_text", autospec=True, side_effect=original) as extract:
            result = self.loader.extract("fast")
            layouts = result.layout_provider()
            self.assertEqual(extract.call_count, 5)
        self.assertEqual(result.effective_mode, "fast")
        self.assertEqual(result.empty_page_indexes, (0, 2, 4))
        self.assertEqual(result.text, "A readable native paragraph.\nA readable native paragraph.")
        for span in result.boundary_evidence.boundaries:
            self.assertEqual(result.text[span.char_start:span.char_end], span.text)
        self.assertEqual(result.boundary_evidence.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(len(layouts), 5)
        self.assertTrue(all(item.ocr_pass_count == 0 and not item.high_cost_analysis_permitted for item in layouts))
        self.assertTrue(all("image_count_unknown" in item.evidence for item in layouts))

    def test_watermarked_and_mixed_native_text_does_not_trigger_routing(self):
        for kind in ("watermarked", "mixed"):
            with self.subTest(kind=kind):
                self.write(kind)
                result = self.loader.extract("fast", mode="fast")
                self.assertIn("readable native", result.text)
                self.assertIn("native_only_visual_coverage_unverified", result.warnings)
                self.assertFalse(result.outline_provider().usable)
                self.assertEqual(result.layout_provider()[0].ocr_pass_count, 0)

    def test_scan_blank_and_zero_pages_fail_without_escalation(self):
        for kinds in (("scan",), ("blank",), ()):
            with self.subTest(kinds=kinds):
                self.write(*kinds)
                with self.assertRaises(PdfExtractionError) as caught:
                    self.loader.extract("fast")
                self.assertEqual(caught.exception.reason, PdfExtractionFailureReason.NO_TEXT)
                self.assertIn("OCR", str(caught.exception))

    def test_mixed_book_reports_empty_scan_page_without_visual_completeness_claim(self):
        self.write("native", "scan")
        result = self.loader.extract("fast")
        self.assertEqual(result.empty_page_indexes, (1,))
        self.assertIn("native_text_empty:page=2", result.warnings)
        self.assertEqual(result.layout_provider()[1].ocr_text, "")

    def test_deferred_providers_use_original_snapshot(self):
        self.write("native", outline=True)
        result = self.loader.extract("fast")
        self.write("blank")
        outline = result.outline_provider()
        self.assertTrue(outline.usable)
        self.assertEqual(outline.entries[0].title, "Chapter One")
        self.assertEqual(outline.source_sha256, result.boundary_evidence.source_sha256)
        self.assertEqual(result.layout_provider()[0].ocr_text, result.text)

    def test_corrupt_and_encrypted_sources_fail_explicitly(self):
        self.path.write_bytes(b"not a PDF")
        with self.assertRaises(PdfExtractionError) as caught:
            self.loader.extract("fast")
        self.assertEqual(caught.exception.reason, PdfExtractionFailureReason.EXTRACTION_FAILED)
        self.write("native", encrypted=True)
        with self.assertRaises(PdfExtractionError) as caught:
            self.loader.extract("fast")
        self.assertEqual(caught.exception.reason, PdfExtractionFailureReason.EXTRACTION_FAILED)

    def test_page_failure_never_returns_partial_text(self):
        self.write("native", "native")
        with patch.object(PageObject, "extract_text", side_effect=["first page", ValueError("bad stream")]):
            with self.assertRaises(PdfExtractionError) as caught:
                self.loader.extract("fast")
        self.assertEqual(caught.exception.reason, PdfExtractionFailureReason.EXTRACTION_FAILED)

    def test_cheap_encoding_checks_without_language_heuristics(self):
        self.write("native")
        for text in ("12345", "\u4e2d\u6587", "English", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"):
            with patch.object(PageObject, "extract_text", return_value=text):
                self.assertEqual(self.loader.extract("fast").text, text)
        with patch.object(PageObject, "extract_text", return_value="\ufffd\x00"):
            with self.assertRaises(PdfExtractionError) as caught:
                self.loader.extract("fast")
        self.assertEqual(caught.exception.reason, PdfExtractionFailureReason.QUALITY_FAILED)

    def test_invalid_or_pending_modes_do_not_read_source(self):
        with patch.object(self.loader, "_resolve_file_path") as resolve:
            for mode in (None, "FAST", "unknown"):
                with self.assertRaises(ValueError):
                    self.loader.extract("fast", mode=mode)
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
