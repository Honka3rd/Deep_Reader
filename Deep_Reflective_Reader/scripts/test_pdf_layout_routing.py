#!/usr/bin/env python3
"""Native-first layout routing without real OCR, services or user artifacts."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PageObject, PdfWriter
from pypdf.generic import RectangleObject
from PIL import Image

from doc_loaders.document_load_errors import RawTextOcrFailedError
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_ocr_layout_fixtures import OCR_LAYOUT_FIXTURES
from test_pdf_page_inspection import add_image, page_fixture


NATIVE = (
    b"BT /F1 3 Tf 10 70 Td (A useful native paragraph with enough readable words.) Tj ET "
    b"BT /F1 3 Tf 10 40 Td (Another distinct line supplies reliable page positions.) Tj ET"
)


class PdfLayoutRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "layout.pdf"
        self.render_path = Path(self.temp.name) / "render.png"
        Image.new("L", (700, 1000)).save(self.render_path)
        self.loader = PdfDocumentLoader(base_dir=self.temp.name, ocr_enabled=True)

    def write_pages(self, *kinds):
        writer = PdfWriter()
        for kind in kinds:
            content = NATIVE if kind in {"native", "rotated", "uncertain", "mixed"} else b""
            if kind == "uncertain":
                content = b"q /GS1 gs " + content + b" Q"
            if kind in {"scan", "mixed"}:
                content += b" q 100 0 0 100 0 0 cm /Im Do Q"
            _, page = page_fixture(content, with_font=bool(content and kind != "scan"))
            if kind in {"scan", "mixed"}:
                add_image(page)
            if kind == "scan":
                page.mediabox = RectangleObject((0, 0, 700, 1000))
            if kind == "rotated":
                page.rotate(90)
            writer.add_page(page)
        writer.write(self.path)

    def forbid_expensive(self, stack):
        render = stack.enter_context(patch.object(self.loader, "_render_page_images"))
        process = stack.enter_context(patch("doc_loaders.pdf_document_loader.subprocess.run"))
        images = stack.enter_context(patch.object(
            PageObject, "images", new_callable=PropertyMock,
            side_effect=AssertionError("inventory must not decode images"),
        ))
        return render, process, images

    def run_candidate(self, *, fixture="horizontal_scan_leader_lines", error=None, **kwargs):
        result = subprocess.CompletedProcess(
            [], 0, stdout=OCR_LAYOUT_FIXTURES[fixture].tsv_text(), stderr="",
        )
        with patch("doc_loaders.pdf_document_loader.shutil.which", return_value="tesseract"), \
                patch.object(self.loader, "_render_page_images", return_value=[self.render_path]) as render, \
                patch("doc_loaders.pdf_document_loader.subprocess.run", return_value=result,
                      side_effect=error) as process, \
                patch.object(PageObject, "images", new_callable=PropertyMock) as images:
            evidence = self.loader.load_page_layout_evidence("layout", **kwargs)
        images.assert_not_called()
        return evidence, render, process

    def test_native_and_blank_inventory_never_render_or_decode(self):
        self.write_pages("native", "blank")
        original = PageObject.extract_text
        with ExitStack() as stack:
            render, process, images = self.forbid_expensive(stack)
            extract = stack.enter_context(patch.object(
                PageObject, "extract_text", autospec=True, side_effect=original,
            ))
            pages = self.loader.load_page_layout_evidence("layout", candidate_page_limit=30)
        self.assertEqual(extract.call_count, 2)
        render.assert_not_called()
        process.assert_not_called()
        images.assert_not_called()
        self.assertEqual([v.page_index for v in pages], [0, 1])
        self.assertTrue(all(v.analysis_stage == "inventory" and v.ocr_pass_count == 0 for v in pages))
        self.assertIn("layout_ocr_reason:native_layout_sufficient", pages[0].evidence)
        self.assertIn("layout_ocr_reason:blank_page", pages[1].evidence)
        self.assertIn("native paragraph", pages[0].ocr_text)
        self.assertEqual(pages[0].source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(pages[0].ocr_words, [])

    def test_uncertain_native_only_is_fallback_not_a_skip_claim(self):
        self.write_pages("uncertain")
        with ExitStack() as stack:
            render, process, images = self.forbid_expensive(stack)
            page = self.loader.load_page_layout_evidence("layout")[0]
        self.assertIn("layout_ocr_decision:uncertain", page.evidence)
        self.assertIn("layout_uncertain_inventory_fallback", page.evidence)
        self.assertIn("graphics_state_uninspected", page.evidence)
        self.assertIn("native paragraph", page.ocr_text)
        render.assert_not_called()
        process.assert_not_called()
        images.assert_not_called()

    def test_zero_candidates_and_disabled_ocr_preserve_all_inventory(self):
        self.write_pages("scan", "native", "scan")
        for enabled, limit in ((True, 0), (False, None)):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                self.loader.ocr_enabled = enabled
                render, process, images = self.forbid_expensive(stack)
                pages = self.loader.load_page_layout_evidence("layout", candidate_page_limit=limit)
                self.assertEqual(len(pages), 3)
                self.assertEqual([v.image_count for v in pages], [1, 0, 1])
                render.assert_not_called()
                process.assert_not_called()
                images.assert_not_called()

    def test_limit_is_original_page_prefix_not_number_of_selected_candidates(self):
        self.write_pages("native", "scan", "scan")
        pages, render, process = self.run_candidate(candidate_page_limit=2)
        self.assertEqual([v.analysis_stage for v in pages], ["inventory", "coordinate_ocr", "inventory"])
        self.assertEqual(render.call_count, 1)
        self.assertEqual(render.call_args.kwargs["page_index"], 1)
        self.assertEqual(process.call_count, 2)
        self.assertIn("layout_candidate_limit", pages[2].evidence)

    def test_scan_retains_vertical_and_horizontal_coordinate_candidates(self):
        self.write_pages("scan")
        for name in ("horizontal_scan_leader_lines", "guo_fu_lun_vertical_body_text"):
            with self.subTest(fixture=name):
                pages, render, process = self.run_candidate(fixture=name)
                self.assertEqual(pages[0].writing_mode, OCR_LAYOUT_FIXTURES[name].expected_writing_mode)
                self.assertGreater(pages[0].word_count, 0)
                self.assertEqual(render.call_count, 1)
                self.assertEqual([call.args[0][4] for call in process.call_args_list], ["11", "5"])
                self.assertIn("layout_ocr_decision:uncertain", pages[0].evidence)

    def test_rotated_and_mixed_pages_are_still_candidates(self):
        self.write_pages("rotated", "mixed")
        pages, render, process = self.run_candidate()
        self.assertEqual(render.call_count, 2)
        self.assertEqual(process.call_count, 4)
        self.assertTrue(all(v.analysis_stage == "coordinate_ocr" for v in pages))

    def test_coordinate_evidence_uses_render_pixels_not_pdf_points(self):
        self.write_pages("mixed")
        Image.new("L", (1400, 2000)).save(self.render_path)
        pages, _, _ = self.run_candidate()
        self.assertEqual((pages[0].width, pages[0].height), (1400, 2000))
        self.assertEqual(pages[0].analysis_stage, "coordinate_ocr")

    def test_missing_engine_does_not_render(self):
        self.write_pages("scan")
        with ExitStack() as stack:
            render, process, images = self.forbid_expensive(stack)
            stack.enter_context(patch("doc_loaders.pdf_document_loader.shutil.which", return_value=None))
            page = self.loader.load_page_layout_evidence("layout")[0]
        self.assertEqual(page.failure_reason, "tesseract_unavailable")
        render.assert_not_called()
        process.assert_not_called()
        images.assert_not_called()

    def test_optional_ocr_failures_keep_native_and_other_pages(self):
        self.write_pages("mixed", "native")
        for error in (subprocess.TimeoutExpired("tesseract", 60), OSError("not executable")):
            with self.subTest(error=type(error).__name__):
                pages, _, _ = self.run_candidate(error=error)
                self.assertEqual(len(pages), 2)
                expected_reason = (
                    "layout_analysis_time_budget_exhausted"
                    if isinstance(error, subprocess.TimeoutExpired) else "layout_ocr_failed"
                )
                self.assertEqual(pages[0].failure_reason, expected_reason)
                self.assertIn("native paragraph", pages[0].ocr_text)
                self.assertIn("layout_ocr_not_needed", pages[1].evidence)

    def test_empty_or_unsuccessful_ocr_keeps_native_inventory(self):
        self.write_pages("mixed")
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), \
                    patch("doc_loaders.pdf_document_loader.shutil.which", return_value="tesseract"), \
                    patch.object(self.loader, "_render_page_images", return_value=[self.render_path]), \
                    patch("doc_loaders.pdf_document_loader.subprocess.run", return_value=
                          subprocess.CompletedProcess([], returncode, stdout="", stderr="")):
                page = self.loader.load_page_layout_evidence("layout")[0]
                self.assertEqual(page.failure_reason, "layout_ocr_empty")
                self.assertEqual(page.analysis_stage, "inventory")
                self.assertIn("native paragraph", page.ocr_text)

    def test_render_and_decode_failure_are_page_local(self):
        self.write_pages("scan", "native")
        with patch("doc_loaders.pdf_document_loader.shutil.which", return_value="tesseract"), \
                patch.object(self.loader, "_render_page_images", side_effect=
                             RawTextOcrFailedError(doc_name="layout", detail="decode")), \
                patch.object(PageObject, "images", new_callable=PropertyMock,
                             side_effect=ValueError("invalid image")) as images, \
                patch("doc_loaders.pdf_document_loader.subprocess.run") as process:
            pages = self.loader.load_page_layout_evidence("layout")
        self.assertEqual(pages[0].failure_reason, "layout_image_decode_failed")
        self.assertIn("native paragraph", pages[1].ocr_text)
        self.assertEqual(images.call_count, 1)
        process.assert_not_called()

    def test_renderer_failure_retains_embedded_image_fallback(self):
        self.write_pages("scan")
        fixture = OCR_LAYOUT_FIXTURES["guo_fu_lun_vertical_body_text"]
        with patch("doc_loaders.pdf_document_loader.shutil.which", return_value="tesseract"), \
                patch.object(self.loader, "_render_page_images", side_effect=
                             RawTextOcrFailedError(doc_name="layout", detail="render")), \
                patch.object(PageObject, "images", new_callable=PropertyMock,
                             return_value=[SimpleNamespace(name="scan.png", data=b"image")]) as images, \
                patch("doc_loaders.pdf_document_loader.subprocess.run", return_value=
                      subprocess.CompletedProcess([], 0, stdout=fixture.tsv_text(), stderr="")) as process:
            page = self.loader.load_page_layout_evidence("layout")[0]
        self.assertEqual(page.writing_mode, "vertical")
        self.assertIn("layout_renderer_failed", page.evidence)
        self.assertEqual(images.call_count, 1)
        self.assertEqual(process.call_count, 2)

    def test_layout_does_not_change_native_raw_text_or_boundaries(self):
        self.write_pages("native", "blank", "native")
        before_text = self.loader.load("layout")
        before_boundaries = self.loader.load_page_boundary_evidence("layout")
        before_bytes = self.path.read_bytes()
        self.loader.load_page_layout_evidence("layout")
        self.assertEqual(self.loader.load("layout"), before_text)
        self.assertEqual(self.loader.load_page_boundary_evidence("layout"), before_boundaries)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertIsNone(self.loader.last_ocr_provenance)
        self.assertIsNone(self.loader.last_ocr_pages)

    def test_invalid_limits_rejected_before_reading(self):
        for limit in (-1, True, 1.5, "1"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.loader.load_page_layout_evidence("missing", candidate_page_limit=limit)


if __name__ == "__main__":
    unittest.main()
