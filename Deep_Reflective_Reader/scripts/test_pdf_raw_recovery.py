#!/usr/bin/env python3
"""Selective raw recovery, atomic failures and canonical page offset parity."""

from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, FloatObject, NameObject
from PIL import Image

from doc_loaders.document_load_errors import (
    RawTextOcrFailedError, RawTextOcrLowQualityError, RawTextRequiresOcrError,
)
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_contract import PdfExtractionError
from doc_loaders.pdf_page_inspection import PdfNativeTextQuality, inspect_pdf_page
from test_pdf_page_inspection import add_image, page_fixture


NATIVE = b"BT /F1 3 Tf 10 60 Td (  Original native paragraph with enough readable words.  ) Tj ET"
RECOVERED = "Recovered scanned paragraph with a complete set of readable words."
HEADER = "A native header with several readable words"


class PdfRawRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.pdf"
        self.loader = PdfDocumentLoader(base_dir=self.temp.name, ocr_enabled=True, ocr_language="eng")

    def write_pages(self, *kinds):
        writer = PdfWriter()
        for kind in kinds:
            content = NATIVE if kind in {"native", "opaque", "transparent"} else b""
            if kind == "short":
                content = b"BT /F1 3 Tf 10 95 Td (Title) Tj ET"
            if kind == "mixed":
                content = f"BT /F1 3 Tf 10 95 Td ({HEADER}) Tj ET".encode()
            if kind in {"scan", "mixed"}:
                content += b" q 100 0 0 100 0 0 cm /Im Do Q"
            if kind in {"opaque", "transparent"}:
                content = b"q /GS gs " + content + b" Q"
            _, page = page_fixture(content, with_font=kind not in {"scan", "blank"})
            if kind in {"scan", "mixed"}:
                add_image(page)
            if kind in {"opaque", "transparent"}:
                page["/Resources"][NameObject("/ExtGState")] = DictionaryObject({
                    NameObject("/GS"): DictionaryObject({
                        NameObject("/ca"): FloatObject(1 if kind == "opaque" else 0.2),
                        NameObject("/BM"): NameObject("/Normal"),
                    }),
                })
            writer.add_page(page)
        writer.write(self.path)

    def fake_ocr(self, stack, texts=None, errors=None):
        texts, errors = texts or {}, errors or {}
        stack.enter_context(patch.object(self.loader, "_binary_identity", side_effect=lambda cmd: (cmd, "fixture-v1")))
        stack.enter_context(patch("doc_loaders.pdf_document_loader.shutil.which", return_value="engine"))
        stack.enter_context(patch.object(self.loader, "_tesseract_version", return_value="test-version"))
        render = stack.enter_context(patch.object(
            self.loader, "_render_page_images", side_effect=lambda **kw: [Path(f"page-{kw['page_index']}.png")],
        ))

        def recognize(**kw):
            index = kw["page_index"]
            if index in errors:
                raise errors[index]
            text = texts.get(index, RECOVERED)
            candidate = self.loader._score_ocr_candidate(psm="default", text=text)
            return (text if candidate.passed else ""), [candidate]

        ocr = stack.enter_context(patch.object(self.loader, "_ocr_image_text_with_quality", side_effect=recognize))
        return render, ocr

    def assert_parity(self):
        with self.loader.preparation_scope():
            text = self.loader.load("sample")
            pages = self.loader.load_pages("sample")
            evidence = self.loader.load_page_boundary_evidence("sample")
        self.assertEqual(len(pages), evidence.page_count)
        self.assertEqual([b.text for b in evidence.boundaries], pages)
        self.assertEqual([b.page_index for b in evidence.boundaries], list(range(len(pages))))
        for boundary in evidence.boundaries:
            self.assertEqual(text[boundary.char_start:boundary.char_end], boundary.text)
            self.assertEqual(boundary.source_sha256, evidence.source_sha256)
        self.assertEqual(evidence.boundaries[-1].char_end, len(text))
        return text, pages, evidence

    def test_native_blank_and_short_pages_need_no_ocr_and_keep_whitespace(self):
        self.write_pages("blank", "native", "blank", "short", "blank")
        with patch("doc_loaders.pdf_document_loader.subprocess.run", side_effect=AssertionError("no OCR")), \
                patch.object(PageObject, "images", new_callable=PropertyMock,
                             side_effect=AssertionError("no decoding")):
            text, pages, _ = self.assert_parity()
        self.assertTrue(pages[1].startswith("  "))
        self.assertTrue(pages[1].endswith("  "))
        self.assertEqual(text, pages[1] + "\nTitle")
        self.assertIsNone(self.loader.last_ocr_pages)

    def test_mixed_pdf_recovers_only_missing_page_once_across_entrypoints(self):
        self.write_pages("native", "scan", "blank", "native")
        with ExitStack() as stack:
            render, ocr = self.fake_ocr(stack)
            text, pages, _ = self.assert_parity()
        self.assertEqual(pages[1], RECOVERED)
        self.assertEqual(text.count(RECOVERED), 1)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(render.call_args.kwargs["page_index"], 1)
        self.assertEqual(self.loader.last_ocr_pages, ["", RECOVERED, "", ""])
        self.assertEqual(self.loader.last_ocr_provenance["recovery_page_indexes"], [1])

    def test_full_scan_and_blank_offsets_share_the_same_separator(self):
        self.write_pages("blank", "scan", "blank", "scan", "blank")
        with ExitStack() as stack:
            self.fake_ocr(stack, {3: "A second scanned paragraph with different readable words."})
            text, pages, _ = self.assert_parity()
        self.assertEqual(text, pages[1] + "\n\n" + pages[3])

    def test_mixed_page_full_fallback_replaces_instead_of_duplicating_header(self):
        self.write_pages("mixed", "native")
        with ExitStack() as stack:
            self.fake_ocr(stack, {0: HEADER + "\n" + RECOVERED})
            text, pages, _ = self.assert_parity()
        self.assertEqual(text.count(HEADER), 1)
        self.assertEqual(pages[0], HEADER + "\n" + RECOVERED)

    def test_mixed_page_cannot_lose_or_only_repeat_native_header(self):
        self.write_pages("mixed")
        for output in (RECOVERED, HEADER):
            with self.subTest(output=output), ExitStack() as stack:
                self.fake_ocr(stack, {0: output})
                with self.assertRaisesRegex(RawTextOcrFailedError, "native_coverage_unresolved"):
                    self.loader.load("sample")
                self.assertIsNone(self.loader.last_ocr_pages)

    def test_coverage_failure_stops_before_later_pages_and_retry_is_atomic(self):
        self.write_pages("mixed", "scan", "scan")
        with self.loader.preparation_scope(), ExitStack() as stack:
            render, ocr = self.fake_ocr(stack)
            with self.assertRaisesRegex(RawTextOcrFailedError, "page=1:native_coverage_unresolved"):
                self.loader.load("sample")
            self.assertEqual(render.call_count, 1)
            self.assertEqual(ocr.call_count, 1)
            self.assertIsNone(self.loader.last_ocr_provenance)
            self.assertIsNone(self.loader.last_ocr_pages)
            self.assertEqual(self.loader.last_ocr_quality, [])
        with ExitStack() as stack:
            render, _ = self.fake_ocr(stack, {0: HEADER + " " + RECOVERED})
            self.assert_parity()
            self.assertEqual(render.call_count, 3)

    def test_blank_render_cannot_drop_native_text_or_continue(self):
        self.write_pages("mixed", "scan")
        with ExitStack() as stack:
            render, ocr = self.fake_ocr(stack)
            stack.enter_context(patch.object(self.loader, "_render_is_uniform", return_value=True))
            with self.assertRaisesRegex(RawTextOcrFailedError, "page=1:native_coverage_unresolved"):
                self.loader.load("sample")
            self.assertEqual(render.call_count, 1)
            ocr.assert_not_called()

    def test_auto_coverage_error_is_actionable_without_raw_text(self):
        self.write_pages("mixed", "scan")
        with ExitStack() as stack:
            render, _ = self.fake_ocr(stack)
            with self.assertRaises(PdfExtractionError) as raised:
                self.loader.extract("sample", mode="auto")
            message = str(raised.exception)
            self.assertIn("page 1", message)
            self.assertIn("native_coverage_unresolved", message)
            self.assertIn("Retry fast", message)
            self.assertIn("explicit OCR", message)
            self.assertNotIn(HEADER, message)
            self.assertNotIn(RECOVERED, message)
            self.assertEqual(raised.exception.reason.value, "extraction_failed")
            self.assertEqual(render.call_count, 1)

    def test_cached_ocr_still_checks_native_coverage(self):
        self.write_pages("mixed")
        with self.loader.preparation_scope(), ExitStack() as stack:
            render, _ = self.fake_ocr(stack)
            self.loader._load_pages_with_ocr(
                doc_name="sample", file_path=self.path, reader=PdfReader(self.path),
            )
            with self.assertRaisesRegex(RawTextOcrFailedError, "native_coverage_unresolved"):
                self.loader.load("sample")
            self.assertEqual(render.call_count, 1)
            self.assertIsNone(self.loader.last_ocr_pages)
            self.assertEqual(self.loader.last_ocr_quality, [])

    def test_disabled_recovery_fails_all_entrypoints_without_partial_native_text(self):
        self.write_pages("native", "scan")
        self.loader.ocr_enabled = False
        for method in (self.loader.load, self.loader.load_pages, self.loader.load_page_boundary_evidence):
            with self.subTest(method=method.__name__), \
                    patch("doc_loaders.pdf_document_loader.subprocess.run") as process:
                with self.assertRaisesRegex(RawTextRequiresOcrError, "page=2"):
                    method("sample")
                process.assert_not_called()

    def test_candidate_prefix_budget_preflight_rejects_truncated_document(self):
        self.write_pages("native", "scan", "scan")
        for limit in (0, 1, 2):
            with self.subTest(limit=limit), ExitStack() as stack:
                self.loader.ocr_page_limit = limit
                render, ocr = self.fake_ocr(stack)
                with self.assertRaisesRegex(RawTextOcrFailedError, "excludes_required_page"):
                    self.loader.load("sample")
                render.assert_not_called()
                ocr.assert_not_called()

    def test_page_failure_is_atomic_and_retry_does_not_reuse_partial_results(self):
        self.write_pages("scan", "native", "scan")
        with ExitStack() as stack:
            self.fake_ocr(stack, {2: ""})
            with self.assertRaisesRegex(RawTextOcrFailedError, "empty_ocr_page:page=3"):
                self.loader.load("sample")
        self.assertIsNone(self.loader.last_ocr_provenance)
        self.assertIsNone(self.loader.last_ocr_pages)
        with ExitStack() as stack:
            render, _ = self.fake_ocr(stack)
            self.assert_parity()
        self.assertEqual(render.call_count, 2)

    def test_low_quality_single_page_is_not_hidden_by_good_pages(self):
        self.write_pages("scan", "native", "scan")
        with ExitStack() as stack:
            self.fake_ocr(stack, {2: "[[[[[[[[[]]]]]]]]]"})
            with self.assertRaisesRegex(RawTextOcrLowQualityError, "page=3"):
                self.loader.load("sample")
        self.assertIsNone(self.loader.last_ocr_pages)

    def test_ocr_timeout_and_oserror_are_typed_failures(self):
        self.write_pages("scan")
        for error in (OSError("engine failed"), subprocess.TimeoutExpired("ocr", 1)):
            with self.subTest(error=type(error).__name__), ExitStack() as stack:
                self.fake_ocr(stack, errors={0: error})
                with self.assertRaisesRegex(RawTextOcrFailedError, "ocr_process_failed"):
                    self.loader.load("sample")

    def test_unavailable_engine_cannot_render_or_publish_native_only(self):
        self.write_pages("native", "scan")
        with patch("doc_loaders.pdf_document_loader.shutil.which", return_value=None), \
                patch.object(self.loader, "_render_page_images") as render:
            with self.assertRaisesRegex(RawTextOcrFailedError, "tesseract_not_found"):
                self.loader.load("sample")
        render.assert_not_called()

    def test_source_replacement_clears_previous_ocr_provenance(self):
        self.write_pages("scan")
        with ExitStack() as stack:
            self.fake_ocr(stack)
            self.loader.load("sample")
        self.write_pages("native")
        text = self.loader.load("sample")
        self.assertNotIn(RECOVERED, text)
        self.assertIsNone(self.loader.last_ocr_pages)
        self.assertIsNone(self.loader.last_ocr_provenance)

    def test_source_change_during_recovery_is_rejected(self):
        self.write_pages("scan")
        with ExitStack() as stack:
            _, ocr = self.fake_ocr(stack)
            recognize = ocr.side_effect

            def replace_source(**kw):
                self.write_pages("native")
                return recognize(**kw)

            ocr.side_effect = replace_source
            with self.assertRaisesRegex(RawTextOcrFailedError, "source_changed"):
                self.loader.load("sample")
        self.assertIsNone(self.loader.last_ocr_provenance)

    def test_native_extraction_failure_is_not_treated_as_blank(self):
        self.write_pages("native")
        self.loader.ocr_enabled = False
        with patch.object(PageObject, "extract_text", side_effect=ValueError("broken cmap")):
            with self.assertRaises(RawTextRequiresOcrError):
                self.loader.load("sample")

    def test_default_opaque_graphics_state_does_not_force_ocr(self):
        self.write_pages("opaque")
        self.loader.ocr_enabled = False
        self.assert_parity()
        self.write_pages("transparent")
        with self.assertRaises(RawTextRequiresOcrError):
            self.loader.load("sample")

    def test_unknown_graphics_keys_stay_unresolved(self):
        _, page = page_fixture(b"q /GS gs " + NATIVE + b" Q", with_font=True)
        page["/Resources"][NameObject("/ExtGState")] = DictionaryObject({
            NameObject("/GS"): DictionaryObject({NameObject("/SMask"): NameObject("/None")}),
        })
        inspection = inspect_pdf_page(page, source_sha256="a" * 64, page_index=0)
        self.assertIn("graphics_state_uninspected", inspection.flags)

    def test_only_complete_background_fill_before_text_allows_native_preservation(self):
        for drawing, expected in (
            (b"0 0 100 100 re f " + NATIVE, True),
            (b"0 0 50 50 re f " + NATIVE, False),
            (NATIVE + b" 0 0 100 100 re f", False),
            (b"0 0 m 100 0 l 100 100 l 0 100 l h f " + NATIVE, False),
        ):
            with self.subTest(drawing=drawing):
                writer, page = page_fixture(drawing, with_font=True)
                inspection = inspect_pdf_page(page, source_sha256="a" * 64, page_index=0)
                self.assertTrue(inspection.has_vector_paint)
                self.assertEqual(inspection.has_only_page_background, expected)
                writer.write(self.path)
                self.loader.ocr_enabled = False
                if expected:
                    self.assert_parity()
                else:
                    with self.assertRaises(RawTextRequiresOcrError):
                        self.loader.load("sample")

    def test_single_glyph_lines_are_not_duplicate_passages(self):
        self.assertEqual(PdfNativeTextQuality.from_text("a\nb\na\nb\na\nb").duplicate_line_ratio, 0)
        self.assertEqual(PdfNativeTextQuality.from_text("duplicate line\nduplicate line").duplicate_line_ratio, 0.5)

    def test_corrupt_native_text_can_be_replaced_after_quality_gated_ocr(self):
        writer, _ = page_fixture(b"BT /F1 3 Tf 10 60 Td (\\001\\001\\001) Tj ET", with_font=True)
        writer.write(self.path)
        with ExitStack() as stack:
            self.fake_ocr(stack)
            text, _, _ = self.assert_parity()
        self.assertEqual(text, RECOVERED)

    def test_disabled_after_success_clears_stale_ocr_state(self):
        self.write_pages("scan")
        with ExitStack() as stack:
            self.fake_ocr(stack)
            self.loader.load("sample")
        self.loader.ocr_enabled = False
        with self.assertRaises(RawTextRequiresOcrError):
            self.loader.load("sample")
        self.assertIsNone(self.loader.last_ocr_pages)
        self.assertEqual(self.loader.last_ocr_quality, [])

    def test_native_extracted_once_per_page_in_one_load(self):
        self.write_pages("native", "short", "blank")
        original = PageObject.extract_text
        with patch.object(PageObject, "extract_text", autospec=True, side_effect=original) as extract:
            self.loader.load("sample")
        self.assertEqual(extract.call_count, 3)

    def test_invalid_raw_page_limits_are_rejected(self):
        for limit in (-1, True, 1.5, "1"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                PdfDocumentLoader(ocr_page_limit=limit)

    def test_uniform_full_page_is_confirmed_blank_not_empty_ocr_failure(self):
        self.write_pages("scan", "native", "scan")
        image_path = Path(self.temp.name) / "blank.png"
        Image.new("RGB", (30, 40), "white").save(image_path)
        with ExitStack() as stack:
            render, ocr = self.fake_ocr(stack)
            render.side_effect = lambda **kw: [image_path]
            text, pages, _ = self.assert_parity()
        self.assertEqual(pages[0], "")
        self.assertEqual(pages[2], "")
        self.assertEqual(text, pages[1])
        ocr.assert_not_called()
        self.assertTrue(all(item["blank_render_confirmed"] for item in self.loader.last_ocr_quality))
        Image.new("RGBA", (30, 40), (0, 0, 0, 0)).save(image_path)
        self.assertFalse(self.loader._render_is_uniform(image_path))
        Image.new("RGB", (30, 40), "white").save(image_path)
        with Image.open(image_path) as nonuniform:
            nonuniform.putpixel((1, 1), (0, 0, 0))
            nonuniform.save(image_path)
        self.assertFalse(self.loader._render_is_uniform(image_path))

    def test_multiple_or_missing_render_results_cannot_publish_partial_page(self):
        self.write_pages("scan")
        for paths in ([], [Path("a.png"), Path("b.png")]):
            with self.subTest(paths=paths), ExitStack() as stack:
                render, ocr = self.fake_ocr(stack)
                render.side_effect = None
                render.return_value = paths
                with self.assertRaisesRegex(RawTextOcrFailedError, "expected_one_full_page_render"):
                    self.loader.load("sample")
                ocr.assert_not_called()

    def test_text_only_page_does_not_require_layout_coordinates(self):
        self.write_pages("native")
        original = inspect_pdf_page
        from dataclasses import replace

        def without_positions(*args, **kwargs):
            return replace(original(*args, **kwargs), text_origins=(), text_positions_complete=False,
                           flags=("text_position_unavailable",))

        with patch("doc_loaders.pdf_document_loader.inspect_pdf_page", side_effect=without_positions):
            self.loader.ocr_enabled = False
            self.assert_parity()


if __name__ == "__main__":
    unittest.main()
