#!/usr/bin/env python3
"""Cheap page evidence regression using synthetic PDFs and poisoned OCR paths."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PageObject, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, EncodedStreamObject,
    NameObject, NumberObject, RectangleObject,
)

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_page_inspection import (
    MAX_IMAGE_RECTS, MAX_TEXT_ORIGINS, PdfNativeTextQuality, inspect_pdf_page,
)


def page_fixture(content=b"", *, with_font=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    resources = DictionaryObject()
    if with_font:
        resources[NameObject("/Font")] = DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }),
        })
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject("/Contents")] = stream
    return writer, page


def add_image(page, name="/Im"):
    image = EncodedStreamObject()
    image.update({
        NameObject("/Type"): NameObject("/XObject"),
        NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(100),
        NameObject("/Height"): NumberObject(100),
        NameObject("/BitsPerComponent"): NumberObject(8),
        NameObject("/ColorSpace"): NameObject("/DeviceGray"),
        NameObject("/Filter"): NameObject("/DCTDecode"),
    })
    image._data = b"not-a-decodable-image"
    resources = page["/Resources"]
    if "/XObject" not in resources:
        resources[NameObject("/XObject")] = DictionaryObject()
    resources["/XObject"][NameObject(name)] = image
    return image


class PdfPageInspectionTests(unittest.TestCase):
    def inspect(self, page):
        return inspect_pdf_page(page, source_sha256="ab" * 32, page_index=0)

    def test_text_extracted_once_and_retained_without_content_serialization(self):
        _, page = page_fixture(b"BT /F1 12 Tf 10 70 Td (PRIVATE_TEXT) Tj ET", with_font=True)
        with patch.object(page, "extract_text", wraps=page.extract_text) as extract:
            result = self.inspect(page)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(result.native_text, "PRIVATE_TEXT")
        self.assertEqual(result.text_quality.non_whitespace_count, 12)
        self.assertTrue(result.has_font_resources)
        self.assertEqual(result.text_origins, ((10.0, 70.0),))
        self.assertTrue(result.text_positions_complete)
        self.assertEqual(result.image_coverage_ratio, 0.0)
        self.assertNotIn("PRIVATE_TEXT", json.dumps(result.to_metadata()))
        self.assertNotIn("PRIVATE_TEXT", repr(result))
        result.to_metadata()["flags"].append("changed")
        self.assertNotIn("changed", result.flags)
        with self.assertRaises(FrozenInstanceError):
            result.page_index = 1

    def test_quality_signals_are_script_neutral_and_do_not_decide_ocr(self):
        for text in ("\u4e2d\u6587\u6bb5\u843d", "English text", "123456789", "caf\u00e9", "\u4e2dA12"):
            with self.subTest(text=text):
                quality = PdfNativeTextQuality.from_text(text)
                self.assertEqual(quality.printable_ratio, 1.0)
                self.assertEqual(quality.control_ratio, 0.0)
                self.assertEqual(quality.replacement_ratio, 0.0)
        quality = PdfNativeTextQuality.from_text("A\ufffd\x00\n")
        self.assertEqual(quality.replacement_ratio, 0.25)
        self.assertEqual(quality.control_ratio, 0.25)
        self.assertEqual(PdfNativeTextQuality.from_text("aaaa").longest_repeat_ratio, 1.0)
        self.assertEqual(PdfNativeTextQuality.from_text("abc\nabc").duplicate_line_ratio, 0.5)
        self.assertEqual(PdfNativeTextQuality.from_text("").character_count, 0)

    def test_blank_and_vector_pages_remain_distinguishable(self):
        _, blank = page_fixture()
        _, vector = page_fixture(b"0 0 50 50 re f")
        empty = self.inspect(blank)
        art = self.inspect(vector)
        self.assertEqual(empty.native_text, "")
        self.assertFalse(empty.has_vector_paint)
        self.assertTrue(art.has_vector_paint)
        self.assertEqual(empty.image_draw_count, 0)
        self.assertEqual(art.image_coverage_ratio, 0.0)

    def test_image_union_clips_and_deduplicates_without_decoding(self):
        _, page = page_fixture(
            b"q 100 0 0 100 -50 0 cm /Im Do /Im Do Q "
            b"q 100 0 0 100 25 0 cm /Im Do Q"
        )
        image = add_image(page)
        with patch.object(image, "get_data", side_effect=AssertionError("image decoded")) as decode:
            result = self.inspect(page)
        decode.assert_not_called()
        self.assertEqual(result.image_draw_count, 3)
        self.assertEqual(result.image_rects, ((0, 0, 50, 100), (0, 0, 50, 100), (25, 0, 100, 100)))
        self.assertEqual(result.image_coverage_ratio, 1.0)

    def test_overlapping_images_have_union_not_sum_coverage(self):
        _, page = page_fixture(b"q 50 0 0 100 0 0 cm /Im Do Q q 50 0 0 100 25 0 cm /Im Do Q")
        add_image(page)
        self.assertEqual(self.inspect(page).image_coverage_ratio, 0.75)

    def test_unused_image_resource_is_not_a_drawn_image(self):
        _, page = page_fixture()
        add_image(page)
        result = self.inspect(page)
        self.assertEqual(result.image_draw_count, 0)
        self.assertEqual(result.image_coverage_ratio, 0.0)

    def test_matrix_composition_crop_offset_rotation_and_stack(self):
        _, page = page_fixture(b"q 2 0 0 2 10 10 cm 10 0 0 20 5 5 cm /Im Do Q /Im Do")
        page.cropbox = RectangleObject((10, 10, 90, 90))
        page[NameObject("/Rotate")] = NumberObject(90)
        add_image(page)
        result = self.inspect(page)
        self.assertEqual(result.image_rects, ((20, 20, 40, 60),))
        self.assertEqual(result.image_coverage_ratio, 800 / 6400)
        self.assertEqual(result.rotation_degrees, 90)

    def test_inline_image_supported_without_image_decode(self):
        _, page = page_fixture(b"q 50 0 0 50 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x00 EI Q")
        result = self.inspect(page)
        self.assertEqual(result.image_draw_count, 1)
        self.assertEqual(result.image_coverage_ratio, 0.25)
        self.assertIn("inline_image_present", result.flags)

    def test_nested_forms_preserve_unknown_not_zero(self):
        _, page = page_fixture(b"/Fm Do")
        inner = DecodedStreamObject()
        inner.set_data(b"q 100 0 0 100 0 0 cm /Im Do Q")
        image = add_image(page)
        inner.update({NameObject("/Subtype"): NameObject("/Form"),
                      NameObject("/BBox"): RectangleObject((0, 0, 100, 100)),
                      NameObject("/Resources"): DictionaryObject({
                          NameObject("/XObject"): DictionaryObject({NameObject("/Im"): image}),
                      })})
        outer = DecodedStreamObject()
        outer.set_data(b"/Inner Do")
        outer.update({NameObject("/Subtype"): NameObject("/Form"),
                      NameObject("/BBox"): RectangleObject((0, 0, 100, 100)),
                      NameObject("/Resources"): DictionaryObject({
                          NameObject("/XObject"): DictionaryObject({NameObject("/Inner"): inner}),
                      })})
        page["/Resources"]["/XObject"][NameObject("/Fm")] = outer
        with patch.object(image, "get_data", side_effect=AssertionError("image decoded")) as decode:
            result = self.inspect(page)
        decode.assert_not_called()
        self.assertIsNone(result.image_draw_count)
        self.assertIsNone(result.image_coverage_ratio)
        self.assertIsNone(result.has_font_resources)
        self.assertFalse(result.text_positions_complete)
        self.assertIn("form_or_unknown_xobject_uninspected", result.flags)

    def test_custom_clipping_and_skew_are_unknown(self):
        for content, flag in (
            (b"0 0 50 50 re W n /Im Do", "custom_clipping_unresolved"),
            (b"50 20 0 50 0 0 cm /Im Do", "image_transform_unresolved"),
        ):
            with self.subTest(flag=flag):
                _, page = page_fixture(content)
                add_image(page)
                result = self.inspect(page)
                self.assertIsNone(result.image_coverage_ratio)
                self.assertIn(flag, result.flags)

    def test_resource_and_stream_errors_are_explicit(self):
        _, page = page_fixture(b"/Missing Do")
        result = self.inspect(page)
        self.assertIsNone(result.image_coverage_ratio)
        self.assertIsNone(result.image_draw_count)
        self.assertIn("content_stream_inspection_failed", result.flags)
        _, page = page_fixture(b"Q")
        self.assertIn("content_stream_inspection_failed", self.inspect(page).flags)

    def test_extraction_failure_is_not_empty_text(self):
        _, page = page_fixture()
        with patch.object(page, "extract_text", side_effect=ValueError("PRIVATE_ERROR")) as extract:
            result = self.inspect(page)
        self.assertEqual(extract.call_count, 1)
        self.assertIsNone(result.native_text)
        self.assertIsNone(result.text_quality)
        self.assertFalse(result.text_positions_complete)
        self.assertIn("native_text_extraction_failed", result.flags)
        self.assertNotIn("PRIVATE_ERROR", json.dumps(result.to_metadata()))

    def test_missing_geometry_and_unreadable_resources_stay_unknown(self):
        _, page = page_fixture()
        page.cropbox = RectangleObject((0, 0, 0, 100))
        result = self.inspect(page)
        self.assertIsNone(result.crop_box)
        self.assertIsNone(result.image_coverage_ratio)
        self.assertFalse(result.text_positions_complete)
        self.assertIn("page_geometry_unavailable", result.flags)
        _, page = page_fixture()
        page[NameObject("/Resources")] = NumberObject(1)
        result = self.inspect(page)
        self.assertIsNone(result.has_font_resources)
        self.assertIsNone(result.image_coverage_ratio)
        self.assertIsNone(result.image_draw_count)
        self.assertIn("resources_unavailable", result.flags)

    def test_empty_extraction_with_text_operators_is_not_a_confirmed_blank(self):
        _, page = page_fixture(b"BT /F1 12 Tf (Text) Tj ET", with_font=True)
        with patch.object(page, "extract_text", return_value=""):
            result = self.inspect(page)
        self.assertIn("text_operators_without_extracted_text", result.flags)
        self.assertFalse(result.text_positions_complete)

    def test_image_geometry_is_bounded_and_not_reported_as_complete(self):
        _, page = page_fixture(b"/Im Do " * (MAX_IMAGE_RECTS + 1))
        add_image(page)
        result = self.inspect(page)
        self.assertEqual(result.image_draw_count, MAX_IMAGE_RECTS + 1)
        self.assertEqual(len(result.image_rects), MAX_IMAGE_RECTS)
        self.assertIsNone(result.image_coverage_ratio)
        self.assertIn("image_geometry_limit", result.flags)

    def test_unreliable_or_excess_text_positions_do_not_look_complete(self):
        _, page = page_fixture()

        def extract(**kwargs):
            for _ in range(MAX_TEXT_ORIGINS + 1):
                kwargs["visitor_text"]("a", [1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 10, 10], {}, 12)
            return "a" * (MAX_TEXT_ORIGINS + 1)

        with patch.object(page, "extract_text", side_effect=extract):
            result = self.inspect(page)
        self.assertFalse(result.text_positions_complete)
        self.assertEqual(result.text_origins, ())
        self.assertIn("text_position_limit", result.flags)

    def test_hidden_text_and_annotations_are_not_silently_trusted(self):
        _, page = page_fixture(b"BT /F1 12 Tf 3 Tr 10 70 Td (Hidden) Tj ET", with_font=True)
        page[NameObject("/Annots")] = ArrayObject([DictionaryObject()])
        result = self.inspect(page)
        self.assertEqual(result.native_text, "Hidden")
        self.assertIn("invisible_text_mode", result.flags)
        self.assertIn("annotations_uninspected", result.flags)

    def test_loader_snapshot_identity_page_order_no_ocr_no_stale_reuse(self):
        writer, _ = page_fixture(b"BT /F1 12 Tf 10 70 Td (One) Tj ET", with_font=True)
        writer.add_blank_page(width=100, height=100)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.pdf"
            writer.write(source)
            loader = PdfDocumentLoader(base_dir=directory, ocr_enabled=True)
            with patch.object(PageObject, "images", new=property(lambda _: self.fail("image decoded"))), \
                 patch("subprocess.run", side_effect=AssertionError("external process")), \
                 patch.object(loader, "load", side_effect=AssertionError("raw path entered")):
                result = loader.inspect_pages("test")
                self.assertEqual([item.page_index for item in result], [0, 1])
                self.assertEqual([item.native_text for item in result], ["One", ""])
                first_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                self.assertTrue(all(item.source_sha256 == first_hash for item in result))
                updated, _ = page_fixture(b"BT /F1 12 Tf 10 70 Td (Two) Tj ET", with_font=True)
                updated.write(source)
                newer = loader.inspect_pages("test.pdf")
                self.assertEqual(newer[0].native_text, "Two")
                self.assertNotEqual(newer[0].source_sha256, first_hash)
            self.assertIsNone(loader.last_ocr_provenance)
            self.assertEqual(list(Path(directory).iterdir()), [source])
            with self.assertRaises(FileNotFoundError):
                loader.inspect_pages("missing")


if __name__ == "__main__":
    unittest.main()
