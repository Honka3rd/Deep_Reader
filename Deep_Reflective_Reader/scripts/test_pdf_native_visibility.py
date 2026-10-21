"""Scoped native visibility proof; unsupported content must remain uncertain."""
from pathlib import Path
from io import BytesIO
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pypdf import PdfReader
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject, NameObject, NumberObject
from doc_loaders.pdf_native_visibility import proves_native_visibility
from doc_loaders.pdf_page_inspection import inspect_pdf_page
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from test_pdf_page_inspection import page_fixture, add_image

TEXT = b"BT /F1 10 Tf 1 0 0 1 20 40 Tm (ABC) Tj ET"


def fixture(content, alpha=.4684):
    _, page = page_fixture(content, with_font=True)
    font = page["/Resources"]["/Font"]["/F1"]
    font.update({NameObject("/Subtype"): NameObject("/TrueType"),
                 NameObject("/FirstChar"): NumberObject(65),
                 NameObject("/Widths"): ArrayObject([NumberObject(500)] * 3),
                 NameObject("/FontDescriptor"): DictionaryObject({
                     NameObject("/FontBBox"): ArrayObject([NumberObject(n) for n in (0, -200, 500, 800)])})})
    page["/Resources"][NameObject("/ExtGState")] = DictionaryObject({
        NameObject("/GS"): DictionaryObject({NameObject("/ca"): FloatObject(alpha)})})
    return page


class NativeVisibilityTests(unittest.TestCase):
    def proof(self, page):
        return proves_native_visibility(page, (0, 0, 100, 100))

    def test_scoped_clipping_translucency_and_exact_glyph_edges(self):
        for clip in (b"10 20 40 40 re", b"20 38 15 10 re"):
            page = fixture(b"q " + clip + b" W n /GS gs " + TEXT + b" Q " + TEXT)
            self.assertTrue(self.proof(page))
            result = inspect_pdf_page(page, source_sha256="a" * 64, page_index=0)
            self.assertEqual(result.flags, ())
            self.assertEqual(result.native_text, page.extract_text())
            self.assertEqual(result.image_coverage_ratio, 0)

    def test_nested_clips_intersect_and_restore(self):
        self.assertTrue(self.proof(fixture(b"q 0 0 100 100 re W n q 20 38 15 10 re W n " + TEXT + b" Q Q " + TEXT)))
        self.assertFalse(self.proof(fixture(b"q 0 0 10 10 re W n q 0 0 100 100 re W n " + TEXT + b" Q Q")))

    def test_clipped_hidden_masked_forms_and_unknown_fail_closed(self):
        for content in (b"q 20 38 14 10 re W n " + TEXT + b" Q",
                        b"q 0 0 m 80 0 l 90 90 l h W n " + TEXT + b" Q",
                        b"3 Tr " + TEXT, b"7 Tr " + TEXT, b"/Fm Do " + TEXT,
                        TEXT + b" 0 0 100 100 re f", TEXT + b" Q", b"q " + TEXT,
                        b"/OC /Hidden BDC " + TEXT + b" EMC", b"/Pattern cs " + TEXT):
            with self.subTest(content=content):
                self.assertFalse(self.proof(fixture(content)))
        for alpha in (0, .001, 1.1):
            self.assertFalse(self.proof(fixture(b"/GS gs " + TEXT, alpha)))
        page = fixture(b"/GS gs " + TEXT)
        page["/Resources"]["/ExtGState"]["/GS"][NameObject("/SMask")] = NameObject("/None")
        self.assertFalse(self.proof(page))

    def test_images_remain_uncovered_and_unknown_is_not_zero(self):
        page = fixture(b"0 0 100 100 re W n /Im Do " + TEXT)
        add_image(page)
        self.assertFalse(self.proof(page))
        result = inspect_pdf_page(page, source_sha256="a" * 64, page_index=0)
        self.assertEqual(result.image_draw_count, 1)
        self.assertIsNone(result.image_coverage_ratio)

    def test_spacing_and_background(self):
        self.assertTrue(self.proof(fixture(b"0 100 m 100 100 l 100 0 l 0 0 l h f 0 0 100 100 re W n " + TEXT)))
        self.assertFalse(self.proof(fixture(b"q 20 38 15 10 re W n 10 Tc " + TEXT + b" Q")))

    def test_embedded_glyph_bounds_and_unknown_mapping(self):
        from fontTools.fontBuilder import FontBuilder
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        builder = FontBuilder(1000, isTTF=True)
        builder.setupGlyphOrder([".notdef", "A", "B", "C"])
        pen = TTGlyphPen(None)
        pen.moveTo((0, 0)); pen.lineTo((300, 0)); pen.lineTo((300, 600)); pen.lineTo((0, 600)); pen.closePath()
        builder.setupGlyf({name: pen.glyph() for name in (".notdef", "A", "B", "C")})
        builder.setupHorizontalMetrics({name: (500, 0) for name in (".notdef", "A", "B", "C")})
        builder.setupHorizontalHeader(ascent=800, descent=-200)
        builder.setupCharacterMap({65: "A", 66: "B", 67: "C"})
        builder.setupNameTable({"familyName": "Fixture", "styleName": "Regular"})
        builder.setupOS2(); builder.setupPost()
        stream = BytesIO(); builder.save(stream)
        page = fixture(b"q 20 40 13 6 re W n " + TEXT + b" Q")
        font = page["/Resources"]["/Font"]["/F1"]
        embedded = DecodedStreamObject(); embedded.set_data(stream.getvalue())
        font["/FontDescriptor"][NameObject("/FontFile2")] = embedded
        font[NameObject("/Encoding")] = NameObject("/WinAnsiEncoding")
        self.assertTrue(self.proof(page))
        font[NameObject("/Encoding")] = NameObject("/Unknown")
        self.assertFalse(self.proof(page))

    def test_auto_identity_changes_without_touching_fast_or_ocr(self):
        from document_preparation.pdf_extraction_identity import identity_for_loader
        loader = PdfDocumentLoader()
        with patch("doc_loaders.pdf_extraction_identity.shutil.which", return_value=None):
            for mode in ("fast", "ocr", "auto"):
                before = identity_for_loader(loader, source_sha256="a" * 64, mode=mode)
                with patch("doc_loaders.pdf_native_visibility.VISIBILITY_VERSION", "older"):
                    after = identity_for_loader(loader, source_sha256="a" * 64, mode=mode)
                self.assertEqual(before == after, mode != "auto")

    def test_actual_fixture_preserves_all_pages_without_subprocess(self):
        files = list((Path(__file__).resolve().parents[1] / "data/raw").glob("20250515-*.pdf"))
        if len(files) != 1:
            self.skipTest("private fixture unavailable")
        pages = PdfReader(files[0]).pages
        self.assertEqual(len(pages), 13)
        with patch("subprocess.run", side_effect=AssertionError("no render/OCR")):
            result = PdfDocumentLoader(files[0].parent, ocr_enabled=True).extract(files[0].stem, mode="auto")
        self.assertEqual(result.pages, tuple(page.extract_text() for page in pages))
        for boundary in result.boundary_evidence.boundaries:
            self.assertEqual(result.text[boundary.char_start:boundary.char_end], boundary.text)


if __name__ == "__main__":
    unittest.main()
