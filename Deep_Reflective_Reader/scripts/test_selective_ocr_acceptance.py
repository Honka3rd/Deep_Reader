#!/usr/bin/env python3
"""No-engine tests of the isolated end-to-end acceptance harness."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader, PdfWriter

from accept_selective_ocr import SCAN_LINES, accept, make_fixtures
from document_structure.heading_acceptance import heading_decision
from test_pdf_page_inspection import add_image, page_fixture


class SelectiveOcrAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.pdf"

    def test_synthetic_bookmarks_match_source_heading_lines(self):
        root = Path(self.temp.name)
        make_fixtures(root)
        reader = PdfReader(root / "mixed-outline.pdf")
        self.assertEqual(len(reader.pages), 3)
        self.assertEqual(reader.pages[2].extract_text(), "")
        titles = [bookmark.title for bookmark in reader.outline]
        self.assertEqual(titles, ["Chapter One", "Chapter Two"])
        sources = [reader.pages[0].extract_text(), "\n".join(SCAN_LINES)]
        for title, source in zip(titles, sources):
            with self.subTest(title=title):
                self.assertEqual(source.splitlines()[0], title)
                self.assertEqual(heading_decision(source, 0, title, corroborated=True), "accepted")
                self.assertEqual(heading_decision(title + ". Body text.", 0, title,
                                                 corroborated=True), "rejected")

    def test_native_cold_persistence_warm_readback_and_exact_text(self):
        writer, _ = page_fixture(
            b"BT /F1 3 Tf 5 70 Td (Readable text with enough detail to remain native.) Tj ET",
            with_font=True,
        )
        writer.write(self.path)
        with patch("subprocess.run", side_effect=AssertionError("native subprocess")):
            report = accept(self.path, expected=True, required_ocr_pages=0)
        self.assertEqual(report["evidence"]["quality_check"], "exact_native_text")
        self.assertEqual(report["evidence"]["boundaries"], "exact_ordered_spans")
        self.assertEqual(report["warm"]["save_calls_total"], 1)
        self.assertEqual(report["warm"]["process_calls"], {})
        self.assertIn("pdf_layout_evidence", report["cold"]["stages_ms"])
        self.assertNotIn("raw_load", report["warm"]["stages_ms"])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_disabled_recovery_rejects_without_saving_or_warm_success(self):
        writer, page = page_fixture(b"q 100 0 0 100 0 0 cm /Im Do Q")
        add_image(page)
        writer.write(self.path)
        with patch("subprocess.run", side_effect=AssertionError("disabled subprocess")):
            report = accept(self.path, ocr=False, expected=False)
        self.assertEqual(report["settings"]["mode"], "auto")
        self.assertEqual(report["cold"]["typed_failure"], "engine_unavailable")
        self.assertIsNone(report["warm"])
        self.assertEqual(report["cold"]["save_calls_total"], 0)
        self.assertIn("load_raw_text_requires_ocr:sample", report["cold"]["errors"])
        self.assertEqual(report["evidence"]["quality_check"], "rejected_no_persistence")

    def test_page_excerpt_does_not_change_original_or_claim_full_book(self):
        writer = PdfWriter()
        for number in (1, 2):
            _, page = page_fixture(
                f"BT /F1 3 Tf 5 70 Td (Chapter {number}. Readable native body.) Tj ET".encode(),
                with_font=True,
            )
            writer.add_page(page)
        writer.write(self.path)
        before = self.path.read_bytes()
        report = accept(self.path, page=2, expected=True)
        self.assertEqual(report["page_excerpt"], 2)
        self.assertEqual(report["evidence"]["page_count"], 1)
        self.assertNotEqual(report["source_sha256"], report["fixture_sha256"])
        self.assertEqual(before, self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
