#!/usr/bin/env python3
"""Outline-first scheduling using real synthetic PDFs and isolated dependencies."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypdf import PageObject, PdfWriter

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_outline import PdfOutlineEntry, PdfOutlineResult
from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from document_structure.section_splitter_selector import SectionSplitterMode
from document_structure.structured_document import StructuredSection
from document_structure.structured_document_builder import StructuredDocumentBuilder
from language.language_code import LanguageCode
from test_pdf_page_inspection import add_image, page_fixture
from test_prepare_structured_reuse_before_raw_load import (
    _build_pipeline, _fake_document, _StructuredStore,
)


PAGES = ["Chapter One\n\nA useful native paragraph.", "Chapter Two\nAnother useful paragraph."]
RAW = "\n".join(PAGES)
BOUNDARIES = [
    PdfPageTextBoundary(0, 0, len(PAGES[0]), PAGES[0], source_sha256="a" * 64),
    PdfPageTextBoundary(1, len(PAGES[0]) + 1, len(RAW), PAGES[1], source_sha256="a" * 64),
]
OUTLINE = PdfOutlineResult(
    present=True, usable=True, source_sha256="a" * 64,
    entries=[PdfOutlineEntry("Chapter One", 0, 0), PdfOutlineEntry("Chapter Two", 0, 1)],
)


class BuilderLazyLayoutTests(unittest.TestCase):
    def build(self, **overrides):
        kwargs = dict(document_id="sample", title="sample", raw_text=RAW,
                      language=LanguageCode.EN, page_boundaries=BOUNDARIES, outline=OUTLINE)
        kwargs.update(overrides)
        return StructuredDocumentBuilder().build(**kwargs)

    def test_validated_outline_skips_provider_even_with_internal_double_newlines(self):
        provider = Mock(side_effect=AssertionError("optional layout must stay lazy"))
        document = self.build(page_evidence_provider=provider)
        provider.assert_not_called()
        self.assertEqual(document.parse_provenance["effective_parser_mode"], "native_pdf_outline")
        self.assertEqual([chapter.title for chapter in document.chapters], ["Chapter One", "Chapter Two"])
        self.assertEqual(document.parse_provenance["layout_evidence"]["status"], "skipped_validated_outline")
        self.assertNotIn("sections", document.to_dict())
        self.assertNotIn("structure_nodes", document.to_dict())

    def test_present_usable_outline_still_requires_structure_owned_validation(self):
        cases = [
            {"outline": replace(OUTLINE, entries=[PdfOutlineEntry("bad", 0, 99)])},
            {"outline": replace(OUTLINE, entries=list(reversed(OUTLINE.entries)))},
            {"outline": replace(OUTLINE, source_sha256="b" * 64)},
            {"page_boundaries": None},
            {"page_boundaries": [BOUNDARIES[0], replace(BOUNDARIES[1], char_start=0)]},
            {"page_boundaries": [replace(BOUNDARIES[0], text="wrong"), BOUNDARIES[1]]},
            {"page_boundaries": [BOUNDARIES[0], replace(BOUNDARIES[1], source_sha256="b" * 64)]},
        ]
        for case in cases:
            with self.subTest(case=case):
                provider = Mock(return_value=None)
                document = self.build(page_evidence_provider=provider, **case)
                provider.assert_called_once_with()
                self.assertNotEqual(document.parse_provenance["effective_parser_mode"], "native_pdf_outline")
                self.assertIsNone(document.structure_error_code)

    def test_missing_and_loader_rejected_outline_fall_back_once(self):
        for outline in (None, replace(OUTLINE, usable=False), replace(OUTLINE, entries=[])):
            with self.subTest(outline=outline):
                provider = Mock(return_value=[])
                document = self.build(outline=outline, page_evidence_provider=provider)
                provider.assert_called_once_with()
                self.assertEqual(document.parse_provenance["layout_evidence"]["status"], "loaded")
                self.assertIsNone(document.structure_error_code)

    def test_optional_provider_failure_retains_common_parser(self):
        baseline = self.build(outline=None)
        provider = Mock(side_effect=RuntimeError("PRIVATE_ERROR_SENTINEL"))
        document = self.build(outline=None, page_evidence_provider=provider)
        provider.assert_called_once_with()
        self.assertIsNone(document.structure_error_code)
        self.assertEqual(document.chapters, baseline.chapters)
        self.assertEqual(document.parse_provenance["layout_evidence"], {"status": "failed", "error_type": "RuntimeError"})
        self.assertNotIn("PRIVATE_ERROR_SENTINEL", str(document.parse_provenance))

    def test_explicit_eager_evidence_does_not_call_provider(self):
        provider = Mock()
        document = self.build(outline=None, page_evidence=[], page_evidence_provider=provider)
        provider.assert_not_called()
        self.assertEqual(document.parse_provenance["layout_evidence"]["status"], "provided")

    def test_explicit_enhancement_is_not_overridden_by_outline_or_layout(self):
        selector = Mock()
        selector.get_splitter.return_value.split_candidate_with_provenance.return_value = (
            [StructuredSection("section-0", 0, "Chapter", 1, RAW, 0, len(RAW))],
            {"effective_parser_mode": "llm_enhanced"},
        )
        builder = StructuredDocumentBuilder(section_splitter_selector=selector)
        for outline in (OUTLINE, None):
            selector.reset_mock()
            provider = Mock(return_value=None)
            document = builder.build("sample", "sample", RAW, LanguageCode.EN,
                                     parser_mode=SectionSplitterMode.LLM_ENHANCED,
                                     outline=outline, page_boundaries=BOUNDARIES,
                                     page_evidence_provider=provider)
            self.assertIsNone(document.structure_error_code)
            self.assertEqual(provider.call_count, 0)
            self.assertEqual(selector.get_splitter.return_value.split_candidate_with_provenance.call_count, 1)


class PreparationLazyLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.pdf"
        self.loader = PdfDocumentLoader(base_dir=self.temp.name, ocr_enabled=False)
        self.store = _StructuredStore(exists_value=False, document=_fake_document("sample"))
        self.pipeline = _build_pipeline(loader=self.loader, store=self.store)
        self.builder = StructuredDocumentBuilder()
        self.pipeline.structured_document_builder = self.builder

    def write_pdf(self, *, outline=True, scan=False, suffix=""):
        writer = PdfWriter()
        for index in range(2):
            content = f"BT /F1 3 Tf 10 60 Td (Chapter {index + 1}) Tj 0 -10 Td (A complete native paragraph with readable words. {suffix}) Tj ET".encode()
            if scan and index == 1:
                content = b"q 100 0 0 100 0 0 cm /Im Do Q"
            _, page = page_fixture(content, with_font=not (scan and index == 1))
            if scan and index == 1:
                add_image(page)
            writer.add_page(page)
        if outline:
            writer.add_outline_item("Chapter 1", 0)
            writer.add_outline_item("Chapter 2", 1)
        writer.write(self.path)

    def prepare(self, **kwargs):
        return self.pipeline.prepare("sample", mode="base", pdf_extraction_mode="auto", **kwargs)

    def test_pdf_preparation_extracts_once_and_builds_once_without_layout(self):
        self.write_pdf()
        with patch.object(PageObject, "extract_text", autospec=True, side_effect=PageObject.extract_text) as extract, \
                patch.object(self.loader, "load_page_text_boundaries") as reread, \
                patch.object(self.loader, "load_page_layout_evidence") as layout, \
                patch.object(self.loader, "_render_page_images") as render, \
                patch("doc_loaders.pdf_document_loader.subprocess.run") as process, \
                patch.object(self.builder, "build", wraps=self.builder.build) as build:
            assets = self.prepare()
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertEqual(extract.call_count, 2)
        reread.assert_not_called()
        layout.assert_not_called()
        render.assert_not_called()
        process.assert_not_called()
        build.assert_called_once()
        self.assertEqual(self.store.save_calls, 1)
        self.assertEqual(self.pipeline.language_detector.detect_calls, 1)
        self.assertEqual(self.pipeline.profile_builder.build_calls, 1)
        for boundary in build.call_args.kwargs["page_boundaries"]:
            self.assertEqual(assets.raw_text[boundary.char_start:boundary.char_end], boundary.text)

    def test_force_rebuild_does_not_force_optional_ocr(self):
        self.write_pdf()
        self.store.exists_value = True
        with patch.object(self.loader, "load_page_layout_evidence") as layout:
            assets = self.prepare(force_rebuild=True)
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertEqual(self.store.save_calls, 1)
        layout.assert_not_called()

    def test_cache_reuse_does_not_load_any_source_or_call_llm(self):
        self.write_pdf()
        self.assertTrue(self.prepare().structured_document_ready)
        before = (self.store.save_calls, self.pipeline.language_detector.detect_calls,
                  self.pipeline.profile_builder.build_calls)
        with patch.object(self.loader, "extract", side_effect=AssertionError("extraction")), \
                patch.object(self.loader, "load", side_effect=AssertionError("legacy load")):
            result = self.pipeline.prepare_and_load("sample", mode="base", pdf_extraction_mode="auto")
        self.assertIs(result.structured_document, self.store.document)
        self.assertEqual(before, (self.store.save_calls, self.pipeline.language_detector.detect_calls,
                                 self.pipeline.profile_builder.build_calls))

    def test_missing_outline_loads_layout_once_with_existing_budget(self):
        self.write_pdf(outline=False)
        with patch.object(self.loader, "load_page_layout_evidence", wraps=self.loader.load_page_layout_evidence) as layout:
            assets = self.prepare()
        self.assertTrue(assets.structured_document_ready, assets.errors)
        layout.assert_called_once_with("sample.pdf", candidate_page_limit=30)
        self.assertEqual(self.store.document.parse_provenance["layout_evidence"]["status"], "loaded")

    def test_layout_failure_is_advisory_and_logged(self):
        self.write_pdf(outline=False)
        with patch.object(self.loader, "load_page_layout_evidence", side_effect=TimeoutError("PRIVATE")) as layout, \
                self.assertLogs("document_preparation.document_preparation_pipeline", level="WARNING") as logs:
            assets = self.prepare()
        self.assertTrue(assets.structured_document_ready, assets.errors)
        layout.assert_called_once()
        self.assertIn("error_type=TimeoutError", " ".join(logs.output))
        self.assertNotIn("PRIVATE", " ".join(logs.output))
        self.assertEqual(self.store.document.parse_provenance["layout_evidence"]["status"], "unavailable")

    def test_outline_cannot_bypass_required_body_recovery(self):
        from doc_loaders.pdf_extraction_contract import PdfExtractionError
        self.write_pdf(scan=True)
        with patch.object(self.loader, "load_outline") as outline, \
                patch.object(self.loader, "load_page_layout_evidence") as layout:
            with self.assertRaises(PdfExtractionError):
                self.prepare()
        outline.assert_not_called()
        layout.assert_not_called()
        self.assertEqual(self.store.save_calls, 0)
        self.assertEqual(self.pipeline.profile_builder.build_calls, 0)

    def test_recovered_body_precedes_outline_and_skips_only_layout(self):
        self.write_pdf(scan=True)
        self.loader.ocr_enabled = True
        events = []
        load_outline = self.loader._read_native_outline

        def recover(**kwargs):
            events.append("recover")
            self.assertEqual(kwargs["page_indexes"], [1])
            return ["", "Chapter 2\nRecovered complete scanned body text."]

        def outline(*args):
            events.append("outline")
            return load_outline(*args)

        with patch.object(self.loader, "_load_pages_with_ocr", side_effect=recover) as recovery, \
                patch.object(self.loader, "_read_native_outline", side_effect=outline), \
                patch.object(self.loader, "load_page_layout_evidence") as layout:
            assets = self.prepare()
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertEqual(events, ["recover", "outline"])
        recovery.assert_called_once()
        layout.assert_not_called()
        self.assertIn("Recovered complete scanned body", assets.raw_text)
        self.assertEqual(self.store.document.parse_provenance["effective_parser_mode"], "native_pdf_outline")

    def test_changed_source_evidence_does_not_authorize_old_raw_text(self):
        self.write_pdf()

        def change_source(_):
            self.write_pdf(suffix="new revision")
            return "zh"

        with patch.object(self.pipeline.language_detector, "detect", side_effect=change_source), \
                patch.object(self.loader, "load_page_layout_evidence", wraps=self.loader.load_page_layout_evidence) as layout:
            assets = self.prepare()
        self.assertFalse(assets.structured_document_ready)
        self.assertEqual(self.store.save_calls, 0)
        self.assertTrue(any("source changed" in error for error in assets.errors))
        self.assertNotIn("new revision", assets.raw_text)

    def test_new_pass_does_not_reuse_old_source_snapshot(self):
        self.write_pdf()
        first = self.prepare()
        self.write_pdf(suffix="new revision")
        second = self.prepare(force_rebuild=True)
        self.assertTrue(second.structured_document_ready, second.errors)
        self.assertNotEqual(first.raw_text, second.raw_text)
        self.assertIn("new revision", second.raw_text)
        self.assertEqual(self.store.save_calls, 2)


if __name__ == "__main__":
    unittest.main()
