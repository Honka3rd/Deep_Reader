#!/usr/bin/env python3
"""Mode propagation with real extraction and isolated preparation substitutes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.pdf_extraction_contract import PdfExtractionError
from test_pdf_page_inspection import page_fixture
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _fake_document


class ModePreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "sample.pdf"
        writer, _ = page_fixture(b"BT /F1 12 Tf 10 70 Td (A readable paragraph about science and evidence.) Tj ET", with_font=True)
        writer.write(self.path)
        self.loader = PdfDocumentLoader(base_dir=temp.name, ocr_enabled=False)
        self.store = _StructuredStore(exists_value=False, document=_fake_document("sample"))
        self.pipeline = _build_pipeline(loader=self.loader, store=self.store)
        def build(**kwargs):
            kwargs["page_evidence_provider"]()
            document = _fake_document("sample")
            return replace(document, raw_text=kwargs["raw_text"])
        self.pipeline.structured_document_builder = Mock()
        self.pipeline.structured_document_builder.build.side_effect = build

    def prepare(self, mode="fast", **kwargs):
        return self.pipeline.prepare("sample", mode="base", pdf_extraction_mode=mode, **kwargs)

    def test_fast_optional_fallback_never_calls_legacy_loaders_and_reuses(self):
        with patch.object(self.loader, "_inspect_page", side_effect=AssertionError("auto inspection")), \
                patch.object(self.loader, "load_page_layout_evidence", side_effect=AssertionError("auto layout")), \
                patch.object(self.loader, "_render_page_images", side_effect=AssertionError("render")):
            first = self.prepare()
            self.assertTrue(first.structured_document_ready, first.errors)
            self.assertEqual(first.pdf_extraction_provenance, "applied")
            with patch.object(self.loader, "extract", side_effect=AssertionError("cache miss")):
                second = self.prepare()
            self.assertEqual(second.pdf_extraction_provenance, "reused")
            self.assertEqual(self.store.save_calls, 1)

    def test_mode_switch_rejects_legacy_cache_and_preserves_on_failure(self):
        self.assertTrue(self.prepare().structured_document_ready)
        old = self.store.document
        with patch("doc_loaders.pdf_full_ocr_extraction.shutil.which", return_value=None):
            with self.assertRaises(PdfExtractionError):
                self.prepare("ocr")
        self.assertIs(self.store.document, old)
        self.assertEqual(self.store.save_calls, 1)

    def test_auto_uses_existing_selective_path_on_native_source(self):
        assets = self.prepare("auto")
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertEqual(assets.effective_pdf_extraction_mode, "auto")
        self.assertEqual(assets.pdf_extraction_identity.mode, "auto")

    def test_manual_evidence_reuses_saved_mode_without_ocr_or_language_call(self):
        self.assertTrue(self.prepare().structured_document_ready)
        with patch.object(self.loader, "load", side_effect=AssertionError("legacy load")), \
                patch.object(self.pipeline.language_detector, "detect", side_effect=AssertionError("language LLM")):
            evidence = self.pipeline.load_manual_structure_source_evidence("sample")
            self.assertEqual(evidence.raw_text, self.store.document.raw_text)
            self.assertEqual(len(evidence.page_boundaries), 1)

    def test_changed_source_rejects_manual_evidence_without_rebuild(self):
        self.assertTrue(self.prepare().structured_document_ready)
        self.path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.pipeline.load_manual_structure_source_evidence("sample")
        self.assertEqual(self.store.save_calls, 1)

    def test_optional_provider_failure_keeps_parser_fallback(self):
        result = self.loader.extract("sample")
        result = replace(result, outline_provider=Mock(side_effect=ValueError("bad outline")),
                         layout_provider=Mock(side_effect=ValueError("bad optional layout")))
        with patch.object(self.loader, "extract", return_value=result):
            assets = self.prepare()
        self.assertTrue(assets.structured_document_ready, assets.errors)

    def test_request_modes_do_not_leak_between_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        barrier = Barrier(2)
        original = self.loader.extract
        def extract(*args, **kwargs):
            barrier.wait(timeout=10)
            return original(*args, **kwargs)
        with patch.object(self.loader, "extract", side_effect=extract), ThreadPoolExecutor(max_workers=2) as executor:
            jobs = [executor.submit(self.prepare, mode, force_rebuild=True) for mode in ("fast", "auto")]
            results = [job.result(timeout=15) for job in jobs]
        self.assertEqual([result.effective_pdf_extraction_mode for result in results], ["fast", "auto"])
        self.assertTrue(all(result.structured_document_ready for result in results))

    def test_prepare_api_uses_default_fast_and_reports_reuse(self):
        import main
        from document_structure.structured_document_builder import StructuredDocumentBuilder
        from fastapi.testclient import TestClient
        self.pipeline.structured_document_builder = StructuredDocumentBuilder()
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", self.pipeline):
            client = TestClient(main.app)
            for provenance in ("applied", "reused"):
                response = client.post("/documents/prepare", json={"doc_name": "sample"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["pdf_extraction"]["effective_mode"], "fast")
                self.assertEqual(response.json()["pdf_extraction"]["provenance"], provenance)

    def test_full_ocr_executor_reaches_structured_save_with_mode_and_spans(self):
        from test_pdf_full_ocr_extraction import FullOcrTests
        fixture = FullOcrTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.pipeline.loader_factory.loader = fixture.loader
        with patch("shutil.which", return_value=sys.executable):
            assets = self.prepare("ocr")
        self.assertTrue(assets.structured_document_ready, assets.errors)
        self.assertEqual(assets.effective_pdf_extraction_mode, "ocr")
        self.assertEqual(fixture.render.call_count, 3)
        self.assertEqual(len(self.store.document.parse_provenance["pdf_page_spans"]), 3)
        self.assertEqual(self.store.document.parse_provenance["pdf_extraction"]["identity"]["mode"], "ocr")
        with patch.object(fixture.loader, "load", side_effect=AssertionError("read OCR")), \
                patch.object(fixture.loader, "extract", side_effect=AssertionError("read extraction")), \
                patch.object(self.pipeline.language_detector, "detect", side_effect=AssertionError("read language")):
            boundaries = self.pipeline.load_task_layout_page_boundaries("sample", document=self.store.document)
        self.assertEqual(len(boundaries), 3)
        self.assertEqual(fixture.render.call_count, 3)

    def test_prepare_layout_api_reports_applied_mode(self):
        import main
        from api_schemas import DocumentTaskLayoutResponse
        from fastapi.testclient import TestClient
        layout = DocumentTaskLayoutResponse(document_id="sample", title="sample", language="zh", chapters=[], enhanced_parse_recommendation=None)
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", self.pipeline), \
                patch.object(main.section_task_coordinator, "ensure_document_task_layout") as ensure, \
                patch.object(main, "get_document_task_layout", return_value=layout) as project:
            response = TestClient(main.app).post("/documents/prepare-task-layout", json={"doc_name": "sample"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["pdf_extraction"]["provenance"], "applied")
        project.assert_called_once()
        ensure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
