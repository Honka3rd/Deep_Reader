#!/usr/bin/env python3
"""PDF-MODE-01 boundary tests; no live OCR, LLM, DB or user-document writes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from pydantic import ValidationError
import main
from api_schemas import (
    DocumentTaskLayoutResponse,
    PdfExtractionMetadataResponse,
    PrepareDocumentRequest,
    PrepareTaskLayoutRequest,
)
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_extraction_contract import (
    PdfExtractionError,
    PdfExtractionFailureReason,
    PdfExtractionMode,
    PdfExtractionProvenance,
)
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.prepared_document_assets import PreparedDocumentAssets


ENDPOINTS = ("/documents/prepare", "/documents/prepare-task-layout")


def assets():
    return PreparedDocumentAssets(
        doc_name="Text", raw_text=None, language=None,
        structured_document_ready=True, faiss_ready=False, profile_ready=False,
        bundle_ready=False, structured_document_path="fake-path", faiss_namespace=None,
    )


class ExtractionContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "Book.pdf").write_bytes(b"not parsed by these tests")
        (root / "Text.txt").write_text("not loaded by these tests", encoding="utf-8")
        self.factory = DocumentLoaderFactory(root)
        # Route contract tests stub preparation; integration tests exercise real extraction.
        self.pipeline = object.__new__(DocumentPreparationPipeline)
        self.pipeline.loader_factory = self.factory
        self.client = TestClient(main.app)
        ensure_patch = patch.object(main.section_task_coordinator, "ensure_document_task_layout")
        self.ensure_layout = ensure_patch.start()
        self.addCleanup(ensure_patch.stop)
        self.layout = DocumentTaskLayoutResponse(
            document_id="Text", title="Text", language=None, chapters=[],
            enhanced_parse_recommendation=None,
        )

    def test_request_defaults_and_independent_modes(self):
        for schema in (PrepareDocumentRequest, PrepareTaskLayoutRequest):
            self.assertEqual(schema(doc_name="Book").pdf_extraction_mode, PdfExtractionMode.FAST)
            for mode in PdfExtractionMode:
                request = schema(doc_name="Book", pdf_extraction_mode=mode.value,
                                 structured_parser_mode="llm_enhanced")
                self.assertEqual(request.pdf_extraction_mode, mode)
                self.assertEqual(request.structured_parser_mode, "llm_enhanced")
        self.assertEqual(PrepareDocumentRequest(doc_name="Book", mode="free_qa").mode, "free_qa")

    def test_invalid_modes_rejected_before_pipeline(self):
        with patch.object(main.qa_coordinator, "document_preparation_pipeline") as pipeline:
            pipeline.prepare.side_effect = AssertionError("invalid input reached preparation")
            for endpoint in ENDPOINTS:
                for invalid in (None, "", "FAST", "unknown", True, 42, {}, []):
                    with self.subTest(endpoint=endpoint, invalid=invalid):
                        result = self.client.post(endpoint, json={"doc_name": "Book", "pdf_extraction_mode": invalid})
                        self.assertEqual(result.status_code, 422, result.text)
            pipeline.prepare.assert_not_called()

    def test_pdf_requests_forward_snapshot_and_identity_to_preparation(self):
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", self.pipeline), \
                patch.object(self.pipeline, "_prepare_in_scope", side_effect=PdfExtractionError(PdfExtractionFailureReason.NO_TEXT, "fixture has no text")) as prepare, \
                patch.object(self.factory, "preparation_scope") as scope, \
                patch.object(main, "get_document_task_layout") as project:
            for endpoint in ENDPOINTS:
                for name in ("Book", "Book.pdf"):
                    for mode in (None, "fast", "ocr", "auto"):
                        for force in (False, True):
                            with self.subTest(endpoint=endpoint, name=name, mode=mode, force=force):
                                request = {"doc_name": name, "force_rebuild": force}
                                if mode is not None:
                                    request["pdf_extraction_mode"] = mode
                                response = self.client.post(endpoint, json=request)
                                self.assertEqual(response.status_code, 422, response.text)
                                self.assertEqual(response.json()["reason"], "no_text")
                                forwarded = prepare.call_args.kwargs["source_session"]
                                self.assertEqual(forwarded.identity.mode, mode or "fast")
                                self.assertEqual(forwarded.identity.source_format, "pdf")
                                forwarded.verify_source()
                                self.assertEqual(response.json()["pdf_extraction"], {
                                    "requested_mode": mode or "fast", "effective_mode": None,
                                    "provenance": "unknown",
                                })
            self.assertTrue(prepare.called)
            self.assertTrue(scope.called)
            project.assert_not_called()

    def test_text_fast_is_noop_and_response_is_not_pdf_provenance(self):
        def prepared(*args, **kwargs):
            result = assets()
            result.apply_extraction_status(kwargs["source_session"].compatibility_status("applied"))
            return result
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", self.pipeline), \
                patch.object(self.pipeline, "_prepare_in_scope", side_effect=prepared) as prepare, \
                patch.object(main, "get_document_task_layout", return_value=self.layout) as project:
            for endpoint in ENDPOINTS:
                for explicit in (False, True):
                    request = {"doc_name": "Text", "force_rebuild": True, "structured_parser_mode": "llm_enhanced"}
                    if explicit:
                        request["pdf_extraction_mode"] = "fast"
                    response = self.client.post(endpoint, json=request)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["pdf_extraction"], {
                        "requested_mode": "fast", "effective_mode": None,
                        "provenance": "not_applicable",
                    })
                    self.assertEqual(prepare.call_args.args, ("Text", True, "base", "llm_enhanced"))
            self.assertEqual(project.call_count, 2)
            self.assertNotIn("pdf_extraction", self.layout.model_dump())

    def test_text_ocr_auto_rejected_before_preparation(self):
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", self.pipeline), \
                patch.object(self.pipeline, "_prepare_in_scope") as prepare:
            for endpoint in ENDPOINTS:
                for mode in ("ocr", "auto"):
                    response = self.client.post(endpoint, json={"doc_name": "Text", "pdf_extraction_mode": mode})
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertEqual(response.json()["reason"], "mode_not_applicable")
            prepare.assert_not_called()

    def test_routes_forward_mode_and_preserve_request_options(self):
        pipeline = Mock()
        pipeline.prepare.return_value = assets()
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline), \
                patch.object(main, "get_document_task_layout", return_value=self.layout) as project:
            for mode in PdfExtractionMode:
                response = self.client.post(ENDPOINTS[0], json={
                    "doc_name": "Text", "pdf_extraction_mode": mode.value, "mode": "free_qa",
                    "force_rebuild": True, "structured_parser_mode": "llm_enhanced",
                })
                self.assertEqual(response.status_code, 400)  # Fake assets have no free-QA readiness.
                self.assertEqual(pipeline.prepare.call_args.kwargs, {
                    "doc_name": "Text", "pdf_extraction_mode": mode, "mode": "free_qa",
                    "force_rebuild": True, "structured_parser_mode": "llm_enhanced",
                    "toc_hint": "auto",
                    "enhancement_intent": "restore_source",
                })
                response = self.client.post(ENDPOINTS[1], json={
                    "doc_name": "Text", "pdf_extraction_mode": mode.value,
                    "refresh_task_units": True, "task_unit_split_mode": "semantic_safe",
                    "include_anchor_page_evidence": True,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(pipeline.prepare.call_args.kwargs["pdf_extraction_mode"], mode)
                self.assertTrue(project.call_args.args[0].include_anchor_page_evidence)
                self.assertFalse(project.call_args.args[0].refresh_task_units)
                self.assertTrue(self.ensure_layout.call_args.kwargs["refresh_task_units"])
                self.assertIsNone(response.json()["pdf_extraction"]["effective_mode"])

    def test_typed_failure_mapping_for_both_routes(self):
        for reason, status in main._PDF_EXTRACTION_FAILURE_HTTP_STATUS.items():
            pipeline = Mock()
            pipeline.prepare.side_effect = PdfExtractionError(reason, "Safe fixture detail")
            with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline), \
                    patch.object(main, "get_document_task_layout") as project:
                for endpoint in ENDPOINTS:
                    response = self.client.post(endpoint, json={"doc_name": "Book", "pdf_extraction_mode": "ocr"})
                    self.assertEqual(response.status_code, status, response.text)
                    self.assertEqual(response.json()["reason"], reason.value)
                    self.assertEqual(response.json()["detail"], "Safe fixture detail")
                    self.assertIsNone(response.json()["pdf_extraction"]["effective_mode"])
                project.assert_not_called()

    def test_toc_hint_forwarded_independently_of_extraction_mode(self):
        pipeline = Mock()
        pipeline.prepare.return_value = assets()
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline), \
                patch.object(main, "get_document_task_layout", return_value=self.layout):
            for endpoint in ENDPOINTS:
                for hint in ("auto", "present", "absent"):
                    response = self.client.post(endpoint, json={"doc_name": "Text", "toc_hint": hint})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(pipeline.prepare.call_args.kwargs["toc_hint"], hint)
                    self.assertEqual(pipeline.prepare.call_args.kwargs["pdf_extraction_mode"], "fast")
            pipeline.reset_mock()
            response = self.client.post(ENDPOINTS[1], json={
                "doc_name": "Text", "layout_only": True, "toc_hint": "absent"})
            self.assertEqual(response.status_code, 400)
            pipeline.prepare.assert_not_called()

    def test_reparse_route_forwards_toc_declaration(self):
        result = SimpleNamespace(success=True, doc_name="Text", parser_mode="common",
                                 structured_document_path="fake", section_count=1)
        with patch.object(main.section_task_coordinator, "reparse_document_structure", return_value=result) as reparse:
            for hint in ("auto", "present", "absent"):
                response = self.client.post("/documents/reparse-structure", json={
                    "doc_name": "Text", "parser_mode": "common", "toc_hint": hint})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(reparse.call_args.kwargs["toc_hint"], hint)

    def test_enhancement_intent_handoff_and_execution_error(self):
        pipeline = Mock()
        pipeline.prepare.side_effect = ValueError("enhancement_candidate_rejected")
        with patch.object(main.qa_coordinator, "document_preparation_pipeline", pipeline):
            for endpoint in ENDPOINTS:
                response = self.client.post(endpoint, json={"doc_name": "Text",
                    "structured_parser_mode": "llm_enhanced", "toc_hint": "present",
                    "enhancement_intent": "generate_reading_outline"})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(pipeline.prepare.call_args.kwargs["enhancement_intent"], "generate_reading_outline")
                self.assertEqual(pipeline.prepare.call_args.kwargs["toc_hint"], "present")
        with patch.object(main.section_task_coordinator, "reparse_document_structure",
                          side_effect=ValueError("enhancement_candidate_rejected")) as reparse:
            response = self.client.post("/documents/reparse-structure", json={"doc_name": "Text",
                "parser_mode": "llm_enhanced", "enhancement_intent": "generate_reading_outline"})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["detail"], "confirmation_required")
            reparse.assert_not_called()

    def test_provenance_validation_prevents_invented_effective_mode(self):
        for invalid in (
            {"effective_mode": "fast", "provenance": "unknown"},
            {"effective_mode": "ocr", "provenance": "not_applicable"},
            {"provenance": "applied"},
            {"provenance": "reused"},
            {"requested_mode": "fast", "effective_mode": "auto", "provenance": "applied"},
        ):
            with self.assertRaises(ValidationError):
                PdfExtractionMetadataResponse(**invalid)
        self.assertEqual(PdfExtractionMetadataResponse(
            requested_mode="ocr", effective_mode="ocr", provenance="applied",
        ).effective_mode, PdfExtractionMode.OCR)

    def test_legacy_assets_remain_unknown_and_roundtrip(self):
        payload = assets().to_dict()
        for key in ("requested_pdf_extraction_mode", "effective_pdf_extraction_mode", "pdf_extraction_provenance"):
            payload.pop(key)
        restored = PreparedDocumentAssets.from_dict(payload)
        self.assertIsNone(restored.requested_pdf_extraction_mode)
        self.assertIsNone(restored.effective_pdf_extraction_mode)
        self.assertEqual(restored.pdf_extraction_provenance, PdfExtractionProvenance.UNKNOWN)
        self.assertEqual(PreparedDocumentAssets.from_dict(json.loads(json.dumps(restored.to_dict()))), restored)

    def test_session_default_is_fast_but_unexecuted_result_is_not_labelled_applied(self):
        with patch.object(self.pipeline, "_prepare_in_scope", return_value=assets()) as prepare, \
                patch.object(self.factory, "get", wraps=self.factory.get) as get:
            result = self.pipeline.prepare("Book", mode="base")
            self.assertEqual(result.pdf_extraction_provenance, PdfExtractionProvenance.UNKNOWN)
            self.assertIsNone(result.effective_pdf_extraction_mode)
            get.assert_called_once_with("Book")
            self.assertEqual(prepare.call_args.kwargs["source_session"].identity.mode, "fast")
            prepare.assert_called_once()

    def test_openapi_exposes_only_explicit_preparation_modes(self):
        schema = self.client.get("/openapi.json").json()
        models = schema["components"]["schemas"]
        self.assertEqual(models["PdfExtractionMode"]["enum"], ["fast", "ocr", "auto"])
        for name in ("PrepareDocumentRequest", "PrepareTaskLayoutRequest"):
            self.assertEqual(models[name]["properties"]["pdf_extraction_mode"]["default"], "fast")
        self.assertNotIn("pdf_extraction", models["DocumentTaskLayoutResponse"]["properties"])
        for endpoint in ENDPOINTS:
            for status in ("422", "501", "502", "503"):
                self.assertIn(status, schema["paths"][endpoint]["post"]["responses"])


if __name__ == "__main__":
    unittest.main()
