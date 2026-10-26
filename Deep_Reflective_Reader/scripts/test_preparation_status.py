"""Browser-independent status with real HTTP routes and isolated preparation ports."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
import main
from app.preparation_operations import PreparationOperations
from api_schemas import DocumentTaskLayoutResponse
from document_structure.document_artifact_repository import DocumentListItem


class PreparationStatusTests(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        directory = self.enterContext(TemporaryDirectory())
        self.raw = Path(directory)
        (self.raw / "Book.txt").write_text("fixture", encoding="utf-8")
        self.operations = PreparationOperations()
        self.repository = Mock()
        self.repository.list_documents.return_value = []
        self.pipeline = Mock()
        self.coordinator = Mock()
        for name, value in {
            "_RAW_DOCUMENT_BASE_DIR": self.raw,
            "preparation_operations": self.operations,
            "document_artifact_repository": self.repository,
            "qa_coordinator": SimpleNamespace(document_preparation_pipeline=self.pipeline),
            "section_task_coordinator": self.coordinator,
        }.items():
            self.stack.enter_context(patch.object(main, name, value))
        self.stack.enter_context(patch.object(main, "get_document_task_layout", return_value=
            DocumentTaskLayoutResponse(document_id="Book", title="Book", language="en",
                                       chapters=[], enhanced_parse_recommendation=None)))

    def status(self):
        # A new client models browser reload/re-entry, not local component state.
        response = TestClient(main.app).get("/documents/management")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["items"][0]["preparation_status"]

    def assets(self, ready=True):
        return SimpleNamespace(structured_document_ready=ready, structured_document_path=None,
            faiss_ready=False, profile_ready=False, bundle_ready=False, errors=[],
            effective_pdf_extraction_mode=None, pdf_extraction_provenance="unknown")

    def test_running_survives_new_clients_and_rejects_duplicate_then_ready(self):
        started, release = Event(), Event()
        def prepare(**_):
            started.set()
            if not release.wait(10):
                raise TimeoutError("test worker not released")
            self.repository.list_documents.return_value = [
                DocumentListItem(doc_name="Book", title="Book", source="structured_file")]
            return self.assets()
        self.pipeline.prepare.side_effect = prepare
        self.assertEqual(self.status(), "not_prepared")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(TestClient(main.app).post, "/documents/prepare-task-layout",
                                  json={"doc_name": "Book"})
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(self.status(), "building")
                self.assertEqual(self.status(), "building")
                duplicate = TestClient(main.app).post("/documents/prepare", json={"doc_name": "Book", "mode": "base"})
                self.assertEqual(duplicate.status_code, 409, duplicate.text)
                self.assertEqual(self.status(), "building")
                delete = TestClient(main.app).post("/documents/lifecycle/delete",
                    json={"doc_name": "Book", "action": "delete", "confirm": True})
                self.assertEqual(delete.status_code, 409, delete.text)
                self.assertTrue((self.raw / "Book.txt").exists())
                self.repository.delete_document.assert_not_called()
            finally:
                release.set()
            response = pending.result(timeout=10)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.status(), "ready")
        self.assertEqual(self.pipeline.prepare.call_count, 1)

    def test_failure_is_visible_even_with_old_structure_and_retry_recovers(self):
        self.repository.list_documents.return_value = [
            DocumentListItem(doc_name="Book", title="Book", source="structured_file")]
        self.pipeline.prepare.side_effect = RuntimeError("test failure")
        response = TestClient(main.app).post("/documents/prepare-task-layout", json={"doc_name": "Book"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.status(), "failed")
        self.pipeline.prepare.side_effect = None
        self.pipeline.prepare.return_value = self.assets()
        self.assertEqual(TestClient(main.app).post("/documents/prepare-task-layout", json={"doc_name": "Book"}).status_code, 200)
        self.assertEqual(self.status(), "ready")

    def test_prepare_business_failure_is_not_http_success(self):
        self.pipeline.prepare.return_value = self.assets(False)
        response = TestClient(main.app).post("/documents/prepare", json={"doc_name": "Book", "mode": "base"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.status(), "failed")

    def test_json_error_and_exception_both_record_failure(self):
        for callback in (lambda _: JSONResponse(status_code=422, content={"reason": "extraction_failed"}),
                         lambda _: (_ for _ in ()).throw(HTTPException(status_code=503))):
            with self.subTest(callback=callback):
                wrapped = main._track_preparation(callback)
                try:
                    wrapped(SimpleNamespace(doc_name="Book"))
                except HTTPException:
                    pass
                self.assertEqual(self.status(), "failed")

    def test_reparse_tracks_actual_callback_and_business_failure(self):
        def reparse(**_):
            self.assertEqual(self.status(), "building")
            return SimpleNamespace(success=False, doc_name="Book", parser_mode="common",
                structured_document_path=None, error="invalid_structure", section_count=None)
        self.coordinator.reparse_document_structure.side_effect = reparse
        response = TestClient(main.app).post("/documents/reparse-structure",
            json={"doc_name": "Book", "parser_mode": "common"})
        self.assertFalse(response.json()["success"], response.text)
        self.assertEqual(self.status(), "failed")

    def test_success_never_invents_structure_and_restart_is_not_durable(self):
        with self.operations.track("Book") as outcome:
            outcome["success"] = True
        self.assertEqual(self.status(), "not_prepared")
        with self.operations.track("Book"):
            pass
        self.assertEqual(self.status(), "failed")
        with patch.object(main, "preparation_operations", PreparationOperations()):
            self.assertEqual(self.status(), "not_prepared")
        self.operations.forget("Book")
        self.assertEqual(self.status(), "not_prepared")

    def test_management_read_does_not_start_preparation(self):
        self.assertEqual(self.status(), "not_prepared")
        self.pipeline.prepare.assert_not_called()
        self.coordinator.ensure_document_task_layout.assert_not_called()
        self.assertEqual(self.operations.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
