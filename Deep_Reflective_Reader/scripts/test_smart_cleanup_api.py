"""Actual FastAPI contracts over temporary files and disposable PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from threading import Event
from unittest.mock import patch
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from api_schemas import SmartCleanupExecuteRequest, SmartCleanupOperationResponse, SmartCleanupPreflightResponse
from app.standalone_cleanup import CleanupUnavailable
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
from document_preparation.cleanup_input_adapter import adapt_document_cleanup_input
from document_structure.reading_content import state_for
from document_structure.structured_document import StructuredDocument
from smart_cleanup_routes import get_smart_cleanup_service
from scripts import test_standalone_cleanup_service as fixtures
from scripts.test_ocr_cleanup_contract import item


BASE = "/documents/book/smart-cleanup"


class SmartCleanupApiTests(unittest.TestCase):
    fixture_type = fixtures.StandaloneCleanupServiceTests

    def setUp(self):
        self.f = self.fixture_type()
        if hasattr(self, "scoped"):
            self.f.scoped = self.scoped
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.flow = self.f.flow
        self.flow.read_input = self.read_input
        overrides = dict(main.app.dependency_overrides)
        self.addCleanup(setattr, main.app, "dependency_overrides", overrides)
        main.app.dependency_overrides[get_smart_cleanup_service] = lambda: self.flow
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def read_input(self, doc_name):
        repo = self.f.repo
        if hasattr(repo, "load_enhancement_snapshot"):
            stored = repo.load_enhancement_snapshot(doc_name)
            document, version = stored.document, stored.structure_version
        else:
            document = repo.load_document(doc_name)
            version = document.parse_provenance.get("storage_structure_version", 1)
        source = resolve_accepted_source_snapshot(document=document, namespace=self.f.namespace,
            expected_document_id=doc_name, structure_version=version,
            source_path=Path(document.source_path), accepted=True)
        return adapt_document_cleanup_input(document=document, snapshot=source,
            expected_reading_revision=state_for(document).get("revision", 0))

    def restart(self):
        self.flow = self.f.make_flow()
        self.flow.read_input = self.read_input

    def consent(self):
        response = self.client.post(BASE + "/preflight", json={})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["execution_available"], response.text)
        return {key: body[key] for key in ("operation_id", "confirmation_token")}

    def execute(self, request):
        return self.client.post(BASE, json=request)

    def lookup(self, request):
        return self.client.get(BASE + "/" + request["operation_id"])

    def test_only_three_public_routes_and_no_domain_contract_leak(self):
        schema = self.client.get("/openapi.json").json()
        self.assertEqual({p: set(v) for p, v in schema["paths"].items() if "smart-cleanup" in p}, {
            "/documents/{doc_name}/smart-cleanup/preflight": {"post"},
            "/documents/{doc_name}/smart-cleanup": {"post"},
            "/documents/{doc_name}/smart-cleanup/{operation_id}": {"get"}})
        for name in ("CleanupInput", "CleanupCommitCommand", "CleanupOperationBinding", "StagedCleanup"):
            self.assertNotIn(name, schema["components"]["schemas"])
        properties = schema["components"]["schemas"]["SmartCleanupExecuteRequest"]["properties"]
        self.assertEqual(set(properties), {"operation_id", "confirmation_token"})
        error_schema = schema["paths"]["/documents/{doc_name}/smart-cleanup"]["post"]["responses"]["422"]
        self.assertEqual(error_schema["content"]["application/json"]["schema"]["$ref"],
                         "#/components/schemas/SmartCleanupErrorResponse")

    def test_preflight_lookup_read_only_and_minimal(self):
        before = self.f.snapshot()
        with patch.object(self.f.repo, "commit_cleanup", side_effect=AssertionError("read cannot write")), \
             patch.object(self.flow, "prepare_layout", side_effect=AssertionError("read cannot prepare")):
            request = self.consent()
            response = self.lookup(request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["phase"], "planning")
        self.assertIsNone(response.json()["outcome"])
        self.assertIsNone(response.json()["reading_ready"])
        self.assertEqual(before, self.f.snapshot())
        self.assertFalse(self.f.provider.calls)
        self.assertNotIn(request["confirmation_token"], response.text)
        for field in ("binding", "source_hash", "canonical_text", "prompt_version", "capacity", "namespace"):
            self.assertNotIn(field, response.json())

    def test_success_duplicate_post_and_lookup_preserve_one_commit(self):
        request = self.consent()
        with patch.object(self.f.repo, "commit_cleanup", wraps=self.f.repo.commit_cleanup) as commit:
            first = self.execute(request)
            second = self.execute(request)
            lookup = self.lookup(request)
            self.assertEqual(commit.call_count, 1)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json(), lookup.json())
        result = first.json()
        self.assertEqual(result["commit"], "committed")
        self.assertTrue(result["reading_ready"])
        self.assertEqual(result["outcome"]["applied_count"], 1)
        self.assertEqual(result["pending_review_count"], 0)
        self.assertEqual(len(self.f.provider.calls), 1)
        self.assertNotIn("success", result)
        self.assertNotIn("structure_changed", result)

    def test_noop_preserves_revision_generated_artifacts_and_review(self):
        self.f.plan([])
        before = self.f.snapshot()[0]
        response = self.execute(self.consent())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["outcome"]["status"], "no_changes")
        self.assertEqual(result["outcome"]["reading_revision"], 0)
        self.assertEqual(result["pending_review_count"], 1)
        self.assertEqual(before.document_task_artifacts, self.f.snapshot()[0].document_task_artifacts)

    def test_partial_counts_and_retry_do_not_imply_perfect_ocr(self):
        self.f.plan([item(quote="@@??"), item(quote="not found")])
        response = self.execute(self.consent())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["outcome"]["status"], "partial")
        self.assertEqual(result["outcome"]["skipped_count"], 1)
        self.assertEqual(result["retry_action"], "regenerate_cleanup")

    def test_provider_failure_is_typed_not_commit_or_transport_success(self):
        self.f.provider.error = TimeoutError("PRIVATE_PROVIDER")
        request = self.consent()
        response = self.execute(request)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["outcome"]["status"], "failed")
        self.assertEqual(result["commit"], "not_committed")
        self.assertEqual(result["reason"], "provider_failed")
        self.assertIsNone(result["reading_ready"])
        self.assertEqual(len(self.f.provider.calls), 2)
        self.assertEqual(self.execute(request).json(), result)
        self.assertEqual(len(self.f.provider.calls), 2)
        self.assertNotIn("PRIVATE_PROVIDER", response.text)

    def test_capacity_and_ineligibility_do_not_issue_handles_or_call_model(self):
        self.f.provider.count = 10000000
        response = self.client.post(BASE + "/preflight")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["reason"], "capacity_exceeded")
        self.assertIsNone(response.json()["operation_id"])
        self.f.provider.count = None
        document = self.f.repo.load_document("book")
        provenance = json.loads(json.dumps(document.parse_provenance))
        provenance["extraction"]["identity"]["mode"] = "fast"
        self.f.repo.save_reparsed_document(replace(document, parse_provenance=provenance), "book")
        response = self.client.post(BASE + "/preflight", json={})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["execution_available"])
        self.assertIsNone(response.json()["confirmation_token"])
        self.assertFalse(self.f.provider.calls)

    def test_unknown_fields_parser_namespace_and_plans_rejected_before_work(self):
        request = self.consent()
        for field, value in (("doc_name", "other"), ("namespace", "other"), ("discard_interactions", True),
                             ("parser_mode", "llm_enhanced"), ("plan", {}), ("source", "PRIVATE_SOURCE"),
                             ("offset", 1), ("model", "other"), ("confirmed", True), ("action", "apply")):
            for path, body in ((BASE, {**request, field: value}), (BASE + "/preflight", {field: value})):
                with self.subTest(field=field, path=path):
                    response = self.client.post(path, json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertEqual(response.json(), {"detail": "cleanup_request_invalid"})
        self.assertFalse(self.f.provider.calls)

    def test_handles_reject_coercion_blank_oversize_and_wrong_types(self):
        request = self.consent()
        for field in request:
            for value in (None, True, 1, [], {}, "", " ", "x" * 2048):
                with self.subTest(field=field, value=type(value).__name__):
                    response = self.execute({**request, field: value})
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn(request["confirmation_token"], response.text)
        for payload in ([], "string", True, 3):
            self.assertEqual(self.client.post(BASE + "/preflight", json=payload).status_code, 422)
        self.assertFalse(self.f.provider.calls)

    def test_unknown_queries_and_get_body_cannot_pass_hidden_settings(self):
        request = self.consent()
        for path in (BASE, BASE + "/preflight", BASE + "/" + request["operation_id"]):
            response = self.client.request("GET" if path.endswith(request["operation_id"]) else "POST",
                path + "?namespace=other", json=request if path == BASE else {})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.request("GET", BASE + "/" + request["operation_id"], json={"plan": {}}).status_code, 422)
        self.assertFalse(self.f.provider.calls)

    def test_malformed_json_and_invalid_paths_are_sanitized(self):
        response = self.client.post(BASE, content='{"confirmation_token":"PRIVATE_SECRET",',
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertNotIn("PRIVATE_SECRET", response.text)
        for name in ("%20", "book%5Cother", "book%00", "x" * 256):
            response = self.client.post("/documents/" + name + "/smart-cleanup/preflight", json={})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.get(BASE + "/bad-handle").status_code, 422)

    def test_missing_document_is_404_not_unknown_or_empty_success(self):
        request = self.consent()
        for method, path, body in (("POST", "/preflight", {}), ("POST", "", request),
                                   ("GET", "/" + request["operation_id"], None)):
            response = self.client.request(method, "/documents/missing/smart-cleanup" + path,
                                           **({"json": body} if body is not None else {}))
            self.assertEqual(response.status_code, 404, response.text)
        self.assertFalse(self.f.provider.calls)

    def test_wrong_token_is_409_before_and_after_commit_and_restart(self):
        request = self.consent()
        wrong = {**request, "confirmation_token": "A" * 43}
        self.assertEqual(self.execute(wrong).status_code, 409)
        self.execute(request)
        self.restart()
        self.f.ready = False
        response = self.execute(wrong)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.f.layout_calls, 0)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_expired_consent_is_409(self):
        request = self.consent()
        self.f.now = 300
        response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], "cleanup_confirmation_expired")
        self.assertFalse(self.f.provider.calls)

    def test_cross_document_handle_cannot_execute_or_read_result(self):
        request = self.consent()
        document = self.f.repo.load_document("book")
        self.f.repo.save_document(replace(document, document_id="other"), "other")
        response = self.client.post("/documents/other/smart-cleanup", json=request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["phase"], "unknown")
        self.execute(request)
        response = self.client.get("/documents/other/smart-cleanup/" + request["operation_id"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["outcome"])
        self.assertEqual(response.json()["phase"], "unknown")
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_namespace_owned_by_injected_service_not_headers(self):
        request = self.consent()
        self.execute(request)
        other = self.fixture_type()
        if hasattr(self, "scoped"):
            other.scoped = self.scoped
        other.setUp()
        self.addCleanup(other.doCleanups)
        self.flow = other.flow
        response = self.client.get(BASE + "/" + request["operation_id"], headers={"X-Namespace": self.f.namespace})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["phase"], "unknown")
        self.assertIsNone(response.json()["outcome"])
        self.assertFalse(other.provider.calls)

    def test_stale_source_and_revision_are_typed_conflicts(self):
        request = self.consent()
        block = self.f.service.read("book")["blocks"][1]
        self.f.change("edit", block_id=block["block_id"], text="Keep my correction")
        response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["commit"], "not_committed")
        request = self.consent()
        self.f.path.write_bytes(b"changed source")
        response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["reason"], "source_changed")
        self.assertFalse(self.f.provider.calls)

    def test_restart_unknown_is_readable_but_post_cannot_regenerate(self):
        request = self.consent()
        self.restart()
        response = self.lookup(request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["retry_action"], "reconcile")
        self.assertEqual(self.execute(request).status_code, 409)
        self.assertFalse(self.f.provider.calls)

    def test_read_failure_is_503_with_unknown_not_missing(self):
        request = self.consent()
        with patch.object(self.f.repo, "lookup_cleanup", side_effect=OSError("PRIVATE_DB")):
            for response in (self.lookup(request), self.execute(request)):
                self.assertEqual(response.status_code, 503, response.text)
                self.assertEqual(response.json()["phase"], "unknown")
                self.assertEqual(response.json()["commit"], "unknown")
                self.assertNotIn("PRIVATE_DB", response.text)
        self.assertFalse(self.f.provider.calls)

    def test_commit_failure_remains_unknown_and_never_blindly_replays(self):
        request = self.consent()
        before = self.f.snapshot()
        with self.f.fail_write():
            response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["commit"], "unknown")
        self.assertEqual(before, self.f.snapshot())
        self.assertEqual(self.execute(request).status_code, 409)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_lost_acknowledgement_reconciles_exactly_once(self):
        request = self.consent()
        commit = self.f.repo.commit_cleanup
        def lose(command):
            commit(command)
            raise OSError("lost ack")
        with patch.object(self.f.repo, "commit_cleanup", side_effect=lose) as write:
            self.assertEqual(self.execute(request).status_code, 200)
            self.restart()
            self.assertEqual(self.execute(request).status_code, 200)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_layout_failure_get_is_readonly_same_post_resumes_after_restart(self):
        self.f.ready = False
        request = self.consent()
        with patch.object(self.flow, "prepare_layout", side_effect=OSError("layout failed")):
            response = self.execute(request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["phase"], "layout")
        self.assertFalse(response.json()["reading_ready"])
        self.assertEqual(response.json()["retry_action"], "resume_layout")
        self.assertEqual(self.lookup(request).json()["phase"], "layout")
        self.assertEqual(self.f.layout_calls, 0)
        self.restart()
        with patch.object(self.f.repo, "commit_cleanup", side_effect=AssertionError("no second commit")):
            response = self.execute(request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["reading_ready"])
        self.assertEqual(self.f.layout_calls, 1)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_undo_exposes_current_pending_review_and_supersedes_old_result(self):
        request = self.consent()
        self.execute(request)
        self.f.change("undo")
        result = self.lookup(request).json()
        self.assertEqual(result["phase"], "superseded")
        self.assertEqual(result["pending_review_count"], 1)
        self.assertIsNone(result["reading_ready"])
        self.assertEqual(result["retry_action"], "none")
        self.execute(request)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_concurrent_post_busy_preflight_and_get_never_generate_twice(self):
        request = self.consent()
        entered, release = Event(), Event()
        complete = self.f.provider.complete_text_with_output_budget
        def wait(*args, **kwargs):
            entered.set()
            if not release.wait(15):
                raise AssertionError("provider not released")
            return complete(*args, **kwargs)
        with patch.object(self.f.provider, "complete_text_with_output_budget", side_effect=wait):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self.execute, request)
                try:
                    self.assertTrue(entered.wait(15))
                    self.assertEqual(self.lookup(request).json()["phase"], "planning")
                    self.assertEqual(self.execute(request).status_code, 409)
                    self.assertEqual(self.client.post(BASE + "/preflight", json={}).status_code, 409)
                finally:
                    release.set()
                self.assertEqual(future.result().status_code, 200)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_unconfigured_service_reports_unavailable_without_fallback(self):
        request = self.consent()
        main.app.dependency_overrides.pop(get_smart_cleanup_service)
        original = main.app.state.smart_cleanup_service
        with patch.object(main.app.state, "smart_cleanup_service", None):
            response = self.client.post(BASE + "/preflight", json={})
            self.assertEqual(response.status_code, 503, response.text)
            self.assertFalse(response.json()["execution_available"])
            self.assertIsNone(response.json()["confirmation_token"])
            self.assertEqual(self.execute(request).status_code, 503)
            self.assertEqual(self.lookup(request).status_code, 503)
        self.assertIs(main.app.state.smart_cleanup_service, original)
        self.assertFalse(self.f.provider.calls)

    def test_private_exceptions_unknown_reasons_and_validation_never_leak(self):
        request = self.consent()
        for error in (RuntimeError("PRIVATE_PASSWORD"), CleanupUnavailable("PRIVATE_SECRET"),
                      HTTPException(500, "PRIVATE_HTTP_ERROR")):
            with patch.object(self.flow, "execute", side_effect=error):
                response = self.execute(request)
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json(), {"detail": "cleanup_service_unavailable"})
        for response in (self.execute({**request, "plan": {"quote": "PRIVATE_TEXT"}}), self.lookup(request)):
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertNotIn("PRIVATE_TEXT", response.text)
            self.assertNotIn(request["confirmation_token"], response.text)

    def test_strict_response_contracts_reject_misleading_states(self):
        with self.assertRaises(ValidationError):
            SmartCleanupPreflightResponse(execution_available=True)
        request = self.consent()
        with self.assertRaises(ValidationError):
            SmartCleanupPreflightResponse(execution_available=False, reason="capacity_exceeded", **request)
        with self.assertRaises(ValidationError):
            SmartCleanupExecuteRequest.model_validate({**request, "confirmed": True})
        self.restart()
        result = self.lookup(request).json()
        for update in ({"commit": "committed"}, {"reading_ready": True}, {"retry_action": "regenerate_cleanup"},
                       {"pending_review_count": True}, {"namespace": "private"}, {"reason": "raw error"}):
            with self.assertRaises(ValidationError):
                SmartCleanupOperationResponse.model_validate({**result, **update})

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"), "opt-in immutable backups")
    def test_real_backed_up_books_through_http_contract(self):
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                folder = root / name
                paths = [folder / p for p in ("book.pdf", "document.json", "manifest.json")]
                before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                self.assertEqual(before[paths[0]], json.loads(paths[2].read_text())["source_sha256"])
                self.f.path.write_bytes(paths[0].read_bytes())
                document = StructuredDocument.from_dict(json.loads(paths[1].read_text()))
                self.f.repo.save_reparsed_document(replace(document, document_id="book", source_path=str(self.f.path)), "book")
                items = []
                if name == "sample2":
                    record, = json.loads((folder / "20261023T030347436965Z/deletions.json").read_text())
                    items = [item(quote=record["quote"])]
                self.f.plan(items)
                request = self.consent()
                response = self.execute(request)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["outcome"]["applied_count"], len(items))
                self.restart()
                self.assertEqual(self.lookup(request).json(), response.json())
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class PostgresSmartCleanupApiTests(SmartCleanupApiTests):
    fixture_type = fixtures.PostgresStandaloneCleanupServiceTests
    setUpClass = classmethod(fixtures.PostgresStandaloneCleanupServiceTests.setUpClass.__func__)
    drop_schema = classmethod(fixtures.PostgresStandaloneCleanupServiceTests.drop_schema.__func__)


if __name__ == "__main__":
    unittest.main()
