"""Outcome and reconciliation HTTP contracts on disposable PostgreSQL only."""
from unittest.mock import patch
import unittest
from fastapi.testclient import TestClient
import test_enhancement_conditional_commit as conditional
import main
from app.enhancement_operations import EnhancementOperations


class OutcomeApiTests(conditional.ConditionalCommitTests):
    def lookup(self, token):
        with patch.object(main, "enhancement_preflight_service", self.service):
            return TestClient(main.app).post("/documents/reparse-structure/result",
                json={"doc_name": "doc", "confirmation_token": token})

    def test_http_three_way_flags_and_response_loss_replay(self):
        self.enable_execution()
        for title, kind in [(None, "unchanged"), ("Navigation", "titles_only")]:
            with self.subTest(kind=kind):
                _, candidate = self.matching_candidate(title)
                token = self.token()
                before = self.snapshot()
                self.assertEqual(self.lookup(token).json()["status"], "not_started")
                with patch.object(self.coordinator.document_preparation_pipeline,
                                  "build_enhancement_candidate", return_value=candidate) as build:
                    response = self.execute_route(token)
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    self.assertEqual(body["change_kind"], kind)
                    self.assertEqual(body["committed"], kind == "titles_only")
                    self.assertFalse(body["interactions_cleared"])
                    self.assertFalse(body["reading_layout_required"])
                    self.assertEqual(body["structure_version"], before.structure_version)
                    # Treat the first HTTP response as lost. Lookup and replay
                    # must recover the same result without another generation.
                    lookup = self.lookup(token).json()
                    self.assertEqual(lookup["status"], "completed")
                    self.assertEqual(lookup["result"], body)
                    self.assertEqual(self.execute_route(token).json(), body)
                    self.assertEqual(build.call_count, 1)
        response = self.execute_route()
        body = response.json()
        self.assertEqual(body["change_kind"], "structure_changed")
        self.assertTrue(body["committed"])
        self.assertTrue(body["interactions_cleared"])
        self.assertTrue(body["reading_layout_required"])

    def test_transient_generation_failure_can_retry_same_operation(self):
        self.enable_execution()
        token = self.token()
        before = self.snapshot()
        with patch.object(self.provider, "complete_text", side_effect=TimeoutError("private transport")):
            failed = self.execute_route(token)
        self.assertEqual(failed.status_code, 502, failed.text)
        self.assertTrue(failed.json()["retryable"])
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(self.lookup(token).json()["retryable"])
        result = self.execute_route(token)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.lookup(token).json()["status"], "completed")

    def test_invalid_plan_is_terminal_not_a_generation_retry(self):
        self.enable_execution(lambda payload, context: {"invalid": True})
        token = self.token()
        first = self.execute_route(token)
        self.assertEqual(first.status_code, 422)
        self.assertFalse(first.json()["retryable"])
        calls = self.provider.calls
        self.assertEqual(self.execute_route(token).json(), first.json())
        self.assertEqual(self.provider.calls, calls)

    def test_restart_or_wrong_worker_is_unknown_not_safe_to_regenerate(self):
        self.enable_execution()
        token = self.token()
        self.assertEqual(self.execute_route(token).status_code, 200)
        calls = self.provider.calls
        self.service.operations = EnhancementOperations()
        result = self.lookup(token).json()
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["durability"], "process_local")
        self.assertEqual(self.execute_route(token).status_code, 409)
        self.assertEqual(self.provider.calls, calls)

    def test_duplicate_inflight_http_request_does_not_generate_twice(self):
        self.enable_execution()
        token = self.token()
        complete = self.provider.complete_text
        def during_call(prompt):
            self.assertEqual(self.lookup(token).json()["status"], "running")
            self.assertEqual(self.execute_route(token).status_code, 409)
            return complete(prompt)
        with patch.object(self.provider, "complete_text", side_effect=during_call):
            self.assertEqual(self.execute_route(token).status_code, 200)
        self.assertEqual(self.provider.calls, 1)

    def test_title_edit_invalidates_http_confirmation(self):
        self.enable_execution()
        stale = self.token()
        state, candidate = self.matching_candidate("Concurrent title")
        self.commit_candidate(state, candidate)
        result = self.execute_route(stale)
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.json()["detail"], "confirmation_stale")
        self.assertEqual(self.provider.calls, 0)

    def test_conditional_preflight_and_legacy_response_defaults(self):
        from api_schemas import ReparseDocumentStructureResponse
        self.assertEqual(self.service.preflight(doc_name="doc")["cleanup_condition"], "structure_changed_only")
        legacy = ReparseDocumentStructureResponse(success=True, doc_name="doc", parser_mode="common",
            structured_document_path=None, error=None, section_count=1)
        self.assertIsNone(legacy.change_kind)
        self.assertIsNone(legacy.committed)
        with patch.object(main, "enhancement_preflight_service", self.service):
            self.assertEqual(TestClient(main.app).post("/documents/reparse-structure/result",
                json={"doc_name": "doc"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
