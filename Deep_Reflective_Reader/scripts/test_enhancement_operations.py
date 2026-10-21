"""Reconciliation bounds and failure policy without providers or storage."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.enhancement_operations import EnhancementOperations
from app.enhancement_preflight import EnhancementConflict
from section_tasks.reparse_document_structure_result import ReparseDocumentStructureResult
from llm.llm_provider import is_transient_provider_error


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.operations = EnhancementOperations(clock=lambda: self.now, limit=2, retention=10)
        self.request = dict(doc_name="doc", confirmation_token="synthetic", discard_interactions=True)
        self.operations.register(self.request)
        self.result = ReparseDocumentStructureResult.ok(doc_name="doc", parser_mode="llm_enhanced",
            structured_document_path=None, structure_version=1, change_kind="unchanged")

    def test_eviction_never_means_permission_to_rerun(self):
        callback = Mock(return_value=self.result)
        self.operations.execute(self.request, callback)
        self.now = 11
        self.assertEqual(self.operations.lookup(self.request)["status"], "unknown")
        with self.assertRaisesRegex(EnhancementConflict, "outcome_unknown"):
            self.operations.execute(self.request, callback)
        self.assertEqual(callback.call_count, 1)

    def test_active_record_survives_retention_window(self):
        def callback():
            self.now = 99
            self.assertEqual(self.operations.lookup(self.request)["status"], "running")
            return self.result
        self.operations.execute(self.request, callback)
        self.assertEqual(self.operations.lookup(self.request)["status"], "completed")

    def test_registry_bound_fails_closed(self):
        self.operations.register(dict(self.request, confirmation_token="second"))
        with self.assertRaisesRegex(EnhancementConflict, "operation_capacity"):
            self.operations.register(dict(self.request, confirmation_token="third"))

    def test_identity_and_consent_cannot_be_bypassed_by_replay(self):
        self.operations.execute(self.request, lambda: self.result)
        for request in [dict(self.request, doc_name="other"), dict(self.request, toc_hint="absent")]:
            self.assertEqual(self.operations.lookup(request)["status"], "unknown")
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_required"):
            self.operations.execute(dict(self.request, discard_interactions=False), Mock())

    def test_unexpected_failure_remains_unknown_and_does_not_leak(self):
        with self.assertRaises(RuntimeError):
            self.operations.execute(self.request, Mock(side_effect=RuntimeError("PRIVATE")))
        lookup = self.operations.lookup(self.request)
        self.assertEqual(lookup["status"], "unknown")
        self.assertNotIn("PRIVATE", str(lookup))
        with self.assertRaisesRegex(EnhancementConflict, "outcome_unknown"):
            self.operations.execute(self.request, Mock())

    def test_capacity_conflict_replay_keeps_safe_details(self):
        failure = EnhancementConflict("model_capacity_exceeded", 413, {"status": "exceeded"})
        for _ in range(2):
            with self.assertRaises(EnhancementConflict) as caught:
                self.operations.execute(self.request, Mock(side_effect=failure))
            self.assertEqual(caught.exception.capacity, {"status": "exceeded"})

    def test_transience_requires_transport_or_status_not_error_message(self):
        for error in [TimeoutError(), ConnectionError()]:
            self.assertTrue(is_transient_provider_error(error))
        for status in [408, 429, 500, 502, 503, 504, 400, 401, 403, 422]:
            error = RuntimeError("provider")
            error.status_code = status
            self.assertEqual(is_transient_provider_error(error), status in {408, 429, 500, 502, 503, 504})
        self.assertFalse(is_transient_provider_error(RuntimeError("timeout")))


if __name__ == "__main__":
    unittest.main()
