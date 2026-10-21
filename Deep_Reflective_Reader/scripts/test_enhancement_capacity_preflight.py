"""Capacity preflight, consent binding and zero-call failure with real assembly."""
from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import main
from api_schemas import EnhancementPreflightResponse
from app.enhancement_preflight import EnhancementConflict
from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.structure_enhancement_candidate import EnhancementCandidateRejected
from document_structure.enhancement_resources import EnhancementResourceLimits
from llm.llm_model_capabilities import LLMModelCapabilities
import test_enhancement_preflight as preflight_fixtures


class CapacityPreflightTests(unittest.TestCase):
    def setUp(self):
        preflight_fixtures.PreflightTests.setUp(self)
        self.service.lease = lambda _: nullcontext()
        self.splitter = self.pipeline.structured_document_builder.section_splitter_selector.llm_splitter

    def preflight(self):
        return self.service.preflight(doc_name="doc")

    def request(self, token):
        return dict(doc_name="doc", discard_interactions=True, confirmation_token=token)

    def test_fit_is_read_only_and_execution_uses_identical_prompt(self):
        with patch("subprocess.run", side_effect=AssertionError("OCR forbidden")), \
             patch.object(self.pipeline, "prepare_and_load", side_effect=AssertionError("prepare forbidden")):
            result = self.preflight()
            parsed = EnhancementPreflightResponse.model_validate(result)
            self.assertTrue(parsed.ready and parsed.execution_available)
            self.assertEqual(parsed.capacity.status, "fits")
            self.assertTrue(parsed.capacity.full_source)
            self.assertEqual(parsed.capacity.source_chars, parsed.capacity.visible_source_chars)
            self.assertEqual(self.provider.calls, 0)
            with self.service.authorize_execution(**self.request(result["confirmation_token"])) as state:
                candidate = self.pipeline.build_enhancement_candidate(snapshot=state.source)
            self.assertEqual(candidate.document.raw_text, self.state.source.canonical_text)
            self.assertEqual(self.provider.calls, 1)
            self.assertNotIn(self.state.source.canonical_text, str(result))
            self.assertNotIn("fingerprint", result)

    def test_exceeded_preflight_never_issues_token(self):
        self.provider.get_model_capabilities = lambda: LLMModelCapabilities("small", "responses", 2000, 500)
        result = self.preflight()
        self.assertTrue(result["ready"])
        self.assertFalse(result["execution_available"])
        self.assertIsNone(result["confirmation_token"])
        self.assertEqual(result["capacity"]["reason_code"], "model_capacity_exceeded")
        self.assertGreater(result["capacity"]["required_input_tokens"], result["capacity"]["available_input_tokens"])
        self.assertEqual(self.provider.calls, 0)

    def test_resource_limit_reports_unknown_measurement_not_zero(self):
        self.splitter.resource_limits = EnhancementResourceLimits(max_source_chars=1)
        result = self.preflight()
        self.assertEqual(result["capacity"]["status"], "resource_limited")
        self.assertIsNone(result["capacity"]["required_input_tokens"])
        self.assertIsNone(result["capacity"]["full_source"])
        self.assertIsNone(result["confirmation_token"])
        self.assertEqual(self.provider.calls, 0)

    def test_unavailable_configuration_is_safe_and_closed(self):
        self.service.capacity_reader = Mock(side_effect=RuntimeError("PRIVATE_SECRET"))
        with self.assertLogs("app.enhancement_preflight", level="WARNING") as logs:
            result = self.preflight()
        self.assertEqual(result["capacity"]["status"], "unavailable")
        self.assertIsNone(result["confirmation_token"])
        self.assertNotIn("PRIVATE_SECRET", str(result) + str(logs.output))
        self.assertEqual(self.provider.calls, 0)

    def test_storage_unavailable_no_longer_issues_consent(self):
        self.service.availability = None
        result = self.preflight()
        self.assertTrue(result["ready"])
        self.assertIsNone(result["confirmation_token"])
        self.assertIsNone(result["capacity"])

    def test_model_output_resource_and_count_policy_invalidate_consent(self):
        mutations = [
            lambda: setattr(self.provider, "get_model_capabilities", lambda: LLMModelCapabilities("different", "responses", 1000000, 4000)),
            lambda: setattr(self.provider, "effective_output_budget", 500),
            lambda: setattr(self.splitter, "resource_limits", EnhancementResourceLimits(max_source_chars=1000000)),
            lambda: setattr(self.provider, "count_input_tokens", lambda _: 100),
        ]
        original_get = self.provider.get_model_capabilities
        for mutate in mutations:
            token = self.preflight()["confirmation_token"]
            mutate()
            with self.assertRaisesRegex(EnhancementConflict, "confirmation_stale"):
                self.service.validate(**self.request(token))
            self.provider.get_model_capabilities = original_get
            self.provider.__dict__.pop("effective_output_budget", None)
            self.provider.__dict__.pop("count_input_tokens", None)
            self.splitter.resource_limits = EnhancementResourceLimits()
        self.assertEqual(self.provider.calls, 0)

    def test_late_configuration_change_stops_at_dispatch(self):
        token = self.preflight()["confirmation_token"]
        with self.service.authorize_execution(**self.request(token)) as state:
            self.provider.effective_output_budget = 500
            with self.assertRaisesRegex(EnhancementCandidateRejected, "enhancement_configuration_changed"):
                self.pipeline.build_enhancement_candidate(snapshot=state.source)
        self.assertEqual(self.provider.calls, 0)
        # Scope is reset even on error; a new unscoped candidate is not contaminated.
        self.pipeline.build_enhancement_candidate(snapshot=self.state.source)
        self.assertEqual(self.provider.calls, 1)

    def test_http_capacity_failure_keeps_legacy_detail_and_safe_fields(self):
        token = self.preflight()["confirmation_token"]
        self.provider.get_model_capabilities = lambda: LLMModelCapabilities("small", "responses", 2000, 500)
        coordinator = object.__new__(SectionTaskCoordinator)
        coordinator.document_preparation_pipeline = self.pipeline
        coordinator.document_artifact_repository = Mock()
        with patch.object(main, "section_task_coordinator", coordinator), \
             patch.object(main, "enhancement_preflight_service", self.service), \
             patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            client = TestClient(main.app)
            result = client.post("/documents/reparse-structure", json={
                **self.request(token), "parser_mode": "llm_enhanced"})
            self.assertEqual(result.status_code, 413, result.text)
            self.assertEqual(result.json()["detail"], "model_capacity_exceeded")
            diagnostic = client.post("/documents/reparse-structure/preflight", json={"doc_name": "doc"})
            self.assertEqual(diagnostic.status_code, 200)
            self.assertEqual(result.json()["capacity"], diagnostic.json()["capacity"])
        coordinator.document_artifact_repository.commit_enhancement.assert_not_called()
        self.assertEqual(self.provider.calls, 0)

    def test_tokenizer_fallback_is_labelled_and_invalidates_old_consent(self):
        self.provider.count_input_tokens = lambda _: 100
        result = self.preflight()
        self.assertEqual(result["capacity"]["count_method"], "tokenizer")
        self.provider.count_input_tokens = lambda _: None
        self.assertEqual(self.preflight()["capacity"]["count_method"], "conservative_estimate")
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_stale"):
            self.service.validate(**self.request(result["confirmation_token"]))

    def test_exact_capacity_boundary_and_one_token_over(self):
        self.provider.get_model_capabilities = lambda: LLMModelCapabilities("fixture", "responses", 10000, 1000)
        self.provider.count_input_tokens = lambda _: 8294
        self.assertEqual(self.preflight()["capacity"]["status"], "fits")
        self.provider.count_input_tokens = lambda _: 8295
        result = self.preflight()
        self.assertEqual(result["capacity"]["status"], "exceeded")
        self.assertIsNone(result["confirmation_token"])
        self.assertEqual(self.provider.calls, 0)

    def test_capacity_unavailable_after_confirmation_returns_503(self):
        token = self.preflight()["confirmation_token"]
        self.provider.get_model_capabilities = Mock(side_effect=ValueError("PRIVATE"))
        with self.assertRaises(EnhancementConflict) as caught:
            self.service.validate(**self.request(token))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.capacity["status"], "unavailable")
        self.assertEqual(self.provider.calls, 0)


if __name__ == "__main__":
    unittest.main()
