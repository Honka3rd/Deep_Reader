"""Consent and transport regressions; no user documents or providers."""
from dataclasses import replace
import base64
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import main
from app.enhancement_preflight import (
    EnhancementConflict, EnhancementPreflightService, EnhancementState,
)
from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.accepted_source_snapshot import SourceNotReady
import test_accepted_source_snapshot as source_fixtures


class PreflightTests(unittest.TestCase):
    def setUp(self):
        fixture = source_fixtures.AcceptedSourceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.state = EnhancementState(fixture.resolve(), "revision-1", None, None)
        self.reader = Mock(side_effect=lambda name: self.state)
        self.now = 1000
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        self.provider = Provider()
        self.pipeline = candidate_pipeline(self.provider)
        self.service = EnhancementPreflightService(self.reader, clock=lambda: self.now,
            availability=lambda _: None, capacity_reader=lambda state, hint, intent:
            self.pipeline.inspect_enhancement_capacity(snapshot=state.source, toc_hint=hint, enhancement_intent=intent))

    def token(self):
        return self.service.preflight(doc_name="doc")["confirmation_token"]

    def validate(self, token, **kwargs):
        args = dict(doc_name="doc", discard_interactions=True, confirmation_token=token)
        args.update(kwargs)
        return self.service.validate(**args)

    def test_ready_unknown_counts_and_no_private_content(self):
        result = self.service.preflight(doc_name="doc")
        self.assertTrue(result["ready"])
        self.assertIsNone(result["interaction_count"])
        self.assertIsNone(result["includes_user_notes"])
        self.assertNotIn(self.state.source.canonical_text, str(result))
        self.assertEqual(self.validate(result["confirmation_token"]), self.state)

    def test_missing_consent_does_not_read(self):
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_required"):
            self.service.validate(doc_name="doc")
        self.reader.assert_not_called()

    def test_signature_and_expiry(self):
        token = self.token()
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_invalid"):
            self.validate("!" + token)
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_invalid"):
            self.validate(base64.urlsafe_b64encode("1.binding.nonce.\u00e9".encode()).decode())
        self.now += 300
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_expired"):
            self.validate(token)

    def test_restart_invalidates_token(self):
        token = self.token()
        self.service = EnhancementPreflightService(self.reader, clock=lambda: self.now)
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_invalid"):
            self.validate(token)

    def test_document_options_and_revisions_bound(self):
        token = self.token()
        for kwargs in [dict(doc_name="other"), dict(toc_hint="absent"),
                       dict(enhancement_intent="generate_reading_outline")]:
            with self.assertRaisesRegex(EnhancementConflict, "confirmation_stale"):
                self.validate(token, **kwargs)
        original = self.state
        for state in [replace(original, interaction_revision="edited"),
                      replace(original, source=replace(original.source, structure_version=4)),
                      replace(original, source=replace(original.source, namespace="other")),
                      replace(original, source=replace(original.source, text_sha256="f"*64))]:
            self.state = state
            with self.assertRaisesRegex(EnhancementConflict, "confirmation_stale"):
                self.validate(token)

    def test_readiness_failure_has_no_token(self):
        self.reader.side_effect = SourceNotReady("source_changed")
        result = self.service.preflight(doc_name="doc")
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "source_changed")
        self.assertIsNone(result["confirmation_token"])

    def test_real_routes_and_coordinator_never_prepare(self):
        coordinator = object.__new__(SectionTaskCoordinator)
        coordinator.document_preparation_pipeline = Mock()
        with patch.object(main, "section_task_coordinator", coordinator), \
             patch.object(main, "enhancement_preflight_service", self.service), \
             patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            client = TestClient(main.app)
            response = client.post("/documents/reparse-structure/preflight", json={"doc_name": "doc"})
            self.assertEqual(response.status_code, 200)
            body = {"doc_name": "doc", "parser_mode": "llm_enhanced"}
            response = client.post("/documents/reparse-structure", json=body)
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["detail"], "confirmation_required")
            token = self.token()
            self.service.availability = None
            response = client.post("/documents/reparse-structure", json=dict(body,
                discard_interactions=True, confirmation_token=token))
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()["detail"], "enhancement_execution_unavailable")
        coordinator.document_preparation_pipeline.prepare_and_load.assert_not_called()

    def test_unconfigured_runtime_is_explicitly_unavailable(self):
        with patch.object(main, "enhancement_preflight_service", EnhancementPreflightService()):
            response = TestClient(main.app).post("/documents/reparse-structure/preflight",
                                                json={"doc_name": "doc"})
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["reason"], "enhancement_storage_guards_unavailable")

    def test_transport_rejects_non_boolean_consent_and_wrong_mode(self):
        client = TestClient(main.app)
        for value in ["true", 1]:
            response = client.post("/documents/reparse-structure", json={
                "doc_name": "doc", "parser_mode": "llm_enhanced", "discard_interactions": value})
            self.assertEqual(response.status_code, 422)
        response = client.post("/documents/reparse-structure/preflight", json={
            "doc_name": "doc", "parser_mode": "common"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
