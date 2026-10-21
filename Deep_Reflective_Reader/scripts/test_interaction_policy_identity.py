"""INTERACT-03: real file-store recovery without hidden generation or writes."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_app_reading_interaction_orchestration import _document, _coordinator
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.analysis_interaction_orchestrator import AnalysisInteractionOrchestrator
from section_tasks.quiz_interaction_orchestrator import QuizInteractionOrchestrator
from section_tasks.interaction_policy_identity import effective_policy, policy_matches, POLICY_METADATA_KEY


class InteractionPolicyIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = StructuredDocumentArtifactRepository(base_dir=self.temp.name)
        self.repository.save_document(_document(), doc_name="book")
        self.store = DocumentReadingInteractionArtifactStore(self.repository)
        self.target = ReadingTargetResolver().resolve(document=_document(), target_level="section", section_id="section-1")
        self.calls = []

    def request(self, kind="analysis"):
        return ReadingInteractionRequest(target=self.target, interaction_type=kind,
            context_metadata={"doc_name": "book"}, prompt_instruction_version="client-forgery")

    def generate(self, request, *args):
        self.calls.append(request)
        if request.interaction_type == "quiz":
            return {"items": [{"type": "short_answer", "question": "Which assumptions matter?", "answer": "Assumptions shape conclusions."}]}
        return {"summary": "Assumptions influence conclusions.",
                "reasoning": "The explicit assumptions shape what can be concluded.",
                "explanation": "The contrast between assumptions and conclusions anchors the argument."}

    def orchestrator(self, kind):
        if kind == "analysis":
            return AnalysisInteractionOrchestrator(service=AnalysisInteractionService(self.generate), artifact_store=self.store)
        return QuizInteractionOrchestrator(service=QuizInteractionService(self.generate), artifact_store=self.store)

    def snapshot(self):
        return {str(path): path.read_bytes() for path in Path(self.temp.name).rglob("*") if path.is_file()}

    def save(self, artifact):
        return getattr(self.store, f"save_{artifact.interaction_type}_artifact")(artifact)

    def test_missing_identity_is_stale_until_explicit_generation(self):
        for kind in ["analysis", "quiz"]:
            with self.subTest(kind=kind):
                req = self.request(kind)
                orchestrator = self.orchestrator(kind)
                current = orchestrator.generate(req)
                legacy = replace(current, metadata={"context": {"doc_name": "book"}})
                self.save(legacy)
                before = self.snapshot()
                count = len(self.calls)
                stale = orchestrator.read(req)
                self.assertEqual(stale.status, "stale_target")
                self.assertEqual(stale.reason, "interaction_policy_stale")
                self.assertFalse(stale.payload)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(len(self.calls), count)
                repaired = orchestrator.generate(req)
                self.assertEqual(repaired.status, "completed")
                self.assertTrue(policy_matches(repaired, req))
                self.assertEqual(len(self.calls), count + 1)

    def test_current_reuse_and_refresh_are_explicit(self):
        for kind in ["analysis", "quiz"]:
            req = self.request(kind)
            orchestrator = self.orchestrator(kind)
            current = orchestrator.generate(req)
            before = self.snapshot()
            count = len(self.calls)
            self.assertEqual(orchestrator.read(req).payload, current.payload)
            self.assertEqual(orchestrator.generate(req).payload, current.payload)
            self.assertEqual(len(self.calls), count)
            self.assertEqual(self.snapshot(), before)
            orchestrator.generate(replace(req, refresh=True))
            self.assertEqual(len(self.calls), count + 1)
            self.assertNotEqual(self.calls[-1].prompt_instruction_version, "client-forgery")

    def test_every_policy_dimension_invalidates_current_result(self):
        req = self.request()
        orchestrator = self.orchestrator("analysis")
        current = orchestrator.generate(req)
        for key in effective_policy(req):
            identity = dict(current.metadata[POLICY_METADATA_KEY])
            identity[key] = "obsolete"
            self.save(replace(current, metadata={**current.metadata, POLICY_METADATA_KEY: identity}))
            self.assertEqual(orchestrator.read(req).status, "stale_target", key)
        self.save(current)
        for changed in [replace(req, target=replace(req.target, content=req.target.content + " More evidence.")),
                        replace(req, context_metadata={"doc_name": "book", "document_language": "zh"})]:
            self.assertEqual(orchestrator.read(changed).status, "stale_target")

    def test_client_version_does_not_control_policy(self):
        req = self.request()
        self.assertEqual(effective_policy(req), effective_policy(replace(req, prompt_instruction_version=None)))
        result = self.orchestrator("analysis").generate(req)
        self.assertEqual(result.metadata["prompt_instruction_version"], effective_policy(req)["prompt"])

    def test_failed_refresh_preserves_previous_artifact(self):
        req = self.request()
        self.orchestrator("analysis").generate(req)
        before = self.snapshot()
        failing = AnalysisInteractionOrchestrator(service=AnalysisInteractionService(lambda req: {}), artifact_store=self.store)
        self.assertEqual(failing.generate(replace(req, refresh=True)).status, "generation_failed")
        self.assertEqual(before, self.snapshot())

    def test_old_insufficient_content_rechecked_without_write(self):
        req = self.request()
        orchestrator = self.orchestrator("analysis")
        tiny = replace(req, target=replace(req.target, content="abc"))
        self.assertEqual(orchestrator.generate(tiny).status, "insufficient_content")
        before = self.snapshot()
        self.assertEqual(orchestrator.read(req).status, "not_generated")
        self.assertEqual(orchestrator.read(tiny).status, "insufficient_content")
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.calls)
        self.assertEqual(orchestrator.generate(req).status, "completed")

    def test_thinking_history_survives_stale_read_and_new_question(self):
        count = []

        def question(*args):
            count.append(1)
            return {"question": "Which assumption matters?"}

        service = CriticalThinkingSessionService(question, lambda *args: {})
        coordinator = _coordinator(critical_service=service, critical_store=self.store)
        coordinator.document_artifact_repository = self.repository
        args = {"doc_name": "book", "target_level": "section", "section_id": "section-1"}
        generated = coordinator.generate_critical_thinking_question(**args)
        req = coordinator._build_reading_interaction_request(**args, interaction_type="critical_thinking_session")
        stored = self.store.get_critical_thinking_session(req, generated.session_id)
        submitted = service.submit_answer(stored, "My reasoning must survive policy changes.")
        legacy = replace(submitted, metadata={k: v for k, v in submitted.metadata.items() if k != POLICY_METADATA_KEY})
        self.store.save_critical_thinking_session(legacy)
        before = self.snapshot()
        for extra in [{}, {"session_id": generated.session_id}]:
            result = coordinator.read_critical_thinking_session(**args, **extra)
            self.assertEqual(result.status, "stale_target")
            self.assertEqual(result.session_id, generated.session_id)
            self.assertEqual(result.payload["answer"], submitted.payload["answer"])
        self.assertEqual(before, self.snapshot())
        self.assertEqual(len(count), 1)
        replacement = coordinator.generate_critical_thinking_question(**args)
        self.assertNotEqual(replacement.session_id, generated.session_id)
        historical = self.store.get_critical_thinking_session(req, generated.session_id)
        self.assertEqual(historical.payload, submitted.payload)
        current = coordinator.read_critical_thinking_session(**args)
        self.assertEqual(current.session_id, replacement.session_id)
        self.assertEqual(current.status, "question_generated")
        self.assertEqual(len(count), 2)

    def test_real_api_defaults_and_stale_session_payload(self):
        import main
        from fastapi.testclient import TestClient
        coordinator = _coordinator(
            analysis_orchestrator=self.orchestrator("analysis"),
            quiz_orchestrator=self.orchestrator("quiz"),
            critical_service=CriticalThinkingSessionService(
                lambda *args: {"question": "Which assumption matters?"}, lambda *args: {}),
            critical_store=self.store)
        coordinator.document_artifact_repository = self.repository
        target = {"doc_name": "book", "target_type": "section", "section_id": "section-1"}
        with patch.object(main, "section_task_coordinator", coordinator):
            client = TestClient(main.app)
            for kind, path in [("analysis", "insight"), ("quiz", "quiz")]:
                base = f"/documents/reading-interactions/{path}"
                response = client.post(base + "/generate", json={"target": target})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["envelope"]["prompt_instruction_version"], effective_policy(self.request(kind))["prompt"])
                current = getattr(self.store, f"get_{kind}_artifact")(self.request(kind))
                self.save(replace(current, metadata={"context": {"doc_name": "book"}}))
                before = self.snapshot()
                stale = client.post(base + "/read", json={"target": target})
                self.assertEqual(stale.status_code, 409, stale.text)
                self.assertEqual(stale.json()["envelope"]["reason"], "interaction_policy_stale")
                self.assertEqual(before, self.snapshot())
                repaired = client.post(base + "/generate", json={"target": target, "prompt_instruction_version": "client-forgery"})
                self.assertEqual(repaired.status_code, 200, repaired.text)
                self.assertNotEqual(repaired.json()["envelope"]["prompt_instruction_version"], "client-forgery")
            base = "/documents/reading-interactions/critical-thinking"
            generated = client.post(base + "/generate-question", json={"target": target})
            self.assertEqual(generated.status_code, 200, generated.text)
            session_id = generated.json()["envelope"]["session_id"]
            req = self.request("critical_thinking_session")
            session = self.store.get_critical_thinking_session(req, session_id)
            self.store.save_critical_thinking_session(replace(session,
                status="answer_submitted", payload={**session.payload, "answer": "Keep my answer"},
                metadata={k: v for k, v in session.metadata.items() if k != POLICY_METADATA_KEY}))
            before = self.snapshot()
            stale = client.post(base + "/read", json={"target": target, "session_id": session_id})
            self.assertEqual(stale.status_code, 409, stale.text)
            self.assertIsNone(stale.json()["payload"])
            self.assertEqual(self.store.get_critical_thinking_session(req, session_id).payload["answer"], "Keep my answer")
            self.assertEqual(stale.json()["envelope"]["session_id"], session_id)
            self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
