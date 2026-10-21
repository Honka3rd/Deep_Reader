import json
import tempfile
import unittest
from dataclasses import replace

from scripts.test_reading_interaction_artifact_store import _document
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.summary_interaction_service import SummaryInteractionService
from llm.llm_model_capabilities import LLMModelCapabilities


class Provider:
    effective_output_budget = 1000
    def __init__(self):
        self.prompts = []
        self.fail = False
    def get_model_capabilities(self):
        return LLMModelCapabilities("test", "responses", 100000, 1000)
    def count_input_tokens(self, prompt):
        return len(prompt)
    def complete_text(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            return "invalid"
        return json.dumps({"headline": "Assumptions", "body": "The text discusses assumptions and conclusions.",
                           "key_points": ["Distinguish assumptions from conclusions."]})


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=self.temp.name)
        self.repo.save_document(_document(), "book")
        self.provider = Provider()
        self.service = SummaryInteractionService(self.repo, self.provider)

    def request(self, level="task_unit"):
        fields = {"target_level": level}
        if level in {"chapter", "section", "task_unit", "content_block"}:
            fields["chapter_id"] = "chapter-1"
        if level in {"section", "task_unit", "content_block"}:
            fields["section_id"] = "section-1"
        if level in {"task_unit", "content_block"}:
            fields["task_unit_id"] = "unit-1"
        if level == "content_block":
            fields["content_block_id"] = "unit-1:content:0"
        return ReadingInteractionRequest(
            target=ReadingTargetResolver().resolve(document=self.repo.load_document("book"), **fields),
            interaction_type="summary", context_metadata={"doc_name": "book", "document_language": "en",
                                                          "document_profile_summary": "Profile orientation"})

    def test_read_is_pure_generate_prepares_only_ancestors_and_reuses(self):
        request = self.request()
        self.assertEqual(self.service.read(request).status, "not_generated")
        self.assertEqual(self.provider.prompts, [])
        result = self.service.generate(request)
        self.assertEqual(result.status, "completed", result.reason)
        self.assertEqual(len(self.provider.prompts), 3)
        prompt = self.provider.prompts[-1]
        self.assertIn(request.target.content, prompt)
        self.assertIn("Ancestor Summaries:", prompt)
        self.assertIn("Profile orientation", prompt)
        self.assertEqual(self.service.generate(request), result)
        self.assertEqual(len(self.provider.prompts), 3)
        self.service.generate(replace(request, refresh=True))
        self.assertEqual(len(self.provider.prompts), 4)

    def test_failed_refresh_retains_previous(self):
        request = self.request("document")
        previous = self.service.generate(request)
        self.provider.fail = True
        self.assertEqual(self.service.generate(replace(request, refresh=True)).status, "generation_failed")
        self.assertEqual(self.service.read(request), previous)

    def test_all_five_levels(self):
        for level in ("document", "chapter", "section", "task_unit", "content_block"):
            result = self.service.generate(self.request(level))
            self.assertEqual(result.status, "completed", (level, result.reason))

    def test_capacity_precedes_ancestor_calls(self):
        self.provider.count_input_tokens = lambda prompt: 100001
        result = self.service.generate(self.request())
        self.assertEqual(result.reason, "summary_model_capacity_exceeded")
        self.assertFalse(self.provider.prompts)

    def test_source_abstention_is_not_persisted(self):
        self.provider.complete_text = lambda prompt: json.dumps({"status": "insufficient_content", "reason": "primary_source_not_meaningful"})
        result = self.service.generate(self.request("document"))
        self.assertEqual(result.status, "insufficient_content", result.reason)
        self.assertEqual(self.service.read(self.request("document")).status, "not_generated")

    def test_reparse_during_generation_cannot_restore_cleared_summary(self):
        original = self.provider.complete_text
        def complete(prompt):
            self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
            return original(prompt)
        self.provider.complete_text = complete
        result = self.service.generate(self.request("document"))
        self.assertEqual(result.status, "generation_failed")
        self.assertEqual(self.service.read(self.request("document")).status, "not_generated")

    def test_ancestor_failure_stops_target_and_descendants(self):
        self.provider.fail = True
        result = self.service.generate(self.request("content_block"))
        self.assertEqual(result.status, "generation_failed")
        self.assertEqual(len(self.provider.prompts), 1)

    def test_book_vocabulary_repetition_is_not_a_short_text_failure(self):
        request = self.request("document")
        repeated = replace(request, target=replace(request.target, content=request.target.content * 1000))
        self.assertIsNone(self.service.validity.preflight(repeated))

    def test_routes_use_real_coordinator_and_preserve_output(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import main
        from app.section_task_coordinator import SectionTaskCoordinator
        coordinator = object.__new__(SectionTaskCoordinator)
        coordinator.document_artifact_repository = self.repo
        coordinator.reading_target_resolver = ReadingTargetResolver()
        coordinator.summary_interaction_service = self.service
        coordinator._load_existing_document_profile = lambda name: None
        with patch.object(main, "section_task_coordinator", coordinator):
            client = TestClient(main.app)
            target = {"doc_name": "book", "target_type": "document"}
            base = "/documents/reading-interactions/summary/"
            response = client.post(base + "read", json={"target": target})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["envelope"]["status"], "not_generated")
            self.assertFalse(self.provider.prompts)
            result = client.post(base + "generate", json={"target": target})
            self.assertEqual(result.status_code, 200, result.text)
            self.provider.fail = True
            failed = client.post(base + "refresh", json={"target": target})
            self.assertEqual(failed.status_code, 502, failed.text)
            self.assertEqual(client.post(base + "read", json={"target": target}).json(), result.json())


if __name__ == "__main__":
    unittest.main()
