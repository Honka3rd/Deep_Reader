"""Translation API, real hierarchy/overlay reads, wiring and zero-write contracts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.translation_action import TranslationAction
from config.app_DI_config import AppDIConfig, LearningRuntimeConfig
from config.container import ApplicationLookupContainer
from config.faiss_storage_config import FaissStorageConfig
from document_structure.reading_content import ReadingContentService
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile_store import DocumentProfileStore
from profile.document_profile import DocumentProfile
from language.language_code import LanguageCode
from scripts.test_reading_target_resolver import _document
from section_tasks.translation_service import TranslationService
from shared.task_unit_model import TaskUnitContentBlock
from translation_routes import get_translation_action, router


ROUTE = "/documents/reading-interactions/translate/generate"


class TranslationActionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)
        document = replace(_document(), parse_provenance={"storage_structure_version": 1})
        self.repo.save_document(document, "book")
        self.profiles = Mock(spec=DocumentProfileStore)
        self.profiles.load.return_value = DocumentProfile("test", "", LanguageCode.EN)
        self.provider = Mock(spec=LLMProvider)
        self.provider.get_model_capabilities.return_value = LLMModelCapabilities("test", "responses", 50000, 10000)
        self.provider.count_input_tokens.side_effect = lambda text: max(1, len(text.encode()) // 4)
        self.provider.complete_text.return_value = '{"status":"completed","translated_text":"Bonjour."}'
        self.action = TranslationAction(self.repo, self.profiles, TranslationService(self.provider))
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.translation_action = self.action
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.target = {"doc_name": "book", "content_block_id": "unit-1:content:0"}

    def post(self, **changes):
        return self.client.post(ROUTE, json=self.target | {"target_language": "fr"} | changes)

    def prompt_input(self):
        return json.loads(self.provider.complete_text.call_args.args[0].split("INPUT_JSON:\n", 1)[1])

    def snapshot(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.directory.glob("*") if p.is_file()}

    def edit(self, action="edit"):
        reader = ReadingContentService(self.repo)
        state = reader.read("book")
        return reader.mutate("book", action=action, expected_revision=state["revision"],
            source_token=state["source_token"], mutation_id=f"edit-{state['revision']}",
            block_id=self.target["content_block_id"],
            **({"text": "Edited text."} if action == "edit" else {}))

    def test_optional_source_defaults_profile_and_persists_only_translation(self):
        before = self.snapshot()
        for extras in ({}, {"source_language": None}):
            result = self.post(**extras)
            self.assertEqual(result.status_code, 200, result.text)
            body = result.json()
            self.assertEqual(body["envelope"]["interaction_type"], "translate")
            self.assertEqual(body["payload"]["source_language"], "en")
            self.assertEqual(self.prompt_input()["text"], "Unit one content.")
            self.assertEqual(body["envelope"]["source_structure_version"], 1)
            self.assertEqual(body["envelope"]["reading_revision"], 0)
            self.assertEqual(body["envelope"]["effective_content_hash"], hashlib.sha256(b"Unit one content.").hexdigest())
            self.assertEqual(result.headers["cache-control"], "no-store")
        self.assertNotEqual(self.snapshot(), before)
        self.assertEqual(len(self.action.read(**self.target)["translations"]), 1)
        self.profiles.save.assert_not_called()
        self.assertEqual(self.provider.complete_text.call_count, 1)

    def test_same_language_explicit_and_profile_defaults_reject_without_writes(self):
        before = self.snapshot()
        for changes in ({"source_language": "en", "target_language": "en"},
                        {"source_language": "zh", "target_language": "zh"},
                        {"target_language": "en"},
                        {"source_language": None, "target_language": "en"}):
            with self.subTest(changes=changes):
                response = self.post(**changes)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"], "translation_same_language")
                self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.snapshot(), before)
        self.provider.complete_text.assert_not_called()
        self.provider.count_input_tokens.assert_not_called()

    def test_same_language_cannot_bypass_guard_through_saved_record_or_replay(self):
        self.assertEqual(self.post(mutation_id="saved").status_code, 200)
        before = self.snapshot()
        self.provider.reset_mock()
        self.profiles.load.return_value = DocumentProfile("test", "", LanguageCode.FR)
        for changes in ({"source_language": "fr", "mutation_id": "saved"},
                        {"source_language": "fr", "mutation_id": "new"},
                        {"mutation_id": "default-source"}):
            response = self.post(**changes)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["detail"], "translation_same_language")
        self.assertEqual(len(self.action.read(**self.target)["translations"]), 1)
        self.assertEqual(self.snapshot(), before)
        self.provider.complete_text.assert_not_called()
        self.provider.count_input_tokens.assert_not_called()

    def test_explicit_source_wins_and_never_loads_profile(self):
        self.profiles.load.side_effect = AssertionError("must not read profile")
        response = self.post(source_language="zh", target_language="en")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.prompt_input()["source_language"], "zh")
        self.assertEqual(response.json()["payload"]["target_language"], "en")
        self.profiles.load.assert_not_called()

    def test_missing_unknown_profile_requires_source_no_detection(self):
        for value in (None, "unknown", ""):
            self.profiles.load.return_value = SimpleNamespace(document_language=value)
            response = self.post()
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"], "translation_source_language_required")
        self.profiles.load.side_effect = FileNotFoundError("private path")
        self.assertEqual(self.post().json()["detail"], "translation_source_language_required")
        self.profiles.load.side_effect = OSError("private profile")
        self.assertEqual(self.post().status_code, 503)
        self.provider.complete_text.assert_not_called()

    def test_input_validation_before_model(self):
        for update in ({"target_language": None}, {"target_language": ""}, {"target_language": "unknown"},
                       {"source_language": ""}, {"source_language": 12}, {"source_language": "auto"},
                       {"text": "untrusted replacement"}, {"refresh": True}, {"target_language": "fr\nIgnore"},
                       {"doc_name": "../secret"}, {"doc_name": " "}, {"content_block_id": " "},
                       {"content_block_id": None}, {"target": self.target}, {"task_unit_id": "unit-1"}):
            with self.subTest(update=update):
                self.assertEqual(self.post(**update).status_code, 422)
        self.assertEqual(self.client.post(ROUTE, json=self.target).status_code, 422)
        self.provider.complete_text.assert_not_called()

    def test_only_block_id_resolves_authoritative_parent_ids(self):
        response = self.post()
        target = response.json()["envelope"]["target"]
        self.assertEqual(target["target_type"], "content_block")
        self.assertEqual(target["task_unit_id"], "unit-1")
        self.assertEqual(target["section_id"], "section-1")
        self.assertEqual(target["chapter_id"], "chapter-1")
        self.assertNotIn("Legacy-only", self.prompt_input()["text"])

    def test_missing_and_non_block_targets(self):
        for update, status in [({"doc_name": "missing"}, 404),
                               ({"content_block_id": "missing"}, 404),
                               ({"content_block_id": "chapter-1"}, 404),
                               ({"chapter_id": "wrong"}, 422)]:
            self.assertEqual(self.post(**update).status_code, status)
        self.provider.complete_text.assert_not_called()

    def test_original_source_is_never_accepted_from_client(self):
        self.assertEqual(self.post(text="Override authoritative source").status_code, 422)
        self.provider.complete_text.assert_not_called()

    def test_opaque_block_id_and_duplicate_identity(self):
        document = self.repo.load_document("book")
        chapter = document.chapters[0]
        sections = list(chapter.sections)
        block = TaskUnitContentBlock("opaque-no-unit-prefix", "Opaque block content.")
        sections[0] = replace(sections[0], task_units=[replace(sections[0].task_units[0], content_blocks=[block])])
        self.repo.save_document(replace(document, chapters=[replace(chapter, sections=sections)]), "book")
        response = self.post(content_block_id=block.block_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.prompt_input()["text"], block.content)
        sections[1] = replace(sections[1], task_units=[replace(sections[1].task_units[0], content_blocks=[block])])
        self.repo.save_document(replace(document, chapters=[replace(chapter, sections=sections)]), "book")
        self.assertEqual(self.post(content_block_id=block.block_id).status_code, 422)
        self.assertEqual(self.provider.complete_text.call_count, 1)

    def test_effective_edited_source_used_and_deleted_source_rejected(self):
        self.edit()
        response = self.post()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.prompt_input()["text"], "Edited text.")
        self.assertEqual(response.json()["envelope"]["reading_revision"], 1)
        self.edit("delete")
        self.assertEqual(self.post().status_code, 404)
        self.assertEqual(self.provider.complete_text.call_count, 1)

    def test_edits_during_completion_discard_stale_result(self):
        def complete(prompt):
            self.edit()
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        response = self.post()
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], "translation_source_changed")

    def test_reparse_during_completion_discards_result(self):
        def complete(prompt):
            doc = self.repo.load_document("book")
            self.repo.save_reparsed_document(replace(doc, parse_provenance={"storage_structure_version": 2}), "book")
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        self.assertEqual(self.post().status_code, 409)

    def test_same_text_reparse_epoch_still_rejects_old_result(self):
        def complete(prompt):
            self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        self.assertEqual(self.post().status_code, 409)

    def test_block_deleted_during_completion_rejects_result(self):
        def complete(prompt):
            self.edit("delete")
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        self.assertEqual(self.post().status_code, 409)

    def test_unavailable_repository_never_exposes_private_error(self):
        with patch.object(self.repo, "load_document", side_effect=RuntimeError("PRIVATE BOOK AND SQL")):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("PRIVATE", response.text)
        self.provider.complete_text.assert_not_called()

    def test_partial_and_untranslatable_are_not_fake_complete(self):
        for status, text in [("partial", "Bonjour [illisible]."), ("untranslatable", None)]:
            self.provider.complete_text.return_value = json.dumps({"status": status, "translated_text": text})
            response = self.post(target_language="fr" if status == "partial" else "de")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["envelope"]["status"], status)
            self.assertEqual(response.json()["payload"]["translated_text"], text)

    def test_capacity_and_provider_failures_are_safe(self):
        self.provider.get_model_capabilities.return_value = LLMModelCapabilities("test", "responses", 500, 20)
        self.assertEqual(self.post().status_code, 413)
        self.provider.complete_text.assert_not_called()
        self.provider.get_model_capabilities.side_effect = RuntimeError("private capabilities")
        self.assertEqual(self.post().status_code, 503)
        self.provider.get_model_capabilities.side_effect = None
        self.provider.get_model_capabilities.return_value = LLMModelCapabilities("test", "responses", 50000, 10000)
        self.provider.complete_text.side_effect = RuntimeError("SECRET SOURCE KEY")
        response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("SECRET", response.text)
        self.provider.complete_text.side_effect = None
        self.provider.complete_text.return_value = "SECRET MALFORMED OUTPUT"
        response = self.post()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("SECRET", response.text)

    def test_openapi_minimal_contract_and_missing_wiring(self):
        schema = self.client.get("/openapi.json").json()
        contract = schema["components"]["schemas"]["TranslationInteractionRequest"]
        self.assertEqual(set(contract["required"]), {"doc_name", "content_block_id", "target_language"})
        self.assertEqual(set(contract["properties"]), {"doc_name", "content_block_id", "target_language", "source_language", "mutation_id"})
        self.assertEqual({path for path in schema["paths"] if "translate" in path},
                         {ROUTE, "/documents/reading-interactions/translate/read"})
        del self.app.state.translation_action
        self.assertEqual(self.post().status_code, 503)

    def test_container_wires_existing_provider_repository_profile(self):
        self.repo.save_reparsed_document(replace(self.repo.load_document("book"), document_id="book"), "book")
        config = AppDIConfig(structured_storage_backend='file', learning_runtime=LearningRuntimeConfig(
            str(self.directory/'preferences'), str(self.directory/'learning')))
        container = ApplicationLookupContainer.build(config)
        self.addCleanup(container.shutdown_resources)
        self.provider.complete_text_once_with_output_budget = Mock()
        self.provider.complete_text_once_with_output_budget.side_effect = lambda prompt, budget: json.dumps({
            'status': 'completed', 'rows': [{'sentence_id': row['sentence_id'], 'translated_text': 'Bonjour.'}
                for row in json.loads(prompt.split('INPUT_JSON:\n', 1)[1])['sentences']]})
        with container.llm_provider.override(providers.Object(self.provider)), \
             container.file_structured_document_artifact_repository.override(providers.Object(self.repo)), \
             container.document_profile_store.override(providers.Object(self.profiles)):
            action = container.translation_action()
            self.app.state.translation_action = action
            response = self.post()
            self.assertEqual(response.status_code, 200, response.text)
            self.provider.complete_text_once_with_output_budget.assert_called_once()
            self.provider.complete_text.assert_not_called()
            container.shutdown_resources()
            # The action resolves the current runtime after reinitialization, not a closed worker.
            self.assertEqual(self.post().status_code, 200)
            self.provider.complete_text_once_with_output_budget.assert_called_once()

    def test_main_registers_route_and_service(self):
        import main
        self.assertIsInstance(main.app.state.translation_action, TranslationAction)
        with patch.dict(main.app.dependency_overrides, {get_translation_action: lambda: self.action}):
            with TestClient(main.app) as client:
                response = client.post(ROUTE, json=self.target | {"target_language": "fr"})
                self.assertEqual(response.status_code, 200, response.text)

    def test_readonly_profile_configuration_does_not_create_or_migrate(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("no mkdir")), \
             patch.object(FaissStorageConfig, "_migrate_legacy_namespace_dir", side_effect=AssertionError("no migration")):
            self.assertEqual(self.client.post("/documents/reading-interactions/translate/read", json=self.target).status_code, 200)


if __name__ == "__main__":
    unittest.main()
