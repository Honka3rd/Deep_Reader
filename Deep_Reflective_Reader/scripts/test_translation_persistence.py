"""Persistent translation, exact body equality and source-dependent lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Barrier
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.translation_action import TranslationAction
from document_structure.reading_content import ReadingContentService, state_for
from language.language_profile_registry import LanguageProfileRegistry
from scripts import test_translation_action as action_tests
from section_tasks.translation_contracts import TranslationError
from section_tasks.translation_records import interactive_language, translation_languages
from section_tasks.translation_service import TranslationService
from shared.task_artifacts import DocumentTaskArtifacts, TaskArtifacts, SummaryArtifact, QuizArtifact


class PersistentTranslations(unittest.TestCase):
    setUp = action_tests.TranslationActionTests.setUp
    post = action_tests.TranslationActionTests.post

    def saved(self):
        return self.action.read(**self.target)["translations"]

    def mutate(self, action="edit", text="New body.", keep=True, block_id=None):
        service = ReadingContentService(self.repo)
        state = service.read("book")
        args = dict(action=action, expected_revision=state["revision"], source_token=state["source_token"],
                    mutation_id=uuid4().hex)
        if action != "undo":
            args["block_id"] = block_id or self.target["content_block_id"]
        if action == "edit":
            args.update(text=text, discard_interactions=not keep)
        return service.mutate("book", **args)

    def seed(self):
        def write(document):
            artifacts = document.document_task_artifacts or DocumentTaskArtifacts()
            return replace(artifacts, metadata={**artifacts.metadata,
                "public_summary": {"sample": {"payload": "summary"}},
                "ancestor_summary_artifacts": {"sample": {"payload": "ancestor"}},
                "reading_interaction_artifacts": {"quiz": {"sample": {"payload": "quiz"}}},
                "note": {"n": {"target": ["content_block", self.target["content_block_id"], "c", "s", "u",
                                             self.target["content_block_id"]], "text": "note"}}}), None
        self.repo.mutate_document_artifacts("book", write)

    def test_registry_contract_is_exact(self):
        self.assertEqual(translation_languages(), LanguageProfileRegistry.get_supported_language_codes())
        for code in translation_languages():
            self.assertEqual(interactive_language(code, field="target_language"), code)
        for code in ("unknown", "zh-Hant", "en-US", "FR", "auto", "", "fr "):
            self.assertEqual(self.post(target_language=code).status_code, 422)
        self.provider.complete_text.assert_not_called()

    def test_read_is_readonly_with_missing_profile(self):
        self.profiles.load.side_effect = FileNotFoundError()
        before = self.repo.load_document("book")
        response = self.client.post("/documents/reading-interactions/translate/read", json=self.target)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["default_source_language"])
        self.assertEqual(response.json()["translations"], [])
        self.assertEqual(self.repo.load_document("book"), before)
        self.provider.complete_text.assert_not_called()

    def test_saved_reload_multilanguage_reuse_and_replay(self):
        first = self.post(mutation_id="first").json()
        self.assertTrue(first["envelope"]["persisted"])
        self.assertNotIn("binding", first["record"])
        self.assertEqual(self.post(mutation_id="first").json(), first)
        self.assertEqual(self.provider.complete_text.call_count, 1)
        self.post(target_language="de", mutation_id="german")
        self.assertEqual(len(self.saved()), 2)
        before = self.repo.load_document("book")
        self.assertEqual(self.post(mutation_id="another-click", source_language="de").json(), first)
        self.profiles.load.side_effect = RuntimeError("profile must not be needed for saved text")
        self.assertEqual(self.post(mutation_id="without-profile", source_language="en").json(), first)
        self.assertEqual(self.post(mutation_id="default-unavailable").status_code, 503)
        self.assertEqual(self.repo.load_document("book"), before)
        self.assertEqual(self.provider.complete_text.call_count, 2)
        self.profiles.load.side_effect = None
        self.action = TranslationAction(self.repo, self.profiles, TranslationService(self.provider))
        records = {r["target_language"]: r for r in self.saved()}
        self.assertEqual(records["fr"]["revision"], 1)
        self.assertEqual(records["de"]["revision"], 1)
        self.assertEqual(self.post(mutation_id="first").json(), first)

    def test_failure_or_untranslatable_preserves_previous(self):
        self.post()
        before = self.saved()
        self.provider.complete_text.side_effect = RuntimeError("provider failure")
        self.assertEqual(self.post(target_language="de").status_code, 502)
        self.provider.complete_text.side_effect = None
        self.provider.complete_text.return_value = '{"status":"untranslatable","translated_text":null}'
        result = self.post(target_language="de").json()
        self.assertFalse(result["envelope"]["persisted"])
        self.assertIsNone(result["record"])
        self.assertEqual(self.saved(), before)

    def test_partial_is_saved(self):
        self.provider.complete_text.return_value = '{"status":"partial","translated_text":"Bonjour [illisible]."}'
        self.assertEqual(self.post().json()["record"]["status"], "partial")
        self.assertEqual(self.post().json()["record"]["status"], "partial")
        self.assertEqual(self.provider.complete_text.call_count, 1)

    def test_changed_source_generates_again(self):
        self.post()
        self.mutate()
        self.assertEqual(self.post().status_code, 200)
        self.assertEqual(self.provider.complete_text.call_count, 2)

    def test_same_text_noop_with_both_consents_and_replay(self):
        self.seed()
        self.post()
        service = ReadingContentService(self.repo)
        initial = service.read("book")
        text = next(b["content"] for b in initial["blocks"] if b["block_id"] == self.target["content_block_id"])
        artifacts = self.repo.load_document("book").document_task_artifacts
        for keep in (True, False):
            result = self.mutate(text=text, keep=keep)
            self.assertEqual(result["revision"], initial["revision"])
            self.assertFalse(result["refresh"]["body_changed"])
            self.assertFalse(result["can_undo"])
            self.assertFalse(result["blocks"][0]["manually_edited"])
            self.assertEqual(self.repo.load_document("book").document_task_artifacts, artifacts)
        self.assertEqual(len(self.saved()), 1)

    def test_actual_keep_clears_translations_summaries_not_quiz_note(self):
        self.seed()
        self.post()
        other = ReadingContentService(self.repo).read("book")["blocks"][1]["block_id"]
        self.assertEqual(self.post(content_block_id=other).status_code, 200)
        result = self.mutate(keep=True)
        self.assertTrue(result["refresh"]["summaries_invalidated"])
        self.assertFalse(result["refresh"]["generated_invalidated"])
        metadata = self.repo.load_document("book").document_task_artifacts.metadata
        self.assertNotIn("public_summary", metadata)
        self.assertNotIn("ancestor_summary_artifacts", metadata)
        self.assertTrue(metadata["reading_interaction_artifacts"]["quiz"])
        self.assertTrue(metadata["note"])
        self.assertEqual(self.saved(), [])
        self.assertEqual(len(self.action.read(doc_name="book", content_block_id=other)["translations"]), 1)

    def test_change_clear_and_delete_clear_other_metadata(self):
        self.seed()
        self.post()
        self.mutate(keep=False)
        metadata = self.repo.load_document("book").document_task_artifacts.metadata
        self.assertFalse(metadata.get("reading_interaction_artifacts"))
        self.assertFalse(metadata.get("note"))
        self.post()
        self.mutate(action="delete")
        self.assertFalse(self.repo.load_document("book").document_task_artifacts.metadata.get("translations"))

    def test_overlay_same_text_preserves_flag_version_undo(self):
        self.mutate(text="Exact  text, punctuation.")
        self.post()
        before = deepcopy(state_for(self.repo.load_document("book")))
        self.mutate(text="Exact  text, punctuation.")
        after = state_for(self.repo.load_document("book"))
        for key in ("revision", "undo", "audit", "block_edit_flags", "block_content_epochs"):
            self.assertEqual(before.get(key), after.get(key))
        self.assertEqual(len(self.saved()), 1)
        self.mutate(text="Exact text, punctuation.")
        self.assertFalse(self.saved())

    def test_undo_does_not_resurrect_and_old_writer_preserves_current(self):
        self.seed()
        self.post()
        old = self.repo.load_document("book")
        self.mutate()
        self.repo.save_document(old, "book")
        self.assertFalse(self.saved())
        self.mutate(action="undo")
        self.assertFalse(self.saved())
        self.assertTrue(self.repo.load_document("book").document_task_artifacts.metadata["note"])

    def test_legacy_save_cannot_erase_new_translation(self):
        old = self.repo.load_document("book")
        self.post()
        self.repo.save_document(old, "book")
        self.assertEqual(len(self.saved()), 1)

    def test_rename_rebinds_saved_records(self):
        from section_tasks.interaction_rename import renamed_interactions
        self.post()
        before = self.saved()
        self.repo.rename_document("book", "renamed", renamed_interactions)
        self.assertEqual(self.action.read(doc_name="renamed", content_block_id=self.target["content_block_id"])["translations"], before)

    def concurrent(self, codes):
        barrier = Barrier(2)
        def complete(prompt):
            barrier.wait(timeout=10)
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        def generate(code):
            try:
                return self.action.generate(**self.target, target_language=code, mutation_id=uuid4().hex)
            except TranslationError as error:
                return error.code
        with ThreadPoolExecutor(2) as pool:
            return list(pool.map(generate, codes))

    def test_same_language_concurrent_write_has_one_winner(self):
        results = self.concurrent(["fr", "fr"])
        self.assertEqual(sum(not isinstance(r, str) for r in results), 1)
        self.assertIn("translation_mutation_conflict", results)

    def test_different_languages_merge(self):
        self.assertTrue(all(not isinstance(r, str) for r in self.concurrent(["fr", "de"])))
        self.assertEqual(len(self.saved()), 2)

    def test_edit_then_undo_during_generation_rejects_old_epoch(self):
        def complete(prompt):
            self.mutate()
            self.mutate(action="undo")
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        self.assertEqual(self.post().status_code, 409)
        self.assertFalse(self.saved())

    def test_commit_failure_preserves_existing(self):
        self.post()
        before = self.saved()
        with patch.object(self.repo, "mutate_document_artifacts", side_effect=RuntimeError("commit failed")):
            self.assertEqual(self.post(target_language="de").status_code, 503)
        self.assertEqual(self.saved(), before)

    def test_identical_command_replay_and_stale_editor(self):
        from document_structure.reading_content import ReadingContentConflict
        service = ReadingContentService(self.repo)
        state = service.read("book")
        block = next(b for b in state["blocks"] if b["block_id"] == self.target["content_block_id"])
        command = dict(action="edit", text=block["content"], block_id=block["block_id"],
            expected_revision=state["revision"], source_token=state["source_token"], mutation_id="noop", discard_interactions=True)
        first = service.mutate("book", **command)
        self.assertEqual(service.mutate("book", **command), first)
        self.mutate(text=block["content"] + " ")
        with self.assertRaises(ReadingContentConflict):
            service.mutate("book", **{**command, "mutation_id": "stale"})

    def test_derived_summary_only_cleanup_preserves_quiz(self):
        from document_structure.reading_content_artifacts import clear_source_dependents
        document = self.repo.load_document("book")
        mixed = TaskArtifacts(summary=SummaryArtifact("summary"), quiz=QuizArtifact([{"question": "quiz"}]))
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = section.task_units[0]
        document = replace(document, document_task_artifacts=DocumentTaskArtifacts(chapter_artifacts={"c": mixed}),
            chapters=[replace(chapter, task_artifacts=mixed, sections=[replace(section, task_artifacts=mixed,
                task_units=[replace(unit, task_artifacts=mixed)])])])
        updated = clear_source_dependents(document, [self.target["content_block_id"]])
        values = [updated.document_task_artifacts.chapter_artifacts["c"], updated.chapters[0].task_artifacts,
            updated.chapters[0].sections[0].task_artifacts, updated.chapters[0].sections[0].task_units[0].task_artifacts]
        self.assertTrue(all(value.summary is None and value.quiz == mixed.quiz for value in values))

    def test_reparse_and_delete_during_model_never_save(self):
        def complete(prompt):
            self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
            return '{"status":"completed","translated_text":"Bonjour."}'
        self.provider.complete_text.side_effect = complete
        self.assertEqual(self.post().status_code, 409)

    def test_same_command_concurrent_outputs_return_committed_record(self):
        barrier = Barrier(2)
        def complete(prompt):
            barrier.wait(timeout=10)
            return '{"status":"completed","translated_text":"Bonjour ' + uuid4().hex + '."}'
        self.provider.complete_text.side_effect = complete
        def generate(_):
            return self.action.generate(**self.target, target_language="fr", mutation_id="same")
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(generate, range(2)))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
