"""New actions under the real conditional enhancement commit transaction."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_enhancement_conditional_commit import ConditionalCommitTests
from test_summary_interaction_service import Provider
from section_tasks.note_interaction_service import NoteInteractionService, NoteConflict
from section_tasks.summary_interaction_service import SummaryInteractionService
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.reading_target_resolver import ReadingTargetResolver


class NewActionLifecycle(ConditionalCommitTests):
    def seed_new_actions(self):
        self.notes = NoteInteractionService(self.repository)
        token = self.notes.read("doc", target_level="document")["target_token"]
        self.note = self.notes.save("doc", target_level="document", text="Keep my note",
                                    mutation_id="first", expected_revision=0, target_token=token)
        self.summaries = SummaryInteractionService(self.repository, Provider())
        target = ReadingTargetResolver().resolve(document=self.repository.load_document("doc"), target_level="document")
        target = replace(target, source_structure_version=1)
        self.summary_request = ReadingInteractionRequest(target=target, interaction_type="summary",
            context_metadata={"doc_name": "doc", "document_language": "en"})
        self.summary = self.summaries.generate(self.summary_request)
        self.assertEqual(self.summary.status, "completed", self.summary.reason)

    def test_new_actions_noop_retained(self):
        self.seed_new_actions()
        state, candidate = self.matching_candidate()
        self.assertEqual(self.commit_candidate(state, candidate).change_kind, "unchanged")
        self.assertEqual(self.notes.read("doc", target_level="document"), self.note)
        self.assertEqual(self.summaries.read(self.summary_request), self.summary)

    def test_new_actions_title_only_retained(self):
        self.seed_new_actions()
        state, candidate = self.matching_candidate("New title")
        self.assertEqual(self.commit_candidate(state, candidate).change_kind, "titles_only")
        self.assertEqual(self.notes.read("doc", target_level="document")["note"], self.note["note"])
        self.assertEqual(self.summaries.read(self.summary_request), self.summary)

    def test_new_actions_structural_clear_rejects_delayed_note(self):
        self.seed_new_actions()
        self.assertTrue(self.snapshot().includes_user_notes)
        state, candidate = self.make_candidate()
        self.assertEqual(self.commit_candidate(state, candidate).change_kind, "structure_changed")
        self.assertIsNone(self.notes.read("doc", target_level="document")["note"])
        with self.assertRaises(NoteConflict):
            self.notes.save("doc", target_level="document", text="late", mutation_id="late",
                            expected_revision=1, target_token=self.note["target_token"])
        self.assertEqual(self.summaries.read(self.summary_request).status, "not_generated")

    def test_new_actions_failed_commit_retained(self):
        self.seed_new_actions()
        state, candidate = self.matching_candidate("New title")
        from unittest.mock import patch
        with patch.object(self.store, "_update_enhancement_labels", side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError):
                self.commit_candidate(state, candidate)
        self.assertEqual(self.notes.read("doc", target_level="document"), self.note)
        self.assertEqual(self.summaries.read(self.summary_request), self.summary)


if __name__ == "__main__":
    suite = unittest.TestSuite(NewActionLifecycle(name) for name in
        unittest.defaultTestLoader.getTestCaseNames(NewActionLifecycle) if name.startswith("test_new_actions_"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
