"""Reading overlays, review guards and mutation contracts in isolated storage."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.test_reading_interaction_artifact_store import _document
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.reading_content import ReadingContentService, ReadingContentConflict, project_document
from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.artifact_validity import ReadingInteractionValidityPolicy
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.note_interaction_service import NoteInteractionService


class ReadingContentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=self.temp.name)
        self.repo.save_document(_document(),"book")
        self.service = ReadingContentService(self.repo)
        self.initial = self.service.read("book")
        self.block = self.initial["blocks"][0]

    def change(self, action="edit", **changes):
        state = self.service.read("book")
        kwargs = dict(action=action,expected_revision=state["revision"],source_token=state["source_token"],
                      mutation_id=f"mutation-{state['revision']}",block_id=self.block["block_id"])
        if action=="edit":
            kwargs["text"] = "Corrected readable source text with enough details for testing."
        return self.service.mutate("book",**(kwargs|changes))

    def test_edit_immutable_source_stable_ids_and_parent_projection(self):
        original = self.repo.load_document("book")
        result = self.change()
        stored = self.repo.load_document("book")
        self.assertEqual(original.raw_text,stored.raw_text)
        self.assertEqual(original.chapters,stored.chapters)
        self.assertEqual([b["block_id"] for b in result["blocks"]],[b["block_id"] for b in self.initial["blocks"]])
        resolver = ReadingTargetResolver()
        self.assertIn("Corrected readable",resolver.resolve(document=stored,target_level="document").content)
        target = resolver.resolve(document=stored,target_level="content_block",task_unit_id=self.block["task_unit_id"],content_block_id=self.block["block_id"])
        self.assertIn("Corrected readable",target.content)
        self.assertEqual(target.reading_revision,1)

    def test_delete_and_undo_preserve_identity(self):
        result = self.change("delete")
        self.assertTrue(result["blocks"][0]["deleted"])
        projected = project_document(self.repo.load_document("book"))
        with self.assertRaises(ValueError):
            ReadingTargetResolver().resolve(document=projected,target_level="content_block",
                task_unit_id=self.block["task_unit_id"],content_block_id=self.block["block_id"])
        restored = self.change("undo")
        self.assertEqual(restored["blocks"],self.initial["blocks"])
        self.assertFalse(restored["can_undo"])

    def test_idempotency_and_conflict(self):
        args = dict(action="edit",expected_revision=0,source_token=self.initial["source_token"],mutation_id="retry",
                    block_id=self.block["block_id"],text="Corrected source")
        first = self.service.mutate("book",**args)
        self.assertEqual(first,self.service.mutate("book",**args))
        with self.assertRaises(ReadingContentConflict):
            self.service.mutate("book",**(args|{"text":"different"}))
        with self.assertRaises(ReadingContentConflict):
            self.service.mutate("book",**(args|{"mutation_id":"stale"}))

    def test_concurrent_writers_one_winner(self):
        def write(index):
            try:
                self.change(expected_revision=0,mutation_id=str(index))
                return True
            except ReadingContentConflict:
                return False
        with ThreadPoolExecutor(4) as pool:
            self.assertEqual(sum(pool.map(write,range(4))),1)

    def test_invalid_text_missing_block_and_no_add(self):
        for text in ["", " ", "x"*10001, "\U0001f600"*10001]:
            with self.assertRaises(ValueError): self.change(text=text)
        with self.assertRaises(ValueError): self.change("insert")
        with self.assertRaises(ReadingContentConflict): self.change(block_id="unknown")
        self.assertEqual(self.service.read("book")["revision"],0)
        self.change(text="\U0001f600"*10000)

    def test_late_metadata_save_cannot_erase_overlay(self):
        from shared.task_artifacts import DocumentTaskArtifacts
        old = self.repo.load_document("book")
        old = replace(old, document_task_artifacts=DocumentTaskArtifacts(metadata={"reading_interaction_artifacts":{"obsolete":True}}))
        self.change()
        self.repo.save_document(old,"book")
        self.assertEqual(self.service.read("book")["revision"],1)
        artifacts = self.repo.load_document("book").document_task_artifacts
        self.assertNotIn("reading_interaction_artifacts", artifacts.metadata if artifacts else {})

    def test_late_generated_artifact_rejected_and_current_write_allowed(self):
        from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact
        from section_tasks.interaction_policy_identity import policy_metadata
        def artifact():
            target = ReadingTargetResolver().resolve(document=self.repo.load_document("book"), target_level="document")
            request = ReadingInteractionRequest(target=target, interaction_type="analysis", context_metadata={"doc_name":"book"})
            return ReadingInteractionArtifact.from_target(target=target, interaction_type="analysis", status="completed",
                payload={"interpretation":"A supported interpretation"}, metadata={**policy_metadata(request), "context":{"doc_name":"book"}})
        store = DocumentReadingInteractionArtifactStore(self.repo)
        stale = artifact()
        self.change()
        with self.assertRaisesRegex(ValueError,"stale_interaction_write"):
            store.save_analysis_artifact(stale)
        store.save_analysis_artifact(artifact())

    def test_reparse_clears_overlay_and_invalidates_editor(self):
        self.change()
        self.repo.save_reparsed_document(self.repo.load_document("book"),"book")
        self.assertEqual(self.service.read("book")["revision"],0)
        with self.assertRaises(ReadingContentConflict): self.change(source_token=self.initial["source_token"])

    def test_notes_survive_edit_with_notice(self):
        notes = NoteInteractionService(self.repo)
        target = {"target_level":"section","section_id":self.block["section_id"]}
        token = notes.read("book",**target)["target_token"]
        note = notes.save("book",text="My original note",expected_revision=0,mutation_id="n",target_token=token,**target)["note"]
        self.change()
        response = notes.read("book",**target)
        self.assertEqual(response["note"],note)
        self.assertTrue(response["source_stale"])

    def test_review_blocks_generation_and_discard_can_undo(self):
        original = self.repo.load_document("book")
        self.repo.save_document(replace(original,parse_provenance={**original.parse_provenance,
            "source_review":{"source_sha256":"a"*64,"regions":[{"id":"page-1-region-0","page_index":1,
            "bbox":[0,0,100,100],"render_dpi":200,"reason":"recognition_uncertain","raw_candidate":""}]}}),"book")
        target = ReadingTargetResolver().resolve(document=self.repo.load_document("book"),target_level="document")
        policy = ReadingInteractionValidityPolicy(interaction_label="test",min_content_chars=0,min_alnum_chars=0)
        for kind in ["analysis","quiz","critical_thinking_session","summary"]:
            decision = policy.preflight(ReadingInteractionRequest(target=target,interaction_type=kind))
            self.assertEqual(decision.reason,"source_needs_review")
        self.assertEqual(self.service.read("book")["disposition"],"needs_review")
        self.change("discard_region",region_id="page-1-region-0")
        self.assertEqual(self.service.read("book")["disposition"],"completed")
        self.change("undo")
        self.assertEqual(self.service.read("book")["disposition"],"needs_review")

    def test_failed_write_preserves_source_and_revision(self):
        with patch.object(self.repo,"_atomic_save",side_effect=OSError("disk full")):
            with self.assertRaises(OSError): self.change()
        self.assertEqual(self.service.read("book"),self.initial)

    def test_routes_have_no_insert_and_stale_conflicts(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main,"_reading_content_service",return_value=self.service):
            client = TestClient(main.app)
            self.assertEqual(client.get("/documents/book/reading-content").json()["blocks"],[])
            body = dict(action="edit",expected_revision=0,source_token=self.initial["source_token"],
                        mutation_id="api",block_id=self.block["block_id"],text="New source")
            self.assertEqual(client.post("/documents/book/reading-content",json=body).status_code,200)
            self.assertEqual(client.post("/documents/book/reading-content",json=body|{"mutation_id":"late"}).status_code,409)
            self.assertEqual(client.post("/documents/book/reading-content",json=body|{"action":"insert"}).status_code,422)


if __name__ == "__main__": unittest.main()
