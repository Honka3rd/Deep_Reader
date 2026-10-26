"""CLEAN-08 read-only projection and public route contracts, isolated files only."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
from document_structure.reading_content import cleanup_audit_for, state_for
from document_structure.source_mapped_text import PageRange
from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests


class CleanupAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = CleanupOverlayTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def apply(self):
        candidate, application = self.f.prepare()
        self.f.commit(candidate, replace(application, pages=(PageRange(0, 0, len(self.f.text)),)))

    def test_read_only_default_and_opt_in_original_evidence(self):
        self.apply()
        before = self.f.snapshot()[2]
        with patch.object(main, "document_artifact_repository", self.f.repo):
            client = TestClient(main.app)
            self.assertNotIn("cleanup_audit", client.get("/documents/book/reading-content").json())
            result = client.get("/documents/book/reading-content?include_cleanup_audit=true").json()
            self.assertEqual(result["blocks"], [])
            self.assertEqual(result["undo_action"], "ocr_cleanup")
            batch = result["cleanup_audit"]["items"][0]
            self.assertTrue(batch["undoable"])
            deletion, = batch["deletions"]
            self.assertEqual({key: deletion[key] for key in ("quote", "reason", "page_indices")},
                             {"quote": "@@??", "reason": "ocr_garbage", "page_indices": [0]})
            self.assertTrue(deletion["can_restore"])
            self.assertEqual(deletion["links"][0]["block_id"], "u1:content:0")
            self.assertEqual(result["regions"], [])
            self.assertEqual(client.get("/documents/book/reading-content?cleanup_offset=-1").status_code, 422)
            self.assertEqual(client.get("/documents/book/reading-content?cleanup_offset=1001").status_code, 422)
        self.assertEqual(self.f.snapshot()[2], before)

    def test_undo_restores_queue_and_preserves_audit_source_notes(self):
        original = self.f.snapshot()[0]
        self.apply()
        self.f.change("undo")
        result = self.f.service.read("book", include_cleanup_audit=True)
        batch = result["cleanup_audit"]["items"][0]
        self.assertEqual(batch["remaining_count"], 0)
        self.assertFalse(batch["undoable"])
        self.assertFalse(result["can_undo"])
        self.assertEqual(len(result["regions"]), 1)
        current = self.f.snapshot()[0]
        self.assertEqual(current.raw_text, original.raw_text)
        self.assertEqual(current.parse_provenance["source_page_spans"], original.parse_provenance["source_page_spans"])
        self.assertIn("note", current.document_task_artifacts.metadata)
        self.assertNotIn("public_summary", current.document_task_artifacts.metadata)

    def test_manual_edit_supersedes_batch_undo(self):
        self.apply()
        self.f.change("edit", block_id="u1:content:1", text="Manual correction")
        result = self.f.service.read("book", include_cleanup_audit=True)
        self.assertEqual(result["undo_action"], "edit")
        self.assertFalse(result["cleanup_audit"]["items"][0]["undoable"])
        self.assertEqual(result["cleanup_audit"]["items"][0]["remaining_count"], 1)

    def test_pagination_is_bounded_and_does_not_leak_internal_binding(self):
        self.apply()
        document = deepcopy(self.f.snapshot()[0])
        state = state_for(document)
        entry = state["audit"][0]
        state["audit"] = [{**entry, "revision": i + 1} for i in range(23)]
        page = cleanup_audit_for(document, 10)
        self.assertEqual(page["total"], 23)
        self.assertEqual(len(page["items"]), 10)
        self.assertEqual(page["items"][0]["revision"], 13)
        self.assertEqual(len(cleanup_audit_for(document, 20)["items"]), 3)
        self.assertEqual(cleanup_audit_for(document, 1000)["items"], [])
        self.assertNotIn("operation_id", repr(page))
        self.assertNotIn("char_start", repr(page))

    def test_explicit_undo_advances_only_current_layout_stamp_without_changing_units(self):
        self.apply()
        def stamp(document):
            artifacts = document.document_task_artifacts
            return replace(document, document_task_artifacts=replace(artifacts, metadata={
                **artifacts.metadata, "task_layout": {"reading_revision": 1, "source_hash": "unchanged"}})), None
        self.f.repo.mutate_reading_content("book", stamp)
        before = self.f.snapshot()[0]
        self.f.change("undo")
        after = self.f.snapshot()[0]
        self.assertEqual(after.chapters, before.chapters)
        self.assertEqual(after.document_task_artifacts.metadata["task_layout"],
                         {"reading_revision": 2, "source_hash": "unchanged"})
        self.f.repo.mutate_reading_content("book", stamp)
        self.f.change("edit", block_id="u1:content:1", text="Manual correction")
        self.assertEqual(self.f.snapshot()[0].document_task_artifacts.metadata["task_layout"]["reading_revision"], 1)


if __name__ == "__main__":
    unittest.main()
