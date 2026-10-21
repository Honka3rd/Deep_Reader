import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from scripts.test_reading_interaction_artifact_store import _document
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.note_interaction_service import NoteConflict, NoteInteractionService


class NoteServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=self.temp.name)
        self.repo.save_document(_document(), "book")
        self.service = NoteInteractionService(self.repo)
        self.target = {"target_level": "section", "section_id": "section-1"}
        self.token = self.service.read("book", **self.target)["target_token"]

    def save(self, **changes):
        kwargs = dict(text="my note", expected_revision=0, mutation_id="first",
                      target_token=self.token, **self.target)
        return self.service.save("book", **(kwargs | changes))

    def test_save_reload_retry_and_revision(self):
        first = self.save()
        self.assertEqual(first, self.save())
        self.assertEqual(self.service.read("book", **self.target)["note"], first["note"])
        with self.assertRaises(NoteConflict):
            self.save(text="overwrite", mutation_id="second")
        second = self.save(text="edited", expected_revision=1, mutation_id="second")
        self.assertEqual(second["note"]["revision"], 2)
        self.assertEqual(second["note"]["note_id"], first["note"]["note_id"])

    def test_delete_and_delayed_create(self):
        self.save()
        result = self.service.delete("book", expected_revision=1,
                                     target_token=self.token, **self.target)
        self.assertIsNone(result["note"])
        self.assertNotEqual(result["target_token"], self.token)
        with self.assertRaises(NoteConflict):
            self.save()

    def test_reparse_and_delete_do_not_resurrect_notes(self):
        self.save()
        self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
        with self.assertRaises(NoteConflict):
            self.save()
        self.repo.delete_document("book")
        with self.assertRaises(FileNotFoundError):
            self.save()

    def test_concurrent_creates_have_one_winner(self):
        def create(index):
            try:
                self.save(mutation_id=str(index))
                return True
            except NoteConflict:
                return False
        with ThreadPoolExecutor(4) as pool:
            self.assertEqual(sum(pool.map(create, range(4))), 1)

    def test_rename_preserves_note_identity_and_content(self):
        from section_tasks.interaction_rename import renamed_interactions
        note = self.save()["note"]
        self.repo.rename_document("book", "renamed", renamed_interactions)
        self.assertEqual(self.service.read("renamed", **self.target)["note"], note)
        with self.assertRaises(FileNotFoundError):
            self.service.read("book", **self.target)

    def test_titles_preserve_token_and_invalid_text_never_writes(self):
        doc = self.repo.load_document("book")
        self.repo.save_document(replace(doc, title="new title"), "book")
        self.assertEqual(self.service.read("book", **self.target)["target_token"], self.token)
        with self.assertRaises(ValueError):
            self.save(text="x" * 10001)
        self.assertIsNone(self.service.read("book", **self.target)["note"])

    def test_delayed_legacy_metadata_writer_cannot_erase_note(self):
        earlier = self.repo.load_document("book")
        note = self.save()["note"]
        self.repo.save_document(earlier, "book")
        self.assertEqual(self.service.read("book", **self.target)["note"], note)

    def test_real_route_and_coordinator(self):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import main
        from app.section_task_coordinator import SectionTaskCoordinator
        coordinator = object.__new__(SectionTaskCoordinator)
        coordinator.document_artifact_repository = self.repo
        with patch.object(main, "section_task_coordinator", coordinator):
            client = TestClient(main.app)
            target = {"doc_name": "book", "target_type": "section", "section_id": "section-1"}
            base = "/documents/reading-interactions/note/"
            response = client.post(base + "read", json={"target": target})
            self.assertEqual(response.status_code, 200, response.text)
            payload = dict(target=target, text="note", expected_revision=0,
                           mutation_id="id", target_token=response.json()["target_token"])
            saved = client.post(base + "save", json=payload)
            self.assertEqual(saved.status_code, 200, saved.text)
            self.assertEqual(client.post(base + "save", json=payload).json(), saved.json())
            conflict = client.post(base + "save", json={**payload, "mutation_id": "other"})
            self.assertEqual(conflict.status_code, 409)
            deleted = client.post(base + "delete", json={"target": target,
                                  "expected_revision": 1, "target_token": payload["target_token"]})
            self.assertEqual(deleted.status_code, 200, deleted.text)


if __name__ == "__main__":
    unittest.main()
