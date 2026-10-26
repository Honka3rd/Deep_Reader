"""Actual PostgreSQL transactions in a disposable schema, no live book writes."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest

import psycopg
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests
from scripts.test_summary_note_postgres import PostgresNotes


class PostgresCleanupOverlayTests(CleanupOverlayTests):
    setUpClass = classmethod(PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(PostgresNotes.drop_schema.__func__)

    def repository(self):
        self.store = PostgresStructuredDocumentStore(self.scoped, namespace=self.namespace)
        return PostgresStructuredDocumentArtifactRepository(self.store)

    def snapshot(self):
        snapshot = self.repo.load_enhancement_snapshot("book")
        return snapshot.document, snapshot.structure_version, snapshot.interaction_revision

    def commit(self, candidate, application):
        return self.repo.commit_enhancement("book", candidate=candidate, cleanup=application,
            interaction_revision=application.plan.binding.structure_fingerprint, source_path=self.path)

    def test_failed_final_file_write_preserves_everything(self):
        command = self.prepare("structure_changed")
        before = self.snapshot()
        save = self.store.save
        def fail_after_hierarchy(*args, **kwargs):
            save(*args, **kwargs)
            raise RuntimeError("injected after hierarchy save")
        with patch.object(self.store, "save", side_effect=fail_after_hierarchy):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.commit(*command)
        self.assertEqual(before, self.snapshot())

    def test_cleanup_only_update_rolls_back_revision_labels_and_generated(self):
        command = self.prepare("titles_only")
        before = self.snapshot()
        update = self.store._commit_cleanup_state
        def fail_after_cleanup(*args):
            update(*args)
            raise RuntimeError("injected after cleanup")
        with patch.object(self.store, "_commit_cleanup_state", side_effect=fail_after_cleanup):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.commit(*command)
        self.assertEqual(before, self.snapshot())

    def test_single_atomic_save_no_saved_candidate_reload(self):
        command = self.prepare("structure_changed")
        with patch.object(self.store, "save", wraps=self.store.save) as save:
            self.commit(*command)
        self.assertEqual(save.call_count, 1)
        self.assertIsNotNone(save.call_args.kwargs["_connection"])
        self.assertTrue(save.call_args.kwargs["_preserve_reading_state"])
        # Precondition read and post-write result read occur inside the same transaction;
        # neither result is input to a second cleanup/application pass.

    def test_concurrent_commits_do_not_duplicate_cleanup(self):
        command = self.prepare("structure_changed")
        def run(_):
            try:
                self.commit(*command)
                return True
            except ValueError as error:
                self.assertIn(str(error), {"enhancement_commit_busy", "confirmation_stale"})
                return False
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, range(2)))
        self.assertTrue(any(results))
        state = self.repo.load_document("book").parse_provenance["reading_content"]
        self.assertEqual(state["revision"], 1)
        self.assertEqual(len(state["audit"]), 1)
        self.assertEqual(self.snapshot()[1], 2)

    def test_generated_rows_deleted_but_note_rows_preserved_for_cleanup_only(self):
        with psycopg.connect(self.scoped) as connection:
            doc_id = connection.execute("SELECT id FROM documents WHERE namespace=%s", (self.namespace,)).fetchone()[0]
            for kind in ("analysis", "quiz", "note"):
                connection.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,"
                    "source_structure_version,payload) VALUES (%s,%s,'document',%s,1,'{}')", (doc_id,kind,doc_id))
        self.commit(*self.prepare())
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT artifact_type FROM artifacts WHERE document_id=%s",
                (doc_id,)).fetchall(), [("note",)])


if __name__ == "__main__":
    unittest.main()
