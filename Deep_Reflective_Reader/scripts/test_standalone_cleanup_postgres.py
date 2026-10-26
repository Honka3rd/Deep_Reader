"""Real SQL cleanup parity, rollback and races in a disposable schema."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch
import unittest

import psycopg
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from document_structure.reading_content import ReadingContentConflict
from scripts import test_standalone_cleanup_commit as shared
from scripts import test_summary_note_postgres as harness


class PostgresStandaloneCleanupTests(shared.StandaloneCleanupCommitTests):
    setUpClass = classmethod(harness.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(harness.PostgresNotes.drop_schema.__func__)

    def repository(self):
        self.store = PostgresStructuredDocumentStore(self.scoped, namespace=self.namespace)
        return PostgresStructuredDocumentArtifactRepository(self.store)

    def snapshot(self):
        value = self.repo.load_enhancement_snapshot("book")
        return value.document, value.structure_version, value.interaction_revision

    def fail_write(self):
        write = self.store._commit_cleanup_state
        def fail(*args):
            write(*args)
            raise RuntimeError("injected after all SQL updates")
        return patch.object(self.store, "_commit_cleanup_state", side_effect=fail)

    def test_relational_notes_and_profile_preserved_snapshot_consistent(self):
        with psycopg.connect(self.scoped) as conn:
            doc_id = conn.execute("SELECT id FROM documents WHERE namespace=%s", (self.namespace,)).fetchone()[0]
            for kind in ("analysis", "quiz", "note"):
                conn.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,"
                    "source_structure_version,payload) VALUES (%s,%s,'document',%s,1,'{}')", (doc_id, kind, doc_id))
            conn.execute("INSERT INTO document_profile(document_id,profile_payload) VALUES (%s,'{\"summary\":\"Keep accepted profile\"}')",
                         (doc_id,))
            profile_before = conn.execute("SELECT * FROM document_profile WHERE document_id=%s", (doc_id,)).fetchall()
            hierarchy_before = {t: conn.execute(f"SELECT id FROM {t} WHERE document_id=%s ORDER BY id", (doc_id,)).fetchall()
                                for t in ("chapters", "sections", "task_units", "content_blocks")}
        self.repo.commit_cleanup(self.prepare())
        with psycopg.connect(self.scoped) as conn:
            self.assertEqual(conn.execute("SELECT artifact_type FROM artifacts WHERE document_id=%s", (doc_id,)).fetchall(), [("note",)])
            self.assertEqual(conn.execute("SELECT * FROM document_profile WHERE document_id=%s", (doc_id,)).fetchall(), profile_before)
            for table, ids in hierarchy_before.items():
                self.assertEqual(conn.execute(f"SELECT id FROM {table} WHERE document_id=%s ORDER BY id", (doc_id,)).fetchall(), ids)
            saved = conn.execute("SELECT structured_document_payload FROM structured_document_snapshots WHERE document_id=%s", (doc_id,)).fetchone()[0]
            metadata = conn.execute("SELECT metadata_payload FROM documents WHERE id=%s", (doc_id,)).fetchone()[0]
            self.assertEqual(saved["parse_provenance"]["reading_content"], metadata["parse_provenance"]["reading_content"])
            self.assertEqual(saved["document_task_artifacts"], metadata["document_task_artifacts"])

    def test_noop_does_not_touch_child_or_generated_rows(self):
        def rows():
            with psycopg.connect(self.scoped) as conn:
                doc_id = conn.execute("SELECT id FROM documents WHERE namespace=%s", (self.namespace,)).fetchone()[0]
                return {t: conn.execute(f"SELECT id,xmin::text FROM {t} WHERE document_id=%s ORDER BY id", (doc_id,)).fetchall()
                        for t in ("chapters", "sections", "task_units", "artifacts", "content_blocks")}
        before = rows()
        self.repo.commit_cleanup(self.prepare([]))
        self.assertEqual(rows(), before)

    def test_receipt_write_failure_rolls_back_sql(self):
        command = self.prepare()
        before = self.snapshot()
        execute = psycopg.Connection.execute
        def fail(conn, query, *args, **kwargs):
            if isinstance(query, str) and query.startswith("UPDATE structured_document_snapshots SET structured_document_payload=%s"):
                raise RuntimeError("injected receipt snapshot failure")
            return execute(conn, query, *args, **kwargs)
        with patch.object(psycopg.Connection, "execute", fail), self.assertRaisesRegex(RuntimeError, "injected"):
            self.repo.commit_cleanup(command)
        self.assertEqual(before, self.snapshot())
        self.assertIsNone(self.lookup())

    def test_concurrent_manual_writer_wins_then_cleanup_is_stale(self):
        command = self.prepare()
        block_id = self.service.read("book")["blocks"][1]["block_id"]
        locked, release = Event(), Event()
        original = self.store.load_enhancement_snapshot
        def held(*args, **kwargs):
            value = original(*args, **kwargs)
            if kwargs.get("_connection") is not None and not locked.is_set():
                locked.set()
                if not release.wait(5):
                    raise RuntimeError("test barrier timeout")
            return value
        with patch.object(self.store, "load_enhancement_snapshot", held), ThreadPoolExecutor(2) as pool:
            manual = pool.submit(self.change, "edit", block_id=block_id, text="Concurrent manual correction")
            try:
                self.assertTrue(locked.wait(5))
                cleanup = pool.submit(self.repo.commit_cleanup, command)
            finally:
                release.set()
            manual.result(timeout=10)
            with self.assertRaisesRegex(ReadingContentConflict, "cleanup_confirmation_stale"):
                cleanup.result(timeout=10)
        self.assertIn("Concurrent manual correction", self.content())
        self.assertIsNone(self.lookup())

    def test_concurrent_reparse_wins_then_cleanup_is_stale(self):
        command = self.prepare()
        document = self.repo.load_document("book")
        locked, release = Event(), Event()
        execute = psycopg.Connection.execute
        def held(conn, query, *args, **kwargs):
            result = execute(conn, query, *args, **kwargs)
            if isinstance(query, str) and "SELECT id," in query and "FOR UPDATE" in query and not locked.is_set():
                locked.set()
                if not release.wait(5):
                    raise RuntimeError("test barrier timeout")
            return result
        with patch.object(psycopg.Connection, "execute", held), ThreadPoolExecutor(2) as pool:
            reparse = pool.submit(self.repo.save_reparsed_document, document, "book")
            try:
                self.assertTrue(locked.wait(5))
                cleanup = pool.submit(self.repo.commit_cleanup, command)
            finally:
                release.set()
            reparse.result(timeout=10)
            with self.assertRaisesRegex(ReadingContentConflict, "cleanup_confirmation_stale"):
                cleanup.result(timeout=10)
        self.assertIsNone(self.lookup())


if __name__ == "__main__":
    unittest.main()
