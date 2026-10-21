"""Real disposable-schema checks for notes, summaries and lifecycle cleanup."""
import os
from pathlib import Path
import unittest
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from scripts.test_note_interaction_service import NoteServiceTests
from scripts.test_summary_interaction_service import SummaryTests, Provider
from scripts.test_reading_interaction_artifact_store import _document
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from section_tasks.note_interaction_service import NoteInteractionService
from section_tasks.summary_interaction_service import SummaryInteractionService


class PostgresNotes(NoteServiceTests):
    @classmethod
    def setUpClass(cls):
        cls.dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
        cls.schema = "summary_note_" + uuid4().hex
        with psycopg.connect(cls.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.scoped = make_conninfo(cls.dsn, options=f"-c search_path={cls.schema}")
        with psycopg.connect(cls.scoped) as conn:
            for path in sorted((Path(__file__).resolve().parents[1] / "db/migrations").glob("*.sql")):
                conn.execute(path.read_text())

    @classmethod
    def drop_schema(cls):
        with psycopg.connect(cls.dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        store = PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex)
        self.repo = PostgresStructuredDocumentArtifactRepository(store)
        self.repo.save_reparsed_document(_document(), "book")
        self.service = NoteInteractionService(self.repo)
        section = self.repo.load_document("book").chapters[0].sections[0]
        self.target = {"target_level": "section", "section_id": section.section_id}
        self.token = self.service.read("book", **self.target)["target_token"]

    def test_real_route_and_coordinator(self):
        # Route shape is covered by the file-adapter test; here use actual DB IDs.
        self.save()
        self.assertEqual(self.service.read("book", **self.target)["note"]["text"], "my note")

    def test_consent_inventory_includes_notes(self):
        self.save()
        self.assertTrue(self.repo.load_enhancement_snapshot("book").includes_user_notes)

    def test_summary_and_note_do_not_overwrite_each_other(self):
        from section_tasks.reading_target_resolver import ReadingTargetResolver
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
        self.save()
        target = ReadingTargetResolver().resolve(document=self.repo.load_document("book"), **self.target)
        from dataclasses import replace
        target = replace(target, source_structure_version=1)
        request = ReadingInteractionRequest(target=target, interaction_type="summary",
                                            context_metadata={"doc_name": "book", "document_language": "en"})
        summary = SummaryInteractionService(self.repo, Provider())
        result = summary.generate(request)
        self.assertEqual(result.status, "completed", result.reason)
        self.assertIsNotNone(self.service.read("book", **self.target)["note"])
        self.save(text="next", expected_revision=1, mutation_id="second")
        self.assertEqual(summary.read(request), result)

    def test_rename_route_preserves_new_actions_and_rolls_back_conflict(self):
        import tempfile
        from dataclasses import replace
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import main
        from section_tasks.reading_target_resolver import ReadingTargetResolver
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            (raw / "book.txt").write_text("source", encoding="utf-8")
            document = self.repo.load_document("book")
            self.repo.save_document(replace(document, source_path=str(raw / "book.txt")), "book")
            note = self.save()["note"]
            summaries = SummaryInteractionService(self.repo, Provider())
            target = replace(ReadingTargetResolver().resolve(document=self.repo.load_document("book"),
                             target_level="document"), source_structure_version=1)
            request = ReadingInteractionRequest(target=target, interaction_type="summary",
                                               context_metadata={"doc_name": "book", "document_language": "en"})
            summary = summaries.generate(request)
            self.assertEqual(summary.status, "completed", summary.reason)
            with patch.object(main, "document_artifact_repository", self.repo), \
                 patch.object(main, "_RAW_DOCUMENT_BASE_DIR", raw), \
                 patch.object(main, "_DERIVED_DOCUMENT_BASE_DIR", Path(directory) / "derived"):
                client = TestClient(main.app)
                result = client.post("/documents/lifecycle/rename", json={
                    "doc_name": "book", "new_doc_name": "renamed", "action": "rename"})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertFalse((raw / "book.txt").exists())
                self.assertTrue((raw / "renamed.txt").exists())
                self.assertEqual(self.service.read("renamed", **self.target)["note"], note)
                target = replace(target, document_id="renamed", target_id="renamed")
                request = replace(request, target=target, context_metadata={"doc_name": "renamed", "document_language": "en"})
                self.assertEqual(summaries.read(request).payload, summary.payload)
                self.repo.save_reparsed_document(_document(), "occupied")
                conflict = client.post("/documents/lifecycle/rename", json={
                    "doc_name": "renamed", "new_doc_name": "occupied", "action": "rename"})
                self.assertEqual(conflict.status_code, 409, conflict.text)
                self.assertTrue((raw / "renamed.txt").exists())
                self.assertFalse((raw / "occupied.txt").exists())


if __name__ == "__main__":
    unittest.main()
