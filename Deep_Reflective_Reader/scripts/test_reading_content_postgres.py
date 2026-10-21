"""Reading mutation contracts on a disposable PostgreSQL schema."""

from uuid import uuid4
import psycopg

from scripts.test_reading_content import ReadingContentTests
from scripts.test_summary_note_postgres import PostgresNotes
from scripts.test_reading_interaction_artifact_store import _document
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.reading_content import ReadingContentService


class PostgresReadingContent(ReadingContentTests):
    setUpClass = classmethod(PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(PostgresNotes.drop_schema.__func__)

    def setUp(self):
        store = PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex)
        self.repo = PostgresStructuredDocumentArtifactRepository(store)
        self.repo.save_reparsed_document(_document(), "book")
        self.service = ReadingContentService(self.repo)
        self.initial = self.service.read("book")
        self.block = self.initial["blocks"][0]

    def test_failed_write_preserves_source_and_revision(self):
        # Fail after the document update, during hierarchy invalidation.
        with psycopg.connect(self.scoped) as conn:
            conn.execute("CREATE OR REPLACE FUNCTION fail_content_update() RETURNS trigger "
                         "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected failure'; END $$")
            conn.execute("CREATE TRIGGER reject_content_update BEFORE UPDATE ON task_units "
                         "FOR EACH ROW EXECUTE FUNCTION fail_content_update()")
        try:
            with self.assertRaises(psycopg.Error):
                self.change()
            self.assertEqual(self.service.read("book"), self.initial)
        finally:
            with psycopg.connect(self.scoped) as conn:
                conn.execute("DROP TRIGGER reject_content_update ON task_units")


if __name__ == "__main__":
    import unittest
    unittest.main()
