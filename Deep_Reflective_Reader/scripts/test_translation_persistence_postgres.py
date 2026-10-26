"""Real disposable PostgreSQL parity and post-write rollback for translations."""

from unittest.mock import patch
from uuid import uuid4

import psycopg

from app.translation_action import TranslationAction
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.reading_content import ReadingContentService
from scripts import test_summary_note_postgres as fixtures
from scripts import test_translation_persistence as contracts
from section_tasks.translation_service import TranslationService


class PostgresTranslations(contracts.PersistentTranslations):
    setUpClass = classmethod(fixtures.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(fixtures.PostgresNotes.drop_schema.__func__)

    def setUp(self):
        super().setUp()
        document = self.repo.load_document("book")
        self.repo = PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))
        self.repo.save_reparsed_document(document, "book")
        self.target["content_block_id"] = ReadingContentService(self.repo).read("book")["blocks"][0]["block_id"]
        self.action = TranslationAction(self.repo, self.profiles, TranslationService(self.provider))
        self.app.state.translation_action = self.action

    def test_post_write_rollback_keeps_translation_and_body(self):
        self.post()
        previous = self.repo.load_document("book")
        with psycopg.connect(self.scoped) as conn:
            conn.execute("CREATE FUNCTION reject_translation_snapshot() RETURNS trigger LANGUAGE plpgsql AS "
                         "$$ BEGIN RAISE EXCEPTION 'injected snapshot failure'; END $$")
            conn.execute("CREATE TRIGGER reject_translation_snapshot AFTER UPDATE ON documents "
                         "FOR EACH ROW EXECUTE FUNCTION reject_translation_snapshot()")
        try:
            self.assertEqual(self.post(target_language="de").status_code, 503)
            self.assertEqual(self.repo.load_document("book"), previous)
            with self.assertRaises(psycopg.Error):
                self.mutate()
            self.assertEqual(self.repo.load_document("book"), previous)
        finally:
            with psycopg.connect(self.scoped) as conn:
                conn.execute("DROP TRIGGER reject_translation_snapshot ON documents")
                conn.execute("DROP FUNCTION reject_translation_snapshot()")

    def test_same_text_does_not_touch_document_timestamp(self):
        self.post()
        state = ReadingContentService(self.repo).read("book")
        unit_id = state["blocks"][0]["task_unit_id"]
        with psycopg.connect(self.scoped) as conn:
            before = conn.execute("SELECT d.updated_at FROM documents d JOIN task_units u ON u.document_id=d.id "
                                  "WHERE u.id=%s", (int(unit_id),)).fetchone()[0]
        self.mutate(text=state["blocks"][0]["content"])
        with psycopg.connect(self.scoped) as conn:
            after = conn.execute("SELECT d.updated_at FROM documents d JOIN task_units u ON u.document_id=d.id "
                                 "WHERE u.id=%s", (int(unit_id),)).fetchone()[0]
        self.assertEqual(before, after)


if __name__ == "__main__":
    import unittest
    unittest.main()
