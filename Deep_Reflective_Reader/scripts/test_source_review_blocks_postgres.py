"""Same source-review contract on PostgreSQL, including post-update rollback."""

from uuid import uuid4
import psycopg

from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from scripts import test_source_review_blocks as review_tests
from scripts import test_summary_note_postgres as note_tests


class PostgresSourceReviewBlocks(review_tests.SourceReviewBlockTests):
    setUpClass = classmethod(note_tests.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(note_tests.PostgresNotes.drop_schema.__func__)

    def repository(self):
        return PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))

    def remove_persisted_mapping(self):
        unit = self.service.read("book")["blocks"][0]["task_unit_id"]
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE documents SET metadata_payload = metadata_payload #- "
                "'{parse_provenance,source_review_mapping}' WHERE id = "
                "(SELECT document_id FROM task_units WHERE id = %s)", (int(unit),))

    def replace_persisted_mapping(self, mapping):
        from psycopg.types.json import Jsonb
        unit = self.service.read("book")["blocks"][0]["task_unit_id"]
        with psycopg.connect(self.scoped) as connection:
            result = connection.execute(
                "UPDATE documents SET metadata_payload = jsonb_set(metadata_payload, "
                "'{parse_provenance,source_review_mapping}', %s) WHERE id = "
                "(SELECT document_id FROM task_units WHERE id = %s)", (Jsonb(mapping), int(unit)))
            self.assertEqual(result.rowcount, 1)

    def storage_snapshot(self):
        unit = self.service.read("book")["blocks"][0]["task_unit_id"]
        with psycopg.connect(self.scoped) as connection:
            return connection.execute("SELECT d.xmin::text, d.updated_at, d.metadata_payload "
                "FROM documents d JOIN task_units t ON t.document_id=d.id WHERE t.id=%s",
                (int(unit),)).fetchone()

    def test_physical_note_rows_follow_exact_block_ownership(self):
        from psycopg.types.json import Jsonb
        blocks = self.service.read("book")["blocks"][:2]
        with psycopg.connect(self.scoped) as connection:
            document_id = connection.execute("SELECT document_id FROM task_units WHERE id=%s",
                                             (int(blocks[0]["task_unit_id"]),)).fetchone()[0]
            for index, block in enumerate(blocks):
                row_id = connection.execute(
                    "INSERT INTO content_blocks(document_id,task_unit_id,block_order,content,source_structure_version,metadata_payload) "
                    "VALUES (%s,%s,%s,%s,1,%s) RETURNING id",
                    (document_id, int(block["task_unit_id"]), index, block["content"], Jsonb({"block_id": block["block_id"]}))).fetchone()[0]
                connection.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,source_structure_version,payload) "
                                   "VALUES (%s,'note','content_block',%s,1,%s)", (document_id, row_id, Jsonb({"text": "Physical note"})))
        self.change(discard_interactions=False)
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM artifacts WHERE document_id=%s", (document_id,)).fetchone()[0], 2)
        self.change("delete")
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM artifacts WHERE document_id=%s", (document_id,)).fetchone()[0], 1)

    def test_failed_delete_keeps_note_region_content(self):
        service, target, before = self.add_note()
        initial = self.service.read("book")
        persisted = self.storage_snapshot()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("CREATE FUNCTION reject_review_write() RETURNS trigger LANGUAGE plpgsql AS "
                               "$$ BEGIN RAISE EXCEPTION 'injected write failure'; END $$")
            connection.execute("CREATE TRIGGER reject_review_write BEFORE UPDATE ON task_units "
                               "FOR EACH ROW EXECUTE FUNCTION reject_review_write()")
        try:
            with self.assertRaises(psycopg.Error):
                self.change("delete", region_id="r1")
            self.assertEqual(self.service.read("book"), initial)
            self.assertEqual(service.read("book", **target)["note"], before["note"])
            self.assertEqual(self.storage_snapshot(), persisted)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("DROP TRIGGER reject_review_write ON task_units")
                connection.execute("DROP FUNCTION reject_review_write()")


if __name__ == "__main__":
    import unittest
    unittest.main()
