#!/usr/bin/env python3
"""Run deletion/cascade regression in a unique disposable PostgreSQL schema.

Requires DEEP_READER_TEST_POSTGRES_DSN. No existing document rows are modified.
"""

import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore


def main() -> None:
    dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    schema = "delete_test_" + uuid4().hex
    migrations = Path(__file__).resolve().parents[1] / "db" / "migrations"
    scoped_dsn = make_conninfo(dsn, options=f"-c search_path={schema}")
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(scoped_dsn) as connection:
                for migration in sorted(migrations.glob("*.sql")):
                    connection.execute(migration.read_text(encoding="utf-8"))
                connection.execute("""
                    INSERT INTO documents (id, namespace, document_name, current_structure_version, status)
                    VALUES (1, 'delete-test', 'Book', 1, 'active'),
                           (2, 'other-tenant', 'Book', 1, 'active'),
                           (3, 'delete-test', 'Other Book', 1, 'active');
                    INSERT INTO raw_source_metadata (document_id, source_location) VALUES (1, 'missing.txt');
                    INSERT INTO document_profile (document_id, profile_payload) VALUES (1, '{}');
                    INSERT INTO parse_events (document_id, event_type, new_structure_version) VALUES (1, 'initial_parse', 1);
                    INSERT INTO chapters (id, document_id, chapter_order, level) VALUES (1, 1, 0, 1);
                    INSERT INTO sections (id, document_id, chapter_id, section_order, level) VALUES (1, 1, 1, 0, 1);
                    INSERT INTO task_units (id, document_id, section_id, task_unit_order) VALUES (1, 1, 1, 0);
                    INSERT INTO content_blocks (document_id, task_unit_id, block_order, content, source_structure_version) VALUES (1, 1, 0, 'body', 1);
                    INSERT INTO artifacts (document_id, artifact_type, target_type, target_id, source_structure_version, payload) VALUES (1, 'analysis', 'document', 1, 1, '{}');
                    INSERT INTO structured_document_snapshots (document_id, source_structure_version, structured_document_payload) VALUES (1, 1, '{}');
                    INSERT INTO ocr_runs (id, document_id, source_file_sha256, ocr_engine, ocr_language, page_count, status) VALUES (1, 1, 'hash', 'test', 'eng', 1, 'completed');
                    INSERT INTO ocr_pages (ocr_run_id, page_index, text) VALUES (1, 0, 'body');
                """)
            store = PostgresStructuredDocumentStore(scoped_dsn, namespace="delete-test")
            repository = PostgresStructuredDocumentArtifactRepository(store)
            assert repository.delete_document("Book.txt") is True
            assert repository.delete_document("Book") is False
            assert [item.doc_name for item in repository.list_documents()] == ["Other Book"]
            with psycopg.connect(scoped_dsn) as connection:
                assert connection.execute("SELECT id FROM documents ORDER BY id").fetchall() == [(2,), (3,)]
                for table in (
                    "raw_source_metadata", "document_profile", "parse_events", "chapters",
                    "sections", "task_units", "content_blocks", "artifacts",
                    "structured_document_snapshots", "ocr_runs", "ocr_pages",
                ):
                    count = connection.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
                    assert count == 0, f"orphan rows in {table}"
            print("OK: PostgreSQL delete, cascade, namespace isolation, and missing retry passed")
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__ == "__main__":
    main()
