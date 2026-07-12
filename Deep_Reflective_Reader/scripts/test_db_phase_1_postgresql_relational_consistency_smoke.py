#!/usr/bin/env python3
"""Optional PostgreSQL smoke test for Phase 1 same-document FK consistency.

Set DEEP_READER_TEST_POSTGRES_DSN to run against a disposable PostgreSQL database.
The script creates and drops its own schema, then applies the Phase 1 migration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "001_phase_1_core_hierarchy.sql"
)
SCHEMA_NAME = "deep_reader_phase_1_fk_smoke"


SMOKE_SQL = f"""
DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE;
CREATE SCHEMA {SCHEMA_NAME};
SET search_path TO {SCHEMA_NAME};

{{migration_sql}}

-- PostgreSQL catalog inspection for Phase 1 index policy.
DO $$
DECLARE
    index_name TEXT;
BEGIN
    FOREACH index_name IN ARRAY ARRAY[
        'idx_raw_source_metadata_document_id',
        'idx_documents_namespace_document_name',
        'idx_document_profile_document_id',
        'idx_document_profile_document_id_source_structure_version',
        'idx_sections_chapter_id',
        'idx_chapters_document_id_order',
        'idx_task_units_section_id',
        'idx_artifacts_created_at',
        'idx_structured_document_snapshots_document_id_version'
    ] LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = '{SCHEMA_NAME}'
              AND indexname = index_name
        ) THEN
            RAISE EXCEPTION 'unexpected redundant/premature index exists: %', index_name;
        END IF;
    END LOOP;
END $$;

DO $$
DECLARE
    index_name TEXT;
BEGIN
    FOREACH index_name IN ARRAY ARRAY[
        'idx_parse_events_document_id_occurred_at',
        'idx_parse_events_document_id_new_structure_version',
        'idx_sections_document_id_chapter_id_order',
        'idx_task_units_document_id_section_id_order',
        'idx_content_blocks_document_id_task_unit_id_order',
        'idx_content_blocks_document_id_source_structure_version',
        'idx_artifacts_document_id_artifact_type',
        'idx_artifacts_document_id_target',
        'idx_artifacts_document_id_source_structure_version'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = '{SCHEMA_NAME}'
              AND indexname = index_name
        ) THEN
            RAISE EXCEPTION 'expected retained query index is missing: %', index_name;
        END IF;
    END LOOP;
END $$;

DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    FOREACH constraint_name IN ARRAY ARRAY[
        'uq_documents_namespace_document_name',
        'uq_chapters_document_order',
        'raw_source_metadata_document_id_key',
        'document_profile_document_id_key',
        'uq_structured_document_snapshots_document_version',
        'uq_sections_chapter_order',
        'uq_task_units_section_order'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE connamespace = '{SCHEMA_NAME}'::regnamespace
              AND conname = constraint_name
              AND contype = 'u'
        ) THEN
            RAISE EXCEPTION 'expected UNIQUE-backed index constraint is missing: %', constraint_name;
        END IF;
    END LOOP;
END $$;

INSERT INTO documents (id, namespace, document_name, current_structure_version)
VALUES
    (1, 'tenant-a', 'doc-a', 1),
    (2, 'tenant-a', 'doc-b', 1);

-- Application-managed document timestamp updates are explicit.
UPDATE documents
SET status = 'active', updated_at = NOW()
WHERE id = 1;

INSERT INTO chapters (id, document_id, chapter_order, level, title)
VALUES
    (10, 1, 0, 1, 'Chapter A'),
    (20, 2, 0, 1, 'Chapter B');

-- Numeric boundary values: zero file size is accepted.
INSERT INTO raw_source_metadata (document_id, source_location, file_size)
VALUES (1, 'data/raw/doc-a.txt', 0);

-- Negative file size must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO raw_source_metadata (document_id, source_location, file_size)
        VALUES (2, 'data/raw/doc-b.txt', -1);
        RAISE EXCEPTION 'expected negative file_size to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Negative chapter order must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO chapters (id, document_id, chapter_order, level, title)
        VALUES (30, 1, -1, 1, 'Invalid chapter order');
        RAISE EXCEPTION 'expected negative chapter_order to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Valid initial parse event succeeds.
INSERT INTO parse_events (
    id,
    document_id,
    event_type,
    previous_structure_version,
    new_structure_version
)
VALUES (500, 1, 'initial_parse', NULL, 1);

INSERT INTO parse_events (
    id,
    document_id,
    event_type,
    previous_structure_version,
    new_structure_version
)
VALUES (501, 2, 'initial_parse', NULL, 1);

-- Initial parse with previous version must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version
        )
        VALUES (1, 'initial_parse', 1, 1);
        RAISE EXCEPTION 'expected initial_parse with previous version to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Initial parse with new version other than 1 must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version
        )
        VALUES (1, 'initial_parse', NULL, 2);
        RAISE EXCEPTION 'expected initial_parse with new version other than 1 to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Initial parse with reparse-only metadata must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version,
            reparse_reason
        )
        VALUES (1, 'initial_parse', NULL, 1, 'not allowed');
        RAISE EXCEPTION 'expected initial_parse with reparse_reason to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Valid hard reparse from N to N+1 succeeds.
INSERT INTO parse_events (
    id,
    document_id,
    event_type,
    previous_structure_version,
    new_structure_version,
    invalidated_artifact_count,
    invalidated_content_block_count,
    reparse_reason
)
VALUES (600, 1, 'hard_reparse', 1, 2, 0, 0, 'manual recovery');

-- Hard reparse with null previous version must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version
        )
        VALUES (1, 'hard_reparse', NULL, 2);
        RAISE EXCEPTION 'expected hard_reparse with null previous version to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Hard reparse that skips a version must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version
        )
        VALUES (1, 'hard_reparse', 1, 3);
        RAISE EXCEPTION 'expected hard_reparse version skip to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Negative invalidation counts must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO parse_events (
            document_id,
            event_type,
            previous_structure_version,
            new_structure_version,
            invalidated_artifact_count
        )
        VALUES (1, 'hard_reparse', 2, 3, -1);
        RAISE EXCEPTION 'expected negative invalidated_artifact_count to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Valid same-document section insert succeeds.
INSERT INTO sections (id, document_id, chapter_id, section_order, level, content)
VALUES (100, 1, 10, 0, 2, 'Section A');


-- Valid positive char span range succeeds.
INSERT INTO sections (id, document_id, chapter_id, section_order, level, content, char_start, char_end)
VALUES (110, 1, 10, 1, 2, 'Section A positive span', 5, 10);

-- Negative section order must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO sections (id, document_id, chapter_id, section_order, level, content)
        VALUES (111, 1, 10, -1, 2, 'Invalid section order');
        RAISE EXCEPTION 'expected negative section_order to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Negative section char span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO sections (id, document_id, chapter_id, section_order, level, content, char_start, char_end)
        VALUES (112, 1, 10, 2, 2, 'Invalid section char span', -1, 2);
        RAISE EXCEPTION 'expected negative section char span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Reversed section char span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO sections (id, document_id, chapter_id, section_order, level, content, char_start, char_end)
        VALUES (113, 1, 10, 3, 2, 'Invalid reversed section span', 5, 4);
        RAISE EXCEPTION 'expected reversed section char span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;
-- Cross-document section parent must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO sections (id, document_id, chapter_id, section_order, level, content)
        VALUES (101, 1, 20, 1, 2, 'Invalid section');
        RAISE EXCEPTION 'expected cross-document section insert to fail';
    EXCEPTION WHEN foreign_key_violation THEN
        NULL;
    END;
END $$;

INSERT INTO sections (id, document_id, chapter_id, section_order, level, content)
VALUES (200, 2, 20, 0, 2, 'Section B');

-- Existing section ordering constraint remains valid.
DO $$
BEGIN
    BEGIN
        INSERT INTO sections (id, document_id, chapter_id, section_order, level, content)
        VALUES (102, 1, 10, 0, 2, 'Duplicate section order');
        RAISE EXCEPTION 'expected duplicate section order insert to fail';
    EXCEPTION WHEN unique_violation THEN
        NULL;
    END;
END $$;

-- Valid same-document task unit insert succeeds.
INSERT INTO task_units (id, document_id, section_id, task_unit_order, content_payload)
VALUES (1000, 1, 100, 0, 'Task A');


-- Valid positive task-unit order succeeds.
INSERT INTO task_units (id, document_id, section_id, task_unit_order, content_payload)
VALUES (1010, 1, 100, 1, 'Task A positive order');

-- Negative task-unit order must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO task_units (id, document_id, section_id, task_unit_order, content_payload)
        VALUES (1011, 1, 100, -1, 'Invalid task order');
        RAISE EXCEPTION 'expected negative task_unit_order to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;
-- Cross-document task-unit parent must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO task_units (id, document_id, section_id, task_unit_order, content_payload)
        VALUES (1001, 1, 200, 1, 'Invalid task');
        RAISE EXCEPTION 'expected cross-document task-unit insert to fail';
    EXCEPTION WHEN foreign_key_violation THEN
        NULL;
    END;
END $$;

-- Existing task-unit ordering constraint remains valid.
DO $$
BEGIN
    BEGIN
        INSERT INTO task_units (id, document_id, section_id, task_unit_order, content_payload)
        VALUES (1002, 1, 100, 0, 'Duplicate task order');
        RAISE EXCEPTION 'expected duplicate task-unit order insert to fail';
    EXCEPTION WHEN unique_violation THEN
        NULL;
    END;
END $$;

-- Valid same-document content block insert succeeds.
INSERT INTO content_blocks (
    id,
    document_id,
    task_unit_id,
    block_order,
    content,
    source_structure_version
)
VALUES (10000, 1, 1000, 0, 'Block A', 1);


-- Valid positive content-block order and one-sided spans succeed.
INSERT INTO content_blocks (
    id,
    document_id,
    task_unit_id,
    block_order,
    content,
    source_structure_version,
    quote_span_start,
    quote_span_end
)
VALUES
    (10010, 1, 1000, 1, 'Block positive', 1, 0, 5),
    (10011, 1, 1000, 2, 'Block one-sided start', 1, 3, NULL),
    (10012, 1, 1000, 3, 'Block one-sided end', 1, NULL, 7);

-- Negative content-block order must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO content_blocks (
            id,
            document_id,
            task_unit_id,
            block_order,
            content,
            source_structure_version
        )
        VALUES (10013, 1, 1000, -1, 'Invalid block order', 1);
        RAISE EXCEPTION 'expected negative content block order to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Negative content-block quote span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO content_blocks (
            id,
            document_id,
            task_unit_id,
            block_order,
            content,
            source_structure_version,
            quote_span_start
        )
        VALUES (10014, 1, 1000, 4, 'Invalid block span', 1, -1);
        RAISE EXCEPTION 'expected negative content block quote span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Reversed content-block quote span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO content_blocks (
            id,
            document_id,
            task_unit_id,
            block_order,
            content,
            source_structure_version,
            quote_span_start,
            quote_span_end
        )
        VALUES (10015, 1, 1000, 4, 'Invalid reversed block span', 1, 9, 8);
        RAISE EXCEPTION 'expected reversed content block quote span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Valid artifact positive and one-sided quote spans succeed.
INSERT INTO artifacts (
    id,
    document_id,
    artifact_type,
    target_type,
    target_id,
    source_structure_version,
    quote_span_start,
    quote_span_end,
    payload
)
VALUES
    (20000, 1, 'note', 'task_unit', 1000, 1, 0, 4, '{{{{}}}}'::jsonb),
    (20001, 1, 'note', 'task_unit', 1000, 1, 2, NULL, '{{{{}}}}'::jsonb),
    (20002, 1, 'note', 'task_unit', 1000, 1, NULL, 5, '{{{{}}}}'::jsonb);

-- Artifact updated_at remains nullable until first explicit mutation.
DO $$
DECLARE
    nullable_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO nullable_count
    FROM artifacts
    WHERE id IN (20000, 20001, 20002)
      AND updated_at IS NULL;

    IF nullable_count <> 3 THEN
        RAISE EXCEPTION 'expected inserted artifacts to have nullable updated_at';
    END IF;
END $$;

UPDATE artifacts
SET metadata_payload = '{{{{}}}}'::jsonb, updated_at = NOW()
WHERE id = 20000;

DO $$
DECLARE
    mutated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO mutated_count
    FROM artifacts
    WHERE id = 20000
      AND updated_at IS NOT NULL;

    IF mutated_count <> 1 THEN
        RAISE EXCEPTION 'expected application-managed artifact mutation to set updated_at';
    END IF;
END $$;

-- Negative artifact quote span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO artifacts (
            document_id,
            artifact_type,
            target_type,
            target_id,
            source_structure_version,
            quote_span_start,
            payload
        )
        VALUES (1, 'note', 'task_unit', 1000, 1, -1, '{{{{}}}}'::jsonb);
        RAISE EXCEPTION 'expected negative artifact quote span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;

-- Reversed artifact quote span must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO artifacts (
            document_id,
            artifact_type,
            target_type,
            target_id,
            source_structure_version,
            quote_span_start,
            quote_span_end,
            payload
        )
        VALUES (1, 'note', 'task_unit', 1000, 1, 4, 3, '{{{{}}}}'::jsonb);
        RAISE EXCEPTION 'expected reversed artifact quote span to fail';
    EXCEPTION WHEN check_violation THEN
        NULL;
    END;
END $$;
-- Cross-document content-block parent must fail.
DO $$
BEGIN
    BEGIN
        INSERT INTO content_blocks (
            id,
            document_id,
            task_unit_id,
            block_order,
            content,
            source_structure_version
        )
        VALUES (10001, 2, 1000, 0, 'Invalid block', 1);
        RAISE EXCEPTION 'expected cross-document content-block insert to fail';
    EXCEPTION WHEN foreign_key_violation THEN
        NULL;
    END;
END $$;

-- Document delete still cascades parse events.
DELETE FROM documents WHERE id = 2;
DO $$
DECLARE
    remaining_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO remaining_count
    FROM parse_events
    WHERE document_id = 2;

    IF remaining_count <> 0 THEN
        RAISE EXCEPTION 'expected document delete to cascade parse events';
    END IF;
END $$;

DROP SCHEMA {SCHEMA_NAME} CASCADE;
"""


def main() -> None:
    dsn = os.environ.get("DEEP_READER_TEST_POSTGRES_DSN")
    if not dsn:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "DEEP_READER_TEST_POSTGRES_DSN is not set",
                    "tests": ["db_phase_1_postgresql_relational_consistency_smoke"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    psql_path = shutil.which("psql")
    if psql_path is None:
        raise RuntimeError("psql is required when DEEP_READER_TEST_POSTGRES_DSN is set")

    sql = SMOKE_SQL.format(
        migration_sql=MIGRATION_PATH.read_text(encoding="utf-8"),
    )
    subprocess.run(
        [psql_path, dsn, "--set", "ON_ERROR_STOP=1"],
        input=sql,
        text=True,
        check=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": ["db_phase_1_postgresql_relational_consistency_smoke"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - script-style failure reporting.
        print(str(exc), file=sys.stderr)
        raise
