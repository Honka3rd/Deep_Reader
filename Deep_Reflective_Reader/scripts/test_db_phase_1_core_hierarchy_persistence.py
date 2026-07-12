#!/usr/bin/env python3
"""Isolated validation for Phase 1 core DB hierarchy persistence."""

from __future__ import annotations

import json
import sqlite3

from db.phase_1_core_schema import apply_phase_1_core_schema
from db.sqlite_core_document_store import RawSourceMetadata, SQLiteCoreDocumentStore
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_document() -> StructuredDocument:
    section = StructuredSection(
        section_id="legacy-section-1",
        section_index=0,
        title="Section One",
        level=2,
        content="Section content",
        char_start=0,
        char_end=15,
        parent_chapter_id="legacy-chapter-1",
        task_units=[
            TaskUnit(
                unit_id="legacy-unit-1",
                title="Task One",
                container_title="Section One",
                content="Task content",
                source_section_ids=["legacy-section-1"],
                is_fallback_generated=False,
                parent_section_id="legacy-section-1",
            )
        ],
    )
    return StructuredDocument(
        document_id="legacy-document-id",
        title="DB Core Slice",
        source_path="data/raw/db-core-slice.txt",
        language="en",
        raw_text="Section content",
        sections=[],
        chapters=[
            StructuredChapter(
                chapter_id="legacy-chapter-1",
                title="Chapter One",
                level=1,
                chapter_role="main_body",
                sections=[section],
            )
        ],
    )


def test_create_and_load_current_hierarchy() -> None:
    connection = sqlite3.connect(":memory:")
    apply_phase_1_core_schema(connection)
    store = SQLiteCoreDocumentStore(connection)

    result = store.create_document_with_initial_structure(
        _build_document(),
        namespace="tenant-a",
        document_name="db-core-slice",
        raw_source_metadata=RawSourceMetadata(
            source_location="data/raw/db-core-slice.txt",
            original_filename="db-core-slice.txt",
            mime_type="text/plain",
            file_size=15,
            checksum="checksum-reference",
        ),
        parser_name="isolated-test-parser",
        preparation_mode="base",
    )

    _assert(result.document_id == 1, "document should use DB-generated id")
    _assert(
        result.current_structure_version == 1,
        "initial parse should create structure_version 1",
    )

    lifecycle = store.load_document_lifecycle(result.document_id)
    _assert(
        lifecycle["current_structure_version"] == 1,
        "document lifecycle should expose current structure version",
    )
    _assert(lifecycle["namespace"] == "tenant-a", "document namespace is required")
    _assert(
        lifecycle["document_name"] == "db-core-slice",
        "document_name should be namespace-facing identity",
    )
    _assert(
        store.find_document_by_name(
            namespace="tenant-a",
            document_name="db-core-slice",
        )
        == result.document_id,
        "namespace/document_name lookup should resolve document",
    )

    loaded = store.load_current_structure(result.document_id)
    _assert(loaded.document_id == "1", "loaded document should expose DB id")
    _assert(loaded.raw_text == "", "raw text should not be stored in documents")
    _assert(
        loaded.source_path == "data/raw/db-core-slice.txt",
        "source path should come from raw_source_metadata",
    )
    _assert(loaded.chapters[0].chapter_id == "1", "chapter id should be DB-generated")
    _assert(
        loaded.chapters[0].sections[0].section_id == "1",
        "section id should be DB-generated",
    )
    _assert(
        loaded.chapters[0].sections[0].task_units[0].unit_id == "1",
        "task-unit id should be DB-generated",
    )
    _assert(
        loaded.chapters[0].sections[0].task_units[0].unit_id != "legacy-unit-1",
        "task-unit runtime id must not depend on Python-generated unit_id",
    )
    _assert(
        not loaded.sections,
        "DB hierarchy read must not create root sections as runtime truth",
    )
    _assert(
        not loaded.structure_nodes,
        "DB hierarchy read must not create structure_nodes",
    )

    raw_count = connection.execute(
        "SELECT COUNT(*) FROM raw_source_metadata WHERE document_id = ?",
        (result.document_id,),
    ).fetchone()[0]
    _assert(raw_count == 1, "raw-source metadata should be persisted")

    document_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(documents)").fetchall()
    }
    _assert("raw_text" not in document_columns, "documents must not store raw text")
    _assert("namespace" in document_columns, "documents must include namespace")
    _assert(
        "document_name" in document_columns,
        "documents must include document_name",
    )

    second_result = store.create_document_with_initial_structure(
        _build_document(),
        namespace="tenant-b",
        document_name="db-core-slice",
        raw_source_metadata=RawSourceMetadata(
            source_location="data/raw/db-core-slice-tenant-b.txt",
        ),
    )
    _assert(
        second_result.document_id == 2,
        "same document_name should be allowed in a different namespace",
    )

    parse_event = connection.execute(
        """
        SELECT event_type, previous_structure_version, new_structure_version
        FROM parse_events
        WHERE document_id = ?
        """,
        (result.document_id,),
    ).fetchone()
    _assert(parse_event[0] == "initial_parse", "parse event should be initial_parse")
    _assert(parse_event[1] is None, "initial parse should not have previous version")
    _assert(parse_event[2] == 1, "parse event should record new version 1")


def main() -> None:
    test_create_and_load_current_hierarchy()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": ["db_phase_1_core_hierarchy_persistence"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
