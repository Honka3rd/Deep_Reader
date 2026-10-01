#!/usr/bin/env python3
"""Live PostgreSQL regression for manual reparse hierarchy replacement."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.postgres_structured_document_artifact_repository import (  # noqa: E402
    PostgresStructuredDocumentArtifactRepository,
)
from db.postgres_structured_document_store import (  # noqa: E402
    PostgresStructuredDocumentStore,
)
from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _dsn() -> str | None:
    return (
        os.environ.get("DEEP_READER_TEST_POSTGRES_DSN")
        or os.environ.get("DEEP_READER_POSTGRES_DSN")
    )


def _load_psycopg() -> Any | None:
    try:
        import psycopg
    except ModuleNotFoundError:
        return None
    return psycopg


def _document(
    *,
    doc_name: str,
    title: str,
    chapter_ids: list[str],
) -> StructuredDocument:
    chapters: list[StructuredChapter] = []
    cursor = 0
    for index, chapter_id in enumerate(chapter_ids):
        content = f"{title} chapter {index} body"
        section = StructuredSection(
            section_id=f"{chapter_id}-section",
            section_index=index,
            title=f"{title} Section {index}",
            level=1,
            content=content,
            char_start=cursor,
            char_end=cursor + len(content),
            parent_chapter_id=chapter_id,
        )
        cursor += len(content) + 1
        chapters.append(
            StructuredChapter(
                chapter_id=chapter_id,
                title=f"{title} Chapter {index}",
                level=1,
                chapter_role=None,
                sections=[section],
            )
        )
    return StructuredDocument(
        document_id=doc_name,
        title=title,
        source_path=doc_name,
        language="en",
        raw_text="\n".join(
            section.content
            for chapter in chapters
            for section in chapter.sections
        ),
        chapters=chapters,
        parse_provenance={
            "requested_parser_mode": "manual_structure",
            "effective_parser_mode": "manual_structure_projection",
        },
    )


def _fetch_document_state(psycopg: Any, dsn: str, doc_name: str) -> dict[str, Any]:
    with psycopg.connect(dsn) as connection:
        document_row = connection.execute(
            """
            SELECT id, current_structure_version
            FROM documents
            WHERE namespace = 'default'
              AND document_name = %s
            """,
            (doc_name,),
        ).fetchone()
        _assert(document_row is not None, "expected document row to exist")
        document_id = int(document_row[0])
        chapter_rows = connection.execute(
            """
            SELECT chapter_order, reference_chapter_id, title
            FROM chapters
            WHERE document_id = %s
            ORDER BY chapter_order
            """,
            (document_id,),
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT event_type, previous_structure_version, new_structure_version
            FROM parse_events
            WHERE document_id = %s
            ORDER BY id
            """,
            (document_id,),
        ).fetchall()
    return {
        "document_id": document_id,
        "current_structure_version": int(document_row[1]),
        "chapters": chapter_rows,
        "events": event_rows,
    }


def _cleanup(psycopg: Any, dsn: str, doc_name: str) -> None:
    with psycopg.connect(dsn) as connection:
        connection.execute(
            """
            DELETE FROM documents
            WHERE namespace = 'default'
              AND document_name = %s
            """,
            (doc_name,),
        )
        connection.commit()


class _FailingReplacementStore(PostgresStructuredDocumentStore):
    def _replace_current_hierarchy(  # type: ignore[override]
        self,
        *,
        connection: Any,
        document: StructuredDocument,
        document_id: int,
        Jsonb: Any,
    ) -> None:
        _ = (connection, document, document_id, Jsonb)
        raise RuntimeError("forced replacement failure")


def test_duplicate_chapter_order_reparse_replaces_current_hierarchy() -> None:
    dsn = _dsn()
    psycopg = _load_psycopg()
    if not dsn or psycopg is None:
        print("SKIP: PostgreSQL DSN or psycopg is unavailable")
        return

    doc_name = f"codex_manual_reparse_duplicate_{int(time.time() * 1000)}"
    store = PostgresStructuredDocumentStore(dsn)
    repository = PostgresStructuredDocumentArtifactRepository(store)
    try:
        repository.save_reparsed_document(
            _document(
                doc_name=doc_name,
                title="Initial",
                chapter_ids=["initial-chapter-0", "initial-chapter-1"],
            ),
            doc_name=doc_name,
        )
        repository.save_reparsed_document(
            _document(
                doc_name=doc_name,
                title="Manual",
                chapter_ids=["manual-chapter-0"],
            ),
            doc_name=doc_name,
        )

        loaded = repository.load_document(doc_name)
        state = _fetch_document_state(psycopg, dsn, doc_name)
        _assert(len(loaded.chapters) == 1, "replacement should load one chapter")
        _assert(
            loaded.chapters[0].title == "Manual Chapter 0",
            "replacement should expose manual chapter",
        )
        _assert(
            state["current_structure_version"] == 2,
            "successful replacement should advance structure version",
        )
        _assert(
            [row[0] for row in state["chapters"]] == [0],
            "old chapter rows must be replaced before inserting new order 0",
        )
        _assert(
            [row[1] for row in state["chapters"]] == ["manual-chapter-0"],
            "current hierarchy should contain only manual replacement chapter",
        )
        _assert(
            [row[0] for row in state["events"]] == ["initial_parse", "hard_reparse"],
            "replacement should record initial_parse then hard_reparse",
        )
    finally:
        _cleanup(psycopg, dsn, doc_name)


def test_failed_replacement_rolls_back_current_hierarchy() -> None:
    dsn = _dsn()
    psycopg = _load_psycopg()
    if not dsn or psycopg is None:
        print("SKIP: PostgreSQL DSN or psycopg is unavailable")
        return

    doc_name = f"codex_manual_reparse_rollback_{int(time.time() * 1000)}"
    store = PostgresStructuredDocumentStore(dsn)
    repository = PostgresStructuredDocumentArtifactRepository(store)
    failing_repository = PostgresStructuredDocumentArtifactRepository(
        _FailingReplacementStore(dsn)
    )
    try:
        repository.save_reparsed_document(
            _document(
                doc_name=doc_name,
                title="Initial",
                chapter_ids=["initial-chapter-0", "initial-chapter-1"],
            ),
            doc_name=doc_name,
        )
        try:
            failing_repository.save_reparsed_document(
                _document(
                    doc_name=doc_name,
                    title="Manual",
                    chapter_ids=["manual-chapter-0"],
                ),
                doc_name=doc_name,
            )
        except RuntimeError as error:
            _assert(
                "forced replacement failure" in str(error),
                "expected forced replacement failure",
            )
        else:
            raise AssertionError("forced replacement failure should propagate")

        state = _fetch_document_state(psycopg, dsn, doc_name)
        _assert(
            state["current_structure_version"] == 1,
            "failed replacement must roll back structure version",
        )
        _assert(
            [row[0] for row in state["chapters"]] == [0, 1],
            "failed replacement must preserve original chapter rows",
        )
        _assert(
            [row[1] for row in state["chapters"]]
            == ["initial-chapter-0", "initial-chapter-1"],
            "failed replacement must not leave manual chapter rows",
        )
        _assert(
            [row[0] for row in state["events"]] == ["initial_parse"],
            "failed replacement must not leave a hard_reparse event",
        )
    finally:
        _cleanup(psycopg, dsn, doc_name)


def main() -> None:
    test_duplicate_chapter_order_reparse_replaces_current_hierarchy()
    test_failed_replacement_rolls_back_current_hierarchy()
    print("postgres manual reparse duplicate chapter_order regression passed")


if __name__ == "__main__":
    main()
