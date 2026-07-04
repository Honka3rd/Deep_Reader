from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from typing import Any

from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_unit_model import TaskUnit


@dataclass(frozen=True)
class RawSourceMetadata:
    """Metadata-only reference to file-backed or object-backed raw source bytes."""

    source_location: str | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class InitialStructureWriteResult:
    """Result for an accepted initial hierarchy write."""

    document_id: int
    current_structure_version: int
    parse_event_id: int


class SQLiteCoreDocumentStore:
    """SQLite-backed isolated store for Phase 1 core hierarchy validation.

    This adapter is intentionally detached from production runtime selection.
    It persists already accepted `StructuredDocument` hierarchy and reads it
    back through DB-generated identities.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def create_document_with_initial_structure(
        self,
        document: StructuredDocument,
        *,
        raw_source_metadata: RawSourceMetadata | None = None,
        parser_name: str | None = None,
        preparation_mode: str | None = None,
    ) -> InitialStructureWriteResult:
        """Persist one accepted current hierarchy with structure version 1."""
        self._validate_accepted_hierarchy(document)
        with self._connection:
            document_cursor = self._connection.execute(
                """
                INSERT INTO documents (
                    reference_document_id,
                    title,
                    source_path,
                    language,
                    raw_text,
                    current_structure_version
                )
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    document.document_id,
                    document.title,
                    document.source_path,
                    document.language,
                    document.raw_text,
                ),
            )
            db_document_id = int(document_cursor.lastrowid)
            if raw_source_metadata is not None:
                self._connection.execute(
                    """
                    INSERT INTO raw_source_metadata (
                        document_id,
                        source_location,
                        original_filename,
                        mime_type,
                        file_size,
                        checksum
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_document_id,
                        raw_source_metadata.source_location,
                        raw_source_metadata.original_filename,
                        raw_source_metadata.mime_type,
                        raw_source_metadata.file_size,
                        raw_source_metadata.checksum,
                    ),
                )

            for chapter_order, chapter in enumerate(document.chapters):
                chapter_cursor = self._connection.execute(
                    """
                    INSERT INTO chapters (
                        document_id,
                        chapter_order,
                        reference_chapter_id,
                        title,
                        level,
                        chapter_role,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_document_id,
                        chapter_order,
                        chapter.chapter_id,
                        chapter.title,
                        chapter.level,
                        chapter.chapter_role,
                        json.dumps(chapter.metadata, ensure_ascii=False),
                    ),
                )
                db_chapter_id = int(chapter_cursor.lastrowid)
                for section_order, section in enumerate(chapter.sections):
                    section_cursor = self._connection.execute(
                        """
                        INSERT INTO sections (
                            document_id,
                            chapter_id,
                            section_order,
                            reference_section_id,
                            title,
                            level,
                            content,
                            char_start,
                            char_end,
                            container_title,
                            section_role,
                            section_kind,
                            is_implicit_section
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            db_document_id,
                            db_chapter_id,
                            section_order,
                            section.section_id,
                            section.title,
                            section.level,
                            section.content,
                            section.char_start,
                            section.char_end,
                            section.container_title,
                            (
                                None
                                if section.section_role is None
                                else section.section_role.value
                            ),
                            section.section_kind,
                            1 if section.is_implicit_section else 0,
                        ),
                    )
                    db_section_id = int(section_cursor.lastrowid)
                    for task_unit_order, task_unit in enumerate(section.task_units):
                        self._connection.execute(
                            """
                            INSERT INTO task_units (
                                document_id,
                                section_id,
                                task_unit_order,
                                reference_unit_id,
                                title,
                                container_title,
                                content,
                                source_section_ids_json,
                                is_fallback_generated
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                db_document_id,
                                db_section_id,
                                task_unit_order,
                                task_unit.unit_id,
                                task_unit.title,
                                task_unit.container_title,
                                task_unit.content,
                                json.dumps(
                                    task_unit.source_section_ids,
                                    ensure_ascii=False,
                                ),
                                1 if task_unit.is_fallback_generated else 0,
                            ),
                        )

            parse_event_cursor = self._connection.execute(
                """
                INSERT INTO parse_events (
                    document_id,
                    event_type,
                    previous_structure_version,
                    new_structure_version,
                    parser_name,
                    preparation_mode,
                    metadata_json
                )
                VALUES (?, 'initial_parse', NULL, 1, ?, ?, ?)
                """,
                (
                    db_document_id,
                    parser_name,
                    preparation_mode,
                    json.dumps(
                        {"reference_document_id": document.document_id},
                        ensure_ascii=False,
                    ),
                ),
            )
        return InitialStructureWriteResult(
            document_id=db_document_id,
            current_structure_version=1,
            parse_event_id=int(parse_event_cursor.lastrowid),
        )

    def load_current_structure(self, document_id: int) -> StructuredDocument:
        """Read the current hierarchy using DB-generated identities."""
        document_row = self._connection.execute(
            """
            SELECT id, title, source_path, language, raw_text, current_structure_version
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if document_row is None:
            raise KeyError(f"document not found: {document_id}")

        chapter_rows = self._connection.execute(
            """
            SELECT id, title, level, chapter_role, metadata_json
            FROM chapters
            WHERE document_id = ?
            ORDER BY chapter_order ASC
            """,
            (document_id,),
        ).fetchall()
        if not chapter_rows:
            raise ValueError(
                f"document has no current hierarchy chapters: {document_id}"
            )

        chapters: list[StructuredChapter] = []
        for chapter_row in chapter_rows:
            db_chapter_id = int(chapter_row["id"])
            section_rows = self._connection.execute(
                """
                SELECT
                    id,
                    title,
                    level,
                    content,
                    char_start,
                    char_end,
                    container_title,
                    section_role,
                    section_kind,
                    is_implicit_section
                FROM sections
                WHERE document_id = ? AND chapter_id = ?
                ORDER BY section_order ASC
                """,
                (document_id, db_chapter_id),
            ).fetchall()
            sections: list[StructuredSection] = []
            for section_index, section_row in enumerate(section_rows):
                db_section_id = int(section_row["id"])
                task_unit_rows = self._connection.execute(
                    """
                    SELECT
                        id,
                        title,
                        container_title,
                        content,
                        is_fallback_generated
                    FROM task_units
                    WHERE document_id = ? AND section_id = ?
                    ORDER BY task_unit_order ASC
                    """,
                    (document_id, db_section_id),
                ).fetchall()
                task_units = [
                    TaskUnit(
                        unit_id=str(task_unit_row["id"]),
                        title=task_unit_row["title"],
                        container_title=task_unit_row["container_title"],
                        content=task_unit_row["content"],
                        source_section_ids=[str(db_section_id)],
                        is_fallback_generated=bool(
                            task_unit_row["is_fallback_generated"]
                        ),
                        parent_section_id=str(db_section_id),
                    )
                    for task_unit_row in task_unit_rows
                ]
                sections.append(
                    StructuredSection.from_dict(
                        {
                            "section_id": str(db_section_id),
                            "section_index": section_index,
                            "title": section_row["title"],
                            "level": section_row["level"],
                            "content": section_row["content"],
                            "char_start": section_row["char_start"],
                            "char_end": section_row["char_end"],
                            "container_title": section_row["container_title"],
                            "section_role": section_row["section_role"],
                            "parent_chapter_id": str(db_chapter_id),
                            "section_kind": section_row["section_kind"],
                            "is_implicit_section": bool(
                                section_row["is_implicit_section"]
                            ),
                            "task_units": [
                                task_unit.to_dict() for task_unit in task_units
                            ],
                        }
                    )
                )
            chapters.append(
                StructuredChapter(
                    chapter_id=str(db_chapter_id),
                    title=chapter_row["title"],
                    level=int(chapter_row["level"]),
                    chapter_role=chapter_row["chapter_role"],
                    sections=sections,
                    metadata=self._decode_json_object(chapter_row["metadata_json"]),
                )
            )

        return StructuredDocument(
            document_id=str(document_row["id"]),
            title=document_row["title"],
            source_path=document_row["source_path"],
            language=document_row["language"],
            raw_text=document_row["raw_text"],
            chapters=chapters,
            sections=[],
            structure_nodes=[],
        )

    def load_document_lifecycle(self, document_id: int) -> dict[str, Any]:
        """Return lightweight document lifecycle metadata for validation paths."""
        row = self._connection.execute(
            """
            SELECT id, title, current_structure_version
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {document_id}")
        return {
            "document_id": int(row["id"]),
            "title": row["title"],
            "current_structure_version": int(row["current_structure_version"]),
        }

    @staticmethod
    def _decode_json_object(value: str | None) -> dict[str, Any]:
        if value is None:
            return {}
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            return {}
        return decoded

    @staticmethod
    def _validate_accepted_hierarchy(document: StructuredDocument) -> None:
        if not document.chapters:
            raise ValueError("accepted hierarchy must include chapters")
        for chapter in document.chapters:
            if not chapter.sections:
                raise ValueError(
                    f"accepted chapter must include sections: {chapter.chapter_id}"
                )
