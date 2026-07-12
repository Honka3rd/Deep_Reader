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
    """Metadata-only reference to file-backed or object-backed raw source bytes/text."""

    source_location: str = ""
    original_filename: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    checksum: str | None = None
    ownership_scope: str | None = None
    metadata_payload: dict[str, Any] | None = None


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
        namespace: str,
        document_name: str | None = None,
        raw_source_metadata: RawSourceMetadata | None = None,
        parser_name: str | None = None,
        preparation_mode: str | None = None,
    ) -> InitialStructureWriteResult:
        """Persist one accepted current hierarchy with structure version 1."""
        self._validate_accepted_hierarchy(document)
        normalized_namespace = self._normalize_required_text(namespace, "namespace")
        normalized_document_name = self._normalize_required_text(
            document_name if document_name is not None else document.document_id,
            "document_name",
        )
        document_metadata = {
            "title": document.title,
            "language": document.language,
        }
        with self._connection:
            document_cursor = self._connection.execute(
                """
                INSERT INTO documents (
                    namespace,
                    document_name,
                    current_structure_version,
                    status,
                    metadata_payload
                )
                VALUES (?, ?, 1, 'active', ?)
                """,
                (
                    normalized_namespace,
                    normalized_document_name,
                    json.dumps(document_metadata, ensure_ascii=False),
                ),
            )
            db_document_id = int(document_cursor.lastrowid)
            if raw_source_metadata is not None:
                normalized_source_location = self._normalize_required_text(
                    raw_source_metadata.source_location,
                    "source_location",
                )
                self._connection.execute(
                    """
                    INSERT INTO raw_source_metadata (
                        document_id,
                        source_location,
                        original_filename,
                        mime_type,
                        file_size,
                        checksum,
                        ownership_scope,
                        metadata_payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_document_id,
                        normalized_source_location,
                        raw_source_metadata.original_filename,
                        raw_source_metadata.mime_type,
                        raw_source_metadata.file_size,
                        raw_source_metadata.checksum,
                        raw_source_metadata.ownership_scope,
                        json.dumps(
                            raw_source_metadata.metadata_payload or {},
                            ensure_ascii=False,
                        ),
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
                        source_anchor,
                        metadata_payload
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_document_id,
                        chapter_order,
                        chapter.chapter_id,
                        chapter.title,
                        chapter.level,
                        chapter.chapter_role,
                        None,
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
                            is_implicit_section,
                            source_anchor,
                            metadata_payload
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            json.dumps(
                                {
                                    "char_start": section.char_start,
                                    "char_end": section.char_end,
                                },
                                ensure_ascii=False,
                            ),
                            json.dumps({}, ensure_ascii=False),
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
                                content_payload,
                                source_section_ids_payload,
                                is_fallback_generated,
                                metadata_payload
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                json.dumps({}, ensure_ascii=False),
                            ),
                        )

            parse_event_cursor = self._connection.execute(
                """
                INSERT INTO parse_events (
                    document_id,
                    event_type,
                    previous_structure_version,
                    new_structure_version,
                    trigger_source,
                    parser_mode,
                    metadata_payload
                )
                VALUES (?, 'initial_parse', NULL, 1, 'prepare', ?, ?)
                """,
                (
                    db_document_id,
                    parser_name,
                    json.dumps(
                        {
                            "namespace": normalized_namespace,
                            "document_name": normalized_document_name,
                            "preparation_mode": preparation_mode,
                        },
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
            SELECT
                id,
                namespace,
                document_name,
                current_structure_version,
                metadata_payload
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if document_row is None:
            raise KeyError(f"document not found: {document_id}")

        chapter_rows = self._connection.execute(
            """
            SELECT id, title, level, chapter_role, metadata_payload
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

        document_metadata = self._decode_json_object(document_row["metadata_payload"])
        raw_source_row = self._connection.execute(
            """
            SELECT source_location
            FROM raw_source_metadata
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()
        source_path = None if raw_source_row is None else raw_source_row["source_location"]

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
                        content_payload,
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
                        content=task_unit_row["content_payload"],
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
                    metadata=self._decode_json_object(chapter_row["metadata_payload"]),
                )
            )

        return StructuredDocument(
            document_id=str(document_row["id"]),
            title=str(document_metadata.get("title") or document_row["document_name"]),
            source_path=source_path,
            language=(
                None
                if document_metadata.get("language") is None
                else str(document_metadata.get("language"))
            ),
            raw_text="",
            chapters=chapters,
            sections=[],
            structure_nodes=[],
        )

    def load_document_lifecycle(self, document_id: int) -> dict[str, Any]:
        """Return lightweight document lifecycle metadata for validation paths."""
        row = self._connection.execute(
            """
            SELECT id, namespace, document_name, current_structure_version
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {document_id}")
        return {
            "document_id": int(row["id"]),
            "namespace": row["namespace"],
            "document_name": row["document_name"],
            "current_structure_version": int(row["current_structure_version"]),
        }

    def find_document_by_name(
        self,
        *,
        namespace: str,
        document_name: str,
    ) -> int | None:
        """Find a document by namespace/name isolation key."""
        row = self._connection.execute(
            """
            SELECT id
            FROM documents
            WHERE namespace = ? AND document_name = ?
            """,
            (
                self._normalize_required_text(namespace, "namespace"),
                self._normalize_required_text(document_name, "document_name"),
            ),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

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


    @staticmethod
    def _normalize_required_text(value: str, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized
