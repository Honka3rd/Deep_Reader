from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.storage_namespace_helper import StorageNamespaceHelper
from config.structured_document_storage_config import StructuredDocumentStorageConfig
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from shared.task_artifacts import DocumentTaskArtifacts, TaskArtifacts
from shared.task_unit_model import TaskUnit, TaskUnitContentBlock


@dataclass(frozen=True)
class PostgresStructuredDocumentTarget:
    """Logical target for one structured document row in PostgreSQL."""

    namespace: str
    document_name: str


class PostgresStructuredDocumentStore:
    """PostgreSQL-backed current StructuredDocument persistence.

    This store is the runtime bridge for new Docker deployments that no longer
    use ``data/structured`` as the structured document read/write location.
    """

    _NAMESPACE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

    def __init__(
        self,
        dsn: str,
        namespace: str = "default",
        schema_version: str = "structured_document_runtime_v1",
    ) -> None:
        self._dsn = dsn
        self._namespace = namespace.strip() or "default"
        self._schema_version = schema_version

    def save(
        self,
        document: StructuredDocument,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> None:
        """Persist the current structured document payload in PostgreSQL."""
        resolved = self._resolve_target(target)
        payload = document.to_dict()
        metadata_payload = {
            "title": document.title,
            "language": document.language,
            "source_path": document.source_path,
            "structure_error_code": document.structure_error_code,
            "structure_error_message": document.structure_error_message,
            "document_task_artifacts": (
                None
                if document.document_task_artifacts is None
                else document.document_task_artifacts.to_dict()
            ),
        }
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            with connection.transaction():
                row = connection.execute(
                    """
                    INSERT INTO documents (
                        namespace,
                        document_name,
                        current_structure_version,
                        status,
                        metadata_payload,
                        updated_at
                    )
                    VALUES (%s, %s, 1, 'active', %s, NOW())
                    ON CONFLICT (namespace, document_name)
                    DO UPDATE SET
                        status = 'active',
                        metadata_payload = EXCLUDED.metadata_payload,
                        updated_at = NOW()
                    RETURNING id, current_structure_version
                    """,
                    (
                        resolved.namespace,
                        resolved.document_name,
                        Jsonb(metadata_payload),
                    ),
                ).fetchone()
                if row is None:
                    raise RuntimeError("PostgresStructuredDocumentStore.save: document upsert failed")

                document_id = int(row["id"])
                structure_version = int(row["current_structure_version"])
                self._replace_current_hierarchy(
                    connection=connection,
                    document=document,
                    document_id=document_id,
                    Jsonb=Jsonb,
                )
                connection.execute(
                    """
                    INSERT INTO structured_document_snapshots (
                        document_id,
                        source_structure_version,
                        structured_document_payload,
                        schema_version,
                        metadata_payload
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, source_structure_version)
                    DO UPDATE SET
                        structured_document_payload = EXCLUDED.structured_document_payload,
                        schema_version = EXCLUDED.schema_version,
                        metadata_payload = EXCLUDED.metadata_payload,
                        created_at = NOW()
                    """,
                    (
                        document_id,
                        structure_version,
                        Jsonb(payload),
                        self._schema_version,
                        Jsonb({"storage_backend": "postgres"}),
                    ),
                )
                if document.source_path:
                    connection.execute(
                        """
                        INSERT INTO raw_source_metadata (
                            document_id,
                            source_location,
                            original_filename,
                            metadata_payload
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id)
                        DO UPDATE SET
                            source_location = EXCLUDED.source_location,
                            original_filename = EXCLUDED.original_filename,
                            metadata_payload = EXCLUDED.metadata_payload
                        """,
                        (
                            document_id,
                            document.source_path,
                            Path(document.source_path).name,
                            Jsonb({"storage_backend": "postgres"}),
                        ),
                    )

    def load(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> StructuredDocument:
        """Load the current structured document from PostgreSQL relational rows."""
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            document_row = connection.execute(
                """
                SELECT d.id,
                       d.document_name,
                       d.metadata_payload,
                       r.source_location
                FROM documents d
                LEFT JOIN raw_source_metadata r
                  ON r.document_id = d.id
                WHERE d.namespace = %s
                  AND d.document_name = %s
                """,
                (resolved.namespace, resolved.document_name),
            ).fetchone()
            if document_row is None:
                raise FileNotFoundError(
                    "PostgresStructuredDocumentStore.load: document not found: "
                    f"namespace={resolved.namespace} document_name={resolved.document_name}"
                )
            document_id = int(document_row["id"])
            chapter_rows = connection.execute(
                """
                SELECT id,
                       title,
                       level,
                       chapter_role,
                       metadata_payload
                FROM chapters
                WHERE document_id = %s
                ORDER BY chapter_order, id
                """,
                (document_id,),
            ).fetchall()
            section_rows = connection.execute(
                """
                SELECT id,
                       chapter_id,
                       section_order,
                       title,
                       level,
                       content,
                       char_start,
                       char_end,
                       container_title,
                       section_role,
                       section_kind,
                       is_implicit_section,
                       metadata_payload
                FROM sections
                WHERE document_id = %s
                ORDER BY chapter_id, section_order, id
                """,
                (document_id,),
            ).fetchall()
            task_unit_rows = connection.execute(
                """
                SELECT id,
                       section_id,
                       task_unit_order,
                       title,
                       container_title,
                       content_payload,
                       source_section_ids_payload,
                       is_fallback_generated,
                       metadata_payload
                FROM task_units
                WHERE document_id = %s
                ORDER BY section_id, task_unit_order, id
                """,
                (document_id,),
            ).fetchall()
        return self._build_document_from_rows(
            document_row=document_row,
            chapter_rows=chapter_rows,
            section_rows=section_rows,
            task_unit_rows=task_unit_rows,
        )

    def _replace_current_hierarchy(
        self,
        *,
        connection: Any,
        document: StructuredDocument,
        document_id: int,
        Jsonb: Any,
    ) -> None:
        kept_chapter_ids: list[int] = []
        kept_section_ids: list[int] = []
        kept_task_unit_ids: list[int] = []

        for chapter_order, chapter in enumerate(document.chapters):
            chapter_metadata = dict(chapter.metadata)
            if chapter.task_artifacts is not None:
                chapter_metadata["task_artifacts"] = chapter.task_artifacts.to_dict()
            chapter_id = self._upsert_hierarchy_row(
                connection=connection,
                table_name="chapters",
                id_value=chapter.chapter_id,
                document_id=document_id,
                insert_sql="""
                    INSERT INTO chapters (
                        document_id,
                        chapter_order,
                        reference_chapter_id,
                        title,
                        level,
                        chapter_role,
                        metadata_payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                insert_params=(
                    document_id,
                    chapter_order,
                    chapter.chapter_id,
                    chapter.title,
                    chapter.level,
                    chapter.chapter_role,
                    Jsonb(chapter_metadata),
                ),
                update_sql="""
                    UPDATE chapters
                    SET chapter_order = %s,
                        title = %s,
                        level = %s,
                        chapter_role = %s,
                        metadata_payload = %s
                    WHERE id = %s
                      AND document_id = %s
                    RETURNING id
                """,
                update_params=(
                    chapter_order,
                    chapter.title,
                    chapter.level,
                    chapter.chapter_role,
                    Jsonb(chapter_metadata),
                ),
            )
            kept_chapter_ids.append(chapter_id)

            for section_order, section in enumerate(chapter.sections):
                section_metadata: dict[str, Any] = {
                    "section_index": section.section_index,
                }
                if section.task_artifacts is not None:
                    section_metadata["task_artifacts"] = section.task_artifacts.to_dict()
                section_id = self._upsert_hierarchy_row(
                    connection=connection,
                    table_name="sections",
                    id_value=section.section_id,
                    document_id=document_id,
                    insert_sql="""
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
                            metadata_payload
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """,
                    insert_params=(
                        document_id,
                        chapter_id,
                        section_order,
                        section.section_id,
                        section.title,
                        section.level,
                        section.content,
                        section.char_start,
                        section.char_end,
                        section.container_title,
                        None if section.section_role is None else section.section_role.value,
                        section.section_kind,
                        section.is_implicit_section,
                        Jsonb(section_metadata),
                    ),
                    update_sql="""
                        UPDATE sections
                        SET chapter_id = %s,
                            section_order = %s,
                            title = %s,
                            level = %s,
                            content = %s,
                            char_start = %s,
                            char_end = %s,
                            container_title = %s,
                            section_role = %s,
                            section_kind = %s,
                            is_implicit_section = %s,
                            metadata_payload = %s
                        WHERE id = %s
                          AND document_id = %s
                        RETURNING id
                    """,
                    update_params=(
                        chapter_id,
                        section_order,
                        section.title,
                        section.level,
                        section.content,
                        section.char_start,
                        section.char_end,
                        section.container_title,
                        None if section.section_role is None else section.section_role.value,
                        section.section_kind,
                        section.is_implicit_section,
                        Jsonb(section_metadata),
                    ),
                )
                kept_section_ids.append(section_id)

                for task_unit_order, task_unit in enumerate(section.task_units):
                    task_unit_metadata: dict[str, Any] = {}
                    if task_unit.task_artifacts is not None:
                        task_unit_metadata["task_artifacts"] = task_unit.task_artifacts.to_dict()
                    if task_unit.content_blocks:
                        task_unit_metadata["content_blocks"] = [
                            block.to_dict()
                            for block in task_unit.content_blocks
                        ]
                    task_unit_id = self._upsert_hierarchy_row(
                        connection=connection,
                        table_name="task_units",
                        id_value=task_unit.unit_id,
                        document_id=document_id,
                        insert_sql="""
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
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """,
                        insert_params=(
                            document_id,
                            section_id,
                            task_unit_order,
                            task_unit.unit_id,
                            task_unit.title,
                            task_unit.container_title,
                            task_unit.content,
                            Jsonb([str(section_id)]),
                            task_unit.is_fallback_generated,
                            Jsonb(task_unit_metadata),
                        ),
                        update_sql="""
                            UPDATE task_units
                            SET section_id = %s,
                                task_unit_order = %s,
                                title = %s,
                                container_title = %s,
                                content_payload = %s,
                                source_section_ids_payload = %s,
                                is_fallback_generated = %s,
                                metadata_payload = %s
                            WHERE id = %s
                              AND document_id = %s
                            RETURNING id
                        """,
                        update_params=(
                            section_id,
                            task_unit_order,
                            task_unit.title,
                            task_unit.container_title,
                            task_unit.content,
                            Jsonb([str(section_id)]),
                            task_unit.is_fallback_generated,
                            Jsonb(task_unit_metadata),
                        ),
                    )
                    kept_task_unit_ids.append(task_unit_id)

        self._delete_missing_rows(
            connection=connection,
            table_name="task_units",
            document_id=document_id,
            kept_ids=kept_task_unit_ids,
        )
        self._delete_missing_rows(
            connection=connection,
            table_name="sections",
            document_id=document_id,
            kept_ids=kept_section_ids,
        )
        self._delete_missing_rows(
            connection=connection,
            table_name="chapters",
            document_id=document_id,
            kept_ids=kept_chapter_ids,
        )

    def _upsert_hierarchy_row(
        self,
        *,
        connection: Any,
        table_name: str,
        id_value: str,
        document_id: int,
        insert_sql: str,
        insert_params: tuple[Any, ...],
        update_sql: str,
        update_params: tuple[Any, ...],
    ) -> int:
        if self._is_positive_int_text(id_value):
            row = connection.execute(
                update_sql,
                (*update_params, int(id_value), document_id),
            ).fetchone()
            if row is not None:
                return int(row["id"])
        row = connection.execute(insert_sql, insert_params).fetchone()
        if row is None:
            raise RuntimeError(f"PostgresStructuredDocumentStore: failed to upsert {table_name}")
        return int(row["id"])

    @staticmethod
    def _delete_missing_rows(
        *,
        connection: Any,
        table_name: str,
        document_id: int,
        kept_ids: list[int],
    ) -> None:
        if kept_ids:
            connection.execute(
                f"DELETE FROM {table_name} WHERE document_id = %s AND NOT (id = ANY(%s))",
                (document_id, kept_ids),
            )
            return
        connection.execute(
            f"DELETE FROM {table_name} WHERE document_id = %s",
            (document_id,),
        )

    @staticmethod
    def _is_positive_int_text(value: str) -> bool:
        try:
            return int(value) > 0 and str(int(value)) == str(value)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _dict_payload(payload: Any) -> dict[str, Any]:
        return payload if isinstance(payload, dict) else {}

    def _build_document_from_rows(
        self,
        *,
        document_row: dict[str, Any],
        chapter_rows: list[dict[str, Any]],
        section_rows: list[dict[str, Any]],
        task_unit_rows: list[dict[str, Any]],
    ) -> StructuredDocument:
        document_metadata = self._dict_payload(document_row.get("metadata_payload"))
        task_units_by_section_id: dict[int, list[TaskUnit]] = {}
        for task_unit_row in task_unit_rows:
            task_unit_metadata = self._dict_payload(task_unit_row.get("metadata_payload"))
            content_blocks_payload = task_unit_metadata.get("content_blocks", [])
            content_blocks = [
                TaskUnitContentBlock.from_dict(block)
                for block in content_blocks_payload
                if isinstance(block, dict)
            ]
            section_id = int(task_unit_row["section_id"])
            task_units_by_section_id.setdefault(section_id, []).append(
                TaskUnit(
                    unit_id=str(task_unit_row["id"]),
                    title=task_unit_row.get("title"),
                    container_title=task_unit_row.get("container_title"),
                    content=str(task_unit_row.get("content_payload") or ""),
                    source_section_ids=[str(section_id)],
                    is_fallback_generated=bool(task_unit_row.get("is_fallback_generated")),
                    parent_section_id=str(section_id),
                    task_artifacts=TaskArtifacts.from_dict(
                        task_unit_metadata.get("task_artifacts")
                    ),
                    content_blocks=content_blocks,
                )
            )

        sections_by_chapter_id: dict[int, list[StructuredSection]] = {}
        for section_row in section_rows:
            section_metadata = self._dict_payload(section_row.get("metadata_payload"))
            section_id = int(section_row["id"])
            chapter_id = int(section_row["chapter_id"])
            sections_by_chapter_id.setdefault(chapter_id, []).append(
                StructuredSection(
                    section_id=str(section_id),
                    section_index=int(section_metadata.get("section_index", section_row["section_order"])),
                    title=section_row.get("title"),
                    level=int(section_row["level"]),
                    content=str(section_row.get("content") or ""),
                    char_start=int(section_row.get("char_start") or 0),
                    char_end=int(section_row.get("char_end") or 0),
                    container_title=section_row.get("container_title"),
                    section_role=SectionRole.resolve(section_row.get("section_role")),
                    parent_chapter_id=str(chapter_id),
                    section_kind=section_row.get("section_kind"),
                    is_implicit_section=bool(section_row.get("is_implicit_section")),
                    task_units=task_units_by_section_id.get(section_id, []),
                    task_artifacts=TaskArtifacts.from_dict(
                        section_metadata.get("task_artifacts")
                    ),
                )
            )
        chapters: list[StructuredChapter] = []
        for chapter_row in chapter_rows:
            chapter_metadata = self._dict_payload(chapter_row.get("metadata_payload"))
            chapter_id = int(chapter_row["id"])
            chapter_metadata_without_artifacts = {
                key: value
                for key, value in chapter_metadata.items()
                if key != "task_artifacts"
            }
            chapters.append(
                StructuredChapter(
                    chapter_id=str(chapter_id),
                    title=chapter_row.get("title"),
                    level=int(chapter_row["level"]),
                    chapter_role=chapter_row.get("chapter_role"),
                    sections=sections_by_chapter_id.get(chapter_id, []),
                    task_artifacts=TaskArtifacts.from_dict(
                        chapter_metadata.get("task_artifacts")
                    ),
                    metadata=chapter_metadata_without_artifacts,
                )
            )
        if not chapters:
            raise ValueError(
                "PostgresStructuredDocumentStore.load: hierarchy chapters are required"
            )
        return StructuredDocument(
            document_id=str(document_row["document_name"]),
            title=str(document_metadata.get("title") or document_row["document_name"]),
            source_path=document_row.get("source_location") or document_metadata.get("source_path"),
            language=document_metadata.get("language"),
            raw_text="",
            sections=[],
            chapters=chapters,
            structure_nodes=[],
            structure_error_code=document_metadata.get("structure_error_code"),
            structure_error_message=document_metadata.get("structure_error_message"),
            document_task_artifacts=DocumentTaskArtifacts.from_dict(
                document_metadata.get("document_task_artifacts")
            ),
        )

    def exists(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> bool:
        """Return whether a structured document exists in PostgreSQL."""
        resolved = self._resolve_target(target)
        psycopg, _, _ = self._load_psycopg()
        with psycopg.connect(self._dsn) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM documents
                WHERE namespace = %s
                  AND document_name = %s
                """,
                (resolved.namespace, resolved.document_name),
            ).fetchone()
        return row is not None

    def location(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> str:
        """Return a stable logical location for API readiness responses."""
        resolved = self._resolve_target(target)
        return f"postgres://structured/{resolved.namespace}/{resolved.document_name}"

    def target_for_doc_name(self, doc_name: str) -> PostgresStructuredDocumentTarget:
        """Build a DB target from a logical document name."""
        return PostgresStructuredDocumentTarget(
            namespace=self._namespace,
            document_name=self._normalize_document_name(doc_name),
        )

    def _resolve_target(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> PostgresStructuredDocumentTarget:
        if isinstance(target, PostgresStructuredDocumentTarget):
            return target
        if isinstance(target, StructuredDocumentStorageConfig):
            return self.target_for_doc_name(target.get_doc_name())
        return self.target_for_doc_name(self._document_name_from_string(target))

    def _document_name_from_string(self, target: str) -> str:
        raw = target.strip()
        if raw.startswith("postgres://structured/"):
            parts = raw.removeprefix("postgres://structured/").split("/", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[1]
        name = Path(raw).name
        if name.endswith(".structured.json"):
            name = name[: -len(".structured.json")]
        return name or raw

    def _normalize_document_name(self, doc_name: str) -> str:
        return StorageNamespaceHelper.normalize_namespace(
            doc_name,
            known_extensions=self._NAMESPACE_EXTENSIONS,
            fallback_namespace=StorageNamespaceHelper.DEFAULT_NAMESPACE,
        )

    @staticmethod
    def _load_psycopg() -> tuple[Any, Any, Any]:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Postgres structured storage requires psycopg. "
                "Install project requirements or use structured_storage_backend='file'."
            ) from error
        return psycopg, dict_row, Jsonb
