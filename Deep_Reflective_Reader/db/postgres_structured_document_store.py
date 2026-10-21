from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

from config.storage_namespace_helper import StorageNamespaceHelper
from config.structured_document_storage_config import StructuredDocumentStorageConfig
from document_structure.document_artifact_repository import DocumentListItem, preserve_explicit_annotations
from document_structure.enhancement_change import (
    EnhancementChangeKind, EnhancementComparisonError, compare_enhancement_documents,
)
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


@dataclass(frozen=True)
class EnhancementStorageSnapshot:
    namespace: str
    document: StructuredDocument
    structure_version: int
    accepted: bool
    interaction_revision: str
    interaction_count: int | None
    includes_user_notes: bool | None


class EnhancementCommitConflict(ValueError):
    pass


@dataclass(frozen=True)
class EnhancementCommitResult(EnhancementStorageSnapshot):
    change_kind: EnhancementChangeKind


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
        *,
        replace_existing_hierarchy: bool = True,
        _connection=None,
        _preserve_raw_source: bool = False,
    ) -> None:
        """Persist the current structured document payload in PostgreSQL.

        ``replace_existing_hierarchy`` is reserved for parser-level replacement
        saves. Repository metadata/task-layout updates preserve the current
        document structure version.
        """
        resolved = self._resolve_target(target)
        payload = document.to_dict()
        metadata_payload = {
            "title": document.title,
            "language": document.language,
            "source_path": document.source_path,
            "structure_error_code": document.structure_error_code,
            "structure_error_message": document.structure_error_message,
            "parse_provenance": dict(document.parse_provenance),
            "document_task_artifacts": (
                None
                if document.document_task_artifacts is None
                else document.document_task_artifacts.to_dict()
            ),
        }
        if replace_existing_hierarchy:
            metadata_payload["parse_provenance"].pop("reading_content",None)
            metadata_payload["parse_provenance"].pop("reading_projected",None)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with (nullcontext(_connection) if _connection is not None else
              psycopg.connect(self._dsn, row_factory=dict_row)) as connection:
            with connection.transaction():
                existing_row = connection.execute(
                    """
                    SELECT id,
                           current_structure_version, metadata_payload
                    FROM documents
                    WHERE namespace = %s
                      AND document_name = %s
                    FOR UPDATE
                    """,
                    (resolved.namespace, resolved.document_name),
                ).fetchone()
                if existing_row is None:
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
                        RETURNING id, current_structure_version
                        """,
                        (
                            resolved.namespace,
                            resolved.document_name,
                            Jsonb(metadata_payload),
                        ),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError(
                            "PostgresStructuredDocumentStore.save: document insert failed"
                        )
                    document_id = int(row["id"])
                    previous_structure_version = None
                    structure_version = int(row["current_structure_version"])
                    event_type = "initial_parse"
                    invalidated_artifact_count = 0
                    invalidated_content_block_count = 0
                else:
                    document_id = int(existing_row["id"])
                    previous_structure_version = int(
                        existing_row["current_structure_version"]
                    )
                    if replace_existing_hierarchy:
                        structure_version = previous_structure_version + 1
                        invalidated_artifact_count = self._count_rows(
                            connection=connection,
                            table_name="artifacts",
                            document_id=document_id,
                        )
                        invalidated_content_block_count = self._count_rows(
                            connection=connection,
                            table_name="content_blocks",
                            document_id=document_id,
                        )
                        connection.execute(
                            """
                            UPDATE documents
                            SET current_structure_version = %s,
                                status = 'active',
                                metadata_payload = %s,
                                updated_at = NOW()
                            WHERE id = %s
                            """,
                            (structure_version, Jsonb(metadata_payload), document_id),
                        )
                        self._delete_current_hierarchy_and_derived_rows(
                            connection=connection,
                            document_id=document_id,
                        )
                        event_type = "hard_reparse"
                    else:
                        structure_version = previous_structure_version
                        previous_artifacts = DocumentTaskArtifacts.from_dict(
                            (existing_row.get("metadata_payload") or {}).get("document_task_artifacts"))
                        document = preserve_explicit_annotations(
                            replace(document, document_task_artifacts=previous_artifacts,
                                    parse_provenance=(existing_row.get("metadata_payload") or {}).get("parse_provenance",{})), document)
                        metadata_payload["parse_provenance"] = document.parse_provenance
                        metadata_payload["document_task_artifacts"] = (
                            document.document_task_artifacts.to_dict() if document.document_task_artifacts else None)
                        self._validate_metadata_write(connection, document, document_id,
                                                      structure_version, existing_row)
                        document = self._preserve_navigation_labels(connection, document, document_id)
                        payload = document.to_dict()
                        event_type = None
                        invalidated_artifact_count = None
                        invalidated_content_block_count = None
                        connection.execute(
                            """
                            UPDATE documents
                            SET status = 'active',
                                metadata_payload = %s,
                                updated_at = NOW()
                            WHERE id = %s
                            """,
                            (Jsonb(metadata_payload), document_id),
                        )

                if event_type is not None:
                    connection.execute(
                        """
                        INSERT INTO parse_events (
                            document_id,
                            event_type,
                            previous_structure_version,
                            new_structure_version,
                            trigger_source,
                            parser_mode,
                            reparse_reason,
                            invalidated_artifact_count,
                            invalidated_content_block_count,
                            metadata_payload
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            document_id,
                            event_type,
                            previous_structure_version,
                            structure_version,
                            "postgres_structured_document_store.save",
                            self._effective_parser_mode(document.parse_provenance),
                            None if event_type == "initial_parse" else "force_rebuild",
                            invalidated_artifact_count,
                            invalidated_content_block_count,
                            Jsonb(metadata_payload),
                        ),
                    )
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
                if document.source_path and not _preserve_raw_source:
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

    def mutate_reading_content(self, target, mutation):
        resolved = self._resolve_target(target)
        psycopg,dict_row,Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn,row_factory=dict_row) as connection:
            row = connection.execute("SELECT id,metadata_payload FROM documents WHERE namespace=%s AND document_name=%s FOR UPDATE",
                                     (resolved.namespace,resolved.document_name)).fetchone()
            if row is None:
                raise FileNotFoundError("document_missing")
            document = self.load_enhancement_snapshot(target,_connection=connection).document
            updated,result = mutation(document)
            if updated != document:
                metadata = dict(row["metadata_payload"])
                metadata["parse_provenance"] = updated.parse_provenance
                metadata["document_task_artifacts"] = updated.document_task_artifacts.to_dict() if updated.document_task_artifacts else None
                connection.execute("UPDATE documents SET metadata_payload=%s,updated_at=NOW() WHERE id=%s",
                                   (Jsonb(metadata),row["id"]))
                for table in ("chapters","sections","task_units"):
                    connection.execute(f"UPDATE {table} SET metadata_payload=metadata_payload - 'task_artifacts' WHERE document_id=%s",(row["id"],))
                connection.execute("DELETE FROM artifacts WHERE document_id=%s AND artifact_type NOT IN ('note','notes')",(row["id"],))
            return result

    def mutate_document_artifacts(self, target, mutation):
        """Serialize user mutations with deletion/reparse and other note writers."""
        resolved = self._resolve_target(target)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("LOCK TABLE documents IN ROW EXCLUSIVE MODE")
            row = connection.execute(
                "SELECT id FROM documents WHERE namespace=%s AND document_name=%s FOR UPDATE",
                (resolved.namespace, resolved.document_name),
            ).fetchone()
            if row is None:
                raise FileNotFoundError("document_missing")
            document = self.load_enhancement_snapshot(target, _connection=connection).document
            artifacts, result = mutation(document)
            if artifacts != document.document_task_artifacts:
                connection.execute(
                    "UPDATE documents SET metadata_payload=jsonb_set(metadata_payload, "
                    "'{document_task_artifacts}', %s), updated_at=NOW() WHERE id=%s",
                    (Jsonb(artifacts.to_dict() if artifacts else None), row["id"]),
                )
            return result

    def rename_document(self, target, new_name, transform):
        resolved = self._resolve_target(target)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("LOCK TABLE documents IN ROW EXCLUSIVE MODE")
            row = connection.execute("SELECT id, metadata_payload FROM documents "
                "WHERE namespace=%s AND document_name=%s FOR UPDATE",
                (resolved.namespace, resolved.document_name)).fetchone()
            if row is None:
                return False
            if connection.execute("SELECT id FROM documents WHERE namespace=%s AND document_name=%s",
                                  (resolved.namespace, new_name)).fetchone():
                raise ValueError("conflict")
            document = self.load_enhancement_snapshot(target, _connection=connection).document
            renamed = transform(document, new_name)
            metadata = dict(row["metadata_payload"] or {})
            metadata["title"] = renamed.title
            metadata["document_task_artifacts"] = (renamed.document_task_artifacts.to_dict()
                                                    if renamed.document_task_artifacts else None)
            source = connection.execute("SELECT source_location FROM raw_source_metadata WHERE document_id=%s",
                                        (row["id"],)).fetchone()
            if source:
                old_path = Path(source["source_location"])
                new_path = str(old_path.with_name(new_name + old_path.suffix))
                metadata["source_path"] = new_path
                connection.execute("UPDATE raw_source_metadata SET source_location=%s, original_filename=%s WHERE document_id=%s",
                                   (new_path, Path(new_path).name, row["id"]))
            connection.execute("UPDATE documents SET document_name=%s, metadata_payload=%s, updated_at=NOW() WHERE id=%s",
                               (new_name, Jsonb(metadata), row["id"]))
            connection.execute("UPDATE structured_document_snapshots SET structured_document_payload="
                "jsonb_set(jsonb_set(structured_document_payload, '{document_id}', %s), '{title}', %s) WHERE document_id=%s",
                (Jsonb(new_name), Jsonb(renamed.title), row["id"]))
            return True

    def save_ocr_run(
        self,
        *,
        doc_name: str,
        provenance: dict[str, object],
        full_text: str | None,
        pages: list[str] | None = None,
    ) -> None:
        """Persist OCR output and provenance without making it hierarchy authority."""
        resolved = self.target_for_doc_name(doc_name)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            with connection.transaction():
                document = connection.execute(
                    """
                    SELECT id FROM documents
                    WHERE namespace = %s AND document_name = %s
                    """,
                    (resolved.namespace, resolved.document_name),
                ).fetchone()
                if document is None:
                    raise FileNotFoundError(
                        "PostgresStructuredDocumentStore.save_ocr_run: document not found: "
                        f"{resolved.namespace}/{resolved.document_name}"
                    )
                run = connection.execute(
                    """
                    INSERT INTO ocr_runs (
                        document_id, source_file_sha256, ocr_engine,
                        ocr_engine_version, ocr_language, renderer, render_dpi,
                        page_count, page_limit, status, full_text, metadata_payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        int(document["id"]),
                        str(provenance.get("source_file_sha256", "")),
                        str(provenance.get("ocr_engine", "tesseract")),
                        provenance.get("ocr_engine_version"),
                        str(provenance.get("ocr_language", "")),
                        provenance.get("pdf_renderer"),
                        provenance.get("pdf_render_dpi"),
                        int(provenance.get("page_count", 0)),
                        provenance.get("ocr_page_limit"),
                        "completed",
                        full_text,
                        Jsonb(provenance),
                    ),
                ).fetchone()
                if run is None:
                    raise RuntimeError("PostgresStructuredDocumentStore.save_ocr_run: insert failed")
                if pages is not None:
                    with connection.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO ocr_pages (ocr_run_id, page_index, text, metadata_payload)
                            VALUES (%s, %s, %s, %s)
                            """,
                            [
                                (
                                    int(run["id"]),
                                    index,
                                    page,
                                    Jsonb({"source": "tesseract"}),
                                )
                                for index, page in enumerate(pages)
                            ],
                        )

    def load(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> StructuredDocument:
        """Load the current structured document from PostgreSQL relational rows."""
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            document_row = connection.execute(
                """
                SELECT d.id,
                       d.document_name,
                       d.current_structure_version,
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

    def load_enhancement_snapshot(self, target, *, _connection=None) -> EnhancementStorageSnapshot:
        """Read all affected current rows in one read-only MVCC snapshot.

        Row xmin plus payload detects edits even when row counts do not change.
        This is an optimistic preflight fingerprint, not a commit-time lock.
        """
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with (nullcontext(_connection) if _connection is not None else
              psycopg.connect(self._dsn, row_factory=dict_row)) as connection:
            if _connection is None:
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            document_row = connection.execute(
                "SELECT d.*, d.xmin::text AS row_revision FROM documents d "
                "WHERE namespace = %s AND document_name = %s",
                (resolved.namespace, resolved.document_name),
            ).fetchone()
            if document_row is None:
                raise FileNotFoundError("enhancement_document_missing")
            document_id = int(document_row["id"])
            # Table identifiers and order clauses are fixed, never user input.
            rows = {}
            for table, order in (
                ("raw_source_metadata", "document_id"),
                ("chapters", "chapter_order, id"),
                ("sections", "chapter_id, section_order, id"),
                ("task_units", "section_id, task_unit_order, id"),
                ("content_blocks", "id"), ("artifacts", "id"),
            ):
                rows[table] = connection.execute(
                    f"SELECT t.*, t.xmin::text AS row_revision FROM {table} t "
                    f"WHERE document_id = %s ORDER BY {order}", (document_id,),
                ).fetchall()
        digest = hashlib.sha256(json.dumps(
            [document_row, rows], sort_keys=True, default=str,
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        source_rows = rows["raw_source_metadata"]
        document_row["source_location"] = source_rows[0].get("source_location") if source_rows else None
        document = self._build_document_from_rows(
            document_row=document_row, chapter_rows=rows["chapters"],
            section_rows=rows["sections"], task_unit_rows=rows["task_units"],
        )
        # Embedded metadata may contain overlapping/legacy representations.
        # Do not claim an exact logical count or absence of notes in that case.
        metadata = [document_row.get("metadata_payload", {})]
        metadata.extend(row.get("metadata_payload", {}) for table in
                        ("chapters", "sections", "task_units", "content_blocks")
                        for row in rows[table])
        embedded = any(self._contains_interaction_payload(item) for item in metadata)
        has_notes = (any(row.get("artifact_type") in {"note", "notes"} for row in rows["artifacts"])
                     or any(self._contains_interaction_payload(item, {"note", "notes"}) for item in metadata))
        return EnhancementStorageSnapshot(
            namespace=resolved.namespace, document=document,
            structure_version=int(document_row["current_structure_version"]),
            accepted=document_row.get("status") == "active" and not document.structure_error_code,
            interaction_revision=digest,
            interaction_count=None if embedded else len(rows["artifacts"]),
            includes_user_notes=True if has_notes else (None if embedded else False),
        )

    @staticmethod
    def _contains_interaction_payload(value, keys=None):
        keys = keys or {"task_artifacts", "document_task_artifacts", "note", "notes"}
        if isinstance(value, dict):
            for key, item in value.items():
                if key in keys and item:
                    return True
                if PostgresStructuredDocumentStore._contains_interaction_payload(item, keys):
                    return True
        elif isinstance(value, list):
            return any(PostgresStructuredDocumentStore._contains_interaction_payload(item, keys) for item in value)
        return False

    @staticmethod
    def _validate_metadata_write(connection, document, document_id, version, existing_row):
        if document.parse_provenance.get("storage_structure_version") != version:
            raise EnhancementCommitConflict("stale_structure_write")
        current_ids = {str(row["id"]) for row in connection.execute(
            "SELECT id FROM sections WHERE document_id=%s", (document_id,)).fetchall()}
        supplied_ids = {section.section_id for chapter in document.chapters for section in chapter.sections}
        if current_ids != supplied_ids:
            raise EnhancementCommitConflict("stale_structure_write")
        old = (existing_row.get("metadata_payload") or {}).get("document_task_artifacts") or {}
        new = document.document_task_artifacts.to_dict() if document.document_task_artifacts else {}
        chapter_keys = {"chapter_id::" + chapter.chapter_id for chapter in document.chapters}
        for key, value in new.get("chapter_artifacts", {}).items():
            if value != old.get("chapter_artifacts", {}).get(key) and key not in chapter_keys:
                raise EnhancementCommitConflict("stale_interaction_write")

        def check_updates(previous, incoming):
            if incoming == previous:
                return
            if isinstance(incoming, dict):
                if "source_structure_version" in incoming:
                    if incoming["source_structure_version"] != version:
                        raise EnhancementCommitConflict("stale_interaction_write")
                for key, value in incoming.items():
                    check_updates(previous.get(key) if isinstance(previous, dict) else None, value)
            elif isinstance(incoming, list):
                for value in incoming:
                    check_updates(None, value)
        check_updates(old, new)

    @staticmethod
    def _preserve_navigation_labels(connection, document, document_id):
        """Metadata writers may be delayed; they never own navigation labels."""
        chapters = {str(row["id"]): row for row in connection.execute(
            "SELECT id, title FROM chapters WHERE document_id=%s", (document_id,)).fetchall()}
        sections = {str(row["id"]): row for row in connection.execute(
            "SELECT id, title, container_title FROM sections WHERE document_id=%s", (document_id,)).fetchall()}
        if set(chapters) != {chapter.chapter_id for chapter in document.chapters}:
            raise EnhancementCommitConflict("stale_structure_write")
        return replace(document, chapters=[replace(chapter,
            title=chapters[chapter.chapter_id]["title"],
            sections=[replace(section, title=sections[section.section_id]["title"],
                container_title=sections[section.section_id]["container_title"])
                for section in chapter.sections]) for chapter in document.chapters])

    @staticmethod
    def _update_enhancement_labels(connection, current, candidate, document_id, Jsonb):
        # Comparison has proved positional correspondence on the locked baseline.
        # Patch only labels, retaining unknown embedded metadata and all identities.
        for old_chapter, new_chapter in zip(current.document.chapters, candidate.chapters):
            connection.execute("UPDATE chapters SET title=%s WHERE document_id=%s AND id=%s",
                (new_chapter.title, document_id, int(old_chapter.chapter_id)))
            for old_section, new_section in zip(old_chapter.sections, new_chapter.sections):
                connection.execute("UPDATE sections SET title=%s, container_title=%s "
                    "WHERE document_id=%s AND id=%s", (new_section.title,
                    new_section.container_title, document_id, int(old_section.section_id)))
        row = connection.execute("SELECT structured_document_payload FROM structured_document_snapshots "
            "WHERE document_id=%s AND source_structure_version=%s",
            (document_id, current.structure_version)).fetchone()
        if row is None:
            raise EnhancementCommitConflict("enhancement_snapshot_missing")
        payload = row["structured_document_payload"]
        try:
            snapshot_change = compare_enhancement_documents(
                current.document, StructuredDocument.from_dict(payload),
                canonical_text=candidate.raw_text)
        except (ValueError, TypeError, KeyError) as error:
            raise EnhancementCommitConflict("enhancement_snapshot_mismatch") from error
        if snapshot_change.kind == EnhancementChangeKind.STRUCTURE_CHANGED:
            raise EnhancementCommitConflict("enhancement_snapshot_mismatch")
        chapters = payload.get("chapters", [])
        if len(chapters) != len(candidate.chapters):
            raise EnhancementCommitConflict("enhancement_snapshot_mismatch")
        for chapter, new_chapter in zip(chapters, candidate.chapters):
            if len(chapter.get("sections", [])) != len(new_chapter.sections):
                raise EnhancementCommitConflict("enhancement_snapshot_mismatch")
            chapter["title"] = new_chapter.title
            for section, new_section in zip(chapter["sections"], new_chapter.sections):
                section["title"] = new_section.title
                section["container_title"] = new_section.container_title
        connection.execute("UPDATE structured_document_snapshots SET structured_document_payload=%s "
            "WHERE document_id=%s AND source_structure_version=%s",
            (Jsonb(payload), document_id, current.structure_version))

    @staticmethod
    def _enhancement_guards_available(connection):
        return connection.execute("SELECT count(*) AS count FROM pg_trigger WHERE "
            "(tgrelid='artifacts'::regclass AND tgname='artifacts_current_version' OR "
            "tgrelid='content_blocks'::regclass AND tgname='content_blocks_current_version') "
            "AND tgenabled IN ('O','A')").fetchone()["count"] == 2

    def enhancement_guards_available(self):
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            return self._enhancement_guards_available(connection)

    def profile_structure_invalidated(self, target):
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            row = connection.execute("SELECT metadata_payload FROM documents WHERE namespace=%s "
                "AND document_name=%s", (resolved.namespace, resolved.document_name)).fetchone()
        return bool(row and (row["metadata_payload"] or {}).get("parse_provenance", {}).get("enhancement_source"))

    def backfill_txt_extraction(self, target, *, result, source_path):
        """Explicit, idempotent provenance-only migration; never a read fallback."""
        from document_preparation.extraction_backfill import prove_txt_backfill
        from document_preparation.accepted_source_snapshot import source_file_sha256
        resolved = self._resolve_target(target)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("LOCK TABLE documents, chapters, sections, structured_document_snapshots "
                               "IN SHARE ROW EXCLUSIVE MODE NOWAIT")
            current = self.load_enhancement_snapshot(target, _connection=connection)
            record = prove_txt_backfill(current, result, source_path)
            row = connection.execute("SELECT id, metadata_payload FROM documents WHERE namespace=%s "
                "AND document_name=%s FOR UPDATE", (resolved.namespace, resolved.document_name)).fetchone()
            if current.document.parse_provenance.get("extraction") == record:
                return False
            metadata = row["metadata_payload"] or {}
            metadata["parse_provenance"] = {**metadata.get("parse_provenance", {}), "extraction": record}
            snapshot = connection.execute("SELECT structured_document_payload FROM structured_document_snapshots "
                "WHERE document_id=%s AND source_structure_version=%s FOR UPDATE",
                (row["id"], current.structure_version)).fetchone()
            if snapshot is None:
                raise EnhancementCommitConflict("enhancement_snapshot_missing")
            payload = snapshot["structured_document_payload"]
            payload["parse_provenance"] = {**payload.get("parse_provenance", {}), "extraction": record}
            connection.execute("UPDATE documents SET metadata_payload=%s WHERE id=%s", (Jsonb(metadata), row["id"]))
            connection.execute("UPDATE structured_document_snapshots SET structured_document_payload=%s "
                "WHERE document_id=%s AND source_structure_version=%s",
                (Jsonb(payload), row["id"], current.structure_version))
            if source_file_sha256(source_path) != result.identity.source_sha256:
                raise EnhancementCommitConflict("source_changed")
            return True

    def commit_enhancement(self, target, *, candidate, interaction_revision, source_path):
        """Atomic conditional commit. Caller retains its document execution lease."""
        from document_preparation.accepted_source_snapshot import source_file_sha256
        from document_preparation.structure_enhancement_candidate import validate_enhancement_document
        from doc_loaders.extraction_identity import extraction_matches

        resolved = self._resolve_target(target)
        expected = candidate.source
        if resolved.namespace != expected.namespace or resolved.document_name != expected.document_id:
            raise EnhancementCommitConflict("enhancement_identity_changed")
        validate_enhancement_document(candidate.document, expected)
        if not extraction_matches(candidate.document.parse_provenance, expected.extraction_identity, expected.canonical_text):
            raise EnhancementCommitConflict("enhancement_source_identity_invalid")
        psycopg, dict_row, Jsonb = self._load_psycopg()
        try:
            with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
                # Fixed-order, fail-fast table locks include phantom inserts and
                # writers that do not yet use the per-document advisory lease.
                connection.execute("LOCK TABLE documents, raw_source_metadata, chapters, sections, "
                                   "task_units, content_blocks, artifacts, document_profile, "
                                   "structured_document_snapshots, parse_events "
                                   "IN SHARE ROW EXCLUSIVE MODE NOWAIT")
                if not self._enhancement_guards_available(connection):
                    raise EnhancementCommitConflict("enhancement_storage_guards_unavailable")
                connection.execute("SELECT id FROM documents WHERE namespace=%s AND document_name=%s "
                                   "FOR UPDATE NOWAIT", (resolved.namespace, resolved.document_name))
                current = self.load_enhancement_snapshot(target, _connection=connection)
                if (current.structure_version != expected.structure_version
                        or current.interaction_revision != interaction_revision):
                    raise EnhancementCommitConflict("confirmation_stale")
                if (not current.accepted or not extraction_matches(current.document.parse_provenance,
                        expected.extraction_identity, expected.canonical_text)):
                    raise EnhancementCommitConflict("enhancement_source_identity_invalid")
                if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                    raise EnhancementCommitConflict("source_changed")
                try:
                    change = compare_enhancement_documents(current.document, candidate.document,
                                                           canonical_text=expected.canonical_text)
                except EnhancementComparisonError as error:
                    raise EnhancementCommitConflict("enhancement_baseline_invalid") from error
                if change.kind == EnhancementChangeKind.UNCHANGED:
                    return EnhancementCommitResult(**vars(current), change_kind=change.kind)
                if change.kind == EnhancementChangeKind.TITLES_ONLY:
                    document_id = connection.execute(
                        "SELECT id FROM documents WHERE namespace=%s AND document_name=%s",
                        (resolved.namespace, resolved.document_name)).fetchone()["id"]
                    self._update_enhancement_labels(connection, current, candidate.document, document_id, Jsonb)
                    if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                        raise EnhancementCommitConflict("source_changed")
                    committed = self.load_enhancement_snapshot(target, _connection=connection)
                    return EnhancementCommitResult(**vars(committed), change_kind=change.kind)
                # Build fresh containers: no embedded metadata may survive cleanup.
                provenance_keys = {"requested_parser_mode", "effective_parser_mode", "fallback_used",
                    "fallback_reason", "source", "schema_version", "prompt_instruction_version",
                    "heading_policy_version", "enhancement_intent", "proposed_boundary_count",
                    "resolved_boundary_count", "rejected_boundary_count", "accepted_boundary_count",
                    "source_coverage", "boundary_origins", "common_parse_policy", "toc_hint",
                    "toc_observation", "pdf_extraction", "extraction", "enhancement_source"}
                document = replace(candidate.document, title=current.document.title,
                    parse_provenance={**{key: value for key, value in candidate.document.parse_provenance.items()
                                        if key in provenance_keys},
                        "enhancement_source": {"structure_version": expected.structure_version,
                            "text_sha256": expected.text_sha256, "text_length": expected.text_length}},
                    source_path=current.document.source_path, document_task_artifacts=None,
                    chapters=[replace(chapter, task_artifacts=None,
                        metadata={"title_origin": chapter.metadata.get("title_origin", "implicit")},
                        sections=[replace(section, task_artifacts=None, task_units=[])
                                  for section in chapter.sections]) for chapter in candidate.document.chapters])
                document_id = connection.execute(
                    "SELECT id FROM documents WHERE namespace=%s AND document_name=%s",
                    (resolved.namespace, resolved.document_name)).fetchone()["id"]
                connection.execute("DELETE FROM structured_document_snapshots WHERE document_id=%s", (document_id,))
                self.save(document, target, replace_existing_hierarchy=True,
                          _connection=connection, _preserve_raw_source=True)
                connection.execute("UPDATE document_profile SET profile_payload = "
                    "profile_payload - 'post_structure_metadata' - 'structure_profile', "
                    "advisory_diagnostics=NULL, source_structure_version=NULL WHERE document_id=%s", (document_id,))
                # Recheck bytes after all writes, before commit. External source
                # writers must also honor source ownership; this is not a filesystem transaction.
                if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                    raise EnhancementCommitConflict("source_changed")
                committed = self.load_enhancement_snapshot(target, _connection=connection)
                if committed.structure_version != expected.structure_version + 1:
                    raise EnhancementCommitConflict("enhancement_version_invalid")
            return EnhancementCommitResult(**vars(committed), change_kind=change.kind)
        except psycopg.errors.LockNotAvailable as error:
            raise EnhancementCommitConflict("enhancement_commit_busy") from error

    @contextmanager
    def enhancement_lease(self, target):
        """Cross-process single flight without holding a generation transaction."""
        resolved = self._resolve_target(target)
        psycopg, _, _ = self._load_psycopg()
        key = json.dumps(["enhancement", resolved.namespace, resolved.document_name])
        with psycopg.connect(self._dsn, autocommit=True) as connection:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,),
            ).fetchone()[0]
            if not acquired:
                raise BlockingIOError("enhancement_in_progress")
            try:
                yield
            finally:
                connection.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))

    def delete(
        self,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> bool:
        """Permanently remove one document and its FK-owned rows in one transaction."""
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            with connection.transaction():
                deleted = connection.execute(
                    """
                    DELETE FROM documents
                    WHERE namespace = %s AND document_name = %s
                    RETURNING id
                    """,
                    (resolved.namespace, resolved.document_name),
                ).fetchone()
        return deleted is not None

    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        """List lightweight active structured document candidates from PostgreSQL."""
        bounded_limit = max(1, min(limit, 200))
        normalized_query = (query or "").strip()
        psycopg, dict_row, _ = self._load_psycopg()
        query_filter = f"%{normalized_query}%"
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            if normalized_query:
                rows = connection.execute(
                    """
                    SELECT document_name,
                           metadata_payload
                    FROM documents
                    WHERE namespace = %s
                      AND status = 'active'
                      AND (
                          document_name ILIKE %s
                          OR COALESCE(metadata_payload->>'title', '') ILIKE %s
                      )
                    ORDER BY document_name
                    LIMIT %s
                    """,
                    (self._namespace, query_filter, query_filter, bounded_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT document_name,
                           metadata_payload
                    FROM documents
                    WHERE namespace = %s
                      AND status = 'active'
                    ORDER BY document_name
                    LIMIT %s
                    """,
                    (self._namespace, bounded_limit),
                ).fetchall()
        items: list[DocumentListItem] = []
        for row in rows:
            metadata_payload = self._dict_payload(row.get("metadata_payload"))
            title = metadata_payload.get("title")
            items.append(
                DocumentListItem(
                    doc_name=str(row["document_name"]),
                    title=title if isinstance(title, str) and title.strip() else None,
                    source="structured_postgres",
                    review_required=any(not metadata_payload.get("parse_provenance",{}).get("reading_content",{}).get("regions",{}).get(r["id"])
                        for r in metadata_payload.get("parse_provenance",{}).get("source_review",{}).get("regions",[])),
                )
            )
        return items

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
                            ON CONFLICT (section_id, task_unit_order)
                            DO UPDATE SET
                                reference_unit_id = EXCLUDED.reference_unit_id,
                                title = EXCLUDED.title,
                                container_title = EXCLUDED.container_title,
                                content_payload = EXCLUDED.content_payload,
                                source_section_ids_payload = EXCLUDED.source_section_ids_payload,
                                is_fallback_generated = EXCLUDED.is_fallback_generated,
                                metadata_payload = EXCLUDED.metadata_payload
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
    def _count_rows(
        *,
        connection: Any,
        table_name: str,
        document_id: int,
    ) -> int:
        row = connection.execute(
            f"SELECT COUNT(*) AS row_count FROM {table_name} WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        return 0 if row is None else int(row["row_count"])

    @staticmethod
    def _delete_current_hierarchy_and_derived_rows(
        *,
        connection: Any,
        document_id: int,
    ) -> None:
        for table_name in (
            "artifacts",
            "content_blocks",
            "task_units",
            "sections",
            "chapters",
        ):
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
            parse_provenance=(
                {**self._dict_payload(document_metadata.get("parse_provenance")),
                 **({"storage_structure_version": int(document_row["current_structure_version"])}
                    if document_row.get("current_structure_version") is not None else {})}
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

    @staticmethod
    def _effective_parser_mode(parse_provenance: dict[str, Any]) -> str | None:
        value = parse_provenance.get("effective_parser_mode")
        if isinstance(value, str) and value.strip():
            return value.strip()
        value = parse_provenance.get("requested_parser_mode")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
