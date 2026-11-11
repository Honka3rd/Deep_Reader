from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from document_structure.learning_lifecycle import guard_source_transition
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
    cleanup: dict | None = None


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

    @property
    def dsn(self):
        """Connection identity for composition; never expose in transport/logs."""
        return self._dsn

    @property
    def namespace(self):
        return self._namespace

    def save(
        self,
        document: StructuredDocument,
        target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
        *,
        replace_existing_hierarchy: bool = True,
        _connection=None,
        _preserve_raw_source: bool = False,
        _preserve_reading_state: bool = False,
    ) -> None:
        """Persist the current structured document payload in PostgreSQL.

        ``replace_existing_hierarchy`` is reserved for parser-level replacement
        saves. Repository metadata/task-layout updates preserve the current
        document structure version.
        """
        resolved = self._resolve_target(target)
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
            if not (_preserve_reading_state and _connection is not None):
                metadata_payload["parse_provenance"].pop("reading_content",None)
            metadata_payload["parse_provenance"].pop("reading_projected",None)
        if replace_existing_hierarchy:
            metadata_payload["parse_provenance"]["interaction_epoch"] = uuid4().hex
        current_source_document = None
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
                    if not replace_existing_hierarchy and document.parse_provenance.get("interaction_epoch"):
                        from document_structure.reading_content import ReadingContentConflict
                        raise ReadingContentConflict("reading_document_missing_reload_required")
                    metadata_payload["parse_provenance"]["interaction_epoch"] = uuid4().hex
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
                        previous = self.load_enhancement_snapshot(resolved, _connection=connection).document
                        current_source_document = previous
                        document = preserve_explicit_annotations(previous, document)
                        metadata_payload["parse_provenance"] = document.parse_provenance
                        metadata_payload["document_task_artifacts"] = (
                            document.document_task_artifacts.to_dict() if document.document_task_artifacts else None)
                        self._validate_metadata_write(connection, document, document_id,
                                                      structure_version, existing_row)
                        document = self._preserve_navigation_labels(connection, document, document_id)
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
                persisted_document = self._replace_current_hierarchy(
                    connection=connection,
                    document=document,
                    document_id=document_id,
                    Jsonb=Jsonb,
                )
                from document_structure.source_review_blocks import persist_region_mapping

                # Mapping must see allocated hierarchy/block IDs and the committed epoch.
                persisted_document = persist_region_mapping(replace(
                    persisted_document, document_id=resolved.document_name,
                    parse_provenance={**metadata_payload["parse_provenance"],
                                      "storage_structure_version": structure_version},
                ))
                persisted_document = guard_source_transition(current_source_document, persisted_document)
                metadata_payload["parse_provenance"] = persisted_document.parse_provenance
                payload = self._snapshot_payload(persisted_document)
                connection.execute(
                    "UPDATE documents SET metadata_payload=%s,updated_at=NOW() WHERE id=%s",
                    (Jsonb(metadata_payload), document_id),
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
            updated = guard_source_transition(document, updated)
            if updated != document:
                from document_structure.source_review_blocks import persist_region_mapping

                updated = persist_region_mapping(updated)
                metadata = dict(row["metadata_payload"])
                metadata["parse_provenance"] = updated.parse_provenance
                metadata["document_task_artifacts"] = updated.document_task_artifacts.to_dict() if updated.document_task_artifacts else None
                revision_changed = (updated.parse_provenance.get("reading_content", {}).get("revision", 0)
                                    != document.parse_provenance.get("reading_content", {}).get("revision", 0))
                connection.execute("UPDATE documents SET metadata_payload=%s,"
                                   "updated_at=CASE WHEN %s THEN NOW() ELSE updated_at END WHERE id=%s",
                                   (Jsonb(metadata), revision_changed, row["id"]))
                policy = (updated.parse_provenance.get("reading_content", {}).get("audit") or [{}])[-1]
                reading_changed = revision_changed
                if reading_changed and policy.get("invalidate_generated", True):
                    for table in ("chapters","sections","task_units"):
                        connection.execute(f"UPDATE {table} SET metadata_payload=metadata_payload - 'task_artifacts' WHERE document_id=%s",(row["id"],))
                    connection.execute("DELETE FROM artifacts WHERE document_id=%s AND artifact_type NOT IN ('note','notes')",(row["id"],))
                elif reading_changed and policy.get("summaries_invalidated", False):
                    for table in ("chapters", "sections", "task_units"):
                        connection.execute(f"UPDATE {table} SET metadata_payload=metadata_payload #- '{{task_artifacts,summary}}' "
                                           "WHERE document_id=%s", (row["id"],))
                    connection.execute("DELETE FROM artifacts WHERE document_id=%s AND artifact_type IN "
                                       "('summary','public_summary','ancestor_summary')", (row["id"],))
                cleared = list(dict.fromkeys(policy.get("cleared_block_ids", []) +
                    ([policy["cleared_block_id"]] if policy.get("cleared_block_id") else [])))
                if reading_changed and cleared:
                    connection.execute(
                        "DELETE FROM artifacts a USING content_blocks b WHERE a.document_id=%s "
                        "AND b.document_id=a.document_id AND a.target_type='content_block' AND a.target_id=b.id "
                        "AND COALESCE(b.metadata_payload->>'block_id', "
                        "b.task_unit_id::text || ':content:' || b.block_order::text)=ANY(%s)",
                        (row["id"], cleared))
                connection.execute(
                    "UPDATE structured_document_snapshots SET structured_document_payload=%s "
                    "WHERE document_id=%s AND source_structure_version=%s",
                    (Jsonb(self._snapshot_payload(updated)), row["id"],
                     document.parse_provenance["storage_structure_version"]),
                )
            return result

    def commit_cleanup(self, command):
        """One content-only transaction; no provider, hierarchy save or preparation."""
        from time import perf_counter
        from document_structure.cleanup_commit import stage_cleanup_commit, verify_cleanup_source_file, log_cleanup_commit
        from document_structure.reading_content import ReadingContentConflict
        binding = command.staged.binding
        target = self.target_for_doc_name(binding.document_id)
        if target.namespace != binding.namespace or target.document_name != binding.document_id:
            raise ReadingContentConflict("cleanup_identity_changed")
        psycopg, dict_row, Jsonb = self._load_psycopg()
        started = perf_counter()
        try:
            with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
                row = connection.execute("SELECT id FROM documents WHERE namespace=%s AND document_name=%s FOR UPDATE",
                                         (target.namespace, target.document_name)).fetchone()
                if row is None:
                    raise FileNotFoundError("document_missing")
                current = self.load_enhancement_snapshot(target, _connection=connection)
                if not current.accepted:
                    raise ReadingContentConflict("cleanup_document_not_ready")
                verify_cleanup_source_file(current.document, command.snapshot.source,
                                           accepted_source_path=command.snapshot.accepted_source_path)
                updated, result = stage_cleanup_commit(current.document, command,
                    namespace=target.namespace, structure_version=current.structure_version)
                if not result.replayed:
                    self._commit_cleanup_state(connection, row["id"], updated,
                                               result.outcome.model_dump(mode="json"), Jsonb)
                    verify_cleanup_source_file(current.document, command.snapshot.source,
                                               accepted_source_path=command.snapshot.accepted_source_path)
            log_cleanup_commit(binding, "replayed" if result.replayed else "committed", started)
            return result
        except Exception:
            log_cleanup_commit(binding, "commit_unconfirmed", started)
            raise

    def lookup_cleanup(self, *, namespace, document_id, operation_id):
        from document_structure.cleanup_commit import lookup_cleanup_receipt
        from document_structure.reading_content import ReadingContentConflict
        target = self.target_for_doc_name(document_id)
        if target.namespace != namespace or target.document_name != document_id:
            raise ReadingContentConflict("cleanup_identity_changed")
        try:
            document = self.load(target)
        except FileNotFoundError:
            return None
        return lookup_cleanup_receipt(document, namespace=namespace, document_id=document_id, operation_id=operation_id)

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
        from db import learning_lifecycle
        resolved = self._resolve_target(target)
        psycopg, dict_row, Jsonb = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            preferences = learning_lifecycle.lock_preferences(connection, self._dsn, resolved.namespace, resolved.document_name)
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
            from document_structure.learning_lifecycle import effective_sources
            if (renamed.document_id != new_name or effective_sources(renamed) != effective_sources(document)
                    or renamed.parse_provenance != document.parse_provenance):
                raise ValueError('learning_rename_source_changed')
            learning_lifecycle.rename(connection, row['id'], preferences, resolved.document_name, new_name)
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
        self, target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget,
    ) -> StructuredDocument:
        """Load the current hierarchy in one read-only MVCC snapshot."""
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            return self.load_in_transaction(target, connection=connection)

    def load_in_transaction(
        self, target: str | StructuredDocumentStorageConfig | PostgresStructuredDocumentTarget, *, connection,
    ) -> StructuredDocument:
        """Read canonical hierarchy using the caller's snapshot/locks; never commit.

        Caller must own an active transaction, and lock the document first for
        guarded writes. Backend composition supplies the configured namespace.
        """
        from psycopg.pq import TransactionStatus
        if connection.info.transaction_status != TransactionStatus.INTRANS:
            raise ValueError("document_transaction_required")
        resolved = self._resolve_target(target)
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

    def commit_enhancement(self, target, *, candidate, interaction_revision, source_path, cleanup=None, operation=None):
        """Atomic conditional commit. Caller retains its document execution lease."""
        from document_preparation.accepted_source_snapshot import source_file_sha256
        from document_preparation.structure_enhancement_candidate import validate_enhancement_document
        from doc_loaders.extraction_identity import extraction_matches
        from document_structure.ocr_cleanup_overlay import log_cleanup
        from time import perf_counter
        started = perf_counter()

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
                if operation is not None:
                    prior = current.document.parse_provenance.get("enhancement_receipts", {}).get(operation["id"])
                    if prior is not None:
                        from document_structure.reading_content import state_for
                        if (current.structure_version != prior["structure_version"]
                                or state_for(current.document).get("revision", 0) != prior["reading_revision"]):
                            raise EnhancementCommitConflict("confirmation_stale")
                        if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                            raise EnhancementCommitConflict("source_changed")
                        return EnhancementCommitResult(**vars(current),
                            change_kind=EnhancementChangeKind(prior["change_kind"]), cleanup=prior["cleanup"])
                if cleanup is not None and operation is None:
                    from document_structure.ocr_cleanup_overlay import replay_cleanup
                    receipt = replay_cleanup(current.document, cleanup, candidate.document)
                    if receipt is not None:
                        if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                            raise EnhancementCommitConflict("source_changed")
                        log_cleanup(cleanup, "replayed", started=started, outcome=receipt["outcome"])
                        return EnhancementCommitResult(**vars(current),
                            change_kind=EnhancementChangeKind(receipt["change_kind"]), cleanup=receipt["outcome"])
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
                staged, cleanup_outcome = None, None
                if cleanup is not None:
                    from document_structure.ocr_cleanup_overlay import stage_cleanup
                    staged, cleanup_outcome = stage_cleanup(current.document, candidate.document, cleanup,
                        namespace=resolved.namespace, structure_version=current.structure_version,
                        fingerprint=current.interaction_revision, change_kind=change.kind)
                    if operation is not None and operation.get("cleanup") is not None:
                        cleanup_outcome = operation["cleanup"]
                        staged.parse_provenance["reading_content"]["cleanup_receipts"][
                            cleanup.plan.binding.operation_id]["outcome"] = cleanup_outcome
                if change.kind == EnhancementChangeKind.UNCHANGED and cleanup is None and operation is None:
                    return EnhancementCommitResult(**vars(current), change_kind=change.kind)
                if change.kind in {EnhancementChangeKind.TITLES_ONLY, EnhancementChangeKind.UNCHANGED}:
                    document_id = connection.execute(
                        "SELECT id FROM documents WHERE namespace=%s AND document_name=%s",
                        (resolved.namespace, resolved.document_name)).fetchone()["id"]
                    if change.kind == EnhancementChangeKind.TITLES_ONLY:
                        self._update_enhancement_labels(connection, current, candidate.document, document_id, Jsonb)
                    if staged is not None:
                        self._commit_cleanup_state(connection, document_id, staged, cleanup_outcome, Jsonb)
                    if change.kind == EnhancementChangeKind.TITLES_ONLY:
                        labeled = self.load_enhancement_snapshot(target, _connection=connection)
                        self._commit_source_review_mapping(connection, document_id, labeled, Jsonb)
                    if operation is not None:
                        self._commit_enhancement_receipt(connection, document_id, current, staged or current.document,
                            operation, change.kind, current.structure_version, cleanup_outcome, Jsonb)
                    if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                        raise EnhancementCommitConflict("source_changed")
                    committed = self.load_enhancement_snapshot(target, _connection=connection)
                    if cleanup is not None:
                        log_cleanup(cleanup, "commit_ready", started=started, outcome=cleanup_outcome)
                    return EnhancementCommitResult(**vars(committed), change_kind=change.kind, cleanup=cleanup_outcome)
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
                if staged is not None:
                    document = replace(document, parse_provenance={**document.parse_provenance,
                        **{key: staged.parse_provenance[key] for key in (
                            "reading_content", "source_review", "source_page_spans", "actual_ocr_evidence") if key in staged.parse_provenance}})
                else:
                    document = replace(document, parse_provenance={**document.parse_provenance,
                        **{key: current.document.parse_provenance[key] for key in (
                            "source_review", "source_page_spans", "actual_ocr_evidence")
                           if key in current.document.parse_provenance}})
                document_id = connection.execute(
                    "SELECT id FROM documents WHERE namespace=%s AND document_name=%s",
                    (resolved.namespace, resolved.document_name)).fetchone()["id"]
                connection.execute("DELETE FROM structured_document_snapshots WHERE document_id=%s", (document_id,))
                self.save(document, target, replace_existing_hierarchy=True,
                          _connection=connection, _preserve_raw_source=True, _preserve_reading_state=staged is not None)
                if operation is not None:
                    self._commit_enhancement_receipt(connection, document_id, current, document,
                        operation, change.kind, expected.structure_version + 1, cleanup_outcome, Jsonb)
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
            if cleanup is not None:
                log_cleanup(cleanup, "committed", started=started, outcome=cleanup_outcome)
            return EnhancementCommitResult(**vars(committed), change_kind=change.kind, cleanup=cleanup_outcome)
        except psycopg.errors.LockNotAvailable as error:
            if cleanup is not None:
                log_cleanup(cleanup, "commit_busy", started=started)
            raise EnhancementCommitConflict("enhancement_commit_busy") from error
        except Exception:
            if cleanup is not None:
                log_cleanup(cleanup, "rolled_back", started=started)
            raise

    @staticmethod
    def _commit_source_review_mapping(connection, document_id, current, Jsonb):
        from document_structure.source_review_blocks import persist_region_mapping

        document = current.document
        mapped = persist_region_mapping(document)
        key = "source_review_mapping"
        before = {key: document.parse_provenance[key]} if key in document.parse_provenance else {}
        after = {key: mapped.parse_provenance[key]} if key in mapped.parse_provenance else {}
        if before == after:
            return
        # Patch only mapping evidence; retain cleanup state, receipts and unknown metadata.
        connection.execute("UPDATE documents SET metadata_payload=jsonb_set(metadata_payload, "
            "'{parse_provenance}', ((metadata_payload->'parse_provenance') - 'source_review_mapping') || %s) "
            "WHERE id=%s", (Jsonb(after), document_id))
        connection.execute("UPDATE structured_document_snapshots SET structured_document_payload="
            "jsonb_set(structured_document_payload, '{parse_provenance}', "
            "((structured_document_payload->'parse_provenance') - 'source_review_mapping') || %s) "
            "WHERE document_id=%s AND source_structure_version=%s",
            (Jsonb(after), document_id, current.structure_version))

    def _commit_enhancement_receipt(self, connection, document_id, current, document,
                                    operation, change_kind, version, outcome, Jsonb):
        """Minimal commit evidence shares the hierarchy/overlay transaction."""
        from doc_loaders.extraction_identity import read_extraction_record
        from document_structure.reading_content import state_for
        from document_structure.ocr_cleanup_contract import CleanupOutcome
        import re
        key = operation["id"]
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("invalid_enhancement_operation")
        cleanup = CleanupOutcome.model_validate(outcome or operation["cleanup"]).model_dump(mode="json")
        identity, record = read_extraction_record(document.parse_provenance)
        receipts = dict(current.document.parse_provenance.get("enhancement_receipts", {}))
        receipts[key] = dict(structure_version=version, change_kind=str(change_kind), cleanup=cleanup,
            reading_revision=state_for(document).get("revision", 0),
            section_count=sum(len(chapter.sections) for chapter in document.chapters),
            extraction_identity=identity.to_dict(), text_sha256=record["text_sha256"])
        # Bound metadata history, not OCR/model usage. Evicted outcomes remain unknown.
        receipts = dict(list(receipts.items())[-1000:])
        connection.execute("UPDATE documents SET metadata_payload=jsonb_set(metadata_payload, "
            "'{parse_provenance,enhancement_receipts}', %s) WHERE id=%s", (Jsonb(receipts), document_id))
        connection.execute("UPDATE structured_document_snapshots SET structured_document_payload="
            "jsonb_set(structured_document_payload, '{parse_provenance,enhancement_receipts}', %s) "
            "WHERE document_id=%s", (Jsonb(receipts), document_id))

    def _commit_cleanup_state(self, connection, document_id, staged, outcome, Jsonb):
        """Shared transaction-local cleanup write, never an independent commit."""
        from document_structure.ocr_cleanup_overlay import invalidate_generated
        row = connection.execute("SELECT s.structured_document_payload, s.source_structure_version "
                                 "FROM structured_document_snapshots s JOIN documents d ON d.id=s.document_id "
                                 "AND d.current_structure_version=s.source_structure_version WHERE s.document_id=%s", (document_id,)).fetchone()
        if row is None:
            raise EnhancementCommitConflict("enhancement_snapshot_missing")
        snapshot = StructuredDocument.from_dict(row["structured_document_payload"])
        from document_structure.cleanup_history import bind_cleanup_record_links
        before = snapshot
        snapshot = bind_cleanup_record_links(replace(snapshot, parse_provenance={**snapshot.parse_provenance,
            "reading_content": staged.parse_provenance["reading_content"]}))
        snapshot = guard_source_transition(before, snapshot)
        state = snapshot.parse_provenance["reading_content"]
        connection.execute("UPDATE documents SET metadata_payload=jsonb_set(metadata_payload, "
            "'{parse_provenance,reading_content}', %s),updated_at=NOW() WHERE id=%s",
            (Jsonb(state), document_id))
        if outcome["reading_content_changed"]:
            snapshot = replace(invalidate_generated(snapshot), document_task_artifacts=staged.document_task_artifacts)
            connection.execute("UPDATE documents SET metadata_payload=jsonb_set(metadata_payload, "
                "'{document_task_artifacts}', %s) WHERE id=%s",
                (Jsonb(staged.document_task_artifacts.to_dict() if staged.document_task_artifacts else None), document_id))
            for table in ("chapters", "sections", "task_units"):
                connection.execute(f"UPDATE {table} SET metadata_payload=metadata_payload - 'task_artifacts' "
                                   "WHERE document_id=%s", (document_id,))
            connection.execute("DELETE FROM artifacts WHERE document_id=%s AND artifact_type NOT IN ('note','notes')",
                               (document_id,))
        connection.execute("UPDATE structured_document_snapshots SET structured_document_payload=%s "
                           "WHERE document_id=%s AND source_structure_version=%s",
                           (Jsonb(self._snapshot_payload(snapshot)), document_id, row["source_structure_version"]))

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
        from db import learning_lifecycle
        resolved = self._resolve_target(target)
        psycopg, dict_row, _ = self._load_psycopg()
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            with connection.transaction():
                preferences = learning_lifecycle.lock_preferences(connection, self._dsn, resolved.namespace, resolved.document_name)
                deleted = connection.execute(
                    """
                    DELETE FROM documents
                    WHERE namespace = %s AND document_name = %s
                    RETURNING id
                    """,
                    (resolved.namespace, resolved.document_name),
                ).fetchone()
                if deleted is not None:
                    learning_lifecycle.delete_preferences(connection, preferences, resolved.document_name)
        return deleted is not None

    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        """List lightweight active structured document candidates from PostgreSQL."""
        from document_structure.source_review_blocks import needs_review_from_provenance

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
                    review_required=needs_review_from_provenance(
                        metadata_payload.get("parse_provenance", {})),
                )
            )
        return items

    @staticmethod
    def _snapshot_payload(document):
        """Keep explicit block identities alongside their durable mapping evidence."""
        payload = document.to_dict()
        for chapter, chapter_data in zip(document.chapters, payload["chapters"]):
            for section, section_data in zip(chapter.sections, chapter_data["sections"]):
                section_data["task_units"] = [
                    unit.to_dict(include_content_blocks=True) for unit in section.task_units]
        return payload

    def _replace_current_hierarchy(
        self,
        *,
        connection: Any,
        document: StructuredDocument,
        document_id: int,
        Jsonb: Any,
    ) -> StructuredDocument:
        kept_chapter_ids: list[int] = []
        kept_section_ids: list[int] = []
        kept_task_unit_ids: list[int] = []
        persisted_chapters = []

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
            persisted_sections = []

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
                persisted_units = []

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
                    if str(task_unit_id) != task_unit.unit_id and task_unit.content_blocks:
                        task_unit_metadata["content_blocks"] = [block.to_dict() for block in
                            task_unit.with_unit_id(str(task_unit_id)).content_blocks]
                        connection.execute("UPDATE task_units SET metadata_payload=%s WHERE id=%s AND document_id=%s",
                                           (Jsonb(task_unit_metadata), task_unit_id, document_id))
                    kept_task_unit_ids.append(task_unit_id)
                    persisted_units.append(replace(
                        task_unit.with_unit_id(str(task_unit_id)),
                        parent_section_id=str(section_id), source_section_ids=[str(section_id)],
                    ))
                persisted_sections.append(replace(
                    section, section_id=str(section_id), parent_chapter_id=str(chapter_id),
                    task_units=persisted_units,
                ))
            persisted_chapters.append(replace(
                chapter, chapter_id=str(chapter_id), sections=persisted_sections,
            ))

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
        return replace(document, chapters=persisted_chapters, sections=[], structure_nodes=[])

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
