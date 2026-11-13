import os
import fcntl
from contextlib import contextmanager, ExitStack
import tempfile
from collections.abc import Callable
from dataclasses import replace
import json
from uuid import uuid4
from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper
from shared.document_name import document_name
from document_structure.generated_artifact_source import (
    require_generation_source, require_generated_fields, advance_rename_epoch,
)
from document_structure.document_artifact_repository import (
    DocumentArtifactRepository,
    DocumentListItem,
    preserve_explicit_annotations,
)
from document_structure.document_hierarchy_index import (
    build_section_index_from_chapters,
    flatten_sections_from_chapters,
    get_effective_sections,
    is_severe_hierarchy_warning,
    validate_chapter_hierarchy_consistency,
    with_sections_replaced_in_hierarchy,
)
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_document_store import StructuredDocumentStore
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit


class StructuredDocumentArtifactRepository(DocumentArtifactRepository):
    """File-based repository for task-artifact persistence inside structured JSON."""

    _NAMESPACE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

    def __init__(
        self,
        store: StructuredDocumentStore | None = None,
        base_dir: str = "data/structured",
        namespace: str = "default",
        raw_dir: str | None = None,
    ):
        self.store = store or StructuredDocumentStore()
        self.base_dir = Path(base_dir)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else self.base_dir.parent / "raw"
        self._cleanup_namespace = namespace
        self._learning_lifecycle = None

    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        """List structured document files as lightweight discovery candidates."""
        from document_structure.source_review_blocks import needs_review_from_provenance

        bounded_limit = max(1, min(limit, 200))
        normalized_query = (query or "").strip().casefold()
        if not self.base_dir.exists():
            return []

        candidates: list[DocumentListItem] = []
        for path in sorted(self.base_dir.glob("*.structured.json")):
            doc_name = path.name[: -len(".structured.json")]
            try:
                document_name(doc_name)
                self._checked_path(path)
            except ValueError:
                continue
            title = self._read_document_title(path)
            searchable = " ".join(part for part in [doc_name, title] if part).casefold()
            if normalized_query and normalized_query not in searchable:
                continue
            try:
                provenance = json.loads(path.read_text()).get("parse_provenance",{})
                review_required = needs_review_from_provenance(provenance)
            except (ValueError,OSError,TypeError,KeyError,AttributeError):
                review_required = False
            candidates.append(
                DocumentListItem(
                    doc_name=doc_name,
                    title=title,
                    source="structured_file",
                    review_required=review_required,
                )
            )
            if len(candidates) >= bounded_limit:
                break
        return candidates

    def delete_document(self, doc_name: str) -> bool:
        """Delete the structured artifact without requiring valid hierarchy content."""
        doc_name = self.resolve_document_name(doc_name)
        if self._learning_lifecycle is not None:
            return self._learning_lifecycle.delete(doc_name)
        with self._artifact_lock(doc_name):
            try:
                self._resolve_document_path(doc_name).unlink()
            except FileNotFoundError:
                return False
            return True

    @contextmanager
    def _artifact_lock(self, doc_name, *, exact=False):
        artifact = self._exact_document_path(doc_name) if exact else self._resolve_document_path(doc_name)
        path = self._checked_path(artifact.with_suffix(".lock"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def namespace(self) -> str:
        """Server-bound file namespace; not a caller-selected ownership scope."""
        return self._cleanup_namespace

    @contextmanager
    def locked_document(self, doc_name: str):
        """Internal write coordination: hold the existing source lock, never save.

        Lock order for learning writes: preferences -> document -> learning file.
        Do not call document mutation methods while holding this context.
        Ordinary reads must use load_document (no lock-file creation).
        """
        from document_structure.reading_content import ReadingContentService
        ReadingContentService.validate_name(doc_name)
        with self._artifact_lock(doc_name):
            document = self._load_document_for_write(doc_name)
            if document.document_id != doc_name:
                raise ValueError("learning_document_identity_mismatch")
            yield document

    def install_learning_lifecycle(self, lifecycle):
        """Explicit composition; no filesystem writes or hidden reads."""
        if self._learning_lifecycle is not None and self._learning_lifecycle is not lifecycle:
            raise ValueError('learning_lifecycle_already_installed')
        self._learning_lifecycle = lifecycle

    def lifecycle_path(self, doc_name):
        from reading_preferences.models import identifier
        identifier(doc_name)
        # Lifecycle journals contain canonical identities, never dynamic aliases.
        return self._exact_document_path(doc_name)

    def lifecycle_lock(self, doc_name):
        return self._artifact_lock(doc_name, exact=True)

    @staticmethod
    def lifecycle_payload(document):
        data = document.to_dict()
        for chapter, chapter_data in zip(document.chapters, data['chapters']):
            for section, section_data in zip(chapter.sections, chapter_data['sections']):
                section_data['task_units'] = [unit.to_dict(include_content_blocks=True) for unit in section.task_units]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def mutate_document_artifacts(self, doc_name, mutation):
        with self._artifact_lock(doc_name):
            document = self._load_document_for_write(doc_name)
            artifacts, result = mutation(document)
            if artifacts != document.document_task_artifacts:
                self._atomic_save(replace(document, document_task_artifacts=artifacts),
                                  self._resolve_document_path(doc_name))
            return result

    def mutate_reading_content(self, doc_name, mutation):
        with self._artifact_lock(doc_name):
            document = self._load_document_for_write(doc_name)
            updated,result = mutation(document)
            if updated != document:
                self._atomic_save(updated,self._resolve_document_path(doc_name))
            return result

    def commit_cleanup_enhancement(self, doc_name, *, candidate, cleanup, source_path):
        """Internal file-store equivalent: one locked atomic hierarchy/overlay replace."""
        from document_preparation.accepted_source_snapshot import source_file_sha256
        from document_preparation.structure_enhancement_candidate import validate_enhancement_document
        from document_structure.enhancement_change import compare_enhancement_documents
        from document_structure.ocr_cleanup_overlay import (
            CleanupFileCommitResult, canonical_text, replay_cleanup, stage_cleanup, log_cleanup,
        )
        from document_structure.reading_content import digest, ReadingContentConflict
        from doc_loaders.extraction_identity import extraction_matches
        from time import perf_counter
        started = perf_counter()
        validate_enhancement_document(candidate.document, candidate.source)
        with self._artifact_lock(doc_name):
            current = self._load_document_for_write(doc_name)
            expected = candidate.source
            if (doc_name != expected.document_id or current.document_id != expected.document_id
                    or source_file_sha256(source_path) != expected.extraction_identity.source_sha256
                    or not extraction_matches(current.parse_provenance, expected.extraction_identity,
                                              expected.canonical_text)):
                raise ReadingContentConflict("cleanup_source_changed")
            receipt = replay_cleanup(current, cleanup, candidate.document)
            if receipt is not None:
                log_cleanup(cleanup, "replayed", started=started, outcome=receipt["outcome"])
                return CleanupFileCommitResult(current, receipt["change_kind"], receipt["outcome"])
            version = current.parse_provenance.get("storage_structure_version", 1)
            change = compare_enhancement_documents(current, candidate.document,
                                                   canonical_text=canonical_text(current))
            staged, outcome = stage_cleanup(current, candidate.document, cleanup,
                namespace=expected.namespace, structure_version=version,
                fingerprint=digest(current.to_dict()), change_kind=change.kind)
            if change.kind == "structure_changed":
                document = self.cleanup_reparsed_document_derived_artifacts(candidate.document)
                document = replace(document, chapters=[replace(c, sections=[replace(s, task_units=[])
                    for s in c.sections]) for c in document.chapters], parse_provenance={
                        **document.parse_provenance, "storage_structure_version": version + 1,
                        "interaction_epoch": uuid4().hex})
            else:
                document = replace(current, chapters=[replace(old, title=new.title, sections=[
                    replace(old_s, title=new_s.title, container_title=new_s.container_title)
                    for old_s, new_s in zip(old.sections, new.sections)])
                    for old, new in zip(current.chapters, candidate.document.chapters)])
            document = replace(document, parse_provenance={**document.parse_provenance,
                **{key: staged.parse_provenance[key] for key in (
                    "reading_content", "source_review", "source_page_spans", "actual_ocr_evidence") if key in staged.parse_provenance}})
            if outcome["reading_content_changed"]:
                from document_structure.ocr_cleanup_overlay import invalidate_generated
                from document_structure.reading_content_artifacts import clear_source_dependents
                from document_structure.reading_refresh import changed_blocks
                document = clear_source_dependents(document, changed_blocks(current, document))
                document = invalidate_generated(document)
            if source_file_sha256(source_path) != expected.extraction_identity.source_sha256:
                raise ReadingContentConflict("cleanup_source_changed")
            try:
                document = self._atomic_save(document, self._resolve_document_path(doc_name))
            except Exception:
                log_cleanup(cleanup, "commit_failed", started=started)
                raise
            log_cleanup(cleanup, "committed", started=started, outcome=outcome)
            return CleanupFileCommitResult(document, str(change.kind), outcome)

    def commit_cleanup(self, command):
        """Content-only mutation; the configured base directory owns one namespace."""
        from time import perf_counter
        from document_structure.cleanup_commit import stage_cleanup_commit, verify_cleanup_source_file, log_cleanup_commit
        from document_structure.reading_content import ReadingContentService, ReadingContentConflict
        binding = command.staged.binding
        ReadingContentService.validate_name(binding.document_id)
        if binding.namespace != self._cleanup_namespace:
            raise ReadingContentConflict("cleanup_identity_changed")
        started = perf_counter()
        try:
            with self._artifact_lock(binding.document_id):
                current = self._load_document_for_write(binding.document_id)
                verify_cleanup_source_file(current, command.snapshot.source,
                                           accepted_source_path=command.snapshot.accepted_source_path)
                updated, result = stage_cleanup_commit(current, command, namespace=self._cleanup_namespace,
                    structure_version=current.parse_provenance.get("storage_structure_version", 1))
                if not result.replayed:
                    verify_cleanup_source_file(current, command.snapshot.source,
                                               accepted_source_path=command.snapshot.accepted_source_path)
                    self._atomic_save(updated, self._resolve_document_path(binding.document_id))
            log_cleanup_commit(binding, "replayed" if result.replayed else "committed", started)
            return result
        except Exception:
            log_cleanup_commit(binding, "commit_unconfirmed", started)
            raise

    def lookup_cleanup(self, *, namespace, document_id, operation_id):
        from document_structure.cleanup_commit import lookup_cleanup_receipt
        from document_structure.reading_content import ReadingContentService, ReadingContentConflict
        ReadingContentService.validate_name(document_id)
        if namespace != self._cleanup_namespace:
            raise ReadingContentConflict("cleanup_identity_changed")
        try:
            document = self.load_document(document_id)
        except FileNotFoundError:
            return None
        return lookup_cleanup_receipt(document, namespace=namespace, document_id=document_id, operation_id=operation_id)

    @property
    def lifecycle_storage_identity(self):
        return ('file', self.namespace, str(Path(self.base_dir).resolve()))

    def recover_document_rename(self, doc_name, new_name):
        if self._learning_lifecycle is not None:
            return self._learning_lifecycle.recover(expected_rename=(doc_name, new_name))
        return False

    def rename_document(self, doc_name, new_name, transform):
        doc_name = self.resolve_document_name(doc_name)
        new_name = document_name(new_name)
        if self._learning_lifecycle is not None:
            return self._learning_lifecycle.rename(doc_name, new_name, transform)
        with ExitStack() as stack:
            for name in sorted({doc_name, new_name}):
                stack.enter_context(self._artifact_lock(name, exact=True))
            destination = self._exact_document_path(new_name)
            if destination.exists():
                raise ValueError("conflict")
            original = self._exact_document_path(doc_name)
            if not original.exists():
                return False
            document = advance_rename_epoch(transform(self.load_document(doc_name), new_name))
            self._atomic_save(document, destination)
            try:
                original.unlink()
            except OSError:
                destination.unlink()
                raise
            return True

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name."""
        path = self._resolve_document_path(doc_name)
        if self._learning_lifecycle is not None:
            self._learning_lifecycle.assert_ready()
        document = self.store.load(str(path))
        if self._learning_lifecycle is not None:
            self._learning_lifecycle.assert_ready()
        return document

    def _load_document_for_legacy_migration(self, doc_name: str) -> StructuredDocument:
        """Compatibility-only loader for explicit legacy JSON migration flows."""
        path = self._resolve_document_path(doc_name)
        payload = path.read_text(encoding="utf-8")
        return StructuredDocument.from_legacy_json_for_migration(payload)

    def _load_document_for_write(self, doc_name: str) -> StructuredDocument:
        """Load one document for write with strict hierarchy requirement."""
        document = self.load_document(doc_name)
        if document.chapters:
            return document
        raise ValueError(
            f"load_document_for_write: document '{doc_name}' missing hierarchy chapters; "
            "legacy sections-only document requires explicit migration"
        )

    def save_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist structured document with atomic write."""
        resolved_doc_name = self._write_document_name(document, doc_name)
        path = self._exact_document_path(resolved_doc_name)
        with self._artifact_lock(resolved_doc_name, exact=True):
            if path.exists():
                current = self.load_document(resolved_doc_name)
                version = current.parse_provenance.get("storage_structure_version")
                if version is not None and document.parse_provenance.get("storage_structure_version") != version:
                    from document_structure.reading_content import ReadingContentConflict
                    raise ReadingContentConflict("reading_structure_changed_reload_required")
                document = preserve_explicit_annotations(current, document)
            else:
                if document.parse_provenance.get("interaction_epoch"):
                    from document_structure.reading_content import ReadingContentConflict
                    raise ReadingContentConflict("reading_document_missing_reload_required")
                document = replace(document, parse_provenance={**document.parse_provenance,
                                   "interaction_epoch": uuid4().hex})
            self._atomic_save(document=document, path=path)

    def commit_manual_structure(self, document, *, doc_name, validate_current):
        from document_structure.manual_structure_commit import prepare_manual_replacement

        with self._artifact_lock(doc_name):
            current = self._load_document_for_write(doc_name)
            replacement = prepare_manual_replacement(current, document, validate_current)
            replacement = self.cleanup_reparsed_document_derived_artifacts(replacement)
            replacement = replace(replacement, parse_provenance={
                **replacement.parse_provenance, "interaction_epoch": uuid4().hex,
                "storage_structure_version": current.parse_provenance.get("storage_structure_version", 1) + 1})
            return self._atomic_save(replacement, self._resolve_document_path(doc_name),
                                     before_publish=lambda: validate_current(current))

    def save_reparsed_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist a parser-level replacement after clearing derived artifacts."""
        cleaned_document = self.cleanup_reparsed_document_derived_artifacts(document)
        cleaned_document = replace(cleaned_document, parse_provenance={
            **cleaned_document.parse_provenance, "interaction_epoch": uuid4().hex})
        resolved_doc_name = self._write_document_name(document, doc_name)
        with self._artifact_lock(resolved_doc_name, exact=True):
            self._atomic_save(cleaned_document, self._exact_document_path(resolved_doc_name))

    def _mutate_hierarchy(self, doc_name, mutation):
        """Read, merge and save under the source lock; callbacks do no provider IO."""
        with self._artifact_lock(doc_name):
            current = self._load_document_for_write(doc_name)
            updated = preserve_explicit_annotations(current, mutation(current))
            return self._atomic_save(updated, self._resolve_document_path(doc_name))

    @staticmethod
    def _merge_task_artifacts(current, incoming):
        current = current or TaskArtifacts()
        return TaskArtifacts(
            summary=incoming.summary if incoming.summary is not None else current.summary,
            quiz=incoming.quiz if incoming.quiz is not None else current.quiz,
        )

    def _layout_units_preserving_artifacts(self, section, *, task_units, section_id):
        current = {unit.unit_id: unit for unit in section.task_units}
        return self._with_task_units_parent_section(
            task_units=[replace(unit, task_artifacts=(
                current[unit.unit_id].task_artifacts
                if unit.unit_id in current and current[unit.unit_id].content == unit.content
                else None)) for unit in task_units], section_id=section_id)

    def update_section_artifacts(
        self,
        doc_name: str,
        section_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one section artifact payload and persist new structured document."""
        def mutation(document):
            require_generated_fields(document, artifacts, "section", section_id)
            updated_document = self._update_section_by_id(
                document=document,
                section_id=section_id,
                update_fn=lambda section: replace(section, task_artifacts=self._merge_task_artifacts(section.task_artifacts, artifacts)),
                context="update_section_artifacts",
                doc_name=doc_name,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_section_summary_artifact(
        self,
        doc_name: str,
        section_id: str,
        summary: SummaryArtifact,
    ) -> StructuredDocument:
        """Update one section summary artifact while preserving existing section quiz artifact."""
        def mutation(document):
            require_generation_source(document, summary, "section", section_id)
            def _update_fn(section: StructuredSection) -> StructuredSection:
                existing_artifacts = section.task_artifacts
                merged_artifacts = TaskArtifacts(
                    summary=summary,
                    quiz=(None if existing_artifacts is None else existing_artifacts.quiz),
                )
                return replace(
                    section,
                    task_artifacts=merged_artifacts,
                )

            updated_document = self._update_section_by_id(
                document=document,
                section_id=section_id,
                update_fn=_update_fn,
                context="update_section_summary_artifact",
                doc_name=doc_name,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_section_quiz_artifact(
        self,
        doc_name: str,
        section_id: str,
        quiz: QuizArtifact,
    ) -> StructuredDocument:
        """Update one section quiz artifact while preserving existing section summary artifact."""
        def mutation(document):
            require_generation_source(document, quiz, "section", section_id)
            def _update_fn(section: StructuredSection) -> StructuredSection:
                existing_artifacts = section.task_artifacts
                merged_artifacts = TaskArtifacts(
                    summary=(None if existing_artifacts is None else existing_artifacts.summary),
                    quiz=quiz,
                )
                return replace(
                    section,
                    task_artifacts=merged_artifacts,
                )

            updated_document = self._update_section_by_id(
                document=document,
                section_id=section_id,
                update_fn=_update_fn,
                context="update_section_quiz_artifact",
                doc_name=doc_name,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_chapter_summary_artifact(
        self,
        doc_name: str,
        chapter_key: str,
        summary: SummaryArtifact,
    ) -> StructuredDocument:
        """Update one chapter summary artifact in document-level chapter_artifacts map."""
        normalized_chapter_key = chapter_key.strip()
        if not normalized_chapter_key:
            raise ValueError("update_chapter_summary_artifact: chapter_key cannot be empty")

        def mutation(document):
            require_generation_source(document, summary, "chapter", normalized_chapter_key.removeprefix("chapter_id::"))
            if normalized_chapter_key not in {
                "chapter_id::" + chapter.chapter_id for chapter in document.chapters
            }:
                raise ValueError("unknown_chapter_artifact_target")
            existing_document_artifacts = (
                document.document_task_artifacts or DocumentTaskArtifacts()
            )
            existing_chapter_artifacts = dict(existing_document_artifacts.chapter_artifacts)
            existing_entry = existing_chapter_artifacts.get(normalized_chapter_key)
            existing_quiz = None if existing_entry is None else existing_entry.quiz

            existing_chapter_artifacts[normalized_chapter_key] = TaskArtifacts(
                summary=summary,
                quiz=existing_quiz,
            )
            updated_document_artifacts = replace(
                existing_document_artifacts,
                chapter_artifacts=existing_chapter_artifacts,
            )
            updated_document = replace(
                document,
                document_task_artifacts=updated_document_artifacts,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_chapter_quiz_artifact(
        self,
        doc_name: str,
        chapter_key: str,
        quiz: QuizArtifact,
    ) -> StructuredDocument:
        """Update one chapter quiz artifact in document-level chapter_artifacts map."""
        normalized_chapter_key = chapter_key.strip()
        if not normalized_chapter_key:
            raise ValueError("update_chapter_quiz_artifact: chapter_key cannot be empty")

        def mutation(document):
            require_generation_source(document, quiz, "chapter", normalized_chapter_key.removeprefix("chapter_id::"))
            if normalized_chapter_key not in {
                "chapter_id::" + chapter.chapter_id for chapter in document.chapters
            }:
                raise ValueError("unknown_chapter_artifact_target")
            existing_document_artifacts = (
                document.document_task_artifacts or DocumentTaskArtifacts()
            )
            existing_chapter_artifacts = dict(existing_document_artifacts.chapter_artifacts)
            existing_entry = existing_chapter_artifacts.get(normalized_chapter_key)
            existing_summary = None if existing_entry is None else existing_entry.summary

            existing_chapter_artifacts[normalized_chapter_key] = TaskArtifacts(
                summary=existing_summary,
                quiz=quiz,
            )
            updated_document_artifacts = replace(
                existing_document_artifacts,
                chapter_artifacts=existing_chapter_artifacts,
            )
            updated_document = replace(
                document,
                document_task_artifacts=updated_document_artifacts,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_task_unit_artifacts(
        self,
        doc_name: str,
        task_unit_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one persisted task-unit artifact payload and persist document."""
        def mutation(document):
            section_source = flatten_sections_from_chapters(document)
            matching_paths: list[tuple[str, int]] = []
            for section in section_source:
                for task_unit_index, task_unit in enumerate(section.task_units):
                    if task_unit.unit_id == task_unit_id:
                        matching_paths.append((section.section_id, task_unit_index))

            if not matching_paths:
                raise ValueError(
                    f"update_task_unit_artifacts: unknown task_unit_id='{task_unit_id}' for doc_name='{doc_name}'"
                )
            if len(matching_paths) > 1:
                raise ValueError(
                    "update_task_unit_artifacts: duplicate task_unit_id detected "
                    f"for doc_name='{doc_name}' task_unit_id='{task_unit_id}' "
                    f"match_count={len(matching_paths)}"
                )

            require_generated_fields(document, artifacts, "task_unit", task_unit_id)
            target_section_id, target_task_unit_index = matching_paths[0]

            def _update_fn(section: StructuredSection) -> StructuredSection:
                updated_task_units = list(section.task_units)
                updated_task_units[target_task_unit_index] = replace(
                    updated_task_units[target_task_unit_index],
                    task_artifacts=self._merge_task_artifacts(updated_task_units[target_task_unit_index].task_artifacts, artifacts),
                    parent_section_id=section.section_id,
                )
                return replace(
                    section,
                    task_units=updated_task_units,
                )

            updated_document = self._update_section_by_id(
                document=document,
                section_id=target_section_id,
                update_fn=_update_fn,
                context="update_task_unit_artifacts",
                doc_name=doc_name,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_section_task_units(
        self,
        doc_name: str,
        section_id: str,
        task_units: list[TaskUnit],
    ) -> StructuredDocument:
        """Replace one section task-unit list and persist document."""
        def mutation(document):
            updated_document = self._update_section_by_id(
                document=document,
                section_id=section_id,
                update_fn=lambda section: replace(
                    section,
                    task_units=self._layout_units_preserving_artifacts(section,
                        task_units=task_units,
                        section_id=section.section_id,
                    ),
                ),
                context="update_section_task_units",
                doc_name=doc_name,
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_task_layout(
        self,
        doc_name: str,
        task_units_by_section_id: dict[str, list[TaskUnit]],
        task_layout_metadata: dict[str, str | int | None],
    ) -> StructuredDocument:
        """Bulk update section task units and task-layout metadata in one save."""
        def mutation(document):
            from document_structure.reading_content import state_for, ReadingContentConflict
            if task_layout_metadata.get("reading_revision", 0) != state_for(document).get("revision", 0):
                raise ReadingContentConflict("reading_revision_changed")
            sections = get_effective_sections(document)
            if set(task_units_by_section_id) - {section.section_id for section in sections}:
                raise ValueError("unknown_task_layout_section")
            updated_effective_sections = []
            for section in sections:
                section_units = task_units_by_section_id.get(section.section_id, section.task_units)
                updated_effective_sections.append(
                    replace(
                        section,
                        task_units=self._layout_units_preserving_artifacts(section,
                            task_units=section_units,
                            section_id=section.section_id,
                        ),
                    )
                )
            updated_sections_by_id = {
                section.section_id: section
                for section in updated_effective_sections
            }
            document_with_synced_sections = with_sections_replaced_in_hierarchy(
                document=document,
                sections_by_id=updated_sections_by_id,
            )

            existing_document_artifacts = (
                document_with_synced_sections.document_task_artifacts or DocumentTaskArtifacts()
            )
            metadata = dict(existing_document_artifacts.metadata)
            metadata["task_layout"] = dict(task_layout_metadata)
            updated_document_artifacts = replace(
                existing_document_artifacts,
                metadata=metadata,
            )

            updated_document = replace(
                document_with_synced_sections,
                sections=[],
                document_task_artifacts=updated_document_artifacts,
            )
            if updated_document.chapters:
                warnings = validate_chapter_hierarchy_consistency(updated_document)
                severe_warnings = [
                    warning
                    for warning in warnings
                    if is_severe_hierarchy_warning(warning)
                ]
                if severe_warnings:
                    raise ValueError(
                        "update_task_layout: severe hierarchy inconsistency detected "
                        f"for doc_name='{doc_name}': {' | '.join(severe_warnings)}"
                    )
            self._assert_unique_task_unit_ids(
                document=updated_document,
                context=f"update_task_layout doc_name='{doc_name}'",
            )
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def update_document_artifacts(
        self,
        doc_name: str,
        artifacts: DocumentTaskArtifacts,
    ) -> StructuredDocument:
        """Update document-level task-artifact payload and persist."""
        def mutation(document):
            current = document.document_task_artifacts or DocumentTaskArtifacts()
            chapters = dict(current.chapter_artifacts)
            for key, entry in artifacts.chapter_artifacts.items():
                require_generated_fields(document, entry, "chapter", key.removeprefix("chapter_id::"))
                chapters[key] = self._merge_task_artifacts(chapters.get(key), entry)
            updated_document = replace(document, document_task_artifacts=DocumentTaskArtifacts(
                chapter_artifacts=chapters, metadata={**current.metadata, **artifacts.metadata}))
            return updated_document

        return self._mutate_hierarchy(doc_name, mutation)

    def _write_document_name(self, document, doc_name):
        name = document.document_id if doc_name is None else document_name(doc_name)
        # New imports retain their explicit identity. Existing snapshots can be
        # saved through an alias only when it resolves back to that snapshot.
        if name != document.document_id:
            resolved = self.resolve_document_name(name)
            if resolved == document.document_id:
                return resolved
        return document_name(name)

    def _exact_document_path(self, doc_name):
        return self._checked_path(self.base_dir / f"{document_name(doc_name)}.structured.json")

    def contains_document(self, doc_name: str) -> bool:
        return self._exact_document_path(doc_name).is_file()

    def resolve_document_name(self, doc_name: str) -> str:
        return StorageNamespaceHelper.file_identity(document_name(doc_name), self.base_dir,
                                                   suffix=".structured.json", raw_root=self.raw_dir)

    def _resolve_document_path(self, doc_name: str) -> Path:
        name = self.resolve_document_name(doc_name)
        return self._checked_path(self.base_dir / f"{name}.structured.json")

    def _checked_path(self, path: Path) -> Path:
        """Reject pre-existing links outside the configured root before any IO."""
        try:
            resolved = path.resolve()
            root = self.base_dir.resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError("invalid_document_path") from error
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError("invalid_document_path")
        return path

    @staticmethod
    def _read_document_title(path: Path) -> str | None:
        """Read only lightweight title metadata from structured JSON."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        title = payload.get("title")
        return title.strip() if isinstance(title, str) and title.strip() else None

    def _atomic_save(self, document: StructuredDocument, path: Path, *, before_publish=None) -> StructuredDocument:
        """Persist document atomically through temp-file + replace."""
        self._checked_path(path)
        from document_structure.source_review_blocks import persist_region_mapping

        from document_structure.learning_lifecycle import guard_source_transition
        before = self.store.load(str(path)) if path.exists() else None
        document = guard_source_transition(before, document)
        document = persist_region_mapping(document)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._learning_lifecycle is not None:
            self._learning_lifecycle.assert_ready()
        payload = self.lifecycle_payload(document)
        temp_file = None
        temp_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            )
            temp_path = Path(temp_file.name)
            temp_file.write(payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            temp_file = None
            if before_publish is not None:
                before_publish()
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_file is not None and not temp_file.closed:
                temp_file.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return document

    @staticmethod
    def _assert_unique_task_unit_ids(
        *,
        document: StructuredDocument,
        context: str,
    ) -> None:
        """Defensive check: persisted task-unit ids must be document-scope unique."""
        counts: dict[str, int] = {}
        for section in get_effective_sections(document):
            for task_unit in section.task_units:
                counts[task_unit.unit_id] = counts.get(task_unit.unit_id, 0) + 1
        duplicates = {
            unit_id: count
            for unit_id, count in counts.items()
            if count > 1
        }
        if duplicates:
            duplicate_repr = ", ".join(
                f"{unit_id}:{count}"
                for unit_id, count in sorted(duplicates.items())
            )
            raise ValueError(
                f"{context}: duplicate task_unit_id detected -> {duplicate_repr}"
            )

    @classmethod
    def cleanup_reparsed_document_derived_artifacts(
        cls,
        document: StructuredDocument,
    ) -> StructuredDocument:
        """Remove hierarchy-derived artifacts before hard-reparse replacement saves."""
        cleaned_chapters = [
            cls._cleanup_reparsed_chapter_derived_artifacts(chapter)
            for chapter in document.chapters
        ]
        return replace(
            document,
            sections=[],
            chapters=cleaned_chapters,
            document_task_artifacts=None,
            parse_provenance={k:v for k,v in document.parse_provenance.items() if k not in {"reading_content","reading_projected"}},
        )

    @classmethod
    def _cleanup_reparsed_chapter_derived_artifacts(
        cls,
        chapter: StructuredChapter,
    ) -> StructuredChapter:
        return replace(
            chapter,
            task_artifacts=None,
            sections=[
                cls._cleanup_reparsed_section_derived_artifacts(section)
                for section in chapter.sections
            ],
        )

    @staticmethod
    def _cleanup_reparsed_section_derived_artifacts(
        section: StructuredSection,
    ) -> StructuredSection:
        return replace(
            section,
            task_artifacts=None,
            task_units=[
                replace(task_unit, task_artifacts=None)
                for task_unit in section.task_units
            ],
        )

    @staticmethod
    def _with_task_units_parent_section(
        *,
        task_units: list[TaskUnit],
        section_id: str,
    ) -> list[TaskUnit]:
        """Ensure each task unit carries stable hierarchy owner identity."""
        return [
            replace(task_unit, parent_section_id=section_id)
            for task_unit in task_units
        ]

    @staticmethod
    def _assert_hierarchy_save_consistency(
        *,
        document: StructuredDocument,
        doc_name: str,
        context: str,
    ) -> None:
        if not document.chapters:
            return
        warnings = validate_chapter_hierarchy_consistency(document)
        severe_warnings = [
            warning
            for warning in warnings
            if is_severe_hierarchy_warning(warning)
        ]
        if severe_warnings:
            raise ValueError(
                f"{context}: severe hierarchy inconsistency detected for doc_name='{doc_name}': "
                + " | ".join(severe_warnings)
            )

    def _update_section_by_id(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
        update_fn: Callable[[StructuredSection], StructuredSection],
        context: str,
        doc_name: str,
    ) -> StructuredDocument:
        if not document.chapters:
            raise ValueError(
                f"{context}: document '{doc_name}' missing hierarchy chapters; "
                "legacy sections-only write path is disabled"
            )

        build_section_index_from_chapters(document)
        hierarchy_sections = flatten_sections_from_chapters(document)
        hierarchy_target = next(
            (section for section in hierarchy_sections if section.section_id == section_id),
            None,
        )
        if hierarchy_target is None:
            raise ValueError(
                f"{context}: unknown section_id='{section_id}' for doc_name='{doc_name}'"
            )
        updated_sections_by_id = {
            section.section_id: (
                update_fn(section)
                if section.section_id == section_id
                else section
            )
            for section in hierarchy_sections
        }
        updated_document = with_sections_replaced_in_hierarchy(
            document=document,
            sections_by_id=updated_sections_by_id,
        )
        updated_document = replace(updated_document, sections=[])
        self._assert_hierarchy_save_consistency(
            document=updated_document,
            doc_name=doc_name,
            context=context,
        )
        self._assert_unique_task_unit_ids(
            document=updated_document,
            context=f"{context} doc_name='{doc_name}'",
        )
        return updated_document
