import os
import tempfile
from dataclasses import replace
from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.structured_document import StructuredDocument
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
    ):
        self.store = store or StructuredDocumentStore()
        self.base_dir = Path(base_dir)

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name."""
        path = self._resolve_document_path(doc_name)
        return self.store.load(str(path))

    def save_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist structured document with atomic write."""
        resolved_doc_name = doc_name or document.document_id
        path = self._resolve_document_path(resolved_doc_name)
        self._atomic_save(document=document, path=path)

    def update_section_artifacts(
        self,
        doc_name: str,
        section_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one section artifact payload and persist new structured document."""
        document = self.load_document(doc_name)
        updated_sections = list(document.sections)
        target_index = next(
            (
                index
                for index, section in enumerate(updated_sections)
                if section.section_id == section_id
            ),
            None,
        )
        if target_index is None:
            raise ValueError(
                f"update_section_artifacts: unknown section_id='{section_id}' for doc_name='{doc_name}'"
            )

        updated_sections[target_index] = replace(
            updated_sections[target_index],
            task_artifacts=artifacts,
        )
        updated_document = replace(
            document,
            sections=updated_sections,
        )
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_section_summary_artifact(
        self,
        doc_name: str,
        section_id: str,
        summary: SummaryArtifact,
    ) -> StructuredDocument:
        """Update one section summary artifact while preserving existing section quiz artifact."""
        document = self.load_document(doc_name)
        updated_sections = list(document.sections)
        target_index = next(
            (
                index
                for index, section in enumerate(updated_sections)
                if section.section_id == section_id
            ),
            None,
        )
        if target_index is None:
            raise ValueError(
                f"update_section_summary_artifact: unknown section_id='{section_id}' for doc_name='{doc_name}'"
            )

        existing_artifacts = updated_sections[target_index].task_artifacts
        merged_artifacts = TaskArtifacts(
            summary=summary,
            quiz=(None if existing_artifacts is None else existing_artifacts.quiz),
        )
        updated_sections[target_index] = replace(
            updated_sections[target_index],
            task_artifacts=merged_artifacts,
        )
        updated_document = replace(document, sections=updated_sections)
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_section_quiz_artifact(
        self,
        doc_name: str,
        section_id: str,
        quiz: QuizArtifact,
    ) -> StructuredDocument:
        """Update one section quiz artifact while preserving existing section summary artifact."""
        document = self.load_document(doc_name)
        updated_sections = list(document.sections)
        target_index = next(
            (
                index
                for index, section in enumerate(updated_sections)
                if section.section_id == section_id
            ),
            None,
        )
        if target_index is None:
            raise ValueError(
                f"update_section_quiz_artifact: unknown section_id='{section_id}' for doc_name='{doc_name}'"
            )

        existing_artifacts = updated_sections[target_index].task_artifacts
        merged_artifacts = TaskArtifacts(
            summary=(None if existing_artifacts is None else existing_artifacts.summary),
            quiz=quiz,
        )
        updated_sections[target_index] = replace(
            updated_sections[target_index],
            task_artifacts=merged_artifacts,
        )
        updated_document = replace(document, sections=updated_sections)
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

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

        document = self.load_document(doc_name)
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
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_task_unit_artifacts(
        self,
        doc_name: str,
        task_unit_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one persisted task-unit artifact payload and persist document."""
        document = self.load_document(doc_name)
        updated_sections = list(document.sections)
        has_updated = False
        for section_index, section in enumerate(updated_sections):
            if not section.task_units:
                continue
            updated_task_units: list[TaskUnit] = []
            section_touched = False
            for task_unit in section.task_units:
                if task_unit.unit_id == task_unit_id:
                    updated_task_units.append(
                        replace(task_unit, task_artifacts=artifacts)
                    )
                    section_touched = True
                    has_updated = True
                else:
                    updated_task_units.append(task_unit)
            if section_touched:
                updated_sections[section_index] = replace(
                    section,
                    task_units=updated_task_units,
                )
                break

        if not has_updated:
            raise ValueError(
                f"update_task_unit_artifacts: unknown task_unit_id='{task_unit_id}' for doc_name='{doc_name}'"
            )

        updated_document = replace(document, sections=updated_sections)
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_section_task_units(
        self,
        doc_name: str,
        section_id: str,
        task_units: list[TaskUnit],
    ) -> StructuredDocument:
        """Replace one section task-unit list and persist document."""
        document = self.load_document(doc_name)
        updated_sections = list(document.sections)
        target_index = next(
            (
                index
                for index, section in enumerate(updated_sections)
                if section.section_id == section_id
            ),
            None,
        )
        if target_index is None:
            raise ValueError(
                f"update_section_task_units: unknown section_id='{section_id}' for doc_name='{doc_name}'"
            )

        updated_sections[target_index] = replace(
            updated_sections[target_index],
            task_units=list(task_units),
        )
        updated_document = replace(document, sections=updated_sections)
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_task_layout(
        self,
        doc_name: str,
        task_units_by_section_id: dict[str, list[TaskUnit]],
        task_layout_metadata: dict[str, str | int | None],
    ) -> StructuredDocument:
        """Bulk update section task units and task-layout metadata in one save."""
        document = self.load_document(doc_name)
        updated_sections = []
        for section in document.sections:
            section_units = task_units_by_section_id.get(section.section_id, [])
            updated_sections.append(
                replace(
                    section,
                    task_units=list(section_units),
                )
            )

        existing_document_artifacts = (
            document.document_task_artifacts or DocumentTaskArtifacts()
        )
        metadata = dict(existing_document_artifacts.metadata)
        metadata["task_layout"] = dict(task_layout_metadata)
        updated_document_artifacts = replace(
            existing_document_artifacts,
            metadata=metadata,
        )

        updated_document = replace(
            document,
            sections=updated_sections,
            document_task_artifacts=updated_document_artifacts,
        )
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def update_document_artifacts(
        self,
        doc_name: str,
        artifacts: DocumentTaskArtifacts,
    ) -> StructuredDocument:
        """Update document-level task-artifact payload and persist."""
        document = self.load_document(doc_name)
        updated_document = replace(document, document_task_artifacts=artifacts)
        self.save_document(updated_document, doc_name=doc_name)
        return updated_document

    def _resolve_document_path(self, doc_name: str) -> Path:
        """Resolve logical doc name to structured artifact path."""
        normalized_name = StorageNamespaceHelper.normalize_namespace(
            doc_name,
            known_extensions=self._NAMESPACE_EXTENSIONS,
            fallback_namespace=StorageNamespaceHelper.DEFAULT_NAMESPACE,
        )
        return self.base_dir / f"{normalized_name}.structured.json"

    @staticmethod
    def _atomic_save(document: StructuredDocument, path: Path) -> None:
        """Persist document atomically through temp-file + replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = document.to_json()
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
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_file is not None and not temp_file.closed:
                temp_file.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
