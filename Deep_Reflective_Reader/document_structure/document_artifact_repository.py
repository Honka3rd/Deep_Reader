from abc import ABC, abstractmethod
from dataclasses import dataclass

from document_structure.structured_document import StructuredDocument
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit


@dataclass(frozen=True)
class DocumentListItem:
    """Lightweight document metadata for API discovery/search surfaces."""

    doc_name: str
    title: str | None = None
    source: str = "structured"


class DocumentArtifactRepository(ABC):
    """Repository contract for structured-document task-artifact persistence."""

    @abstractmethod
    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        """List lightweight structured document candidates for discovery/search."""
        raise NotImplementedError

    @abstractmethod
    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load one structured document by logical document name."""
        raise NotImplementedError

    @abstractmethod
    def save_document(self, document: StructuredDocument, doc_name: str | None = None) -> None:
        """Persist one structured document payload."""
        raise NotImplementedError

    @abstractmethod
    def update_section_artifacts(
        self,
        doc_name: str,
        section_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one section-level task artifact payload and persist document."""
        raise NotImplementedError

    @abstractmethod
    def update_section_summary_artifact(
        self,
        doc_name: str,
        section_id: str,
        summary: SummaryArtifact,
    ) -> StructuredDocument:
        """Update one section summary artifact while preserving existing section quiz artifact."""
        raise NotImplementedError

    @abstractmethod
    def update_chapter_summary_artifact(
        self,
        doc_name: str,
        chapter_key: str,
        summary: SummaryArtifact,
    ) -> StructuredDocument:
        """Update one chapter summary artifact at document level while preserving quiz/metadata."""
        raise NotImplementedError

    @abstractmethod
    def update_section_quiz_artifact(
        self,
        doc_name: str,
        section_id: str,
        quiz: QuizArtifact,
    ) -> StructuredDocument:
        """Update one section quiz artifact while preserving existing section summary artifact."""
        raise NotImplementedError

    @abstractmethod
    def update_chapter_quiz_artifact(
        self,
        doc_name: str,
        chapter_key: str,
        quiz: QuizArtifact,
    ) -> StructuredDocument:
        """Update one chapter quiz artifact at document level while preserving summary/metadata."""
        raise NotImplementedError

    @abstractmethod
    def update_task_unit_artifacts(
        self,
        doc_name: str,
        task_unit_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Update one task-unit-level artifact payload (reserved skeleton)."""
        raise NotImplementedError

    @abstractmethod
    def update_section_task_units(
        self,
        doc_name: str,
        section_id: str,
        task_units: list[TaskUnit],
    ) -> StructuredDocument:
        """Replace one section's persisted task-unit list and persist document."""
        raise NotImplementedError

    @abstractmethod
    def update_task_layout(
        self,
        doc_name: str,
        task_units_by_section_id: dict[str, list[TaskUnit]],
        task_layout_metadata: dict[str, str | int | None],
    ) -> StructuredDocument:
        """Bulk update section task units + task-layout metadata with one atomic save."""
        raise NotImplementedError

    @abstractmethod
    def update_document_artifacts(
        self,
        doc_name: str,
        artifacts: DocumentTaskArtifacts,
    ) -> StructuredDocument:
        """Update document-level artifact payload and persist document."""
        raise NotImplementedError
