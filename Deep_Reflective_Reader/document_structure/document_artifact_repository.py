from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from document_structure.structured_document import StructuredDocument
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit


def preserve_explicit_annotations(current, incoming):
    """Only atomic interaction mutations or explicit reparse may replace these fields."""
    current_revision = current.parse_provenance.get("reading_content", {}).get("revision", 0)
    incoming_revision = incoming.parse_provenance.get("reading_content", {}).get("revision", 0)
    if current_revision != incoming_revision:
        # A pre-edit worker must not restore invalidated generated artifacts.
        incoming = replace(incoming, document_task_artifacts=current.document_task_artifacts,
            chapters=[replace(c, task_artifacts=None, sections=[replace(s, task_artifacts=None,
                task_units=[replace(u, task_artifacts=None) for u in s.task_units]) for s in c.sections])
                for c in incoming.chapters])
    provenance = dict(incoming.parse_provenance)
    provenance.pop("reading_projected",None)
    if "reading_content" in current.parse_provenance:
        provenance["reading_content"] = current.parse_provenance["reading_content"]
    else:
        provenance.pop("reading_content",None)
    incoming = replace(incoming,parse_provenance=provenance)
    previous = current.document_task_artifacts
    updated = incoming.document_task_artifacts or DocumentTaskArtifacts()
    metadata = dict(updated.metadata)
    for key in ("note", "public_summary"):
        if previous is not None and key in previous.metadata:
            metadata[key] = previous.metadata[key]
        else:
            metadata.pop(key, None)
    if metadata == updated.metadata:
        return incoming
    return replace(incoming, document_task_artifacts=replace(updated, metadata=metadata))


@dataclass(frozen=True)
class DocumentListItem:
    """Lightweight document metadata for API discovery/search surfaces."""

    doc_name: str
    title: str | None = None
    source: str = "structured"
    review_required: bool = False


class DocumentArtifactRepository(ABC):
    """Repository contract for structured-document task-artifact persistence."""

    def mutate_document_artifacts(self, doc_name, mutation):
        """Atomically apply mutation(current_document) -> (artifacts, result)."""
        raise NotImplementedError("atomic artifact mutation not supported")

    def rename_document(self, doc_name, new_name, transform):
        raise NotImplementedError("document rename not supported")

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

    def save_reparsed_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist a parser-level replacement of the current structured document."""
        self.save_document(document, doc_name=doc_name)

    def delete_document(self, doc_name: str) -> bool:
        """Permanently delete owned structured data; return False if absent.

        Storage failures must propagate so lifecycle callers cannot report success.
        """
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
    def mutate_reading_content(self, doc_name, mutation):
        raise NotImplementedError("atomic reading content mutation not supported")
