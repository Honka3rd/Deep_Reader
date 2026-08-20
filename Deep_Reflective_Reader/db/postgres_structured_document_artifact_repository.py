from __future__ import annotations

from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.document_artifact_repository import DocumentListItem
from document_structure.structured_document import StructuredDocument
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from shared.task_unit_model import TaskUnit


class PostgresStructuredDocumentArtifactRepository(StructuredDocumentArtifactRepository):
    """Structured document artifact repository backed by PostgreSQL."""

    def __init__(self, store: PostgresStructuredDocumentStore) -> None:
        super().__init__(store=store)
        self.store = store

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name from PostgreSQL."""
        return self.store.load(self.store.target_for_doc_name(doc_name))

    def list_documents(
        self,
        query: str | None = None,
        limit: int = 50,
    ) -> list[DocumentListItem]:
        """List structured document candidates from PostgreSQL."""
        return self.store.list_documents(query=query, limit=limit)

    def save_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist structured document by logical doc name to PostgreSQL."""
        resolved_doc_name = doc_name or document.document_id
        self.store.save(
            document=document,
            target=self.store.target_for_doc_name(resolved_doc_name),
            replace_existing_hierarchy=False,
        )

    def update_task_layout(
        self,
        doc_name: str,
        task_units_by_section_id: dict[str, list[TaskUnit]],
        task_layout_metadata: dict[str, str | int | None],
    ) -> StructuredDocument:
        """Update task-layout cache and return the PostgreSQL current hierarchy."""
        super().update_task_layout(
            doc_name=doc_name,
            task_units_by_section_id=task_units_by_section_id,
            task_layout_metadata=task_layout_metadata,
        )
        return self.load_document(doc_name)
