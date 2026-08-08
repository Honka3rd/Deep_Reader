from __future__ import annotations

from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.structured_document import StructuredDocument
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)


class PostgresStructuredDocumentArtifactRepository(StructuredDocumentArtifactRepository):
    """Structured document artifact repository backed by PostgreSQL."""

    def __init__(self, store: PostgresStructuredDocumentStore) -> None:
        super().__init__(store=store)
        self.store = store

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name from PostgreSQL."""
        return self.store.load(self.store.target_for_doc_name(doc_name))

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
        )
