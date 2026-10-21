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

    def delete_document(self, doc_name: str) -> bool:
        """Delete current structured data using the store's namespace policy."""
        return self.store.delete(self.store.target_for_doc_name(doc_name))

    def mutate_document_artifacts(self, doc_name, mutation):
        return self.store.mutate_document_artifacts(
            self.store.target_for_doc_name(doc_name), mutation)

    def mutate_reading_content(self, doc_name, mutation):
        return self.store.mutate_reading_content(self.store.target_for_doc_name(doc_name),mutation)

    def rename_document(self, doc_name, new_name, transform):
        return self.store.rename_document(self.store.target_for_doc_name(doc_name), new_name, transform)

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name from PostgreSQL."""
        return self.store.load(self.store.target_for_doc_name(doc_name))

    def load_enhancement_snapshot(self, doc_name: str):
        return self.store.load_enhancement_snapshot(self.store.target_for_doc_name(doc_name))

    def enhancement_lease(self, doc_name: str):
        return self.store.enhancement_lease(self.store.target_for_doc_name(doc_name))

    def enhancement_guards_available(self):
        return self.store.enhancement_guards_available()

    def profile_structure_invalidated(self, doc_name):
        return self.store.profile_structure_invalidated(self.store.target_for_doc_name(doc_name))

    def commit_enhancement(self, doc_name: str, *, candidate, interaction_revision, source_path):
        return self.store.commit_enhancement(self.store.target_for_doc_name(doc_name),
            candidate=candidate, interaction_revision=interaction_revision, source_path=source_path)

    def backfill_txt_extraction(self, doc_name, *, result, source_path):
        return self.store.backfill_txt_extraction(self.store.target_for_doc_name(doc_name),
            result=result, source_path=source_path)

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
        """Persist structured metadata without replacing the current hierarchy."""
        resolved_doc_name = doc_name or document.document_id
        self.store.save(
            document=document,
            target=self.store.target_for_doc_name(resolved_doc_name),
            replace_existing_hierarchy=False,
        )

    def save_reparsed_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist a parser-level hierarchy replacement to PostgreSQL."""
        resolved_doc_name = doc_name or document.document_id
        cleaned_document = self.cleanup_reparsed_document_derived_artifacts(document)
        self.store.save(
            document=cleaned_document,
            target=self.store.target_for_doc_name(resolved_doc_name),
            replace_existing_hierarchy=True,
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
