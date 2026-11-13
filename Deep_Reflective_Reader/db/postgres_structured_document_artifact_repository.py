from __future__ import annotations

from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.document_artifact_repository import DocumentListItem
from document_structure.structured_document import StructuredDocument
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)


class PostgresStructuredDocumentArtifactRepository(StructuredDocumentArtifactRepository):
    """Structured document artifact repository backed by PostgreSQL."""

    def __init__(self, store: PostgresStructuredDocumentStore) -> None:
        super().__init__(store=store)
        self.store = store

    def contains_document(self, doc_name: str) -> bool:
        from shared.document_name import document_name
        return self.store.exists(self.store.target_for_doc_name(document_name(doc_name)))

    def resolve_document_names(self, names):
        return self.store.resolve_document_names(names)

    def resolve_document_name(self, doc_name: str) -> str:
        return self.store.resolve_document_name(doc_name)

    def delete_document(self, doc_name: str) -> bool:
        """Delete current structured data using the store's namespace policy."""
        return self.store.delete(self.store.target_for_doc_name(self.resolve_document_name(doc_name)))

    def mutate_document_artifacts(self, doc_name, mutation):
        return self.store.mutate_document_artifacts(
            self.store.target_for_doc_name(self.resolve_document_name(doc_name)), mutation)

    def mutate_reading_content(self, doc_name, mutation):
        return self.store.mutate_reading_content(self.store.target_for_doc_name(self.resolve_document_name(doc_name)),mutation)

    def _mutate_hierarchy(self, doc_name, mutation):
        return self.store.mutate_hierarchy(
            self.store.target_for_doc_name(self.resolve_document_name(doc_name)), mutation)

    def commit_cleanup(self, command):
        return self.store.commit_cleanup(command)

    def lookup_cleanup(self, *, namespace, document_id, operation_id):
        return self.store.lookup_cleanup(namespace=namespace, document_id=document_id, operation_id=operation_id)

    @property
    def lifecycle_storage_identity(self):
        from hashlib import sha256
        return ('postgres', self.store.namespace, sha256(self.store.dsn.encode()).hexdigest())

    def recover_document_rename(self, doc_name, new_name):
        # PostgreSQL lifecycle commits or rolls back in one transaction.
        return False

    def rename_document(self, doc_name, new_name, transform):
        return self.store.rename_document(self.store.target_for_doc_name(self.resolve_document_name(doc_name)), new_name, transform)

    def load_document(self, doc_name: str) -> StructuredDocument:
        """Load structured document by logical doc name from PostgreSQL."""
        return self.store.load(self.store.target_for_doc_name(self.resolve_document_name(doc_name)))

    def load_enhancement_snapshot(self, doc_name: str):
        return self.store.load_enhancement_snapshot(self.store.target_for_doc_name(self.resolve_document_name(doc_name)))

    def enhancement_lease(self, doc_name: str):
        return self.store.enhancement_lease(self.store.target_for_doc_name(self.resolve_document_name(doc_name)))

    def enhancement_guards_available(self):
        return self.store.enhancement_guards_available()

    def profile_structure_invalidated(self, doc_name):
        return self.store.profile_structure_invalidated(self.store.target_for_doc_name(self.resolve_document_name(doc_name)))

    def commit_enhancement(self, doc_name: str, *, candidate, interaction_revision, source_path, cleanup=None, operation=None):
        return self.store.commit_enhancement(self.store.target_for_doc_name(self.resolve_document_name(doc_name)),
            candidate=candidate, interaction_revision=interaction_revision, source_path=source_path,
            cleanup=cleanup, operation=operation)

    def backfill_txt_extraction(self, doc_name, *, result, source_path):
        return self.store.backfill_txt_extraction(self.store.target_for_doc_name(self.resolve_document_name(doc_name)),
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
        resolved_doc_name = self._write_document_name(document, doc_name)
        self.store.save(
            document=document,
            target=self.store.target_for_doc_name(resolved_doc_name),
            replace_existing_hierarchy=False,
        )

    def commit_manual_structure(self, document, *, doc_name, validate_current):
        return self.store.commit_manual_structure(
            self.store.target_for_doc_name(self.resolve_document_name(doc_name)),
            document=document, validate_current=validate_current)

    def save_reparsed_document(
        self,
        document: StructuredDocument,
        doc_name: str | None = None,
    ) -> None:
        """Persist a parser-level hierarchy replacement to PostgreSQL."""
        resolved_doc_name = self._write_document_name(document, doc_name)
        cleaned_document = self.cleanup_reparsed_document_derived_artifacts(document)
        self.store.save(
            document=cleaned_document,
            target=self.store.target_for_doc_name(resolved_doc_name),
            replace_existing_hierarchy=True,
        )
