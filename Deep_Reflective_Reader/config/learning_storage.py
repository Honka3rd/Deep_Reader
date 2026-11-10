"""Compose existing storage owners without reads, DDL or recovery side effects."""
from dataclasses import dataclass
from pathlib import Path

from db.postgres_learning_repository import PostgresLearningRepository
from db.postgres_reading_preferences_repository import PostgresReadingPreferencesRepository
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from document_structure.learning_file_lifecycle import FileLearningDocumentLifecycle
from document_structure.learning_file_repository import FileLearningRepository
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from reading_preferences.file_repository import FileReadingPreferencesRepository
from reading_preferences.models import WorkspaceScope


@dataclass(frozen=True)
class LearningStorage:
    documents: object
    preferences: object
    learning: object
    lifecycle: FileLearningDocumentLifecycle | None = None


def build_learning_storage(*, backend, documents, policy, dsn, namespace, admission):
    if backend == "file":
        if not isinstance(documents, StructuredDocumentArtifactRepository) or isinstance(
                documents, PostgresStructuredDocumentArtifactRepository):
            raise ValueError("learning_storage_backend_mismatch")
        scope = WorkspaceScope(documents.namespace)
        preferences = FileReadingPreferencesRepository(Path(policy.preferences_base_dir), scope)
        learning = FileLearningRepository(Path(policy.results_base_dir), documents, preferences,
                                         admission=admission)
        # Install before publishing the document repository to ordinary consumers.
        lifecycle = FileLearningDocumentLifecycle(documents, learning, preferences)
        return LearningStorage(documents, preferences, learning, lifecycle)
    if backend == "postgres":
        if not isinstance(documents, PostgresStructuredDocumentArtifactRepository):
            raise ValueError("learning_storage_backend_mismatch")
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("learning_storage_unavailable")
        store = documents.store
        if store.dsn != dsn or store.namespace != namespace:
            raise ValueError("learning_storage_identity_mismatch")
        scope = WorkspaceScope(store.namespace)
        preferences = PostgresReadingPreferencesRepository(dsn, scope)
        learning = PostgresLearningRepository(dsn, scope, admission=admission,
                                             documents=store, preferences=preferences)
        return LearningStorage(documents, preferences, learning)
    raise ValueError("learning_storage_backend_unavailable")
