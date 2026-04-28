import os
import tempfile
from dataclasses import replace
from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.structured_document import StructuredDocument
from document_structure.structured_document_store import StructuredDocumentStore
from shared.task_artifacts import DocumentTaskArtifacts, TaskArtifacts


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

    def update_task_unit_artifacts(
        self,
        doc_name: str,
        task_unit_id: str,
        artifacts: TaskArtifacts,
    ) -> StructuredDocument:
        """Reserved skeleton for future task-unit level artifact persistence."""
        _ = (doc_name, task_unit_id, artifacts)
        raise NotImplementedError(
            "update_task_unit_artifacts is reserved for a future task-unit persistence round."
        )

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
