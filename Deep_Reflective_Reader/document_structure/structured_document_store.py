from pathlib import Path

from config.structured_document_storage_config import StructuredDocumentStorageConfig
from document_structure.structured_document import StructuredDocument


class StructuredDocumentStore:
    """JSON persistence store for StructuredDocument artifacts."""

    @staticmethod
    def save(
        document: StructuredDocument,
        target: str | StructuredDocumentStorageConfig,
    ) -> None:
        """Save structured document to UTF-8 JSON file."""
        path = StructuredDocumentStore._resolve_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = document.to_json()
        path.write_text(payload, encoding="utf-8")

    @staticmethod
    def load(target: str | StructuredDocumentStorageConfig) -> StructuredDocument:
        """Load structured document from UTF-8 JSON file."""
        path = StructuredDocumentStore._resolve_path(target)
        try:
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"StructuredDocumentStore.load: file not found: {path}"
            ) from error
        except OSError as error:
            raise OSError(
                f"StructuredDocumentStore.load: failed to read file: {path}"
            ) from error

        try:
            return StructuredDocument.from_json(payload)
        except Exception as error:
            raise ValueError(
                f"StructuredDocumentStore.load: invalid structured document JSON: {path}"
            ) from error

    @staticmethod
    def default_path_for_document_id(
        document_id: str,
        base_dir: str = "data/structured",
    ) -> str:
        """Build default JSON artifact path from document id."""
        return StructuredDocumentStorageConfig(
            namespace=document_id,
            base_dir=base_dir,
        ).get_raw_document_path()

    @staticmethod
    def _resolve_path(target: str | StructuredDocumentStorageConfig) -> Path:
        """Resolve save/load target to a concrete filesystem path."""
        if isinstance(target, StructuredDocumentStorageConfig):
            return Path(target.get_raw_document_path())
        return Path(target)
