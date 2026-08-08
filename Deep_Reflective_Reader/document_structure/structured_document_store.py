from pathlib import Path
import os
import tempfile

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
                "StructuredDocumentStore.load: invalid structured document JSON: "
                f"{path} ({error})"
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
    def exists(target: str | StructuredDocumentStorageConfig) -> bool:
        """Return whether a structured document target exists."""
        return StructuredDocumentStore._resolve_path(target).exists()

    @staticmethod
    def location(target: str | StructuredDocumentStorageConfig) -> str:
        """Return a human-readable storage location for the target."""
        return str(StructuredDocumentStore._resolve_path(target))

    @staticmethod
    def _resolve_path(target: str | StructuredDocumentStorageConfig) -> Path:
        """Resolve save/load target to a concrete filesystem path."""
        if isinstance(target, StructuredDocumentStorageConfig):
            return Path(target.get_raw_document_path())
        return Path(target)
