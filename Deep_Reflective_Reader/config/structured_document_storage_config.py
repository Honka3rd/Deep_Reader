from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper


class StructuredDocumentStorageConfig:
    """Resolve filesystem paths for per-document structured document artifacts."""

    _NAMESPACE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

    base_dir: Path
    document_path: Path
    doc_name: str

    def __init__(self, namespace="default", *, base_dir="data/structured", identity_resolver=None):
        self.base_dir = Path(base_dir)
        self.doc_name = (identity_resolver(namespace) if identity_resolver else
                         StorageNamespaceHelper.file_identity(namespace, self.base_dir,
                                                              suffix=".structured.json"))
        self.document_path = StorageNamespaceHelper.checked_path(
            self.base_dir, self.base_dir / f"{self.doc_name}.structured.json")

    normalize_namespace = StorageNamespaceHelper.normalize_namespace

    def get_raw_document_path(self) -> str:
        """Return filesystem path for structured document JSON artifact."""
        return str(self.document_path)

    def exists(self) -> bool:
        """Return whether structured document JSON artifact exists."""
        return self.document_path.exists()

    def get_doc_name(self) -> str:
        """Return normalized logical document namespace."""
        return self.doc_name
