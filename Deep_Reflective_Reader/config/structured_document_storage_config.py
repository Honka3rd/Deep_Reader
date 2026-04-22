from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper


class StructuredDocumentStorageConfig:
    """Resolve filesystem paths for per-document structured document artifacts."""

    _NAMESPACE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

    base_dir: Path
    document_path: Path
    doc_name: str

    def __init__(
        self,
        namespace: str = "default",
    ):
        """Initialize storage paths for one logical document namespace."""
        normalized_namespace = self.normalize_namespace(namespace)
        self.base_dir = Path("data/structured",)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._migrate_legacy_document_file(
            base_root=self.base_dir,
            original_namespace=namespace,
            normalized_namespace=normalized_namespace,
        )

        self.doc_name = normalized_namespace
        self.document_path = self.base_dir / f"{self.doc_name}.structured.json"

    @classmethod
    def normalize_namespace(cls, namespace: str) -> str:
        """Normalize namespace by dropping known document file extensions."""
        return StorageNamespaceHelper.normalize_namespace(
            namespace,
            known_extensions=cls._NAMESPACE_EXTENSIONS,
            fallback_namespace="default",
        )

    @classmethod
    def _migrate_legacy_document_file(
        cls,
        *,
        base_root: Path,
        original_namespace: str,
        normalized_namespace: str,
    ) -> None:
        """Migrate legacy JSON file name with extension to normalized namespace."""
        if original_namespace == normalized_namespace:
            return

        legacy_file = base_root / f"{original_namespace}.structured.json"
        normalized_file = base_root / f"{normalized_namespace}.structured.json"
        if not legacy_file.exists():
            return
        if normalized_file.exists():
            print(
                "StructuredDocumentStorageConfig#namespace_migration: "
                f"skip (legacy={legacy_file}, normalized={normalized_file})"
            )
            return

        legacy_file.rename(normalized_file)
        print(
            "StructuredDocumentStorageConfig#namespace_migration: "
            f"moved {legacy_file} -> {normalized_file}"
        )

    def get_raw_document_path(self) -> str:
        """Return filesystem path for structured document JSON artifact."""
        return str(self.document_path)

    def exists(self) -> bool:
        """Return whether structured document JSON artifact exists."""
        return self.document_path.exists()

    def get_doc_name(self) -> str:
        """Return normalized logical document namespace."""
        return self.doc_name
