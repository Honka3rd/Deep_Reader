import os
from pathlib import Path

from config.storage_namespace_helper import StorageNamespaceHelper


class FaissStorageConfig:
    """Resolve filesystem paths for per-document index artifacts."""
    base_dir: Path
    index_path: Path
    records_path: Path
    meta_path: Path
    doc_name: str
    profile_path: Path
    _NAMESPACE_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")

    def __init__(self, namespace="default", *, read_only=False, base_root="data/faiss",
                 identity_resolver=None):
        # read_only remains accepted for callers; all configuration is now pure.
        root = Path(base_root)
        self.doc_name = (identity_resolver(namespace) if identity_resolver else
                         StorageNamespaceHelper.file_identity(namespace, root))
        physical_name = self.doc_name
        if (identity_resolver and StorageNamespaceHelper.stem(self.doc_name) == self.doc_name
                and not (root / self.doc_name).exists()):
            candidates = [path.name for path in root.iterdir()
                          if path.name != self.doc_name and
                          StorageNamespaceHelper.stem(path.name) == self.doc_name] if root.exists() else []
            if len(candidates) == 1 and identity_resolver(candidates[0]) == self.doc_name:
                physical_name = candidates[0]
        self.base_dir = StorageNamespaceHelper.checked_path(root, root / physical_name)
        self.index_path = self.base_dir / "index.faiss"
        self.records_path = self.base_dir / "records.json"
        self.meta_path = self.base_dir / "meta.json"
        self.profile_path = self.base_dir / "profile.json"

    normalize_namespace = StorageNamespaceHelper.normalize_namespace

    def get_raw_index_path(self) -> str:
        """Return raw index path.

Returns:
    Filesystem path to persisted FAISS index file."""
        return str(self.index_path)

    def get_raw_records_path(self) -> str:
        """Return raw records path.

Returns:
    Filesystem path to persisted records JSON."""
        return str(self.records_path)

    def get_raw_meta_path(self) -> str:
        """Return raw meta path.

Returns:
    Filesystem path to fingerprint metadata JSON."""
        return str(self.meta_path)

    def index_path_exists(self) -> bool:
        """Check whether serialized FAISS index file exists.

Returns:
    ``True`` if index file exists on disk; otherwise ``False``."""
        return os.path.exists(self.index_path)

    def records_path_exists(self) -> bool:
        """Check whether serialized records JSON exists.

Returns:
    ``True`` if records file exists on disk; otherwise ``False``."""
        return os.path.exists(self.records_path)

    def meta_path_exists(self) -> bool:
        """Check whether fingerprint metadata JSON exists.

Returns:
    ``True`` if fingerprint metadata exists; otherwise ``False``."""
        return os.path.exists(self.meta_path)

    def exists(self):
        """Return whether persisted artifact exists.

Returns:
    True when required artifact exists; otherwise False."""
        return self.index_path_exists() and self.records_path_exists() and self.meta_path_exists()

    def get_doc_name(self) -> str:
        """Return doc name.

Returns:
    Storage namespace / logical document name."""
        return self.doc_name

    def get_raw_profile_path(self) -> str:
        """Return raw profile path.

Returns:
    Filesystem path to persisted document profile JSON."""
        return str(self.profile_path)
