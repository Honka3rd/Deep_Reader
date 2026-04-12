import os
from pathlib import Path


class StorageConfig:
    """Resolve filesystem paths for per-document index artifacts."""
    base_dir: Path
    index_path: Path
    records_path: Path
    meta_path: Path
    doc_name: str
    profile_path: Path

    def __init__(self, namespace: str = "default"):
        """Initialize object state and injected dependencies.

Args:
    namespace: Storage namespace, typically same as document name.
"""
        self.base_dir = Path("data/faiss") / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.doc_name = namespace
        self.index_path = self.base_dir / "index.faiss"
        self.records_path = self.base_dir / "records.json"
        self.meta_path = self.base_dir / "meta.json"
        self.profile_path = self.base_dir / "profile.json"

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
