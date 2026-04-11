import os
from pathlib import Path


class StorageConfig:
    base_dir: Path
    index_path: Path
    records_path: Path
    meta_path: Path
    doc_name: str
    profile_path: Path

    def __init__(self, namespace: str = "default"):
        self.base_dir = Path("data/faiss") / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.doc_name = namespace
        self.index_path = self.base_dir / "index.faiss"
        self.records_path = self.base_dir / "records.json"
        self.meta_path = self.base_dir / "meta.json"
        self.profile_path = self.base_dir / "profile.json"

    def get_raw_index_path(self) -> str:
        return str(self.index_path)

    def get_raw_records_path(self) -> str:
        return str(self.records_path)

    def get_raw_meta_path(self) -> str:
        return str(self.meta_path)

    def index_path_exists(self) -> bool:
        return os.path.exists(self.index_path)

    def records_path_exists(self) -> bool:
        return os.path.exists(self.records_path)

    def meta_path_exists(self) -> bool:
        return os.path.exists(self.meta_path)

    def exists(self):
        return self.index_path_exists() and self.records_path_exists() and self.meta_path_exists()

    def get_doc_name(self) -> str:
        return self.doc_name

    def get_raw_profile_path(self) -> str:
        return str(self.profile_path)