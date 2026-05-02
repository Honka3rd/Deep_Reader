import json
import os

from profile.document_profile import DocumentProfile
from config.faiss_storage_config import FaissStorageConfig


class DocumentProfileStore:
    """Persist and load document profile JSON artifacts."""

    @staticmethod
    def save(profile: DocumentProfile, config: FaissStorageConfig) -> None:
        """Persist artifact/data to storage.

Args:
    profile: Document profile with topic/language/summary fields.
    config: FaissStorageConfig describing filesystem artifact paths."""
        payload = profile.to_dict()

        with open(config.get_raw_profile_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(config: FaissStorageConfig) -> DocumentProfile:
        """Load persisted artifact and return parsed object/data.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

Returns:
    Document profile restored from persisted profile JSON."""
        with open(config.get_raw_profile_path(), "r", encoding="utf-8") as f:
            payload = json.load(f)

        return DocumentProfile.from_dict(payload)

    @staticmethod
    def exists(config: FaissStorageConfig) -> bool:
        """Return whether persisted artifact exists.

Args:
    config: FaissStorageConfig describing filesystem artifact paths.

Returns:
    True when required artifact exists; otherwise False."""
        return os.path.exists(config.get_raw_profile_path())

    @staticmethod
    def clear(config: FaissStorageConfig) -> None:
        """Remove persisted artifacts for clean rebuild.

Args:
    config: FaissStorageConfig describing filesystem artifact paths."""
        path = config.get_raw_profile_path()
        if os.path.exists(path):
            os.remove(path)
