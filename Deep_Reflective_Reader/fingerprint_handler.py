import hashlib
import json
import os
from typing import Any


class FingerprintHandler:
    """Manage content/config fingerprints used to decide index reuse."""
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """Initialize object state and injected dependencies.

Args:
    embedding_model: Embedding model name.
    chunk_size: Splitter chunk size.
    chunk_overlap: Splitter chunk overlap size.
"""
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute stable MD5 hash for normalized document text.

Args:
    text: Input text content.

Returns:
    Hex digest string of the normalized text."""
        return hashlib.md5(text.strip().encode()).hexdigest()

    def build_fingerprint(self, text: str) -> dict[str, Any]:
        """Build fingerprint.

Args:
    text: Input text content.

Returns:
    Fingerprint payload containing content hash and index config fields."""
        return {
            "content_hash": self.compute_text_hash(text),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @staticmethod
    def exists(meta_path: str) -> bool:
        """Return whether persisted artifact exists.

Args:
    meta_path: Path to fingerprint metadata JSON file.

Returns:
    True when required artifact exists; otherwise False."""
        return os.path.exists(meta_path)

    def save(self, text: str, meta_path: str) -> None:
        """Persist artifact/data to storage.

Args:
    text: Input text content.
    meta_path: Path to fingerprint metadata JSON file."""
        payload = self.build_fingerprint(text)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(meta_path: str) -> dict[str, Any]:
        """Load persisted artifact and return parsed object/data.

Args:
    meta_path: Path to fingerprint metadata JSON file.

Returns:
    Fingerprint dictionary loaded from ``meta_path`` JSON file."""
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def matches(self, text: str, meta_path: str) -> bool:
        """Return whether current fingerprint matches stored fingerprint.

Args:
    text: Input text content.
    meta_path: Path to fingerprint metadata JSON file.

Returns:
    True when fingerprint payload matches; otherwise False."""
        if not self.exists(meta_path):
            return False

        current = self.build_fingerprint(text)
        stored = self.load(meta_path)

        return current == stored

    @staticmethod
    def clear(meta_path: str) -> None:
        """Remove persisted artifacts for clean rebuild.

Args:
    meta_path: Path to fingerprint metadata JSON file."""
        if os.path.exists(meta_path):
            os.remove(meta_path)
            print("FingerprintHandler#clear: meta removed.")
