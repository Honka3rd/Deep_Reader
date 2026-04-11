import hashlib
import json
import os
from typing import Any


class FingerprintHandler:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def build_fingerprint(self, text: str) -> dict[str, Any]:
        return {
            "content_hash": self.compute_text_hash(text),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    @staticmethod
    def exists(meta_path: str) -> bool:
        return os.path.exists(meta_path)

    def save(self, text: str, meta_path: str) -> None:
        payload = self.build_fingerprint(text)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(meta_path: str) -> dict[str, Any]:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def matches(self, text: str, meta_path: str) -> bool:
        if not self.exists(meta_path):
            return False

        current = self.build_fingerprint(text)
        stored = self.load(meta_path)

        return current == stored

    @staticmethod
    def clear(meta_path: str) -> None:
        """
        刪除 fingerprint（meta.json）
        """
        if os.path.exists(meta_path):
            os.remove(meta_path)
            print("FingerprintHandler#clear: meta removed.")