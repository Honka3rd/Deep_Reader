"""Request-local format-neutral source capture and lazy extraction contract."""
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any

from .extraction_identity import ExtractionIdentity, ExtractionResult


@dataclass(frozen=True)
class SourceTextSpan:
    char_start: int
    char_end: int
    page_index: int | None = None


@dataclass(frozen=True)
class SourceArtifact:
    result: ExtractionResult
    page_boundaries: tuple = ()
    outline_provider: Callable | None = None
    layout_provider: Callable | None = None
    storage_metadata: dict = field(default_factory=dict)
    ocr_run: dict | None = None
    # None means unknown; an empty tuple proves no OCR text in this extraction.
    ocr_spans: tuple[SourceTextSpan, ...] | None = None


class SourceSession:
    """One captured source revision; cache checks never invoke its extractor."""
    def __init__(self, *, identity: ExtractionIdentity, source_path: Path,
                 extract: Callable[[], SourceArtifact], status: Callable[[str], dict] | None = None,
                 source_resolver: Callable[[], Path] | None = None):
        self.identity = identity
        self.source_path = source_path
        self._extract = extract
        self._status = status
        self._artifact = None
        self._source_resolver = source_resolver or (lambda: self.source_path)
        self._captured_target = source_path.resolve()
        self._captured_revision = self._revision(source_path.stat())

    def extract(self) -> SourceArtifact:
        if self._artifact is None:
            artifact = self._extract()
            if artifact.result.identity != self.identity:
                raise ValueError("extraction_identity_mismatch")
            self._artifact = artifact
        return self._artifact

    def verify_source(self):
        path = self._source_resolver()
        if path.resolve() != self._captured_target:
            raise ValueError("Extraction source changed before structured save")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current_path = self._source_resolver()
        current = current_path.stat()
        if (current_path.resolve() != self._captured_target
                or self._captured_revision != self._revision(before)
                or self._revision(before) != self._revision(after)
                or self._revision(after) != self._revision(current)
                or digest.hexdigest() != self.identity.source_sha256):
            raise ValueError("Extraction source changed before structured save")

    @staticmethod
    def _revision(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def compatibility_status(self, state: str) -> dict[str, Any]:
        return self._status(state) if self._status else {}
