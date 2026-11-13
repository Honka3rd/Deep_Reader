from pathlib import Path
import json
import logging
import os
import tempfile
from contextlib import contextmanager

from config.structured_document_storage_config import StructuredDocumentStorageConfig
from document_structure.structured_document import StructuredDocument
from logging_config import log_stage

logger = logging.getLogger(__name__)


class StructuredDocumentStore:
    """JSON persistence store for StructuredDocument artifacts."""

    @staticmethod
    @contextmanager
    def _publication_lock(target):
        """Same lock as repository mutations; never acquire it recursively."""
        import fcntl
        path = StructuredDocumentStore._resolve_path(target)
        lock_path = path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def profile_publication_snapshot(self, target):
        """Hold the snapshot stable through final sidecar validation/replacement."""
        with self._publication_lock(target):
            yield self.load(target) if self.exists(target) else None

    @classmethod
    def save(
        cls,
        document: StructuredDocument,
        target: str | StructuredDocumentStorageConfig,
    ) -> None:
        """Publish through the document boundary, including direct Prepare callers."""
        with cls._publication_lock(target):
            cls._save_unlocked(document, target)

    @classmethod
    def save_and_load(cls, document, target) -> StructuredDocument:
        """Return this write's persisted snapshot before a competitor can replace it."""
        with cls._publication_lock(target):
            cls._save_unlocked(document, target)
            with log_stage(logger, "structured_readback"):
                return cls.load(target)

    @staticmethod
    def _save_unlocked(document, target) -> None:
        """Serialization primitive; caller must already hold the document lock."""
        from document_structure.source_review_blocks import persist_region_mapping
        document = persist_region_mapping(document)
        path = StructuredDocumentStore._resolve_path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = document.to_dict()
        for chapter, chapter_data in zip(document.chapters, data["chapters"]):
            for section, section_data in zip(chapter.sections, chapter_data["sections"]):
                section_data["task_units"] = [unit.to_dict(include_content_blocks=True) for unit in section.task_units]
        payload = json.dumps(data, ensure_ascii=False, indent=2)
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
