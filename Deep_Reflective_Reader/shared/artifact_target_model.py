from dataclasses import dataclass
from enum import Enum
from typing import Any


class ArtifactTargetLevel(str, Enum):
    """Stable artifact-target level labels for shared interaction metadata."""

    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    TASK_UNIT = "task_unit"
    CONTENT_BLOCK = "content_block"


@dataclass(frozen=True)
class ArtifactTargetRef:
    """Caller-neutral artifact target metadata reference."""

    target_level: ArtifactTargetLevel
    document_id: str | None = None
    chapter_id: str | None = None
    section_id: str | None = None
    task_unit_id: str | None = None
    content_block_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize target reference into JSON-friendly dictionary."""
        return {
            "target_level": self.target_level.value,
            "document_id": self.document_id,
            "chapter_id": self.chapter_id,
            "section_id": self.section_id,
            "task_unit_id": self.task_unit_id,
            "content_block_id": self.content_block_id,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactTargetRef":
        """Deserialize target reference from dictionary payload."""
        return cls(
            target_level=ArtifactTargetLevel(str(data["target_level"])),
            document_id=(
                None if data.get("document_id") is None else str(data.get("document_id"))
            ),
            chapter_id=(
                None if data.get("chapter_id") is None else str(data.get("chapter_id"))
            ),
            section_id=(
                None if data.get("section_id") is None else str(data.get("section_id"))
            ),
            task_unit_id=(
                None if data.get("task_unit_id") is None else str(data.get("task_unit_id"))
            ),
            content_block_id=(
                None
                if data.get("content_block_id") is None
                else str(data.get("content_block_id"))
            ),
            metadata=(
                None
                if data.get("metadata") is None
                else {str(key): value for key, value in data.get("metadata", {}).items()}
            ),
        )
