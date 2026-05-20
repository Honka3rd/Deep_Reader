from dataclasses import dataclass
from typing import Any

from shared.task_artifacts import TaskArtifacts


@dataclass(frozen=True)
class TaskUnitContentBlock:
    """Shared-layer render/interaction content block under a task unit."""

    block_id: str
    content: str
    block_type: str | None = None
    artifact_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize content block into JSON-friendly dictionary."""
        return {
            "block_id": self.block_id,
            "content": self.content,
            "block_type": self.block_type,
            "artifact_ids": None if self.artifact_ids is None else list(self.artifact_ids),
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnitContentBlock":
        """Deserialize content block from dictionary payload."""
        artifact_ids_payload = data.get("artifact_ids")
        metadata_payload = data.get("metadata")
        return cls(
            block_id=str(data["block_id"]),
            content=str(data.get("content", "")),
            block_type=None if data.get("block_type") is None else str(data.get("block_type")),
            artifact_ids=(
                None
                if artifact_ids_payload is None
                else [str(value) for value in artifact_ids_payload]
            ),
            metadata=(
                None
                if metadata_payload is None
                else {str(key): value for key, value in metadata_payload.items()}
            ),
        )


def build_default_content_block_id(task_unit_id: str, block_index: int = 0) -> str:
    """Build deterministic task-unit content block id for adapter-generated blocks."""
    return f"{task_unit_id}:content:{block_index}"


@dataclass(frozen=True)
class TaskUnit:
    """Resolved task-time unit for section-based summary/quiz execution."""

    unit_id: str
    title: str | None
    container_title: str | None
    content: str
    source_section_ids: list[str]
    is_fallback_generated: bool
    parent_section_id: str | None = None
    task_artifacts: TaskArtifacts | None = None

    def to_content_blocks(self) -> list[TaskUnitContentBlock]:
        """
        Adapt current string content to content blocks.

        Empty content returns an empty list instead of an empty block payload.
        """
        if self.content == "":
            return []
        return [
            TaskUnitContentBlock(
                block_id=build_default_content_block_id(task_unit_id=self.unit_id, block_index=0),
                content=self.content,
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize task unit into JSON-friendly dictionary."""
        return {
            "unit_id": self.unit_id,
            "title": self.title,
            "container_title": self.container_title,
            "content": self.content,
            "source_section_ids": list(self.source_section_ids),
            "is_fallback_generated": self.is_fallback_generated,
            "parent_section_id": self.parent_section_id,
            "task_artifacts": (
                None if self.task_artifacts is None else self.task_artifacts.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskUnit":
        """Deserialize task unit from dictionary payload."""
        source_ids_payload = data.get("source_section_ids", [])
        source_ids = [str(value) for value in source_ids_payload]
        return cls(
            unit_id=str(data["unit_id"]),
            title=(
                None if data.get("title") is None else str(data.get("title"))
            ),
            container_title=(
                None
                if data.get("container_title") is None
                else str(data.get("container_title"))
            ),
            content=str(data["content"]),
            source_section_ids=source_ids,
            is_fallback_generated=bool(data.get("is_fallback_generated", False)),
            parent_section_id=(
                None
                if data.get("parent_section_id") is None
                else str(data.get("parent_section_id"))
            ),
            task_artifacts=TaskArtifacts.from_dict(data.get("task_artifacts")),
        )
