from dataclasses import dataclass
from typing import Any

from shared.task_artifacts import TaskArtifacts


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
