from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskUnit:
    """Resolved task-time unit for section-based summary/quiz execution."""

    unit_id: str
    title: str | None
    container_title: str | None
    content: str
    source_section_ids: list[str]
    is_fallback_generated: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize task unit into JSON-friendly dictionary."""
        return {
            "unit_id": self.unit_id,
            "title": self.title,
            "container_title": self.container_title,
            "content": self.content,
            "source_section_ids": list(self.source_section_ids),
            "is_fallback_generated": self.is_fallback_generated,
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
        )
