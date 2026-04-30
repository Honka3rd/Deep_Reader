from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from shared.task_artifacts import TaskArtifacts
from shared.task_unit_model import TaskUnit


class StructuredNodeType(StrEnum):
    """Document hierarchy node type."""

    DOCUMENT = "document"
    FRONT_MATTER = "front_matter"
    CHAPTER = "chapter"
    SECTION = "section"
    APPENDIX = "appendix"
    BACK_MATTER = "back_matter"

    @classmethod
    def resolve(cls, value: str | None) -> "StructuredNodeType | None":
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        for member in cls:
            if member.value == normalized:
                return member
        return None


@dataclass(frozen=True)
class StructuredDocumentNode:
    """Hierarchy node derived from flat structured sections (backward-compatible addition)."""

    node_id: str
    node_type: StructuredNodeType
    title: str | None
    level: int
    content: str
    char_start: int
    char_end: int
    section_role: str | None = None
    children: list["StructuredDocumentNode"] = field(default_factory=list)
    task_units: list[TaskUnit] = field(default_factory=list)
    task_artifacts: TaskArtifacts | None = None
    source_section_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_role": self.section_role,
            "children": [child.to_dict() for child in self.children],
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
            "task_artifacts": (
                None if self.task_artifacts is None else self.task_artifacts.to_dict()
            ),
            "source_section_ids": list(self.source_section_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredDocumentNode":
        resolved_type = StructuredNodeType.resolve(data.get("node_type"))
        if resolved_type is None:
            raise ValueError(
                "StructuredDocumentNode.from_dict: invalid or missing node_type"
            )
        children_payload = data.get("children", [])
        return cls(
            node_id=str(data["node_id"]),
            node_type=resolved_type,
            title=(None if data.get("title") is None else str(data.get("title"))),
            level=int(data.get("level", 1)),
            content=str(data.get("content", "")),
            char_start=int(data.get("char_start", 0)),
            char_end=int(data.get("char_end", 0)),
            section_role=(
                None
                if data.get("section_role") is None
                else str(data.get("section_role"))
            ),
            children=[
                StructuredDocumentNode.from_dict(payload)
                for payload in children_payload
                if isinstance(payload, dict)
            ],
            task_units=[
                TaskUnit.from_dict(payload)
                for payload in data.get("task_units", [])
                if isinstance(payload, dict)
            ],
            task_artifacts=TaskArtifacts.from_dict(data.get("task_artifacts")),
            source_section_ids=[
                str(section_id) for section_id in data.get("source_section_ids", [])
            ],
        )
