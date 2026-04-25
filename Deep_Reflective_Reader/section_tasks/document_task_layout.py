from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionTaskMode(str, Enum):
    """Section-to-task-unit relationship mode for section-first UI rendering."""

    DIRECT = "direct"
    SPLIT = "split"
    MERGED = "merged"


@dataclass(frozen=True)
class TaskUnitDTO:
    """Frontend-safe task-unit metadata without large content payload."""

    unit_id: str
    title: str | None
    container_title: str | None
    source_section_ids: list[str]
    is_fallback_generated: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize task-unit metadata into JSON-friendly dictionary."""
        return {
            "unit_id": self.unit_id,
            "title": self.title,
            "container_title": self.container_title,
            "source_section_ids": list(self.source_section_ids),
            "is_fallback_generated": self.is_fallback_generated,
        }


@dataclass(frozen=True)
class DocumentTaskLayoutSectionDTO:
    """Section-first layout node with embedded task-unit metadata."""

    section_id: str
    title: str | None
    container_title: str | None
    task_mode: SectionTaskMode
    task_units: list[TaskUnitDTO]

    def to_dict(self) -> dict[str, Any]:
        """Serialize section layout node into JSON-friendly dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "container_title": self.container_title,
            "task_mode": self.task_mode.value,
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
        }


@dataclass(frozen=True)
class DocumentTaskLayout:
    """Frontend-consumable layout that keeps structured-first and task-unit metadata."""

    document_id: str
    title: str
    language: str | None
    sections: list[DocumentTaskLayoutSectionDTO]
    task_units: list[TaskUnitDTO] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize layout into JSON-friendly dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "language": self.language,
            "sections": [section.to_dict() for section in self.sections],
            "task_units": [task_unit.to_dict() for task_unit in self.task_units],
        }
