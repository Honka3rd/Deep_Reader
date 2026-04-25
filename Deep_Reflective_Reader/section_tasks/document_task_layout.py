from dataclasses import dataclass
from typing import Any

from document_structure.structured_document import StructuredDocument
from section_tasks.task_unit import TaskUnit


@dataclass(frozen=True)
class DocumentTaskLayout:
    """View model that exposes structured view + task-unit view + mappings."""

    structured_document: StructuredDocument
    task_units: list[TaskUnit]
    section_to_task_unit_ids: dict[str, list[str]]
    task_unit_to_section_ids: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize layout into JSON-friendly dictionary."""
        return {
            "structured_document": self.structured_document.to_dict(),
            "task_units": [unit.to_dict() for unit in self.task_units],
            "section_to_task_unit_ids": {
                section_id: list(unit_ids)
                for section_id, unit_ids in self.section_to_task_unit_ids.items()
            },
            "task_unit_to_section_ids": {
                unit_id: list(section_ids)
                for unit_id, section_ids in self.task_unit_to_section_ids.items()
            },
        }
