from __future__ import annotations

from dataclasses import replace

from document_structure.structured_document import StructuredDocument, StructuredSection
from shared.task_unit_model import TaskUnit


class TaskUnitIdNormalizer:
    """Normalize persisted task-unit ids to document-scope unique global ids."""

    _UNIT_ID_PREFIX = "task-unit-"

    def normalize_task_units_by_section_id(
        self,
        *,
        document: StructuredDocument,
        task_units_by_section_id: dict[str, list[TaskUnit]],
    ) -> dict[str, list[TaskUnit]]:
        """Return section->task_units mapping with document-global unique unit ids."""
        normalized: dict[str, list[TaskUnit]] = {
            section.section_id: [] for section in document.sections
        }
        global_index = 0
        for section in document.sections:
            section_units = task_units_by_section_id.get(section.section_id, [])
            normalized_units: list[TaskUnit] = []
            for task_unit in section_units:
                normalized_units.append(
                    replace(
                        task_unit,
                        unit_id=f"{self._UNIT_ID_PREFIX}{global_index}",
                    )
                )
                global_index += 1
            normalized[section.section_id] = normalized_units
        return normalized

    def normalize_document_task_unit_ids(
        self,
        *,
        document: StructuredDocument,
    ) -> StructuredDocument:
        """Return new document with section.task_units ids normalized globally by reading order."""
        global_index = 0
        updated_sections: list[StructuredSection] = []
        for section in document.sections:
            updated_units: list[TaskUnit] = []
            for task_unit in section.task_units:
                updated_units.append(
                    replace(
                        task_unit,
                        unit_id=f"{self._UNIT_ID_PREFIX}{global_index}",
                    )
                )
                global_index += 1
            updated_sections.append(replace(section, task_units=updated_units))
        return replace(document, sections=updated_sections)

    @staticmethod
    def collect_task_unit_id_counts(
        document: StructuredDocument,
    ) -> dict[str, int]:
        """Return task-unit id occurrence counts in one persisted document."""
        counts: dict[str, int] = {}
        for section in document.sections:
            for task_unit in section.task_units:
                counts[task_unit.unit_id] = counts.get(task_unit.unit_id, 0) + 1
        return counts

    def find_duplicate_task_unit_ids(
        self,
        *,
        document: StructuredDocument,
    ) -> dict[str, int]:
        """Return duplicate task-unit ids with occurrence count (>1)."""
        counts = self.collect_task_unit_id_counts(document)
        return {
            unit_id: count
            for unit_id, count in counts.items()
            if count > 1
        }

    def has_duplicate_task_unit_ids(
        self,
        *,
        document: StructuredDocument,
    ) -> bool:
        """Return whether persisted task-unit ids contain duplicates."""
        return bool(self.find_duplicate_task_unit_ids(document=document))

    def assert_unique_task_unit_ids(
        self,
        *,
        document: StructuredDocument,
        context: str,
    ) -> None:
        """Raise ValueError when duplicated task-unit ids are detected."""
        duplicates = self.find_duplicate_task_unit_ids(document=document)
        if duplicates:
            duplicate_repr = ", ".join(
                f"{unit_id}:{count}"
                for unit_id, count in sorted(duplicates.items())
            )
            raise ValueError(
                f"{context}: duplicate task_unit_id detected -> {duplicate_repr}"
            )

