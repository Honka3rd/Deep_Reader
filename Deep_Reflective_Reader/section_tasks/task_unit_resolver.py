from document_structure.structured_document import StructuredDocument, StructuredSection
from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from section_tasks.task_unit_split_resolver_selector import TaskUnitSplitResolverSelector
from section_tasks.task_unit import TaskUnit


class TaskUnitResolver:
    """Resolve task-time units from structured sections with minimal fallbacks."""

    def __init__(
        self,
        task_unit_min_chars: int = 300,
        task_unit_max_chars: int = 1600,
        split_resolver_selector: TaskUnitSplitResolverSelector | None = None,
        split_mode: TaskUnitSplitMode | str = TaskUnitSplitMode.SEMANTIC_SAFE,
    ):
        self.task_unit_min_chars = max(1, int(task_unit_min_chars))
        self.task_unit_max_chars = max(self.task_unit_min_chars, int(task_unit_max_chars))
        self.split_resolver_selector = (
            split_resolver_selector or TaskUnitSplitResolverSelector()
        )
        self.split_mode = TaskUnitSplitMode.resolve(split_mode)

    def resolve(self, document: StructuredDocument) -> list[TaskUnit]:
        """Resolve usable task units from one structured document."""
        if not document.sections:
            return []

        if self._should_split_single_huge_section(document):
            only_section = document.sections[0]
            return self._split_single_huge_section(only_section)

        return self._merge_short_adjacent_sections(document.sections)

    def _should_split_single_huge_section(self, document: StructuredDocument) -> bool:
        """Return True when single-section document needs fallback split."""
        if len(document.sections) != 1:
            return False
        section = document.sections[0]
        return len(section.content.strip()) > self.task_unit_max_chars

    def _split_single_huge_section(self, section: StructuredSection) -> list[TaskUnit]:
        """Split one huge section through selected single-section split resolver."""
        split_units = self._split_huge_section_with_resolver(
            section=section,
            section_index=0,
        )
        if not split_units:
            return []
        return [
            TaskUnit(
                unit_id=f"task-unit-{index}",
                title=unit.title,
                container_title=unit.container_title,
                content=unit.content,
                source_section_ids=unit.source_section_ids,
                is_fallback_generated=unit.is_fallback_generated,
            )
            for index, unit in enumerate(split_units)
        ]

    def _merge_short_adjacent_sections(
        self,
        sections: list[StructuredSection],
    ) -> list[TaskUnit]:
        """Merge adjacent short sections into larger units for task stability."""
        pending_units = self._expand_sections_to_base_units(sections)
        if not pending_units:
            return []

        resolved_units: list[TaskUnit] = []
        pending_index = 0
        while pending_index < len(pending_units):
            current = pending_units[pending_index]
            if len(current.content) >= self.task_unit_min_chars:
                resolved_units.append(current)
                pending_index += 1
                continue

            can_merge_next = pending_index + 1 < len(pending_units)
            can_merge_prev = len(resolved_units) > 0
            if can_merge_next and self._same_container(
                current.container_title,
                pending_units[pending_index + 1].container_title,
            ):
                pending_units[pending_index + 1] = self._merge_two_units(
                    current,
                    pending_units[pending_index + 1],
                )
                pending_index += 1
                continue
            if can_merge_prev and self._same_container(
                current.container_title,
                resolved_units[-1].container_title,
            ):
                resolved_units[-1] = self._merge_two_units(
                    resolved_units[-1],
                    current,
                )
                pending_index += 1
                continue
            if can_merge_next:
                pending_units[pending_index + 1] = self._merge_two_units(
                    current,
                    pending_units[pending_index + 1],
                )
                pending_index += 1
                continue
            if can_merge_prev:
                resolved_units[-1] = self._merge_two_units(
                    resolved_units[-1],
                    current,
                )
                pending_index += 1
                continue

            resolved_units.append(current)
            pending_index += 1

        return [
            TaskUnit(
                unit_id=f"task-unit-{index}",
                title=unit.title,
                container_title=unit.container_title,
                content=unit.content,
                source_section_ids=unit.source_section_ids,
                is_fallback_generated=unit.is_fallback_generated,
            )
            for index, unit in enumerate(resolved_units)
        ]

    def _expand_sections_to_base_units(
        self,
        sections: list[StructuredSection],
    ) -> list[TaskUnit]:
        """Expand sections into base units, splitting only inside huge sections."""
        base_units: list[TaskUnit] = []
        for section_index, section in enumerate(sections):
            content = section.content.strip()
            if not content:
                continue
            if len(content) > self.task_unit_max_chars:
                base_units.extend(
                    self._split_huge_section_with_resolver(
                        section=section,
                        section_index=section_index,
                    )
                )
                continue
            base_units.append(
                TaskUnit(
                    unit_id=f"task-unit-{section_index}",
                    title=self._normalize_optional_text(section.title),
                    container_title=self._normalize_optional_text(section.container_title),
                    content=content,
                    source_section_ids=[section.section_id],
                    is_fallback_generated=False,
                )
            )
        return base_units

    def _split_huge_section_with_resolver(
        self,
        *,
        section: StructuredSection,
        section_index: int,
    ) -> list[TaskUnit]:
        """Delegate section-internal split to selected split resolver."""
        resolver: AbstractTaskUnitSplitResolver = self.split_resolver_selector.get_resolver(
            self.split_mode
        )
        return resolver.split_section(
            section=section,
            section_index=section_index,
            task_unit_min_chars=self.task_unit_min_chars,
            task_unit_max_chars=self.task_unit_max_chars,
        )

    @staticmethod
    def _merge_two_units(left: TaskUnit, right: TaskUnit) -> TaskUnit:
        """Merge two units while preserving source traceability and order."""
        merged_content = f"{left.content}\n\n{right.content}".strip()
        merged_source_ids: list[str] = []
        for section_id in left.source_section_ids + right.source_section_ids:
            if section_id not in merged_source_ids:
                merged_source_ids.append(section_id)

        merged_title = left.title or right.title
        merged_container = left.container_title or right.container_title
        return TaskUnit(
            unit_id=left.unit_id,
            title=merged_title,
            container_title=merged_container,
            content=merged_content,
            source_section_ids=merged_source_ids,
            is_fallback_generated=True,
        )

    @staticmethod
    def _same_container(left: str | None, right: str | None) -> bool:
        """Check whether two units belong to the same container bucket."""
        left_value = TaskUnitResolver._normalize_optional_text(left)
        right_value = TaskUnitResolver._normalize_optional_text(right)
        return left_value == right_value

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize optional text to either None or non-empty string."""
        normalized = (value or "").strip()
        if not normalized:
            return None
        return normalized
