from abc import ABC, abstractmethod

from document_structure.structured_document import StructuredSection
from section_tasks.task_unit import TaskUnit


class AbstractTaskUnitSplitResolver(ABC):
    """Abstract single-section split resolver for task-unit generation."""

    @abstractmethod
    def split_section(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> list[TaskUnit]:
        """Split one section into section-internal task units."""
        raise NotImplementedError
