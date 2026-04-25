from abc import ABC, abstractmethod

from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext


class SectionTaskPromptBuilder(ABC):
    """Abstract base class for task-specific section prompt builders."""

    @abstractmethod
    def build(
        self,
        *,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
    ) -> str:
        """Build one prompt for a concrete section task."""
        raise NotImplementedError
