from abc import ABC, abstractmethod

from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import SectionTaskContext


class AbstractTaskPromptBuilder(ABC):
    """Abstract base class for task-specific section prompt builders."""

    @abstractmethod
    def build(
        self,
        *,
        context: SectionTaskContext,
        document_profile: DocumentProfile | None = None,
        language_code: LanguageCode | None = None,
    ) -> str:
        """Build one prompt for a concrete section task."""
        raise NotImplementedError
