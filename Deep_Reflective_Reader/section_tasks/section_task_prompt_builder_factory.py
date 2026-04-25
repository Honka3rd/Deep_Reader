from enum import StrEnum

from section_tasks.quiz_task_prompt_builder import QuizTaskPromptBuilder
from section_tasks.section_task_prompt_builder import SectionTaskPromptBuilder
from section_tasks.summary_task_prompt_builder import SummaryTaskPromptBuilder


class SectionTaskType(StrEnum):
    """Supported section task types for prompt-builder routing."""

    SUMMARY = "summary"
    QUIZ = "quiz"

    @classmethod
    def resolve(cls, task_type: "SectionTaskType | str") -> "SectionTaskType":
        """Resolve raw input to supported task type enum."""
        if isinstance(task_type, cls):
            return task_type
        normalized = task_type.strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            supported = ", ".join(value.value for value in cls)
            raise ValueError(
                f"unsupported section task type: {task_type!r}. supported: {supported}"
            ) from error


class SectionTaskPromptBuilderFactory:
    """Factory that returns task-specific prompt builders."""

    def __init__(
        self,
        summary_builder: SummaryTaskPromptBuilder | None,
        quiz_builder: QuizTaskPromptBuilder | None,
    ):
        self.summary_builder = summary_builder
        self.quiz_builder = quiz_builder

    def get_builder(
        self, task_type: SectionTaskType | str
    ) -> SectionTaskPromptBuilder | None:
        """Return task-specific prompt builder based on task type."""
        resolved = SectionTaskType.resolve(task_type)
        if resolved == SectionTaskType.SUMMARY:
            return self.summary_builder
        if resolved == SectionTaskType.QUIZ:
            return self.quiz_builder
        return None

    def get_summary_builder(self) -> SummaryTaskPromptBuilder | None:
        """Return summary prompt builder."""
        return self.summary_builder

    def get_quiz_builder(self) -> QuizTaskPromptBuilder | None:
        """Return quiz prompt builder."""
        return self.quiz_builder
