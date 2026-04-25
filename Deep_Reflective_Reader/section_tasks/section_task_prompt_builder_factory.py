from enum import StrEnum

from section_tasks.chapter_quiz_task_prompt_builder import (
    ChapterQuizTaskPromptBuilder,
)
from section_tasks.section_quiz_task_prompt_builder import (
    SectionQuizTaskPromptBuilder,
)
from section_tasks.abstract_task_prompt_builder import AbstractTaskPromptBuilder
from section_tasks.summary_task_prompt_builder import SummaryTaskPromptBuilder


class SectionTaskType(StrEnum):
    """Supported section task types for prompt-builder routing."""

    SUMMARY = "summary"
    SECTION_QUIZ = "section_quiz"
    CHAPTER_QUIZ = "chapter_quiz"

    @classmethod
    def resolve(cls, task_type: "SectionTaskType | str") -> "SectionTaskType":
        """Resolve raw input to supported task type enum."""
        if isinstance(task_type, cls):
            return task_type
        normalized = task_type.strip().lower()
        if normalized == "quiz":
            # Backward compatibility: legacy generic quiz defaults to section quiz.
            normalized = cls.SECTION_QUIZ.value
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
        section_quiz_builder: SectionQuizTaskPromptBuilder | None,
        chapter_quiz_builder: ChapterQuizTaskPromptBuilder | None,
    ):
        self.summary_builder = summary_builder
        self.section_quiz_builder = section_quiz_builder
        self.chapter_quiz_builder = chapter_quiz_builder

    def get_builder(
        self, task_type: SectionTaskType | str
    ) -> AbstractTaskPromptBuilder | None:
        """Return task-specific prompt builder based on task type."""
        resolved = SectionTaskType.resolve(task_type)
        if resolved == SectionTaskType.SUMMARY:
            return self.summary_builder
        if resolved == SectionTaskType.SECTION_QUIZ:
            return self.section_quiz_builder
        if resolved == SectionTaskType.CHAPTER_QUIZ:
            return self.chapter_quiz_builder
        return None

    def get_summary_builder(self) -> SummaryTaskPromptBuilder | None:
        """Return summary prompt builder."""
        return self.summary_builder

    def get_section_quiz_builder(self) -> SectionQuizTaskPromptBuilder | None:
        """Return section-quiz prompt builder."""
        return self.section_quiz_builder

    def get_chapter_quiz_builder(self) -> ChapterQuizTaskPromptBuilder | None:
        """Return chapter-quiz prompt builder."""
        return self.chapter_quiz_builder
