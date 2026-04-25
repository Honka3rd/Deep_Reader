import json

from document_structure.structured_document import StructuredDocument, StructuredSection
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
from section_tasks.quiz_question import QuizQuestion
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder_factory import (
    SectionTaskPromptBuilderFactory,
    SectionTaskType,
)
from section_tasks.section_task_result import SectionTaskResult


class ChapterQuizService:
    """Generate chapter-level reading-comprehension quiz from structured section data."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_builder: SectionTaskContextBuilder,
        prompt_builder_factory: SectionTaskPromptBuilderFactory,
    ):
        """Initialize service with injected dependencies."""
        self.llm_provider = llm_provider
        self.context_builder = context_builder
        self.prompt_builder_factory = prompt_builder_factory

    def generate_quiz(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Generate quiz for one section id from a structured document."""
        task_context = self.context_builder.build_from_document(
            document=document,
            section_id=section_id,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt_builder = self.prompt_builder_factory.get_builder(
            SectionTaskType.QUIZ
        )
        if prompt_builder is None:
            return SectionTaskResult.fail(
                "quiz prompt builder is unavailable"
            )
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
        )
        try:
            raw_response = self.llm_provider.complete_text(prompt).strip()
            quiz_questions = self._parse_and_validate_quiz_questions(raw_response)
            return SectionTaskResult.ok(quiz_questions)
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_chapter_quiz(
        self,
        section: StructuredSection,
        document_title: str | None = None,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Generate quiz text from one structured chapter section."""
        task_context = self.context_builder.build_from_section(
            section=section,
            document_title=document_title,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt_builder = self.prompt_builder_factory.get_builder(
            SectionTaskType.QUIZ
        )
        if prompt_builder is None:
            return SectionTaskResult.fail(
                "quiz prompt builder is unavailable"
            )
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
        )
        try:
            raw_response = self.llm_provider.complete_text(prompt).strip()
            quiz_questions = self._parse_and_validate_quiz_questions(raw_response)
            return SectionTaskResult.ok(quiz_questions)
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    @staticmethod
    def _parse_and_validate_quiz_questions(raw_response: str) -> list[QuizQuestion]:
        """Parse LLM JSON response and validate quiz-question schema."""
        json_text = ChapterQuizService._extract_json_array_text(raw_response)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"quiz_parse_failed: invalid JSON array output: {error}"
            ) from error

        if not isinstance(payload, list):
            raise ValueError("quiz_parse_failed: output must be a JSON array")

        if len(payload) < 4:
            raise ValueError(
                f"quiz_validation_failed: expected at least 4 questions, got {len(payload)}"
            )

        quiz_questions: list[QuizQuestion] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(
                    "quiz_validation_failed: "
                    f"item at index {index} is not an object"
                )
            quiz_questions.append(QuizQuestion.from_dict(item))

        return quiz_questions

    @staticmethod
    def _extract_json_array_text(raw_response: str) -> str:
        """Extract JSON array string from raw LLM output."""
        text = raw_response.strip()
        if not text:
            raise ValueError("quiz_parse_failed: empty LLM output")

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()

        if text.startswith("[") and text.endswith("]"):
            return text

        start_index = text.find("[")
        end_index = text.rfind("]")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise ValueError("quiz_parse_failed: JSON array not found in LLM output")
        return text[start_index : end_index + 1]
