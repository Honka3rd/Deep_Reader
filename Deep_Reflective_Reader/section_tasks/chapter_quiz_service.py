import json

from document_structure.structured_document import StructuredDocument, StructuredSection
from language.language_code import LanguageCode, LanguageCodeResolver
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
from section_tasks.quiz_question import QuizQuestion
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_resolver import TaskUnitResolver
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder_factory import (
    SectionTaskPromptBuilderFactory,
    SectionTaskType,
)
from section_tasks.section_task_result import SectionTaskResult


class ChapterQuizService:
    """Generate section/chapter quiz from structured section data."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_builder: SectionTaskContextBuilder,
        prompt_builder_factory: SectionTaskPromptBuilderFactory,
        task_unit_resolver: TaskUnitResolver,
        quiz_min_section_chars: int = 400,
    ):
        """Initialize service with injected dependencies."""
        self.llm_provider = llm_provider
        self.context_builder = context_builder
        self.prompt_builder_factory = prompt_builder_factory
        self.task_unit_resolver = task_unit_resolver
        self.quiz_min_section_chars = max(1, int(quiz_min_section_chars))

    def generate_section_quiz(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Adapter: generate section quiz by section id via task-unit resolution."""
        task_unit_result = self._resolve_task_unit_for_section(
            document=document,
            section_id=section_id,
        )
        return self.generate_task_unit_quiz(
            task_unit=task_unit_result.task_unit,
            document_title=document.title,
            document_profile=document_profile,
            task_type=SectionTaskType.SECTION_QUIZ,
            task_unit_index=task_unit_result.unit_index,
        )

    def generate_task_unit_quiz(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile: DocumentProfile | None = None,
        *,
        task_type: SectionTaskType = SectionTaskType.SECTION_QUIZ,
        task_unit_index: int = 0,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Canonical quiz execution path based on TaskUnit."""
        task_context = self.context_builder.build_from_task_unit(
            task_unit=task_unit,
            document_title=document_title,
            section_index=task_unit_index,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        skipped_reason = self._build_min_length_skip_reason(task_context.section_content)
        if skipped_reason is not None:
            return SectionTaskResult.fail(skipped_reason)
        prompt_builder = self.prompt_builder_factory.get_builder(task_type)
        if prompt_builder is None:
            return SectionTaskResult.fail(
                f"{task_type.value} prompt builder is unavailable"
            )
        language_code = self._resolve_language_code(document_profile)
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
            language_code=language_code,
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
        """Adapter: generate chapter quiz by section metadata via task-unit resolution."""
        synthetic_document = self._build_synthetic_document_from_section(
            section=section,
            document_title=document_title,
        )
        task_unit_result = self._resolve_task_unit_for_section(
            document=synthetic_document,
            section_id=section.section_id,
        )
        return self.generate_task_unit_quiz(
            task_unit=task_unit_result.task_unit,
            document_title=document_title,
            document_profile=document_profile,
            task_type=SectionTaskType.CHAPTER_QUIZ,
            task_unit_index=task_unit_result.unit_index,
        )

    @staticmethod
    def _resolve_language_code(
        document_profile: DocumentProfile | None,
    ) -> LanguageCode:
        if document_profile is None:
            return LanguageCode.UNKNOWN
        return LanguageCodeResolver.resolve(document_profile.document_language)

    def _resolve_task_unit_for_section(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> "_ResolvedTaskUnit":
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")

        task_units = self.task_unit_resolver.resolve(document)
        if not task_units:
            raise ValueError(
                f"no task units resolved for document '{document.document_id}'"
            )

        for unit_index, task_unit in enumerate(task_units):
            if normalized_section_id in task_unit.source_section_ids:
                return _ResolvedTaskUnit(task_unit=task_unit, unit_index=unit_index)

        raise ValueError(
            f"section_id '{normalized_section_id}' not found in resolved task units "
            f"for document '{document.document_id}'"
        )

    def _build_min_length_skip_reason(self, section_content: str) -> str | None:
        """Return skip reason when section content is shorter than configured threshold."""
        current_chars = len(section_content.strip())
        if current_chars >= self.quiz_min_section_chars:
            return None
        return (
            "quiz_generation_skipped: section content too short "
            f"(current={current_chars}, required>={self.quiz_min_section_chars})"
        )

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

    @staticmethod
    def _build_synthetic_document_from_section(
        *,
        section: StructuredSection,
        document_title: str | None,
    ) -> StructuredDocument:
        resolved_title = (document_title or "").strip() or "Unknown Document"
        content = section.content
        return StructuredDocument(
            document_id=f"synthetic-{section.section_id}",
            title=resolved_title,
            source_path=None,
            language=None,
            raw_text=content,
            sections=[section],
        )


class _ResolvedTaskUnit:
    """Internal holder for resolved task unit + position."""

    def __init__(self, task_unit: TaskUnit, unit_index: int):
        self.task_unit = task_unit
        self.unit_index = unit_index
