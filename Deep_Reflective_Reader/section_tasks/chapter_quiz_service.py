from document_structure.structured_document import StructuredDocument, StructuredSection
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder import (
    SectionTaskPromptBuilder,
    SectionTaskType,
)
from section_tasks.section_task_result import SectionTaskResult


class ChapterQuizService:
    """Generate chapter-level reading-comprehension quiz from structured section data."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_builder: SectionTaskContextBuilder,
        prompt_builder: SectionTaskPromptBuilder,
    ):
        """Initialize service with injected dependencies."""
        self.llm_provider = llm_provider
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder

    def generate_quiz(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult:
        """Generate quiz for one section id from a structured document."""
        task_context = self.context_builder.build_from_document(
            document=document,
            section_id=section_id,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.QUIZ,
            context=task_context,
            document_profile=document_profile,
        )
        try:
            return SectionTaskResult.ok(
                self.llm_provider.complete_text(prompt).strip()
            )
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_chapter_quiz(
        self,
        section: StructuredSection,
        document_title: str | None = None,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult:
        """Generate quiz text from one structured chapter section."""
        task_context = self.context_builder.build_from_section(
            section=section,
            document_title=document_title,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.QUIZ,
            context=task_context,
            document_profile=document_profile,
        )
        try:
            return SectionTaskResult.ok(
                self.llm_provider.complete_text(prompt).strip()
            )
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)
