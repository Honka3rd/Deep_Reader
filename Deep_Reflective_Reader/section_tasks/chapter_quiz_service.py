from document_structure.structured_document import StructuredDocument, StructuredSection
from llm.llm_provider import LLMProvider
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder import (
    SectionTaskPromptBuilder,
    SectionTaskType,
)


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
        document_profile: object | None = None,
    ) -> str:
        """Generate quiz for one section id from a structured document."""
        task_context = self.context_builder.build_from_document(
            document=document,
            section_id=section_id,
        )
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.QUIZ,
            context=task_context,
            document_profile=document_profile,
        )
        return self.llm_provider.complete_text(prompt).strip()

    def generate_chapter_quiz(
        self,
        section: StructuredSection,
        document_title: str | None = None,
        document_profile: object | None = None,
    ) -> str:
        """Generate quiz text from one structured chapter section."""
        task_context = self.context_builder.build_from_section(
            section=section,
            document_title=document_title,
        )
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.QUIZ,
            context=task_context,
            document_profile=document_profile,
        )
        return self.llm_provider.complete_text(prompt).strip()
