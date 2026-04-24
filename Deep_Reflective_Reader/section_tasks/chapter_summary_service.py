from document_structure.structured_document import StructuredDocument, StructuredSection
from llm.llm_provider import LLMProvider
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder import (
    SectionTaskPromptBuilder,
    SectionTaskType,
)


class ChapterSummaryService:
    """Summarize one structured chapter/section without retrieval dependencies."""

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

    def summarize_section(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: object | None = None,
    ) -> str:
        """Summarize a section by id from one structured document."""
        task_context = self.context_builder.build_from_document(
            document=document,
            section_id=section_id,
        )
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.SUMMARY,
            context=task_context,
            document_profile=document_profile,
        )
        return self.llm_provider.complete_text(prompt).strip()

    def summarize_chapter(
        self,
        section: StructuredSection,
        document_title: str | None = None,
        document_profile: object | None = None,
    ) -> str:
        """Summarize one structured section directly from section metadata + content."""
        task_context = self.context_builder.build_from_section(
            section=section,
            document_title=document_title,
        )
        prompt = self.prompt_builder.build(
            task_type=SectionTaskType.SUMMARY,
            context=task_context,
            document_profile=document_profile,
        )
        return self.llm_provider.complete_text(prompt).strip()
