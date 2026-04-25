from document_structure.structured_document import StructuredDocument, StructuredSection
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
from section_tasks.section_task_context_builder import (
    SectionTaskContextBuilder,
)
from section_tasks.section_task_prompt_builder_factory import (
    SectionTaskPromptBuilderFactory,
    SectionTaskType,
)
from section_tasks.section_task_result import SectionTaskResult


class ChapterSummaryService:
    """Summarize one structured chapter/section without retrieval dependencies."""

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

    def summarize_section(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult:
        """Summarize a section by id from one structured document."""
        task_context = self.context_builder.build_from_document(
            document=document,
            section_id=section_id,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt_builder = self.prompt_builder_factory.get_builder(
            SectionTaskType.SUMMARY
        )
        if prompt_builder is None:
            return SectionTaskResult.fail(
                "summary prompt builder is unavailable"
            )
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
        )
        try:
            return SectionTaskResult.ok(
                self.llm_provider.complete_text(prompt).strip()
            )
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def summarize_chapter(
        self,
        section: StructuredSection,
        document_title: str | None = None,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult:
        """Summarize one structured section directly from section metadata + content."""
        task_context = self.context_builder.build_from_section(
            section=section,
            document_title=document_title,
        )
        if not task_context.valid:
            reason = task_context.reason.value if task_context.reason else "invalid section task context"
            return SectionTaskResult.fail(reason)
        prompt_builder = self.prompt_builder_factory.get_builder(
            SectionTaskType.SUMMARY
        )
        if prompt_builder is None:
            return SectionTaskResult.fail(
                "summary prompt builder is unavailable"
            )
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
        )
        try:
            return SectionTaskResult.ok(
                self.llm_provider.complete_text(prompt).strip()
            )
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)
