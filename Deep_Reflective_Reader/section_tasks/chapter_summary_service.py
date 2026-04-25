from document_structure.structured_document import StructuredDocument, StructuredSection
from language.language_code import LanguageCode, LanguageCodeResolver
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
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


class ChapterSummaryService:
    """Summarize one structured chapter/section without retrieval dependencies."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        context_builder: SectionTaskContextBuilder,
        prompt_builder_factory: SectionTaskPromptBuilderFactory,
        task_unit_resolver: TaskUnitResolver,
    ):
        """Initialize service with injected dependencies."""
        self.llm_provider = llm_provider
        self.context_builder = context_builder
        self.prompt_builder_factory = prompt_builder_factory
        self.task_unit_resolver = task_unit_resolver

    def summarize_section(
        self,
        document: StructuredDocument,
        section_id: str,
        document_profile: DocumentProfile | None = None,
    ) -> SectionTaskResult:
        """Adapter: summarize by section id via task-unit resolution."""
        task_unit_result = self._resolve_task_unit_for_section(
            document=document,
            section_id=section_id,
        )
        return self.summarize_task_unit(
            task_unit=task_unit_result.task_unit,
            document_title=document.title,
            document_profile=document_profile,
            task_unit_index=task_unit_result.unit_index,
        )

    def summarize_task_unit(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile: DocumentProfile | None = None,
        *,
        task_unit_index: int = 0,
    ) -> SectionTaskResult:
        """Canonical summary execution path based on TaskUnit."""
        task_context = self.context_builder.build_from_task_unit(
            task_unit=task_unit,
            document_title=document_title,
            section_index=task_unit_index,
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
        language_code = self._resolve_language_code(document_profile)
        prompt = prompt_builder.build(
            context=task_context,
            document_profile=document_profile,
            language_code=language_code,
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
        """Adapter: summarize by chapter section via task-unit resolution."""
        synthetic_document = self._build_synthetic_document_from_section(
            section=section,
            document_title=document_title,
        )
        task_unit_result = self._resolve_task_unit_for_section(
            document=synthetic_document,
            section_id=section.section_id,
        )
        return self.summarize_task_unit(
            task_unit=task_unit_result.task_unit,
            document_title=document_title,
            document_profile=document_profile,
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
