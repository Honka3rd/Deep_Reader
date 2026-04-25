from dataclasses import dataclass, replace

from config.app_DI_config import AppDIConfig
from config.faiss_storage_config import FaissStorageConfig
from config.container import ApplicationLookupContainer
from document_preparation.prepared_document_result import PreparedDocumentResult
from document_preparation.preparation_mode import PreparationMode
from document_structure.structured_document import StructuredDocument
from language.language_code import LanguageCode
from language.language_profile_registry import LanguageProfileRegistry
from llm.openai_llm_provider import OpenAIModelName
from profile.document_profile import DocumentProfile
from retrieval.faiss_index_bundle import FaissIndexBundle
from question.qa_enums import AnswerLevel
from question.standardized.standardized_question import StandardizedQuestion
from section_tasks.document_task_layout import (
    DocumentTaskLayout,
    DocumentTaskLayoutSectionDTO,
    SectionTaskMode,
    TaskUnitDTO,
)
from section_tasks.quiz_question import QuizQuestion
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit import TaskUnit
from session.reading_session import ReadingSession
from session.session_manager import SessionUpdateResult


@dataclass(frozen=True)
class AskExecutionResult:
    """Coordinator ask output with answer text and HTTP status hint."""
    answer_text: str
    is_low_value: bool


@dataclass(frozen=True)
class ResolvedTaskUnit:
    """Resolved task unit plus stable unit index inside current document."""
    task_unit: TaskUnit
    task_unit_index: int


class Coordinator:
    """Coordinate document loading, index readiness, and session-aware QA orchestration."""
    chunk_size: int
    chunk_overlap: int
    embedding_model: str

    def __init__(
            self,
            chunk_size: int | None = None,
            chunk_overlap: int | None = None,
            embedding_model: str | None = None,
            llm_model: OpenAIModelName | str | None = None,
            target_max_input_tokens: int | None = None,
            target_max_output_tokens: int | None = None,
            target_max_context_tokens: int | None = None,
            input_budget_utilization_ratio: float | None = None,
            context_budget_utilization_ratio: float | None = None,
            full_text_input_budget_utilization_ratio: float | None = None,
            full_text_context_budget_utilization_ratio: float | None = None,
            embedding_batch_size: int | None = None,
            bundle_cache_capacity: int | None = None,
            session_recent_limit: int | None = None,
            base_near_chunk_threshold: int | None = None,
            min_near_chunk_threshold: int | None = None,
            max_near_chunk_threshold: int | None = None,
            global_scope_min_top_k: int | None = None,
            global_coverage_chunk_gap: int | None = None,
    ):
        """Initialize runtime dependencies and in-memory session storage.

        Args:
            chunk_size: Optional override for splitter chunk size.
            chunk_overlap: Optional override for splitter chunk overlap size.
            embedding_model: Optional override for embedding model name.
            llm_model: Optional override for LLM model name.
            target_max_input_tokens: Optional override for input budget before model clamp.
            target_max_output_tokens: Optional override for output budget before model clamp.
            target_max_context_tokens: Optional override for retrieval-context token budget.
            input_budget_utilization_ratio: Optional override for input-capacity utilization ratio.
            context_budget_utilization_ratio: Optional override for context budget utilization ratio.
            full_text_input_budget_utilization_ratio: Optional override for global full-text input ratio.
            full_text_context_budget_utilization_ratio: Optional override for global full-text context ratio.
            embedding_batch_size: Optional override for index-time embedding batch size.
            bundle_cache_capacity: Optional override for in-memory bundle cache capacity.
            session_recent_limit: Optional override for per-session recent history length.
            base_near_chunk_threshold: Optional override for local-reading base threshold.
            min_near_chunk_threshold: Optional override for local-reading threshold lower bound.
            max_near_chunk_threshold: Optional override for local-reading threshold upper bound.
            global_scope_min_top_k: Optional override for global-scope minimum retrieval top_k.
            global_coverage_chunk_gap: Optional override for global coverage dedup chunk gap.
        """
        override_values: dict[str, int | float | str | OpenAIModelName] = {}
        if chunk_size is not None:
            override_values["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            override_values["chunk_overlap"] = chunk_overlap
        if embedding_model is not None:
            override_values["embedding_model"] = embedding_model
        if llm_model is not None:
            override_values["llm_model"] = llm_model
        if target_max_input_tokens is not None:
            override_values["target_max_input_tokens"] = target_max_input_tokens
        if target_max_output_tokens is not None:
            override_values["target_max_output_tokens"] = target_max_output_tokens
        if target_max_context_tokens is not None:
            override_values["target_max_context_tokens"] = target_max_context_tokens
        if input_budget_utilization_ratio is not None:
            override_values["input_budget_utilization_ratio"] = input_budget_utilization_ratio
        if context_budget_utilization_ratio is not None:
            override_values["context_budget_utilization_ratio"] = context_budget_utilization_ratio
        if full_text_input_budget_utilization_ratio is not None:
            override_values["full_text_input_budget_utilization_ratio"] = (
                full_text_input_budget_utilization_ratio
            )
        if full_text_context_budget_utilization_ratio is not None:
            override_values["full_text_context_budget_utilization_ratio"] = (
                full_text_context_budget_utilization_ratio
            )
        if embedding_batch_size is not None:
            override_values["embedding_batch_size"] = embedding_batch_size
        if bundle_cache_capacity is not None:
            override_values["bundle_cache_capacity"] = bundle_cache_capacity
        if session_recent_limit is not None:
            override_values["session_recent_limit"] = session_recent_limit
        if base_near_chunk_threshold is not None:
            override_values["base_near_chunk_threshold"] = base_near_chunk_threshold
        if min_near_chunk_threshold is not None:
            override_values["min_near_chunk_threshold"] = min_near_chunk_threshold
        if max_near_chunk_threshold is not None:
            override_values["max_near_chunk_threshold"] = max_near_chunk_threshold
        if global_scope_min_top_k is not None:
            override_values["global_scope_min_top_k"] = global_scope_min_top_k
        if global_coverage_chunk_gap is not None:
            override_values["global_coverage_chunk_gap"] = global_coverage_chunk_gap

        self.app_config = replace(AppDIConfig(), **override_values)

        self.container = ApplicationLookupContainer.build(self.app_config)
        self.bundle_provider = self.container.bundle_provider()
        self.document_preparation_pipeline = self.container.document_preparation_pipeline()
        self.document_profile_store = self.container.document_profile_store()
        self.session_manager = self.container.session_manager()
        self.context_orchestrator = self.container.context_orchestrator()
        self.prompt_assembler = self.container.prompt_assembler()
        self.llm_provider = self.container.llm_provider()
        self.chapter_summary_service = self.container.chapter_summary_service()
        self.chapter_quiz_service = self.container.chapter_quiz_service()
        self.task_unit_resolver = self.container.task_unit_resolver()

    def get_bundle(self, doc_name: str) -> FaissIndexBundle:
        """Ensure index/profile readiness and return query-ready bundle.

        Args:
            doc_name: Logical document name and artifact namespace.

        Returns:
            Ready ``FaissIndexBundle`` for retrieval and answering.
        """
        preparation_result = self.document_preparation_pipeline.prepare_and_load(doc_name)
        if preparation_result.bundle is None:
            error_detail = " | ".join(preparation_result.assets.errors)
            raise RuntimeError(
                "Coordinator#get_bundle: preparation succeeded but runtime bundle is unavailable "
                f"for doc_name='{doc_name}'. errors={error_detail}"
            )
        return preparation_result.bundle

    def get_or_create_session(self, session_id: str, doc_name: str) -> ReadingSession:
        """Get existing session by id or create/reset it."""
        return self.session_manager.get_or_create_session(session_id, doc_name)

    def get_session(self, session_id: str) -> ReadingSession | None:
        """Return current session snapshot for inspection."""
        return self.session_manager.get_session(session_id)

    def ask(
        self,
        question: str,
        doc_name: str,
        top_k: int = 3,
        session_id: str | None = None,
    ) -> AskExecutionResult:
        """Execute one QA turn and optionally advance session reading state.

        Args:
            question: Original user question text.
            doc_name: Target document name.
            top_k: Maximum number of retrieval hits to use.
            session_id: Optional session id for reading-context continuity.

        Returns:
            Ask execution result with answer text and low-value marker.
        """
        session: ReadingSession | None = None
        session_active_chunk_index: int | None = None
        if session_id is not None:
            session = self.session_manager.get_or_create_session(session_id, doc_name)
            session_active_chunk_index = session.active_chunk_index
            print(
                f"Coordinator#ask before: session_id={session_id}, "
                f"active_chunk_index={session_active_chunk_index}"
            )

        preparation_result: PreparedDocumentResult
        preparation_result = self.document_preparation_pipeline.prepare_and_load(doc_name)
        print(
            "Coordinator#prepare:",
            f"doc_name={doc_name}",
            f"structured_document_path={preparation_result.assets.structured_document_path}",
            f"bundle_ready={preparation_result.assets.bundle_ready}",
            f"faiss_ready={preparation_result.assets.faiss_ready}",
            f"profile_ready={preparation_result.assets.profile_ready}",
        )
        bundle = preparation_result.bundle
        if bundle is None:
            error_detail = " | ".join(preparation_result.assets.errors)
            raise RuntimeError(
                "Coordinator#ask: preparation finished but runtime bundle is unavailable "
                f"for doc_name='{doc_name}'. errors={error_detail}"
            )

        context_result = self.context_orchestrator.build(
            query=question,
            bundle=bundle,
            top_k=top_k,
            session_active_chunk_index=session_active_chunk_index,
        )
        is_low_value = context_result.answer_mode.level == AnswerLevel.REJECT
        if context_result.answer_mode.level == AnswerLevel.REJECT:
            answer_language = self._resolve_answer_language(
                context_result.standardized_question
            )
            answer_text = LanguageProfileRegistry.get_low_value_not_found_response(
                answer_language
            )
        else:
            prompt = self.prompt_assembler.build_answer_prompt(
                context=context_result.context_text,
                question=context_result.standardized_question,
                profile=bundle.profile,
                answer_mode=context_result.answer_mode,
                prompt_mode=context_result.prompt_mode,
            )
            answer_text = self.llm_provider.complete_text(prompt)

        if session is not None:
            self.session_manager.update_session(
                session=session,
                result=SessionUpdateResult(
                    question=question,
                    bundle=bundle,
                    results=context_result.results,
                ),
            )
            print(
                f"Coordinator#ask after: session_id={session.session_id}, "
                f"active_chunk_index_before={session_active_chunk_index}, "
                f"active_chunk_index_after={session.active_chunk_index}, "
                f"context_mode={context_result.prompt_mode}"
            )

        return AskExecutionResult(
            answer_text=answer_text,
            is_low_value=is_low_value,
        )

    def summarize_section(self, doc_name: str, section_id: str) -> SectionTaskResult:
        """Run section-summary task by coordinator-level task-unit resolution."""
        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )
        try:
            resolved_task_unit = self._resolve_task_unit_for_section_id(
                document=structured_document,
                section_id=section_id,
            )
            return self.chapter_summary_service.summarize_task_unit(
                task_unit=resolved_task_unit.task_unit,
                document_title=structured_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def summarize_chapter(self, doc_name: str, chapter_title: str) -> SectionTaskResult:
        """Run chapter-summary task by chapter-title -> task-unit resolution."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            return SectionTaskResult.fail("chapter_title cannot be empty")

        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )

        try:
            resolved_task_unit = self._resolve_task_unit_for_chapter_title(
                document=structured_document,
                chapter_title=normalized_chapter_title,
            )
            return self.chapter_summary_service.summarize_task_unit(
                task_unit=resolved_task_unit.task_unit,
                document_title=structured_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_section_quiz(
        self, doc_name: str, section_id: str
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run section-quiz task by coordinator-level task-unit resolution."""
        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )
        try:
            resolved_task_unit = self._resolve_task_unit_for_section_id(
                document=structured_document,
                section_id=section_id,
            )
            return self.chapter_quiz_service.generate_task_unit_quiz(
                task_unit=resolved_task_unit.task_unit,
                document_title=structured_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_chapter_quiz(
        self,
        doc_name: str,
        chapter_title: str,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run chapter-quiz task by chapter-title -> task-unit resolution."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            return SectionTaskResult.fail("chapter_title cannot be empty")

        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )

        try:
            resolved_task_unit = self._resolve_task_unit_for_chapter_title(
                document=structured_document,
                chapter_title=normalized_chapter_title,
            )
            return self.chapter_quiz_service.generate_task_unit_quiz(
                task_unit=resolved_task_unit.task_unit,
                document_title=structured_document.title,
                document_profile=document_profile,
                task_type="chapter_quiz",
                task_unit_index=resolved_task_unit.task_unit_index,
            )
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def _resolve_task_unit_for_section_id(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> ResolvedTaskUnit:
        """Resolve first task unit whose source section ids contain target section id."""
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
                return ResolvedTaskUnit(
                    task_unit=task_unit,
                    task_unit_index=unit_index,
                )
        raise ValueError(
            f"section_id '{normalized_section_id}' not found in resolved task units "
            f"for document '{document.document_id}'"
        )

    def _resolve_task_unit_for_chapter_title(
        self,
        *,
        document: StructuredDocument,
        chapter_title: str,
    ) -> ResolvedTaskUnit:
        """Resolve task unit by first exact chapter-title hit in source sections."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")

        target_section_id: str | None = None
        for section in document.sections:
            if ((section.title or "").strip()) == normalized_chapter_title:
                target_section_id = section.section_id
                break
        if target_section_id is None:
            raise ValueError(
                f"chapter_title '{normalized_chapter_title}' not found in document '{document.title}'"
            )

        return self._resolve_task_unit_for_section_id(
            document=document,
            section_id=target_section_id,
        )

    def _prepare_section_task_inputs(
        self,
        doc_name: str,
    ) -> tuple[StructuredDocument | None, DocumentProfile | None, list[str]]:
        """Prepare base structured artifact and optionally load existing profile."""
        preparation_result = self.document_preparation_pipeline.prepare_and_load(
            doc_name=doc_name,
            mode=PreparationMode.BASE,
        )
        structured_document = preparation_result.structured_document
        if structured_document is None:
            return None, None, list(preparation_result.assets.errors)
        document_profile = self._load_existing_document_profile(doc_name)
        return structured_document, document_profile, list(preparation_result.assets.errors)

    def get_document_task_layout(self, doc_name: str) -> DocumentTaskLayout:
        """Return section-first document layout with embedded task-unit metadata."""
        preparation_result = self.document_preparation_pipeline.prepare_and_load(
            doc_name=doc_name,
            mode=PreparationMode.BASE,
        )
        structured_document = preparation_result.structured_document
        if structured_document is None:
            detail = " | ".join(preparation_result.assets.errors)
            raise ValueError(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )

        resolved_task_units = self.task_unit_resolver.resolve(structured_document)
        task_unit_dtos: list[TaskUnitDTO] = [
            TaskUnitDTO(
                unit_id=task_unit.unit_id,
                title=task_unit.title,
                container_title=task_unit.container_title,
                source_section_ids=list(task_unit.source_section_ids),
                is_fallback_generated=task_unit.is_fallback_generated,
            )
            for task_unit in resolved_task_units
        ]
        task_unit_by_id: dict[str, TaskUnitDTO] = {
            task_unit.unit_id: task_unit for task_unit in task_unit_dtos
        }

        section_to_unit_ids: dict[str, list[str]] = {
            section.section_id: [] for section in structured_document.sections
        }
        for task_unit in task_unit_dtos:
            for source_section_id in task_unit.source_section_ids:
                normalized_section_id = source_section_id.strip()
                if not normalized_section_id:
                    continue
                section_to_unit_ids.setdefault(normalized_section_id, [])
                section_to_unit_ids[normalized_section_id].append(task_unit.unit_id)

        section_layouts: list[DocumentTaskLayoutSectionDTO] = []
        for section in structured_document.sections:
            unit_ids = section_to_unit_ids.get(section.section_id, [])
            section_units = [
                task_unit_by_id[unit_id]
                for unit_id in unit_ids
                if unit_id in task_unit_by_id
            ]
            section_mode = self._resolve_section_task_mode(
                section_id=section.section_id,
                section_task_units=section_units,
            )
            section_layouts.append(
                DocumentTaskLayoutSectionDTO(
                    section_id=section.section_id,
                    title=section.title,
                    container_title=section.container_title,
                    task_mode=section_mode,
                    task_units=section_units,
                )
            )

        return DocumentTaskLayout(
            document_id=structured_document.document_id,
            title=structured_document.title,
            language=structured_document.language,
            sections=section_layouts,
            task_units=task_unit_dtos,
        )

    @staticmethod
    def _resolve_section_task_mode(
        *,
        section_id: str,
        section_task_units: list[TaskUnitDTO],
    ) -> SectionTaskMode:
        """Infer section task mode from section-to-unit relationship."""
        if not section_task_units:
            return SectionTaskMode.DIRECT

        for task_unit in section_task_units:
            source_count = len(task_unit.source_section_ids)
            if source_count > 1:
                return SectionTaskMode.MERGED

        if len(section_task_units) > 1:
            return SectionTaskMode.SPLIT

        only_unit = section_task_units[0]
        if len(only_unit.source_section_ids) == 1 and only_unit.source_section_ids[0] == section_id:
            return SectionTaskMode.DIRECT

        return SectionTaskMode.MERGED

    def _load_existing_document_profile(self, doc_name: str) -> DocumentProfile | None:
        """Load existing profile artifact when available; return None otherwise."""
        config = FaissStorageConfig(namespace=doc_name)
        if not self.document_profile_store.exists(config):
            return None
        try:
            return self.document_profile_store.load(config)
        except Exception as error:
            print(
                "Coordinator#section_task_profile_load_failed:",
                f"doc_name={doc_name}",
                f"error={error}",
            )
            return None

    @staticmethod
    def _resolve_answer_language(question: StandardizedQuestion) -> LanguageCode:
        """Resolve language used for user-facing fallback answer text."""
        if question.user_language != LanguageCode.UNKNOWN:
            return question.user_language
        if question.document_language != LanguageCode.UNKNOWN:
            return question.document_language
        return LanguageCode.EN
