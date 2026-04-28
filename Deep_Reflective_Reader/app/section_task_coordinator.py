import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass

from config.faiss_storage_config import FaissStorageConfig
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.preparation_mode import PreparationMode
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.enhanced_parse_trigger_evaluator import (
    EnhancedParseTriggerDecision,
    EnhancedParseTriggerEvaluator,
)
from document_structure.section_split_plan import SectionParserMode
from document_structure.section_splitter_selector import SectionSplitterMode
from document_structure.structured_document import StructuredDocument, StructuredSection
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.document_task_layout import (
    DocumentTaskLayout,
    DocumentTaskLayoutSectionDTO,
    EnhancedParseRecommendationDTO,
    SectionTaskMode,
    TaskUnitDTO,
)
from section_tasks.quiz_question import QuizQuestion
from section_tasks.reparse_document_structure_result import ReparseDocumentStructureResult
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_resolver import TaskUnitResolver
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import SummaryArtifact, TaskArtifacts


@dataclass(frozen=True)
class ResolvedTaskUnit:
    """Resolved task unit plus stable unit index inside current document."""

    task_unit: TaskUnit
    task_unit_index: int


@dataclass(frozen=True)
class TaskUnitResolveOptions:
    """Request-time resolve options for task-unit split behavior."""

    split_mode: TaskUnitSplitMode
    semantic_top_k_candidates: int | None


class SectionTaskCoordinator:
    """Coordinator for section/chapter task orchestration and layout projection."""
    _TASK_LAYOUT_METADATA_KEY = "task_layout"
    _TASK_LAYOUT_RESOLVER_VERSION = "task_unit_resolver_v1"
    _SECTION_SUMMARY_PROMPT_VERSION = "section_summary_v1"
    _CHAPTER_SUMMARY_PROMPT_VERSION = "chapter_summary_v1"

    def __init__(
        self,
        document_preparation_pipeline: DocumentPreparationPipeline,
        document_artifact_repository: DocumentArtifactRepository,
        document_profile_store: DocumentProfileStore,
        chapter_summary_service: ChapterSummaryService,
        chapter_quiz_service: ChapterQuizService,
        task_unit_resolver: TaskUnitResolver,
        enhanced_parse_trigger_evaluator: EnhancedParseTriggerEvaluator,
        semantic_top_k_candidates_max: int = 20,
    ):
        self.document_preparation_pipeline = document_preparation_pipeline
        self.document_artifact_repository = document_artifact_repository
        self.document_profile_store = document_profile_store
        self.chapter_summary_service = chapter_summary_service
        self.chapter_quiz_service = chapter_quiz_service
        self.task_unit_resolver = task_unit_resolver
        self.enhanced_parse_trigger_evaluator = enhanced_parse_trigger_evaluator
        self.semantic_top_k_candidates_max = max(1, int(semantic_top_k_candidates_max))

    def summarize_section(
        self,
        doc_name: str,
        section_id: str,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_summary: bool = False,
    ) -> SectionTaskResult:
        """Run section-summary task with persisted task-layout + summary-cache reuse."""
        resolve_options = self._resolve_task_unit_request_options(
            task_unit_split_mode=task_unit_split_mode,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )
        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )
        try:
            target_section = self._find_section_or_raise(
                document=structured_document,
                section_id=section_id,
            )
            if (
                not refresh_summary
                and self._is_section_summary_cache_valid(
                    section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._SECTION_SUMMARY_PROMPT_VERSION,
                    expected_scope="section",
                )
            ):
                cached_summary = target_section.task_artifacts.summary
                assert cached_summary is not None
                print(
                    "SectionTaskCoordinator#section_summary_cache_hit:",
                    f"doc_name={doc_name}",
                    f"section_id={target_section.section_id}",
                )
                return SectionTaskResult.ok(cached_summary.content)

            layout_document = self._get_or_refresh_task_layout_document(
                doc_name=doc_name,
                structured_document=structured_document,
                resolve_options=resolve_options,
                refresh_task_units=False,
            )
            resolved_task_unit = self._resolve_task_unit_for_section_id_from_persisted_layout(
                document=layout_document,
                section_id=section_id,
            )
            summary_result = self.chapter_summary_service.summarize_task_unit(
                task_unit=resolved_task_unit.task_unit,
                document_title=layout_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
            if not summary_result.success:
                return summary_result

            summary_artifact = self._build_summary_artifact(
                content=summary_result.payload,
                target_section=target_section,
                document=layout_document,
                document_profile=document_profile,
                resolve_options=resolve_options,
                prompt_version=self._SECTION_SUMMARY_PROMPT_VERSION,
                scope="section",
                source_task_unit_id=resolved_task_unit.task_unit.unit_id,
                chapter_title=None,
            )
            self.document_artifact_repository.update_section_summary_artifact(
                doc_name=doc_name,
                section_id=target_section.section_id,
                summary=summary_artifact,
            )
            print(
                "SectionTaskCoordinator#section_summary_cache_write:",
                f"doc_name={doc_name}",
                f"section_id={target_section.section_id}",
            )
            return summary_result
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def summarize_chapter(
        self,
        doc_name: str,
        chapter_title: str,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_summary: bool = False,
    ) -> SectionTaskResult:
        """Run chapter-summary task with section-level summary cache reuse."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            return SectionTaskResult.fail("chapter_title cannot be empty")
        resolve_options = self._resolve_task_unit_request_options(
            task_unit_split_mode=task_unit_split_mode,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )

        structured_document, document_profile, preparation_errors = (
            self._prepare_section_task_inputs(doc_name)
        )
        if structured_document is None:
            detail = " | ".join(preparation_errors)
            return SectionTaskResult.fail(
                f"structured document unavailable for doc_name='{doc_name}'. errors={detail}"
            )

        try:
            target_section = self._find_section_by_chapter_title_or_raise(
                document=structured_document,
                chapter_title=normalized_chapter_title,
            )
            if (
                not refresh_summary
                and self._is_section_summary_cache_valid(
                    section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
                    expected_scope="chapter",
                    expected_chapter_title=normalized_chapter_title,
                )
            ):
                cached_summary = target_section.task_artifacts.summary
                assert cached_summary is not None
                print(
                    "SectionTaskCoordinator#chapter_summary_cache_hit:",
                    f"doc_name={doc_name}",
                    f"chapter_title={normalized_chapter_title}",
                    f"section_id={target_section.section_id}",
                )
                return SectionTaskResult.ok(cached_summary.content)

            layout_document = self._get_or_refresh_task_layout_document(
                doc_name=doc_name,
                structured_document=structured_document,
                resolve_options=resolve_options,
                refresh_task_units=False,
            )
            resolved_task_unit = self._resolve_task_unit_for_section_id_from_persisted_layout(
                document=layout_document,
                section_id=target_section.section_id,
            )
            summary_result = self.chapter_summary_service.summarize_task_unit(
                task_unit=resolved_task_unit.task_unit,
                document_title=layout_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
            if not summary_result.success:
                return summary_result

            summary_artifact = self._build_summary_artifact(
                content=summary_result.payload,
                target_section=target_section,
                document=layout_document,
                document_profile=document_profile,
                resolve_options=resolve_options,
                prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
                scope="chapter",
                source_task_unit_id=resolved_task_unit.task_unit.unit_id,
                chapter_title=normalized_chapter_title,
            )
            self.document_artifact_repository.update_section_summary_artifact(
                doc_name=doc_name,
                section_id=target_section.section_id,
                summary=summary_artifact,
            )
            print(
                "SectionTaskCoordinator#chapter_summary_cache_write:",
                f"doc_name={doc_name}",
                f"chapter_title={normalized_chapter_title}",
                f"section_id={target_section.section_id}",
            )
            return summary_result
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_section_quiz(
        self,
        doc_name: str,
        section_id: str,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run section-quiz task by coordinator-level task-unit resolution."""
        resolve_options = self._resolve_task_unit_request_options(
            task_unit_split_mode=task_unit_split_mode,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )
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
                resolve_options=resolve_options,
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
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run chapter-quiz task by chapter-title -> task-unit resolution."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            return SectionTaskResult.fail("chapter_title cannot be empty")
        resolve_options = self._resolve_task_unit_request_options(
            task_unit_split_mode=task_unit_split_mode,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )

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
                resolve_options=resolve_options,
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

    def get_document_task_layout(
        self,
        doc_name: str,
        refresh_task_units: bool = False,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> DocumentTaskLayout:
        """Return section-first document layout with embedded task-unit metadata."""
        resolve_options = self._resolve_task_unit_request_options(
            task_unit_split_mode=task_unit_split_mode,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )
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
        cached_document = self._get_or_refresh_task_layout_document(
            doc_name=doc_name,
            structured_document=structured_document,
            resolve_options=resolve_options,
            refresh_task_units=refresh_task_units,
        )

        task_unit_dtos, section_units_by_section_id = self._build_task_unit_dtos_from_document(
            document=cached_document
        )

        section_layouts: list[DocumentTaskLayoutSectionDTO] = []
        for section in cached_document.sections:
            section_units = section_units_by_section_id.get(section.section_id, [])
            section_mode = self._resolve_section_task_mode(
                section_id=section.section_id,
                section_task_units=section_units,
            )
            section_layouts.append(
                DocumentTaskLayoutSectionDTO(
                    section_id=section.section_id,
                    title=section.title,
                    container_title=section.container_title,
                    section_role=(
                        None
                        if section.section_role is None
                        else section.section_role.value
                    ),
                    task_mode=section_mode,
                    task_units=section_units,
                )
            )

        trigger_decision = self._evaluate_enhanced_parse_trigger(
            structured_document=cached_document,
            section_layouts=section_layouts,
            task_unit_dtos=task_unit_dtos,
        )
        self._log_enhanced_parse_trigger_decision(
            doc_name=doc_name,
            decision=trigger_decision,
        )

        return DocumentTaskLayout(
            document_id=cached_document.document_id,
            title=cached_document.title,
            language=cached_document.language,
            sections=section_layouts,
            task_units=task_unit_dtos,
            enhanced_parse_recommendation=EnhancedParseRecommendationDTO(
                should_recommend=trigger_decision.should_recommend,
                score=trigger_decision.score,
                reasons=list(trigger_decision.reasons),
                metrics=dict(trigger_decision.metrics),
            ),
        )

    def reparse_document_structure(
        self,
        doc_name: str,
        parser_mode: SectionParserMode | str,
    ) -> ReparseDocumentStructureResult:
        """Explicitly reparse one document structure with selected parser mode."""
        normalized_doc_name = doc_name.strip()
        if not normalized_doc_name:
            return ReparseDocumentStructureResult.fail(
                doc_name=doc_name,
                parser_mode="common",
                error="doc_name cannot be empty",
            )

        resolved_splitter_mode = SectionSplitterMode.resolve(str(parser_mode))
        resolved_parser_mode = SectionParserMode.resolve(resolved_splitter_mode.value)
        print(
            "SectionTaskCoordinator#reparse_document_structure:",
            f"doc_name={normalized_doc_name}",
            f"parser_mode={resolved_parser_mode.value}",
        )

        preparation_result = self.document_preparation_pipeline.prepare_and_load(
            doc_name=normalized_doc_name,
            force_rebuild=True,
            mode=PreparationMode.BASE,
            structured_parser_mode=resolved_splitter_mode,
        )

        assets = preparation_result.assets
        if (
            assets.structured_document_ready
            and assets.structured_document_path is not None
            and preparation_result.structured_document is not None
        ):
            section_count = len(preparation_result.structured_document.sections)
            return ReparseDocumentStructureResult.ok(
                doc_name=normalized_doc_name,
                parser_mode=resolved_parser_mode.value,
                structured_document_path=assets.structured_document_path,
                section_count=section_count,
            )

        error_detail = " | ".join(assets.errors).strip() or (
            "reparse_document_structure_failed:structured_document_unavailable"
        )
        print(
            "SectionTaskCoordinator#reparse_document_structure_failed:",
            f"doc_name={normalized_doc_name}",
            f"parser_mode={resolved_parser_mode.value}",
            f"error={error_detail}",
        )
        return ReparseDocumentStructureResult.fail(
            doc_name=normalized_doc_name,
            parser_mode=resolved_parser_mode.value,
            structured_document_path=assets.structured_document_path,
            error=error_detail,
        )

    def _evaluate_enhanced_parse_trigger(
        self,
        *,
        structured_document: StructuredDocument,
        section_layouts: list[DocumentTaskLayoutSectionDTO],
        task_unit_dtos: list[TaskUnitDTO],
    ) -> EnhancedParseTriggerDecision:
        """Evaluate whether enhanced parser should be recommended."""
        total_sections = len(section_layouts)
        affected_sections = sum(
            1
            for section in section_layouts
            if section.task_mode in {SectionTaskMode.SPLIT, SectionTaskMode.MERGED}
        )
        affected_section_ratio = (
            affected_sections / total_sections if total_sections > 0 else 0.0
        )

        total_task_units = len(task_unit_dtos)
        fallback_task_units = sum(
            1 for task_unit in task_unit_dtos if task_unit.is_fallback_generated
        )
        fallback_task_unit_ratio = (
            fallback_task_units / total_task_units if total_task_units > 0 else 0.0
        )

        return self.enhanced_parse_trigger_evaluator.evaluate(
            structured_document=structured_document,
            affected_section_ratio=affected_section_ratio,
            fallback_task_unit_ratio=fallback_task_unit_ratio,
            total_task_units=total_task_units,
        )

    def _build_task_unit_dtos_from_document(
        self,
        *,
        document: StructuredDocument,
    ) -> tuple[list[TaskUnitDTO], dict[str, list[TaskUnitDTO]]]:
        """Build deduplicated task-unit DTOs and section->task-unit mapping from persisted data."""
        task_unit_by_id: dict[str, TaskUnitDTO] = {}
        section_units_by_section_id: dict[str, list[TaskUnitDTO]] = {
            section.section_id: [] for section in document.sections
        }

        for section in document.sections:
            section_task_units: list[TaskUnitDTO] = []
            for task_unit in section.task_units:
                if task_unit.unit_id not in task_unit_by_id:
                    task_unit_by_id[task_unit.unit_id] = TaskUnitDTO(
                        unit_id=task_unit.unit_id,
                        title=task_unit.title,
                        container_title=task_unit.container_title,
                        source_section_ids=list(task_unit.source_section_ids),
                        is_fallback_generated=task_unit.is_fallback_generated,
                    )
                section_task_units.append(task_unit_by_id[task_unit.unit_id])
            section_units_by_section_id[section.section_id] = section_task_units
        return list(task_unit_by_id.values()), section_units_by_section_id

    def _build_task_layout_metadata(
        self,
        *,
        document: StructuredDocument,
        resolve_options: TaskUnitResolveOptions,
    ) -> dict[str, str | int | None]:
        """Build deterministic task-layout cache metadata."""
        return {
            "source_hash": self._compute_source_hash(document),
            "task_unit_split_mode": resolve_options.split_mode.value,
            "semantic_top_k_candidates": resolve_options.semantic_top_k_candidates,
            "resolver_version": self._TASK_LAYOUT_RESOLVER_VERSION,
        }

    def _is_task_layout_cache_valid(
        self,
        *,
        document: StructuredDocument,
        resolve_options: TaskUnitResolveOptions,
    ) -> bool:
        """Check persisted section.task_units cache validity against request options."""
        for section in document.sections:
            if section.content.strip() and not section.task_units:
                return False

        document_task_artifacts = document.document_task_artifacts
        if document_task_artifacts is None:
            return False

        metadata = dict(document_task_artifacts.metadata or {})
        raw_task_layout = metadata.get(self._TASK_LAYOUT_METADATA_KEY)
        if not isinstance(raw_task_layout, dict):
            return False

        expected = self._build_task_layout_metadata(
            document=document,
            resolve_options=resolve_options,
        )
        return (
            raw_task_layout.get("source_hash") == expected["source_hash"]
            and raw_task_layout.get("task_unit_split_mode")
            == expected["task_unit_split_mode"]
            and raw_task_layout.get("semantic_top_k_candidates")
            == expected["semantic_top_k_candidates"]
            and raw_task_layout.get("resolver_version") == expected["resolver_version"]
        )

    def _get_or_refresh_task_layout_document(
        self,
        *,
        doc_name: str,
        structured_document: StructuredDocument,
        resolve_options: TaskUnitResolveOptions,
        refresh_task_units: bool,
    ) -> StructuredDocument:
        """Return document with valid persisted section.task_units, recomputing when needed."""
        if (
            not refresh_task_units
            and self._is_task_layout_cache_valid(
                document=structured_document,
                resolve_options=resolve_options,
            )
        ):
            print(
                "SectionTaskCoordinator#task_layout_cache_hit:",
                f"doc_name={doc_name}",
                f"split_mode={resolve_options.split_mode.value}",
                f"semantic_top_k={resolve_options.semantic_top_k_candidates}",
            )
            return structured_document

        resolved_task_units = self.task_unit_resolver.resolve_with_options(
            document=structured_document,
            split_mode=resolve_options.split_mode,
            semantic_top_k_candidates=resolve_options.semantic_top_k_candidates,
        )
        task_units_by_section_id = self._build_task_units_by_section_id(
            document=structured_document,
            task_units=resolved_task_units,
        )
        task_layout_metadata = self._build_task_layout_metadata(
            document=structured_document,
            resolve_options=resolve_options,
        )
        updated_document = self.document_artifact_repository.update_task_layout(
            doc_name=doc_name,
            task_units_by_section_id=task_units_by_section_id,
            task_layout_metadata=task_layout_metadata,
        )
        print(
            "SectionTaskCoordinator#task_layout_cache_write:",
            f"doc_name={doc_name}",
            f"split_mode={resolve_options.split_mode.value}",
            f"semantic_top_k={resolve_options.semantic_top_k_candidates}",
            f"refresh={refresh_task_units}",
        )
        return updated_document

    def _build_task_units_by_section_id(
        self,
        *,
        document: StructuredDocument,
        task_units: list[TaskUnit],
    ) -> dict[str, list[TaskUnit]]:
        """Group resolver output into section-level persisted task-unit lists."""
        task_units_by_section_id: dict[str, list[TaskUnit]] = {
            section.section_id: [] for section in document.sections
        }
        for task_unit in task_units:
            assigned_section_ids = []
            for section_id in task_unit.source_section_ids:
                normalized_section_id = section_id.strip()
                if not normalized_section_id:
                    continue
                if normalized_section_id in task_units_by_section_id:
                    assigned_section_ids.append(normalized_section_id)

            for section_id in dict.fromkeys(assigned_section_ids):
                task_units_by_section_id[section_id].append(task_unit)
        return task_units_by_section_id

    @staticmethod
    def _compute_source_hash(document: StructuredDocument) -> str:
        """Compute stable source hash for task-layout cache invalidation."""
        payload = document.raw_text.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _compute_section_source_hash(section: StructuredSection) -> str:
        """Compute stable source hash for one section-summary cache key."""
        payload = section.content.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _resolve_summary_language(
        *,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
    ) -> str | None:
        """Resolve summary language tag from profile first, then structured document."""
        profile_language = (
            None
            if document_profile is None
            else (document_profile.document_language or "").strip() or None
        )
        if profile_language is not None:
            return profile_language
        return (document.language or "").strip() or None

    def _build_summary_artifact(
        self,
        *,
        content: str,
        target_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        scope: str,
        source_task_unit_id: str,
        chapter_title: str | None,
    ) -> SummaryArtifact:
        """Build persisted summary artifact payload."""
        normalized_content = content.strip()
        metadata: dict[str, str | int | None] = {
            "summary_scope": scope,
            "section_id": target_section.section_id,
            "source_task_unit_id": source_task_unit_id,
        }
        if chapter_title is not None:
            metadata["chapter_title"] = chapter_title

        return SummaryArtifact(
            content=normalized_content,
            language=self._resolve_summary_language(
                document=document,
                document_profile=document_profile,
            ),
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_hash=self._compute_section_source_hash(target_section),
            prompt_version=prompt_version,
            task_unit_split_mode=resolve_options.split_mode.value,
            semantic_top_k_candidates=resolve_options.semantic_top_k_candidates,
            metadata=metadata,
        )

    def _is_section_summary_cache_valid(
        self,
        *,
        section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        expected_scope: str,
        expected_chapter_title: str | None = None,
    ) -> bool:
        """Check whether one section-level summary artifact can be reused safely."""
        artifacts = section.task_artifacts
        if artifacts is None or artifacts.summary is None:
            return False
        summary = artifacts.summary
        if not summary.content.strip():
            return False
        if summary.source_hash != self._compute_section_source_hash(section):
            return False

        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if summary.language != expected_language:
            return False
        if summary.task_unit_split_mode != resolve_options.split_mode.value:
            return False
        if (
            summary.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return False
        if summary.prompt_version != prompt_version:
            return False

        metadata = dict(summary.metadata or {})
        if metadata.get("summary_scope") != expected_scope:
            return False
        if metadata.get("section_id") != section.section_id:
            return False
        if expected_scope == "chapter":
            if (metadata.get("chapter_title") or "") != (
                expected_chapter_title or ""
            ):
                return False
        return True

    @staticmethod
    def _find_section_or_raise(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> StructuredSection:
        """Find one section by id or raise ValueError with clear details."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")
        for section in document.sections:
            if section.section_id == normalized_section_id:
                return section
        raise ValueError(
            f"section_id '{normalized_section_id}' not found in document '{document.document_id}'"
        )

    @staticmethod
    def _find_section_by_chapter_title_or_raise(
        *,
        document: StructuredDocument,
        chapter_title: str,
    ) -> StructuredSection:
        """Find first section by exact chapter title or raise ValueError."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")
        for section in document.sections:
            if ((section.title or "").strip()) == normalized_chapter_title:
                return section
        raise ValueError(
            f"chapter_title '{normalized_chapter_title}' not found in document '{document.title}'"
        )

    @staticmethod
    def _resolve_task_unit_for_section_id_from_persisted_layout(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> ResolvedTaskUnit:
        """Resolve one task unit from persisted section.task_units without recomputing resolver."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")

        ordered_task_units: list[TaskUnit] = []
        seen_unit_ids: set[str] = set()
        for section in document.sections:
            for task_unit in section.task_units:
                if task_unit.unit_id in seen_unit_ids:
                    continue
                seen_unit_ids.add(task_unit.unit_id)
                ordered_task_units.append(task_unit)

        if not ordered_task_units:
            raise ValueError(
                f"no persisted task units found for document '{document.document_id}'"
            )
        for unit_index, task_unit in enumerate(ordered_task_units):
            if normalized_section_id in task_unit.source_section_ids:
                return ResolvedTaskUnit(
                    task_unit=task_unit,
                    task_unit_index=unit_index,
                )
        raise ValueError(
            f"section_id '{normalized_section_id}' not found in persisted task layout "
            f"for document '{document.document_id}'"
        )

    @staticmethod
    def _log_enhanced_parse_trigger_decision(
        *,
        doc_name: str,
        decision: EnhancedParseTriggerDecision,
    ) -> None:
        """Print deterministic recommendation decision for inspection/debug."""
        print(
            "SectionTaskCoordinator#enhanced_parse_trigger:",
            f"doc_name={doc_name}",
            f"should_recommend={decision.should_recommend}",
            f"score={decision.score}",
            f"reasons={decision.reasons}",
            f"metrics={decision.metrics}",
        )

    def _resolve_task_unit_for_section_id(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
        resolve_options: TaskUnitResolveOptions,
    ) -> ResolvedTaskUnit:
        """Resolve first task unit whose source section ids contain target section id."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")

        task_units = self.task_unit_resolver.resolve_with_options(
            document=document,
            split_mode=resolve_options.split_mode,
            semantic_top_k_candidates=resolve_options.semantic_top_k_candidates,
        )
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
        resolve_options: TaskUnitResolveOptions,
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
            resolve_options=resolve_options,
        )

    def _resolve_task_unit_request_options(
        self,
        *,
        task_unit_split_mode: TaskUnitSplitMode | str | None,
        semantic_top_k_candidates: int | None,
    ) -> TaskUnitResolveOptions:
        """Resolve/validate request-time task-unit split options."""
        if task_unit_split_mode is None:
            resolved_mode = self.task_unit_resolver.split_mode
        else:
            resolved_mode = TaskUnitSplitMode.resolve_strict(task_unit_split_mode)

        normalized_top_k: int | None = None
        if semantic_top_k_candidates is not None:
            candidate = int(semantic_top_k_candidates)
            if candidate <= 0 or candidate > self.semantic_top_k_candidates_max:
                raise ValueError(
                    "semantic_top_k_candidates must be a positive integer in "
                    f"[1, {self.semantic_top_k_candidates_max}]"
                )
            if resolved_mode == TaskUnitSplitMode.SEMANTIC_SAFE:
                normalized_top_k = candidate
            else:
                print(
                    "SectionTaskCoordinator#semantic_top_k_ignored:",
                    f"mode={resolved_mode.value}",
                    f"semantic_top_k_candidates={candidate}",
                )

        return TaskUnitResolveOptions(
            split_mode=resolved_mode,
            semantic_top_k_candidates=normalized_top_k,
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
            if len(task_unit.source_section_ids) > 1:
                return SectionTaskMode.MERGED

        if len(section_task_units) > 1:
            return SectionTaskMode.SPLIT

        only_unit = section_task_units[0]
        if (
            len(only_unit.source_section_ids) == 1
            and only_unit.source_section_ids[0] == section_id
        ):
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
                "SectionTaskCoordinator#profile_load_failed:",
                f"doc_name={doc_name}",
                f"error={error}",
            )
            return None
