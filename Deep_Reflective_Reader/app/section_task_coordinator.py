import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, replace

from config.faiss_storage_config import FaissStorageConfig
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.preparation_mode import PreparationMode
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.document_hierarchy_index import (
    find_chapter_by_id_effective,
    find_chapters_by_title_effective,
    find_section_by_chapter_title_effective,
    find_section_by_id_effective,
    find_sections_by_title_effective,
    get_effective_sections,
    is_severe_hierarchy_warning,
    validate_chapter_hierarchy_consistency,
)
from document_structure.enhanced_parse_trigger_evaluator import (
    EnhancedParseTriggerDecision,
    EnhancedParseTriggerEvaluator,
)
from document_structure.section_split_plan import SectionParserMode
from document_structure.section_splitter_selector import SectionSplitterMode
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.document_task_layout import (
    ArtifactAvailabilityDTO,
    DocumentTaskLayoutChapterDTO,
    DocumentTaskLayout,
    DocumentTaskLayoutSectionDTO,
    EnhancedParseRecommendationDTO,
    ProfileStructureDiagnosticsDTO,
    SectionTaskMode,
    TaskUnitDTO,
)
from section_tasks.artifact_validity import ArtifactValidityResult
from section_tasks.quiz_question import QuizQuestion
from section_tasks.reparse_document_structure_result import ReparseDocumentStructureResult
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_id_normalizer import TaskUnitIdNormalizer
from section_tasks.task_unit_resolver import TaskUnitResolver
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import QuizArtifact, SummaryArtifact, TaskArtifacts


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
    _TASK_LAYOUT_RESOLVER_VERSION = "task_unit_resolver_v2"
    _SECTION_SUMMARY_PROMPT_VERSION = "section_summary_v1"
    _CHAPTER_SUMMARY_PROMPT_VERSION = "chapter_summary_v1"
    _SECTION_QUIZ_PROMPT_VERSION = "section_quiz_v1"
    _CHAPTER_QUIZ_PROMPT_VERSION = "chapter_quiz_v1"
    _TASK_UNIT_SUMMARY_PROMPT_VERSION = "task_unit_summary_v1"
    _TASK_UNIT_QUIZ_PROMPT_VERSION = "task_unit_quiz_v1"
    _QUIZ_SCHEMA_VERSION = "quiz_schema_v1"
    _CHAPTER_ARTIFACT_KEY_PREFIX = "chapter::"
    _CHAPTER_ARTIFACT_KEY_BY_ID_PREFIX = "chapter_id::"
    _TASK_UNIT_STATS_AVAILABLE_COVERAGE_THRESHOLD = 0.95
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
        task_unit_id_normalizer: TaskUnitIdNormalizer | None = None,
    ):
        self.document_preparation_pipeline = document_preparation_pipeline
        self.document_artifact_repository = document_artifact_repository
        self.document_profile_store = document_profile_store
        self.chapter_summary_service = chapter_summary_service
        self.chapter_quiz_service = chapter_quiz_service
        self.task_unit_resolver = task_unit_resolver
        self.enhanced_parse_trigger_evaluator = enhanced_parse_trigger_evaluator
        self.semantic_top_k_candidates_max = max(1, int(semantic_top_k_candidates_max))
        self.task_unit_id_normalizer = task_unit_id_normalizer or TaskUnitIdNormalizer()

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
                return SectionTaskResult.ok(cached_summary.content, cache_hit=True)

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
            return SectionTaskResult.ok(summary_result.payload, cache_hit=False)
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def summarize_chapter(
        self,
        doc_name: str,
        chapter_title: str | None = None,
        chapter_id: str | None = None,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_summary: bool = False,
    ) -> SectionTaskResult:
        """Run chapter-summary task with document-level chapter summary cache reuse."""
        normalized_chapter_id, normalized_chapter_title = self._normalize_chapter_target(
            chapter_id=chapter_id,
            chapter_title=chapter_title,
        )
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
            target_chapter = self._find_chapter_or_raise(
                document=structured_document,
                chapter_id=normalized_chapter_id,
                chapter_title=normalized_chapter_title,
            )
            target_section = self._find_section_for_chapter_or_raise(
                chapter=target_chapter,
            )
            resolved_chapter_title = (
                (target_chapter.title or target_section.title or target_chapter.chapter_id).strip()
            )
            chapter_key = self._build_chapter_artifact_key(target_chapter)
            legacy_chapter_key = (
                self._build_legacy_chapter_artifact_key(resolved_chapter_title)
                if resolved_chapter_title
                else None
            )
            chapter_candidate_keys = self._build_chapter_artifact_candidate_keys(
                chapter=target_chapter,
                chapter_title=resolved_chapter_title,
            )
            if (
                normalized_chapter_id is not None
                and normalized_chapter_title is not None
                and normalized_chapter_title != resolved_chapter_title
            ):
                print(
                    "SectionTaskCoordinator#chapter_target_title_ignored:",
                    f"doc_name={doc_name}",
                    f"chapter_id={normalized_chapter_id}",
                    f"request_chapter_title={normalized_chapter_title}",
                    f"resolved_chapter_title={resolved_chapter_title}",
                )
            if (
                not refresh_summary
                and self._is_chapter_summary_cache_valid(
                    chapter_keys=chapter_candidate_keys,
                    chapter_id=target_chapter.chapter_id,
                    chapter_title=resolved_chapter_title,
                    source_section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
                )
            ):
                cached_summary = self._get_chapter_summary_artifact(
                    document=structured_document,
                    chapter_keys=chapter_candidate_keys,
                )
                assert cached_summary is not None
                print(
                    "SectionTaskCoordinator#chapter_summary_cache_hit:",
                    f"doc_name={doc_name}",
                    f"chapter_title={resolved_chapter_title}",
                    f"chapter_id={target_chapter.chapter_id}",
                    f"section_id={target_section.section_id}",
                )
                return SectionTaskResult.ok(cached_summary.content, cache_hit=True)

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
                chapter_title=resolved_chapter_title,
                chapter_id=target_chapter.chapter_id,
                chapter_key=chapter_key,
                legacy_chapter_key=legacy_chapter_key,
            )
            self.document_artifact_repository.update_chapter_summary_artifact(
                doc_name=doc_name,
                chapter_key=chapter_key,
                summary=summary_artifact,
            )
            print(
                "SectionTaskCoordinator#chapter_summary_cache_write:",
                f"doc_name={doc_name}",
                f"chapter_title={resolved_chapter_title}",
                f"chapter_id={target_chapter.chapter_id}",
                f"chapter_key={chapter_key}",
            )
            return SectionTaskResult.ok(summary_result.payload, cache_hit=False)
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
        refresh_quiz: bool = False,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run section-quiz task with persisted task-layout + quiz-cache reuse."""
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
            cached_questions = None
            if not refresh_quiz:
                cached_questions = self._get_valid_cached_quiz_questions(
                    section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._SECTION_QUIZ_PROMPT_VERSION,
                    expected_scope="section",
                )
            if cached_questions is not None:
                print(
                    "SectionTaskCoordinator#section_quiz_cache_hit:",
                    f"doc_name={doc_name}",
                    f"section_id={target_section.section_id}",
                )
                return SectionTaskResult.ok(cached_questions, cache_hit=True)

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
            quiz_result = self.chapter_quiz_service.generate_task_unit_quiz(
                task_unit=resolved_task_unit.task_unit,
                document_title=layout_document.title,
                document_profile=document_profile,
                task_unit_index=resolved_task_unit.task_unit_index,
            )
            if not quiz_result.success:
                return quiz_result

            quiz_artifact = self._build_quiz_artifact(
                questions=quiz_result.payload,
                target_section=target_section,
                document=layout_document,
                document_profile=document_profile,
                resolve_options=resolve_options,
                prompt_version=self._SECTION_QUIZ_PROMPT_VERSION,
                scope="section",
                source_task_unit_id=resolved_task_unit.task_unit.unit_id,
                chapter_title=None,
            )
            self.document_artifact_repository.update_section_quiz_artifact(
                doc_name=doc_name,
                section_id=target_section.section_id,
                quiz=quiz_artifact,
            )
            print(
                "SectionTaskCoordinator#section_quiz_cache_write:",
                f"doc_name={doc_name}",
                f"section_id={target_section.section_id}",
            )
            return SectionTaskResult.ok(quiz_result.payload, cache_hit=False)
        except ValueError as error:
            return SectionTaskResult.fail(str(error))
        except Exception as error:
            return SectionTaskResult.from_llm_error(error)

    def generate_chapter_quiz(
        self,
        doc_name: str,
        chapter_title: str | None = None,
        chapter_id: str | None = None,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_quiz: bool = False,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run chapter-quiz task with document-level chapter quiz cache reuse."""
        normalized_chapter_id, normalized_chapter_title = self._normalize_chapter_target(
            chapter_id=chapter_id,
            chapter_title=chapter_title,
        )
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
            target_chapter = self._find_chapter_or_raise(
                document=structured_document,
                chapter_id=normalized_chapter_id,
                chapter_title=normalized_chapter_title,
            )
            target_section = self._find_section_for_chapter_or_raise(
                chapter=target_chapter,
            )
            resolved_chapter_title = (
                (target_chapter.title or target_section.title or target_chapter.chapter_id).strip()
            )
            chapter_key = self._build_chapter_artifact_key(target_chapter)
            legacy_chapter_key = (
                self._build_legacy_chapter_artifact_key(resolved_chapter_title)
                if resolved_chapter_title
                else None
            )
            chapter_candidate_keys = self._build_chapter_artifact_candidate_keys(
                chapter=target_chapter,
                chapter_title=resolved_chapter_title,
            )
            if (
                normalized_chapter_id is not None
                and normalized_chapter_title is not None
                and normalized_chapter_title != resolved_chapter_title
            ):
                print(
                    "SectionTaskCoordinator#chapter_target_title_ignored:",
                    f"doc_name={doc_name}",
                    f"chapter_id={normalized_chapter_id}",
                    f"request_chapter_title={normalized_chapter_title}",
                    f"resolved_chapter_title={resolved_chapter_title}",
                )
            cached_questions = None
            if not refresh_quiz:
                cached_questions = self._get_valid_cached_chapter_quiz_questions(
                    chapter_keys=chapter_candidate_keys,
                    chapter_id=target_chapter.chapter_id,
                    chapter_title=resolved_chapter_title,
                    source_section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_QUIZ_PROMPT_VERSION,
                )
            if cached_questions is not None:
                print(
                    "SectionTaskCoordinator#chapter_quiz_cache_hit:",
                    f"doc_name={doc_name}",
                    f"chapter_title={resolved_chapter_title}",
                    f"chapter_id={target_chapter.chapter_id}",
                    f"chapter_key={chapter_key}",
                )
                return SectionTaskResult.ok(cached_questions, cache_hit=True)

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
            quiz_result = self.chapter_quiz_service.generate_task_unit_quiz(
                task_unit=resolved_task_unit.task_unit,
                document_title=layout_document.title,
                document_profile=document_profile,
                task_type="chapter_quiz",
                task_unit_index=resolved_task_unit.task_unit_index,
            )
            if not quiz_result.success:
                return quiz_result

            quiz_artifact = self._build_quiz_artifact(
                questions=quiz_result.payload,
                target_section=target_section,
                document=layout_document,
                document_profile=document_profile,
                resolve_options=resolve_options,
                prompt_version=self._CHAPTER_QUIZ_PROMPT_VERSION,
                scope="chapter",
                source_task_unit_id=resolved_task_unit.task_unit.unit_id,
                chapter_title=resolved_chapter_title,
                chapter_id=target_chapter.chapter_id,
                chapter_key=chapter_key,
                legacy_chapter_key=legacy_chapter_key,
            )
            self.document_artifact_repository.update_chapter_quiz_artifact(
                doc_name=doc_name,
                chapter_key=chapter_key,
                quiz=quiz_artifact,
            )
            print(
                "SectionTaskCoordinator#chapter_quiz_cache_write:",
                f"doc_name={doc_name}",
                f"chapter_title={resolved_chapter_title}",
                f"chapter_id={target_chapter.chapter_id}",
                f"chapter_key={chapter_key}",
            )
            return SectionTaskResult.ok(quiz_result.payload, cache_hit=False)
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
        """Return hierarchy-first document layout with embedded task-unit metadata."""
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
        document_profile = self._load_existing_document_profile(doc_name)
        effective_sections = self._resolve_task_layout_sections(
            document=cached_document,
            context="SectionTaskCoordinator#get_document_task_layout",
        )
        self._assert_unique_task_unit_ids_in_sections(
            sections=effective_sections,
            context="SectionTaskCoordinator#get_document_task_layout",
        )

        task_unit_dtos, section_units_by_section_id = self._build_task_unit_dtos_from_document(
            document=cached_document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            sections=effective_sections,
        )

        section_layout_by_id: dict[str, DocumentTaskLayoutSectionDTO] = {}
        section_layouts: list[DocumentTaskLayoutSectionDTO] = []
        for section in effective_sections:
            section_units = section_units_by_section_id.get(section.section_id, [])
            section_mode = self._resolve_section_task_mode(
                section_id=section.section_id,
                section_task_units=section_units,
            )
            section_layout = DocumentTaskLayoutSectionDTO(
                section_id=section.section_id,
                title=section.title,
                container_title=section.container_title,
                section_role=(
                    None
                    if section.section_role is None
                    else section.section_role.value
                ),
                parent_chapter_id=section.parent_chapter_id,
                section_kind=section.section_kind,
                is_implicit_section=section.is_implicit_section,
                task_mode=section_mode,
                task_units=section_units,
                artifacts=self._build_section_artifact_availability(
                    section=section,
                    document=cached_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                ),
            )
            section_layouts.append(section_layout)
            section_layout_by_id[section.section_id] = section_layout

        hierarchy_section_ids = {
            section.section_id
            for chapter in cached_document.chapters
            for section in chapter.sections
        }
        sections_are_hierarchy_source = bool(cached_document.chapters) and all(
            section.section_id in hierarchy_section_ids
            for section in effective_sections
        )
        chapter_layouts = self._build_chapter_layouts(
            document=cached_document,
            section_layout_by_id=section_layout_by_id,
            fallback_section_layouts=section_layouts,
            document_profile=document_profile,
            resolve_options=resolve_options,
            sections_are_hierarchy_source=sections_are_hierarchy_source,
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
            chapters=chapter_layouts,
            sections=section_layouts,
            task_units=task_unit_dtos,
            chapter_artifacts=self._build_chapter_artifact_availability(
                document=cached_document,
                document_profile=document_profile,
                resolve_options=resolve_options,
            ),
            enhanced_parse_recommendation=EnhancedParseRecommendationDTO(
                should_recommend=trigger_decision.should_recommend,
                score=trigger_decision.score,
                reasons=list(trigger_decision.reasons),
                metrics=dict(trigger_decision.metrics),
            ),
            profile_diagnostics=self._build_profile_structure_diagnostics(
                document_profile=document_profile,
                sections=effective_sections,
            ),
        )

    def _compute_current_task_unit_coverage(
        self,
        *,
        sections: list[StructuredSection],
    ) -> tuple[bool, float | None, int]:
        """Compute task-unit coverage from current effective layout sections.

        This is intentionally request-time projection state and should not be
        conflated with prepare-time post_structure_metadata snapshot fields.
        """
        section_count = len(sections)
        sections_with_task_units_count = sum(
            1 for section in sections if len(section.task_units) > 0
        )
        if section_count == 0:
            return False, None, sections_with_task_units_count
        coverage = round(sections_with_task_units_count / section_count, 4)
        available = (
            coverage >= self._TASK_UNIT_STATS_AVAILABLE_COVERAGE_THRESHOLD
        )
        return available, coverage, sections_with_task_units_count

    def _build_profile_structure_diagnostics(
        self,
        *,
        document_profile: DocumentProfile | None,
        sections: list[StructuredSection],
    ) -> ProfileStructureDiagnosticsDTO | None:
        """Build mixed-source diagnostics for task-layout response.

        Persisted profile contributes shape/risk hints, while task-unit stats
        are derived from the *current* section list used for this layout.
        """
        if document_profile is None:
            return None
        parser_metadata = document_profile.parser_metadata
        post_metadata = document_profile.post_structure_metadata
        if parser_metadata is None and post_metadata is None:
            return None

        parser_shape = (
            None
            if parser_metadata is None or parser_metadata.document_structure_shape is None
            else parser_metadata.document_structure_shape.value
        )
        post_shape = (
            None
            if post_metadata is None or post_metadata.actual_structure_shape is None
            else post_metadata.actual_structure_shape.value
        )
        title_uniqueness_risk = (
            None
            if post_metadata is None or post_metadata.title_uniqueness_risk is None
            else post_metadata.title_uniqueness_risk.value
        )
        title_target_requires_id = title_uniqueness_risk in {"medium", "high"}

        (
            task_unit_stats_available,
            task_unit_section_coverage,
            _sections_with_task_units_count,
        ) = self._compute_current_task_unit_coverage(sections=sections)
        task_unit_stats_incomplete = not task_unit_stats_available

        parser_post_shape_mismatch = (
            parser_shape is not None
            and post_shape is not None
            and parser_shape not in {"unknown", "mixed"}
            and post_shape not in {"unknown", "mixed"}
            and parser_shape != post_shape
        )

        warnings: list[str] = []
        if title_target_requires_id:
            warnings.append("chapter_or_section_titles_are_not_unique_use_id_targets")
        if task_unit_stats_incomplete:
            warnings.append("task_unit_stats_not_available_or_incomplete")
            if (
                task_unit_section_coverage is not None
                and task_unit_section_coverage > 0.0
                and task_unit_section_coverage
                < self._TASK_UNIT_STATS_AVAILABLE_COVERAGE_THRESHOLD
            ):
                warnings.append("task_unit_stats_partially_available")
        if parser_post_shape_mismatch:
            warnings.append("pre_structure_shape_differs_from_post_structure_shape")

        enhanced_parse_hint: str | None = None
        post_notes = [] if post_metadata is None else list(post_metadata.notes)
        if (
            post_shape == "flat_long_text"
            and parser_shape in {"essay_sections", "chapter_section", "part_chapter"}
        ):
            enhanced_parse_hint = "common_parser_may_undersegment"
        elif (
            post_shape == "part_chapter"
            and "possible_part_chapter_repeated_local_titles" in post_notes
        ):
            enhanced_parse_hint = "common_parser_may_have_flattened_parent_grouping"

        return ProfileStructureDiagnosticsDTO(
            parser_metadata_shape=parser_shape,
            post_actual_structure_shape=post_shape,
            title_uniqueness_risk=title_uniqueness_risk,
            title_target_requires_id=title_target_requires_id,
            task_unit_stats_available=task_unit_stats_available,
            task_unit_section_coverage=task_unit_section_coverage,
            parser_post_shape_mismatch=parser_post_shape_mismatch,
            enhanced_parse_hint=enhanced_parse_hint,
            warnings=warnings,
        )

    def _build_chapter_layouts(
        self,
        *,
        document: StructuredDocument,
        section_layout_by_id: dict[str, DocumentTaskLayoutSectionDTO],
        fallback_section_layouts: list[DocumentTaskLayoutSectionDTO],
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        sections_are_hierarchy_source: bool,
    ) -> list[DocumentTaskLayoutChapterDTO]:
        """Build hierarchy-first chapter layout tree with legacy synthetic fallback."""
        if document.chapters and sections_are_hierarchy_source:
            chapter_layouts: list[DocumentTaskLayoutChapterDTO] = []
            for chapter in document.chapters:
                section_layouts = [
                    section_layout_by_id[section.section_id]
                    for section in chapter.sections
                    if section.section_id in section_layout_by_id
                ]
                chapter_layouts.append(
                    DocumentTaskLayoutChapterDTO(
                        chapter_id=chapter.chapter_id,
                        title=chapter.title,
                        level=chapter.level,
                        chapter_role=chapter.chapter_role,
                        sections=section_layouts,
                        artifacts=self._build_single_chapter_artifact_availability(
                            chapter=chapter,
                            document=document,
                            document_profile=document_profile,
                            resolve_options=resolve_options,
                        ),
                        metadata=dict(chapter.metadata),
                    )
                )
            return chapter_layouts

        return [
            DocumentTaskLayoutChapterDTO(
                chapter_id="chapter-legacy-0",
                title=document.title,
                level=1,
                chapter_role="legacy_flat_sections",
                sections=list(fallback_section_layouts),
                artifacts=None,
                metadata={
                    "synthetic": True,
                    "reason": "missing_chapters",
                },
            )
        ]

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
            section_count = len(
                get_effective_sections(preparation_result.structured_document)
            )
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
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        sections: list[StructuredSection],
    ) -> tuple[list[TaskUnitDTO], dict[str, list[TaskUnitDTO]]]:
        """Build task-unit DTOs and section->task-unit mapping from persisted data."""
        section_units_by_section_id: dict[str, list[TaskUnitDTO]] = {
            section.section_id: [] for section in sections
        }
        task_unit_dtos: list[TaskUnitDTO] = []

        for section in sections:
            section_task_units: list[TaskUnitDTO] = []
            for task_unit in section.task_units:
                task_unit_dto = TaskUnitDTO(
                    unit_id=task_unit.unit_id,
                    title=task_unit.title,
                    container_title=task_unit.container_title,
                    source_section_ids=list(task_unit.source_section_ids),
                    is_fallback_generated=task_unit.is_fallback_generated,
                    artifacts=self._build_task_unit_artifact_availability(
                        task_unit=task_unit,
                        document=document,
                        document_profile=document_profile,
                        resolve_options=resolve_options,
                    ),
                )
                task_unit_dtos.append(task_unit_dto)
                section_task_units.append(task_unit_dto)
            section_units_by_section_id[section.section_id] = section_task_units
        return task_unit_dtos, section_units_by_section_id

    @staticmethod
    def _build_artifact_availability_from_validity(
        *,
        summary_validity: ArtifactValidityResult,
        quiz_validity: ArtifactValidityResult,
        task_artifacts: TaskArtifacts | None,
    ) -> ArtifactAvailabilityDTO | None:
        """Build lightweight artifact availability metadata for response DTOs."""
        if not summary_validity.exists and not quiz_validity.exists:
            return None
        summary = None if task_artifacts is None else task_artifacts.summary
        quiz = None if task_artifacts is None else task_artifacts.quiz

        return ArtifactAvailabilityDTO(
            has_summary=summary_validity.exists,
            has_quiz=quiz_validity.exists,
            summary_cache_valid=summary_validity.cache_valid,
            quiz_cache_valid=quiz_validity.cache_valid,
            summary_invalid_reason=summary_validity.invalid_reason,
            quiz_invalid_reason=quiz_validity.invalid_reason,
            summary_generated_at=(None if summary is None else summary.generated_at),
            quiz_generated_at=(None if quiz is None else quiz.generated_at),
        )

    def _build_section_artifact_availability(
        self,
        *,
        section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
    ) -> ArtifactAvailabilityDTO | None:
        """Build section-level artifact availability with cache-validity visibility."""
        artifacts = section.task_artifacts
        summary_validity = self._validate_section_summary_artifact(
            section=section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._SECTION_SUMMARY_PROMPT_VERSION,
            expected_scope="section",
            expected_chapter_title=None,
        )
        quiz_validity, _ = self._validate_section_quiz_artifact(
            section=section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._SECTION_QUIZ_PROMPT_VERSION,
            expected_scope="section",
            expected_chapter_title=None,
        )
        return self._build_artifact_availability_from_validity(
            summary_validity=summary_validity,
            quiz_validity=quiz_validity,
            task_artifacts=artifacts,
        )

    def _build_task_unit_artifact_availability(
        self,
        *,
        task_unit: TaskUnit,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
    ) -> ArtifactAvailabilityDTO | None:
        """Build task-unit-level artifact availability with cache-validity visibility."""
        artifacts = task_unit.task_artifacts
        summary_validity = self._validate_task_unit_summary_artifact(
            task_unit=task_unit,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._TASK_UNIT_SUMMARY_PROMPT_VERSION,
        )
        quiz_validity, _ = self._validate_task_unit_quiz_artifact(
            task_unit=task_unit,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._TASK_UNIT_QUIZ_PROMPT_VERSION,
        )
        return self._build_artifact_availability_from_validity(
            summary_validity=summary_validity,
            quiz_validity=quiz_validity,
            task_artifacts=artifacts,
        )

    def _build_chapter_artifact_availability(
        self,
        *,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
    ) -> dict[str, ArtifactAvailabilityDTO]:
        """Build chapter artifact availability map without heavy payload fields."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return {}

        chapter_availability: dict[str, ArtifactAvailabilityDTO] = {}
        if not document.chapters:
            for chapter_key, chapter_artifacts in document_artifacts.chapter_artifacts.items():
                metadata = {}
                if chapter_artifacts.summary is not None:
                    metadata = dict(chapter_artifacts.summary.metadata or {})
                elif chapter_artifacts.quiz is not None:
                    metadata = dict(chapter_artifacts.quiz.metadata or {})
                chapter_title = str(metadata.get("chapter_title") or "")
                if not chapter_title and chapter_key.startswith(self._CHAPTER_ARTIFACT_KEY_PREFIX):
                    chapter_title = chapter_key[len(self._CHAPTER_ARTIFACT_KEY_PREFIX):]
                source_section = self._find_section_by_title(
                    document=document,
                    title=chapter_title,
                )
                chapter_id = str(metadata.get("chapter_id") or "")
                summary_validity = self._validate_chapter_summary_artifact(
                    chapter_keys=[chapter_key],
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    source_section=source_section,
                    document=document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
                )
                quiz_validity, _ = self._validate_chapter_quiz_artifact(
                    chapter_keys=[chapter_key],
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    source_section=source_section,
                    document=document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_QUIZ_PROMPT_VERSION,
                )
                availability = self._build_artifact_availability_from_validity(
                    summary_validity=summary_validity,
                    quiz_validity=quiz_validity,
                    task_artifacts=chapter_artifacts,
                )
                if availability is not None:
                    chapter_availability[chapter_key] = availability
            return chapter_availability

        for chapter in document.chapters:
            chapter_key = self._build_chapter_artifact_key(chapter)
            availability = self._build_single_chapter_artifact_availability(
                chapter=chapter,
                document=document,
                document_profile=document_profile,
                resolve_options=resolve_options,
            )
            if availability is not None:
                chapter_availability[chapter_key] = availability
        return chapter_availability

    def _build_single_chapter_artifact_availability(
        self,
        *,
        chapter: StructuredChapter,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
    ) -> ArtifactAvailabilityDTO | None:
        """Build one chapter artifact availability using legacy chapter-key compatibility."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None

        chapter_title = (chapter.title or "").strip()
        candidate_keys = self._build_chapter_artifact_candidate_keys(
            chapter=chapter,
            chapter_title=chapter_title,
        )

        resolved_chapter_key: str | None = None
        task_artifacts: TaskArtifacts | None = None
        for chapter_key in candidate_keys:
            candidate_artifacts = document_artifacts.chapter_artifacts.get(chapter_key)
            if candidate_artifacts is not None:
                resolved_chapter_key = chapter_key
                task_artifacts = candidate_artifacts
                break
        if resolved_chapter_key is None or task_artifacts is None:
            return None

        source_section = self._find_section_by_title(
            document=document,
            title=chapter_title,
        )
        if source_section is None and chapter.sections:
            source_section = chapter.sections[0]

        summary_validity = self._validate_chapter_summary_artifact(
            chapter_keys=candidate_keys,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter_title,
            source_section=source_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
        )
        quiz_validity, _ = self._validate_chapter_quiz_artifact(
            chapter_keys=candidate_keys,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter_title,
            source_section=source_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._CHAPTER_QUIZ_PROMPT_VERSION,
        )
        return self._build_artifact_availability_from_validity(
            summary_validity=summary_validity,
            quiz_validity=quiz_validity,
            task_artifacts=task_artifacts,
        )

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
        sections = self._resolve_task_layout_sections(
            document=document,
            context="SectionTaskCoordinator#_is_task_layout_cache_valid",
        )
        for section in sections:
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
            duplicate_ids = self.task_unit_id_normalizer.find_duplicate_task_unit_ids(
                document=structured_document
            )
            if duplicate_ids:
                repaired_document = self.task_unit_id_normalizer.normalize_document_task_unit_ids(
                    document=structured_document
                )
                self.document_artifact_repository.save_document(
                    repaired_document,
                    doc_name=doc_name,
                )
                print(
                    "SectionTaskCoordinator#task_layout_cache_repair_duplicate_task_unit_ids:",
                    f"doc_name={doc_name}",
                    f"duplicates={duplicate_ids}",
                )
                return repaired_document
            print(
                "SectionTaskCoordinator#task_layout_cache_hit:",
                f"doc_name={doc_name}",
                f"split_mode={resolve_options.split_mode.value}",
                f"semantic_top_k={resolve_options.semantic_top_k_candidates}",
            )
            return structured_document

        resolved_task_units = self.task_unit_resolver.resolve_with_options(
            document=replace(
                structured_document,
                sections=self._resolve_task_layout_sections(
                    document=structured_document,
                    context="SectionTaskCoordinator#task_layout_cache_write_source",
                ),
            ),
            split_mode=resolve_options.split_mode,
            semantic_top_k_candidates=resolve_options.semantic_top_k_candidates,
        )
        sections_for_write = self._resolve_task_layout_sections(
            document=structured_document,
            context="SectionTaskCoordinator#task_layout_cache_write_target",
        )
        task_units_by_section_id = self._build_task_units_by_section_id(
            sections=sections_for_write,
            task_units=resolved_task_units,
        )
        task_units_by_section_id = self.task_unit_id_normalizer.normalize_task_units_by_section_id(
            document=replace(structured_document, sections=sections_for_write),
            task_units_by_section_id=task_units_by_section_id,
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
        self._assert_unique_task_unit_ids_in_sections(
            sections=self._resolve_task_layout_sections(
                document=updated_document,
                context="SectionTaskCoordinator#task_layout_cache_write",
            ),
            context="SectionTaskCoordinator#task_layout_cache_write",
        )
        print(
            "SectionTaskCoordinator#task_layout_cache_write:",
            f"doc_name={doc_name}",
            f"split_mode={resolve_options.split_mode.value}",
            f"semantic_top_k={resolve_options.semantic_top_k_candidates}",
            f"refresh={refresh_task_units}",
        )
        return updated_document

    def _resolve_task_layout_sections(
        self,
        *,
        document: StructuredDocument,
        context: str,
    ) -> list[StructuredSection]:
        """Resolve section list for task-layout read path with hierarchy-first fallback."""
        if not document.chapters:
            raise ValueError(
                f"{context}: legacy sections-only document requires migration"
            )

        warnings = validate_chapter_hierarchy_consistency(document)
        severe_warnings = [
            warning
            for warning in warnings
            if is_severe_hierarchy_warning(warning)
        ]
        repairable_warnings = [
            warning
            for warning in severe_warnings
            if warning.startswith("duplicate_task_unit_id:")
        ]
        blocking_warnings = [
            warning
            for warning in severe_warnings
            if not warning.startswith("duplicate_task_unit_id:")
        ]
        if blocking_warnings:
            raise ValueError(
                f"{context}: severe hierarchy inconsistency; legacy sections fallback disabled; "
                f"warnings={blocking_warnings}"
            )

        if repairable_warnings:
            print(
                "SectionTaskCoordinator#task_layout_hierarchy_repairable_warnings:",
                f"context={context}",
                f"warnings={repairable_warnings}",
            )

        if warnings:
            print(
                "SectionTaskCoordinator#task_layout_hierarchy_warnings:",
                f"context={context}",
                f"warnings={warnings}",
            )
        sections = get_effective_sections(document)
        print(
            "SectionTaskCoordinator#task_layout_hierarchy_first_read:",
            f"context={context}",
            f"section_count={len(sections)}",
            "source=chapters",
        )
        return sections

    @staticmethod
    def _find_duplicate_task_unit_ids_in_sections(
        *,
        sections: list[StructuredSection],
    ) -> dict[str, int]:
        """Collect duplicate task-unit ids from arbitrary section list."""
        counts: dict[str, int] = {}
        for section in sections:
            for task_unit in section.task_units:
                counts[task_unit.unit_id] = counts.get(task_unit.unit_id, 0) + 1
        return {
            unit_id: count
            for unit_id, count in counts.items()
            if count > 1
        }

    def _assert_unique_task_unit_ids_in_sections(
        self,
        *,
        sections: list[StructuredSection],
        context: str,
    ) -> None:
        """Raise when duplicate task-unit ids are detected in selected sections."""
        duplicates = self._find_duplicate_task_unit_ids_in_sections(sections=sections)
        if duplicates:
            duplicate_repr = ", ".join(
                f"{unit_id}:{count}" for unit_id, count in sorted(duplicates.items())
            )
            raise ValueError(
                f"{context}: duplicate task_unit_id detected -> {duplicate_repr}"
            )

    def _build_task_units_by_section_id(
        self,
        *,
        sections: list[StructuredSection],
        task_units: list[TaskUnit],
    ) -> dict[str, list[TaskUnit]]:
        """Group resolver output into section-level persisted task-unit lists."""
        task_units_by_section_id: dict[str, list[TaskUnit]] = {
            section.section_id: [] for section in sections
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
            else (document_profile.document_language_code or "").strip() or None
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
        chapter_id: str | None = None,
        chapter_key: str | None = None,
        legacy_chapter_key: str | None = None,
    ) -> SummaryArtifact:
        """Build persisted summary artifact payload."""
        normalized_content = content.strip()
        metadata: dict[str, str | int | None] = {
            "summary_scope": scope,
            "source_section_id": target_section.section_id,
            "source_task_unit_id": source_task_unit_id,
        }
        if chapter_title is not None:
            metadata["chapter_title"] = chapter_title
        if chapter_id is not None:
            metadata["chapter_id"] = chapter_id
        if chapter_key is not None:
            metadata["chapter_key"] = chapter_key
        if legacy_chapter_key is not None:
            metadata["legacy_chapter_key"] = legacy_chapter_key

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
        validity = self._validate_section_summary_artifact(
            section=section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=prompt_version,
            expected_scope=expected_scope,
            expected_chapter_title=expected_chapter_title,
        )
        return validity.cache_valid is True

    def _validate_section_summary_artifact(
        self,
        *,
        section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        expected_scope: str,
        expected_chapter_title: str | None = None,
    ) -> ArtifactValidityResult:
        """Validate section-level summary artifact against current runtime context."""
        artifacts = section.task_artifacts
        if artifacts is None or artifacts.summary is None:
            return ArtifactValidityResult.missing()
        summary = artifacts.summary
        if not summary.content.strip():
            return ArtifactValidityResult.invalid("empty_content")
        if summary.source_hash != self._compute_section_source_hash(section):
            return ArtifactValidityResult.invalid("source_hash_mismatch")

        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if summary.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch")
        if summary.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch")
        if (
            summary.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch")
        if summary.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch")

        metadata = dict(summary.metadata or {})
        if metadata.get("summary_scope") != expected_scope:
            return ArtifactValidityResult.invalid("scope_mismatch")
        source_section_id = metadata.get("source_section_id", metadata.get("section_id"))
        if source_section_id != section.section_id:
            return ArtifactValidityResult.invalid("section_id_mismatch")
        if expected_scope == "chapter":
            if (metadata.get("chapter_title") or "") != (
                expected_chapter_title or ""
            ):
                return ArtifactValidityResult.invalid("chapter_title_mismatch")
        return ArtifactValidityResult.valid()

    def _build_quiz_artifact(
        self,
        *,
        questions: list[QuizQuestion],
        target_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        scope: str,
        source_task_unit_id: str,
        chapter_title: str | None,
        chapter_id: str | None = None,
        chapter_key: str | None = None,
        legacy_chapter_key: str | None = None,
    ) -> QuizArtifact:
        """Build persisted quiz artifact payload."""
        metadata: dict[str, str | int | None] = {
            "quiz_scope": scope,
            "section_id": target_section.section_id,
            "source_section_id": target_section.section_id,
            "source_task_unit_id": source_task_unit_id,
        }
        if chapter_title is not None:
            metadata["chapter_title"] = chapter_title
        if chapter_id is not None:
            metadata["chapter_id"] = chapter_id
        if chapter_key is not None:
            metadata["chapter_key"] = chapter_key
        if legacy_chapter_key is not None:
            metadata["legacy_chapter_key"] = legacy_chapter_key
        return self._quiz_artifact_from_questions(
            questions=questions,
            target_section=target_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=prompt_version,
            metadata=metadata,
        )

    def _get_valid_cached_quiz_questions(
        self,
        *,
        section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        expected_scope: str,
        expected_chapter_title: str | None = None,
    ) -> list[QuizQuestion] | None:
        """Return parsed cached quiz questions when section-level quiz artifact is valid."""
        validity, questions = self._validate_section_quiz_artifact(
            section=section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=prompt_version,
            expected_scope=expected_scope,
            expected_chapter_title=expected_chapter_title,
        )
        if validity.cache_valid is not True:
            return None
        return questions

    def _validate_section_quiz_artifact(
        self,
        *,
        section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        expected_scope: str,
        expected_chapter_title: str | None = None,
    ) -> tuple[ArtifactValidityResult, list[QuizQuestion] | None]:
        """Validate section-level quiz artifact and parse persisted quiz payload."""
        artifacts = section.task_artifacts
        if artifacts is None or artifacts.quiz is None:
            return ArtifactValidityResult.missing(), None
        quiz = artifacts.quiz
        parsed_questions = self._quiz_questions_from_artifact(quiz)
        if parsed_questions is None:
            return ArtifactValidityResult.invalid("malformed_quiz_items"), None
        if quiz.source_hash != self._compute_section_source_hash(section):
            return ArtifactValidityResult.invalid("source_hash_mismatch"), None
        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if quiz.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch"), None
        if quiz.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch"), None
        if (
            quiz.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch"), None
        if quiz.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch"), None
        if quiz.quiz_schema_version != self._QUIZ_SCHEMA_VERSION:
            return ArtifactValidityResult.invalid("quiz_schema_version_mismatch"), None
        metadata = dict(quiz.metadata or {})
        if metadata.get("quiz_scope") != expected_scope:
            return ArtifactValidityResult.invalid("scope_mismatch"), None
        if metadata.get("section_id") != section.section_id:
            return ArtifactValidityResult.invalid("section_id_mismatch"), None
        if expected_scope == "chapter":
            if (metadata.get("chapter_title") or "") != (
                expected_chapter_title or ""
            ):
                return ArtifactValidityResult.invalid("chapter_title_mismatch"), None
        return ArtifactValidityResult.valid(), parsed_questions

    @staticmethod
    def _quiz_questions_from_artifact(
        quiz: QuizArtifact,
    ) -> list[QuizQuestion] | None:
        """Parse persisted quiz artifact into structured quiz questions."""
        if not quiz.items:
            return None
        parsed_questions: list[QuizQuestion] = []
        try:
            for item in quiz.items:
                if not isinstance(item, dict):
                    return None
                parsed_questions.append(QuizQuestion.from_dict(item))
        except ValueError:
            return None
        if not parsed_questions:
            return None
        return parsed_questions

    def _quiz_artifact_from_questions(
        self,
        *,
        questions: list[QuizQuestion],
        target_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
        metadata: dict[str, str | int | None],
    ) -> QuizArtifact:
        """Convert quiz question DTO list into persisted quiz artifact payload."""
        items = [
            {
                "question_id": question.question_id,
                "question_text": question.question_text,
                "answer_text": question.answer_text,
            }
            for question in questions
        ]
        return QuizArtifact(
            items=items,
            language=self._resolve_summary_language(
                document=document,
                document_profile=document_profile,
            ),
            generated_at=datetime.now(timezone.utc).isoformat(),
            source_hash=self._compute_section_source_hash(target_section),
            prompt_version=prompt_version,
            quiz_schema_version=self._QUIZ_SCHEMA_VERSION,
            task_unit_split_mode=resolve_options.split_mode.value,
            semantic_top_k_candidates=resolve_options.semantic_top_k_candidates,
            metadata=metadata,
        )

    @classmethod
    def _build_chapter_artifact_key(
        cls,
        chapter: StructuredChapter | str,
    ) -> str:
        """Build chapter artifact key.

        StructuredChapter input -> stable id-based key (preferred).
        str input -> legacy title-based key (compatibility).
        """
        if isinstance(chapter, StructuredChapter):
            chapter_id = chapter.chapter_id.strip()
            if not chapter_id:
                raise ValueError("chapter.chapter_id cannot be empty")
            return f"{cls._CHAPTER_ARTIFACT_KEY_BY_ID_PREFIX}{chapter_id}"

        normalized_chapter_title = chapter.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")
        return f"{cls._CHAPTER_ARTIFACT_KEY_PREFIX}{normalized_chapter_title}"

    @classmethod
    def _build_legacy_chapter_artifact_key(cls, chapter_title: str) -> str:
        """Build legacy title-based chapter artifact key."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")
        return f"{cls._CHAPTER_ARTIFACT_KEY_PREFIX}{normalized_chapter_title}"

    def _build_chapter_artifact_candidate_keys(
        self,
        *,
        chapter: StructuredChapter,
        chapter_title: str | None,
    ) -> list[str]:
        """Return preferred chapter artifact lookup keys (id-first with legacy fallback)."""
        candidate_keys: list[str] = [self._build_chapter_artifact_key(chapter)]
        legacy_chapter_key = chapter.metadata.get("legacy_chapter_key")
        if isinstance(legacy_chapter_key, str) and legacy_chapter_key.strip():
            candidate_keys.append(legacy_chapter_key.strip())
        normalized_chapter_title = (chapter_title or "").strip()
        if normalized_chapter_title:
            candidate_keys.append(self._build_legacy_chapter_artifact_key(normalized_chapter_title))
        deduped: list[str] = []
        seen: set[str] = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    @staticmethod
    def _get_chapter_summary_artifact(
        *,
        document: StructuredDocument,
        chapter_keys: list[str],
    ) -> SummaryArtifact | None:
        """Get document-level chapter summary artifact if present."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None
        for chapter_key in chapter_keys:
            chapter_artifact = document_artifacts.chapter_artifacts.get(chapter_key)
            if chapter_artifact is None:
                continue
            if chapter_artifact.summary is not None:
                return chapter_artifact.summary
        return None

    def _is_chapter_summary_cache_valid(
        self,
        *,
        chapter_keys: list[str],
        chapter_id: str,
        chapter_title: str,
        source_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> bool:
        """Check whether one document-level chapter summary artifact can be reused safely."""
        validity = self._validate_chapter_summary_artifact(
            chapter_keys=chapter_keys,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            source_section=source_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=prompt_version,
        )
        return validity.cache_valid is True

    def _validate_chapter_summary_artifact(
        self,
        *,
        chapter_keys: list[str],
        chapter_id: str,
        chapter_title: str,
        source_section: StructuredSection | None,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> ArtifactValidityResult:
        """Validate document-level chapter summary artifact against current runtime context."""
        summary = self._get_chapter_summary_artifact(
            document=document,
            chapter_keys=chapter_keys,
        )
        if summary is None:
            return ArtifactValidityResult.missing()
        if not summary.content.strip():
            return ArtifactValidityResult.invalid("empty_content")
        metadata = dict(summary.metadata or {})
        metadata_source_section_id = (metadata.get("source_section_id") or "").strip()
        metadata_source_section = None
        if metadata_source_section_id:
            metadata_source_section = self._find_section_by_id(
                document=document,
                section_id=metadata_source_section_id,
            )
            if metadata_source_section is None:
                return ArtifactValidityResult.invalid("source_section_not_found")

        resolved_source_section = metadata_source_section or source_section
        if resolved_source_section is None:
            return ArtifactValidityResult.invalid("source_section_not_found")
        if summary.source_hash != self._compute_section_source_hash(resolved_source_section):
            return ArtifactValidityResult.invalid("source_hash_mismatch")

        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if summary.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch")
        if summary.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch")
        if (
            summary.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch")
        if summary.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch")

        if metadata.get("summary_scope") != "chapter":
            return ArtifactValidityResult.invalid("scope_mismatch")
        metadata_chapter_id = (metadata.get("chapter_id") or "").strip()
        if metadata_chapter_id and metadata_chapter_id != chapter_id:
            return ArtifactValidityResult.invalid("chapter_id_mismatch")
        if (metadata.get("chapter_title") or "") != chapter_title:
            return ArtifactValidityResult.invalid("chapter_title_mismatch")
        metadata_chapter_key = (metadata.get("chapter_key") or "").strip()
        if metadata_chapter_key and metadata_chapter_key not in chapter_keys:
            return ArtifactValidityResult.invalid("chapter_key_mismatch")
        if metadata_source_section_id and metadata_source_section_id != resolved_source_section.section_id:
            return ArtifactValidityResult.invalid("section_id_mismatch")
        return ArtifactValidityResult.valid()

    @staticmethod
    def _get_chapter_quiz_artifact(
        *,
        document: StructuredDocument,
        chapter_keys: list[str],
    ) -> QuizArtifact | None:
        """Get document-level chapter quiz artifact if present."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None
        for chapter_key in chapter_keys:
            chapter_artifact = document_artifacts.chapter_artifacts.get(chapter_key)
            if chapter_artifact is None:
                continue
            if chapter_artifact.quiz is not None:
                return chapter_artifact.quiz
        return None

    def _get_valid_cached_chapter_quiz_questions(
        self,
        *,
        chapter_keys: list[str],
        chapter_id: str,
        chapter_title: str,
        source_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> list[QuizQuestion] | None:
        """Return parsed cached chapter-quiz questions when chapter-level artifact is valid."""
        validity, parsed = self._validate_chapter_quiz_artifact(
            chapter_keys=chapter_keys,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            source_section=source_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=prompt_version,
        )
        if validity.cache_valid is not True:
            return None
        return parsed

    def _validate_chapter_quiz_artifact(
        self,
        *,
        chapter_keys: list[str],
        chapter_id: str,
        chapter_title: str,
        source_section: StructuredSection | None,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> tuple[ArtifactValidityResult, list[QuizQuestion] | None]:
        """Validate document-level chapter quiz artifact and parse cached payload."""
        quiz = self._get_chapter_quiz_artifact(
            document=document,
            chapter_keys=chapter_keys,
        )
        if quiz is None:
            return ArtifactValidityResult.missing(), None
        parsed = self._quiz_questions_from_artifact(quiz)
        if parsed is None:
            return ArtifactValidityResult.invalid("malformed_quiz_items"), None
        metadata = dict(quiz.metadata or {})
        metadata_source_section_id = (metadata.get("source_section_id") or "").strip()
        metadata_source_section = None
        if metadata_source_section_id:
            metadata_source_section = self._find_section_by_id(
                document=document,
                section_id=metadata_source_section_id,
            )
            if metadata_source_section is None:
                return ArtifactValidityResult.invalid("source_section_not_found"), None

        resolved_source_section = metadata_source_section or source_section
        if resolved_source_section is None:
            return ArtifactValidityResult.invalid("source_section_not_found"), None
        if quiz.source_hash != self._compute_section_source_hash(resolved_source_section):
            return ArtifactValidityResult.invalid("source_hash_mismatch"), None
        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if quiz.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch"), None
        if quiz.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch"), None
        if (
            quiz.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch"), None
        if quiz.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch"), None
        if quiz.quiz_schema_version != self._QUIZ_SCHEMA_VERSION:
            return ArtifactValidityResult.invalid("quiz_schema_version_mismatch"), None
        if metadata.get("quiz_scope") != "chapter":
            return ArtifactValidityResult.invalid("scope_mismatch"), None
        metadata_chapter_id = (metadata.get("chapter_id") or "").strip()
        if metadata_chapter_id and metadata_chapter_id != chapter_id:
            return ArtifactValidityResult.invalid("chapter_id_mismatch"), None
        if (metadata.get("chapter_title") or "") != chapter_title:
            return ArtifactValidityResult.invalid("chapter_title_mismatch"), None
        metadata_chapter_key = (metadata.get("chapter_key") or "").strip()
        if metadata_chapter_key and metadata_chapter_key not in chapter_keys:
            return ArtifactValidityResult.invalid("chapter_key_mismatch"), None
        if metadata_source_section_id and metadata_source_section_id != resolved_source_section.section_id:
            return ArtifactValidityResult.invalid("section_id_mismatch"), None
        return ArtifactValidityResult.valid(), parsed

    @staticmethod
    def _find_section_by_id(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> StructuredSection | None:
        """Find section by id using hierarchy-first lookup with legacy fallback."""
        return find_section_by_id_effective(document, section_id)

    def _validate_task_unit_summary_artifact(
        self,
        *,
        task_unit: TaskUnit,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> ArtifactValidityResult:
        """Validate task-unit summary artifact against current task-layout context."""
        artifacts = task_unit.task_artifacts
        if artifacts is None or artifacts.summary is None:
            return ArtifactValidityResult.missing()
        summary = artifacts.summary
        if not summary.content.strip():
            return ArtifactValidityResult.invalid("empty_content")
        source_hash = hashlib.sha256(task_unit.content.encode("utf-8")).hexdigest()
        if summary.source_hash != source_hash:
            return ArtifactValidityResult.invalid("source_hash_mismatch")
        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if summary.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch")
        if summary.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch")
        if (
            summary.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch")
        if summary.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch")
        return ArtifactValidityResult.valid()

    def _validate_task_unit_quiz_artifact(
        self,
        *,
        task_unit: TaskUnit,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> tuple[ArtifactValidityResult, list[QuizQuestion] | None]:
        """Validate task-unit quiz artifact against current task-layout context."""
        artifacts = task_unit.task_artifacts
        if artifacts is None or artifacts.quiz is None:
            return ArtifactValidityResult.missing(), None
        quiz = artifacts.quiz
        parsed_questions = self._quiz_questions_from_artifact(quiz)
        if parsed_questions is None:
            return ArtifactValidityResult.invalid("malformed_quiz_items"), None
        source_hash = hashlib.sha256(task_unit.content.encode("utf-8")).hexdigest()
        if quiz.source_hash != source_hash:
            return ArtifactValidityResult.invalid("source_hash_mismatch"), None
        expected_language = self._resolve_summary_language(
            document=document,
            document_profile=document_profile,
        )
        if quiz.language != expected_language:
            return ArtifactValidityResult.invalid("language_mismatch"), None
        if quiz.task_unit_split_mode != resolve_options.split_mode.value:
            return ArtifactValidityResult.invalid("split_mode_mismatch"), None
        if (
            quiz.semantic_top_k_candidates
            != resolve_options.semantic_top_k_candidates
        ):
            return ArtifactValidityResult.invalid("semantic_top_k_mismatch"), None
        if quiz.prompt_version != prompt_version:
            return ArtifactValidityResult.invalid("prompt_version_mismatch"), None
        if quiz.quiz_schema_version != self._QUIZ_SCHEMA_VERSION:
            return ArtifactValidityResult.invalid("quiz_schema_version_mismatch"), None
        return ArtifactValidityResult.valid(), parsed_questions

    @staticmethod
    def _find_section_by_title(
        *,
        document: StructuredDocument,
        title: str,
    ) -> StructuredSection | None:
        """Find first section by exact normalized title using effective lookup order."""
        matches = find_sections_by_title_effective(document, title)
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _find_section_or_raise(
        *,
        document: StructuredDocument,
        section_id: str,
    ) -> StructuredSection:
        """Find one section by id with hierarchy-first semantics or raise ValueError."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")
        resolved = find_section_by_id_effective(
            document,
            normalized_section_id,
        )
        if resolved is not None:
            return resolved
        raise ValueError(
            f"section_id '{normalized_section_id}' not found in document '{document.document_id}'"
        )

    @staticmethod
    def _find_section_by_chapter_title_or_raise(
        *,
        document: StructuredDocument,
        chapter_title: str,
    ) -> StructuredSection:
        """Find chapter target section by title using hierarchy-first chapter semantics."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")
        resolved = find_section_by_chapter_title_effective(
            document,
            normalized_chapter_title,
        )
        if resolved is not None:
            return resolved
        raise ValueError(
            f"chapter_title '{normalized_chapter_title}' not found in document '{document.title}'"
        )

    @staticmethod
    def _normalize_chapter_target(
        *,
        chapter_id: str | None,
        chapter_title: str | None,
    ) -> tuple[str | None, str | None]:
        """Normalize chapter target identity and enforce at-least-one rule."""
        normalized_chapter_id = (chapter_id or "").strip() or None
        normalized_chapter_title = (chapter_title or "").strip() or None
        if normalized_chapter_id is None and normalized_chapter_title is None:
            raise ValueError("chapter_id or chapter_title must be provided")
        return normalized_chapter_id, normalized_chapter_title

    @staticmethod
    def _find_chapter_or_raise(
        *,
        document: StructuredDocument,
        chapter_id: str | None,
        chapter_title: str | None,
    ) -> StructuredChapter:
        """Resolve one chapter by chapter_id first, then chapter_title as fallback."""
        normalized_chapter_id = (chapter_id or "").strip() or None
        normalized_chapter_title = (chapter_title or "").strip() or None
        if normalized_chapter_id is None and normalized_chapter_title is None:
            raise ValueError("chapter_id or chapter_title must be provided")

        if normalized_chapter_id is not None:
            resolved = find_chapter_by_id_effective(document, normalized_chapter_id)
            if resolved is None:
                raise ValueError(
                    f"chapter_id '{normalized_chapter_id}' not found in document '{document.document_id}'"
                )
            return resolved

        assert normalized_chapter_title is not None
        matched_chapters = find_chapters_by_title_effective(
            document,
            normalized_chapter_title,
        )
        if len(matched_chapters) > 1:
            raise ValueError(
                f"ambiguous chapter title: '{normalized_chapter_title}' matched {len(matched_chapters)} chapters"
            )
        if len(matched_chapters) == 1:
            return matched_chapters[0]

        resolved_section = find_section_by_chapter_title_effective(
            document,
            normalized_chapter_title,
            allow_legacy_fallback=True,
        )
        if resolved_section is not None:
            synthetic_chapter_id = (
                (resolved_section.parent_chapter_id or "").strip()
                or f"legacy::{resolved_section.section_id}"
            )
            return StructuredChapter(
                chapter_id=synthetic_chapter_id,
                title=resolved_section.title,
                level=max(1, int(resolved_section.level)),
                chapter_role=(
                    None
                    if resolved_section.section_role is None
                    else resolved_section.section_role.value
                ),
                sections=[resolved_section],
                metadata={"synthetic_from_legacy": True},
            )

        raise ValueError(
            f"chapter_title '{normalized_chapter_title}' not found in document '{document.title}'"
        )

    @staticmethod
    def _find_section_for_chapter_or_raise(
        *,
        chapter: StructuredChapter,
    ) -> StructuredSection:
        """Resolve task target section for one chapter, preferring chapter_body."""
        if not chapter.sections:
            raise ValueError(
                f"chapter has no sections: chapter_id='{chapter.chapter_id}' title='{chapter.title}'"
            )
        chapter_body_sections = [
            section
            for section in chapter.sections
            if (section.section_kind or "").strip() == "chapter_body"
        ]
        if chapter_body_sections:
            return chapter_body_sections[0]
        return chapter.sections[0]

    def _resolve_task_unit_for_section_id_from_persisted_layout(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
        allow_legacy_ordering_fallback: bool = False,
    ) -> ResolvedTaskUnit:
        """Resolve one task unit from persisted section.task_units without recomputing resolver."""
        normalized_section_id = section_id.strip()
        if not normalized_section_id:
            raise ValueError("section_id cannot be empty")

        target_section = self._find_section_or_raise(
            document=document,
            section_id=normalized_section_id,
        )
        if not target_section.task_units:
            raise ValueError(
                f"section_id '{normalized_section_id}' has no persisted task units in "
                f"document '{document.document_id}' (task-layout cache missing/stale)"
            )

        sections_for_order = self._resolve_task_layout_sections(
            document=document,
            context="SectionTaskCoordinator#resolve_task_unit_for_section_id_from_persisted_layout",
        )
        task_unit_index_by_id: dict[str, int] = {}
        ordered_task_units: list[TaskUnit] = []
        for section in sections_for_order:
            for task_unit in section.task_units:
                if task_unit.unit_id in task_unit_index_by_id:
                    continue
                task_unit_index_by_id[task_unit.unit_id] = len(ordered_task_units)
                ordered_task_units.append(task_unit)

        if not ordered_task_units:
            raise ValueError(
                f"no persisted task units found for document '{document.document_id}'"
            )

        selected_task_unit = target_section.task_units[0]
        selected_index = task_unit_index_by_id.get(selected_task_unit.unit_id)
        if selected_index is None and allow_legacy_ordering_fallback:
            selected_index = self._resolve_legacy_section_ordering_fallback(
                document=document,
                selected_task_unit_id=selected_task_unit.unit_id,
                task_unit_index_by_id=task_unit_index_by_id,
                ordered_task_units=ordered_task_units,
                context="SectionTaskCoordinator#resolve_task_unit_for_section_id_from_persisted_layout",
            )
        if selected_index is None:
            raise ValueError(
                f"section_id '{normalized_section_id}' task units are not aligned with effective "
                f"task-layout ordering for document '{document.document_id}'"
            )

        return ResolvedTaskUnit(
            task_unit=selected_task_unit,
            task_unit_index=selected_index,
        )

    @staticmethod
    def _resolve_legacy_section_ordering_fallback(
        *,
        document: StructuredDocument,
        selected_task_unit_id: str,
        task_unit_index_by_id: dict[str, int],
        ordered_task_units: list[TaskUnit],
        context: str,
    ) -> int | None:
        """Legacy compatibility fallback for ordering alignment only.

        This path is intentionally not hierarchy-first: it scans root legacy
        `document.sections` to align stale persisted task-unit ordering for old
        payloads. It should remain compatibility-only and can be replaced by
        fail-fast once runtime legacy fallback is retired.
        """
        if not document.sections:
            return None
        for section in document.sections:
            for task_unit in section.task_units:
                if task_unit.unit_id in task_unit_index_by_id:
                    continue
                task_unit_index_by_id[task_unit.unit_id] = len(ordered_task_units)
                ordered_task_units.append(task_unit)

        selected_index = task_unit_index_by_id.get(selected_task_unit_id)
        if selected_index is not None:
            print(
                "SectionTaskCoordinator#legacy_section_ordering_fallback_hit:",
                f"context={context}",
                f"document_id={document.document_id}",
                f"task_unit_id={selected_task_unit_id}",
            )
        return selected_index

    def _resolve_task_unit_for_section_id(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
        resolve_options: TaskUnitResolveOptions,
    ) -> ResolvedTaskUnit:
        """Resolve first task unit whose source section ids contain target section id."""
        target_section = self._find_section_or_raise(
            document=document,
            section_id=section_id,
        )
        normalized_section_id = target_section.section_id

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
        target_section = self._find_section_by_chapter_title_or_raise(
            document=document,
            chapter_title=normalized_chapter_title,
        )

        return self._resolve_task_unit_for_section_id(
            document=document,
            section_id=target_section.section_id,
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
