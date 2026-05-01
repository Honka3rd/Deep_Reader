import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, replace

from config.faiss_storage_config import FaissStorageConfig
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.preparation_mode import PreparationMode
from document_structure.document_artifact_repository import DocumentArtifactRepository
from document_structure.document_hierarchy_index import (
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
        chapter_title: str,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_summary: bool = False,
    ) -> SectionTaskResult:
        """Run chapter-summary task with document-level chapter summary cache reuse."""
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
            chapter_key = self._build_chapter_artifact_key(normalized_chapter_title)
            if (
                not refresh_summary
                and self._is_chapter_summary_cache_valid(
                    chapter_key=chapter_key,
                    chapter_title=normalized_chapter_title,
                    source_section=target_section,
                    document=structured_document,
                    document_profile=document_profile,
                    resolve_options=resolve_options,
                    prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
                )
            ):
                cached_summary = self._get_chapter_summary_artifact(
                    document=structured_document,
                    chapter_key=chapter_key,
                )
                assert cached_summary is not None
                print(
                    "SectionTaskCoordinator#chapter_summary_cache_hit:",
                    f"doc_name={doc_name}",
                    f"chapter_title={normalized_chapter_title}",
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
                chapter_title=normalized_chapter_title,
                chapter_key=chapter_key,
            )
            self.document_artifact_repository.update_chapter_summary_artifact(
                doc_name=doc_name,
                chapter_key=chapter_key,
                summary=summary_artifact,
            )
            print(
                "SectionTaskCoordinator#chapter_summary_cache_write:",
                f"doc_name={doc_name}",
                f"chapter_title={normalized_chapter_title}",
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
        chapter_title: str,
        task_unit_split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
        refresh_quiz: bool = False,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        """Run chapter-quiz task with document-level chapter quiz cache reuse."""
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
            chapter_key = self._build_chapter_artifact_key(normalized_chapter_title)
            cached_questions = None
            if not refresh_quiz:
                cached_questions = self._get_valid_cached_chapter_quiz_questions(
                    chapter_key=chapter_key,
                    chapter_title=normalized_chapter_title,
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
                    f"chapter_title={normalized_chapter_title}",
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
                chapter_title=normalized_chapter_title,
                chapter_key=chapter_key,
            )
            self.document_artifact_repository.update_chapter_quiz_artifact(
                doc_name=doc_name,
                chapter_key=chapter_key,
                quiz=quiz_artifact,
            )
            print(
                "SectionTaskCoordinator#chapter_quiz_cache_write:",
                f"doc_name={doc_name}",
                f"chapter_title={normalized_chapter_title}",
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
        for chapter_key, chapter_artifacts in document_artifacts.chapter_artifacts.items():
            chapter_title = chapter_key
            if chapter_key.startswith(self._CHAPTER_ARTIFACT_KEY_PREFIX):
                chapter_title = chapter_key[len(self._CHAPTER_ARTIFACT_KEY_PREFIX):]
            source_section = self._find_section_by_title(document=document, title=chapter_title)
            summary_validity = self._validate_chapter_summary_artifact(
                chapter_key=chapter_key,
                chapter_title=chapter_title,
                source_section=source_section,
                document=document,
                document_profile=document_profile,
                resolve_options=resolve_options,
                prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
            )
            quiz_validity, _ = self._validate_chapter_quiz_artifact(
                chapter_key=chapter_key,
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

        candidate_keys: list[str] = []
        legacy_chapter_key = chapter.metadata.get("legacy_chapter_key")
        if isinstance(legacy_chapter_key, str) and legacy_chapter_key.strip():
            candidate_keys.append(legacy_chapter_key.strip())
        if chapter.title and chapter.title.strip():
            candidate_keys.append(self._build_chapter_artifact_key(chapter.title))
        if chapter.chapter_id.strip():
            candidate_keys.append(self._build_chapter_artifact_key(chapter.chapter_id))

        seen_keys: set[str] = set()
        resolved_chapter_key: str | None = None
        task_artifacts: TaskArtifacts | None = None
        for chapter_key in candidate_keys:
            if chapter_key in seen_keys:
                continue
            seen_keys.add(chapter_key)
            candidate_artifacts = document_artifacts.chapter_artifacts.get(chapter_key)
            if candidate_artifacts is not None:
                resolved_chapter_key = chapter_key
                task_artifacts = candidate_artifacts
                break
        if resolved_chapter_key is None or task_artifacts is None:
            return None

        chapter_title = (
            (chapter.title or "").strip()
            or resolved_chapter_key.removeprefix(self._CHAPTER_ARTIFACT_KEY_PREFIX)
        )
        source_section = self._find_section_by_title(
            document=document,
            title=chapter_title,
        )
        if source_section is None and chapter.sections:
            source_section = chapter.sections[0]

        summary_validity = self._validate_chapter_summary_artifact(
            chapter_key=resolved_chapter_key,
            chapter_title=chapter_title,
            source_section=source_section,
            document=document,
            document_profile=document_profile,
            resolve_options=resolve_options,
            prompt_version=self._CHAPTER_SUMMARY_PROMPT_VERSION,
        )
        quiz_validity, _ = self._validate_chapter_quiz_artifact(
            chapter_key=resolved_chapter_key,
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
            return list(document.sections)

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
            print(
                "SectionTaskCoordinator#task_layout_hierarchy_fallback_legacy:",
                f"context={context}",
                f"severe_warnings={blocking_warnings}",
            )
            if not document.sections:
                raise ValueError(
                    f"{context}: hierarchy is inconsistent and no legacy sections are available"
                )
            return list(document.sections)

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
        chapter_key: str | None = None,
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
        if chapter_key is not None:
            metadata["chapter_key"] = chapter_key

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
        chapter_key: str | None = None,
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
        if chapter_key is not None:
            metadata["chapter_key"] = chapter_key
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
    def _build_chapter_artifact_key(cls, chapter_title: str) -> str:
        """Build stable chapter artifact key used by document-level chapter_artifacts map."""
        normalized_chapter_title = chapter_title.strip()
        if not normalized_chapter_title:
            raise ValueError("chapter_title cannot be empty")
        return f"{cls._CHAPTER_ARTIFACT_KEY_PREFIX}{normalized_chapter_title}"

    @staticmethod
    def _get_chapter_summary_artifact(
        *,
        document: StructuredDocument,
        chapter_key: str,
    ) -> SummaryArtifact | None:
        """Get document-level chapter summary artifact if present."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None
        chapter_artifact = document_artifacts.chapter_artifacts.get(chapter_key)
        if chapter_artifact is None:
            return None
        return chapter_artifact.summary

    def _is_chapter_summary_cache_valid(
        self,
        *,
        chapter_key: str,
        chapter_title: str,
        source_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> bool:
        """Check whether one document-level chapter summary artifact can be reused safely."""
        validity = self._validate_chapter_summary_artifact(
            chapter_key=chapter_key,
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
        chapter_key: str,
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
            chapter_key=chapter_key,
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
        if (metadata.get("chapter_title") or "") != chapter_title:
            return ArtifactValidityResult.invalid("chapter_title_mismatch")
        if (metadata.get("chapter_key") or "") != chapter_key:
            return ArtifactValidityResult.invalid("chapter_key_mismatch")
        if metadata_source_section_id and metadata_source_section_id != resolved_source_section.section_id:
            return ArtifactValidityResult.invalid("section_id_mismatch")
        return ArtifactValidityResult.valid()

    @staticmethod
    def _get_chapter_quiz_artifact(
        *,
        document: StructuredDocument,
        chapter_key: str,
    ) -> QuizArtifact | None:
        """Get document-level chapter quiz artifact if present."""
        document_artifacts = document.document_task_artifacts
        if document_artifacts is None:
            return None
        chapter_artifact = document_artifacts.chapter_artifacts.get(chapter_key)
        if chapter_artifact is None:
            return None
        return chapter_artifact.quiz

    def _get_valid_cached_chapter_quiz_questions(
        self,
        *,
        chapter_key: str,
        chapter_title: str,
        source_section: StructuredSection,
        document: StructuredDocument,
        document_profile: DocumentProfile | None,
        resolve_options: TaskUnitResolveOptions,
        prompt_version: str,
    ) -> list[QuizQuestion] | None:
        """Return parsed cached chapter-quiz questions when chapter-level artifact is valid."""
        validity, parsed = self._validate_chapter_quiz_artifact(
            chapter_key=chapter_key,
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
        chapter_key: str,
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
            chapter_key=chapter_key,
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
        if (metadata.get("chapter_title") or "") != chapter_title:
            return ArtifactValidityResult.invalid("chapter_title_mismatch"), None
        if (metadata.get("chapter_key") or "") != chapter_key:
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

    def _resolve_task_unit_for_section_id_from_persisted_layout(
        self,
        *,
        document: StructuredDocument,
        section_id: str,
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
        if selected_index is None:
            if document.sections:
                # Backward-compatible extension path for old sections-only payloads.
                for section in document.sections:
                    for task_unit in section.task_units:
                        if task_unit.unit_id in task_unit_index_by_id:
                            continue
                        task_unit_index_by_id[task_unit.unit_id] = len(ordered_task_units)
                        ordered_task_units.append(task_unit)
                selected_index = task_unit_index_by_id.get(selected_task_unit.unit_id)
            if selected_index is None:
                raise ValueError(
                    f"section_id '{normalized_section_id}' task units are not aligned with effective "
                    f"task-layout ordering for document '{document.document_id}'"
                )

        return ResolvedTaskUnit(
            task_unit=selected_task_unit,
            task_unit_index=selected_index,
        )

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
