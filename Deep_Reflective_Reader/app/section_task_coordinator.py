from dataclasses import dataclass

from config.faiss_storage_config import FaissStorageConfig
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_preparation.preparation_mode import PreparationMode
from document_structure.structured_document import StructuredDocument
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from section_tasks.chapter_quiz_service import ChapterQuizService
from section_tasks.chapter_summary_service import ChapterSummaryService
from section_tasks.document_task_layout import (
    DocumentTaskLayout,
    DocumentTaskLayoutSectionDTO,
    SectionTaskMode,
    TaskUnitDTO,
)
from section_tasks.quiz_question import QuizQuestion
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_resolver import TaskUnitResolver


@dataclass(frozen=True)
class ResolvedTaskUnit:
    """Resolved task unit plus stable unit index inside current document."""

    task_unit: TaskUnit
    task_unit_index: int


class SectionTaskCoordinator:
    """Coordinator for section/chapter task orchestration and layout projection."""

    def __init__(
        self,
        document_preparation_pipeline: DocumentPreparationPipeline,
        document_profile_store: DocumentProfileStore,
        chapter_summary_service: ChapterSummaryService,
        chapter_quiz_service: ChapterQuizService,
        task_unit_resolver: TaskUnitResolver,
    ):
        self.document_preparation_pipeline = document_preparation_pipeline
        self.document_profile_store = document_profile_store
        self.chapter_summary_service = chapter_summary_service
        self.chapter_quiz_service = chapter_quiz_service
        self.task_unit_resolver = task_unit_resolver

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
        self,
        doc_name: str,
        section_id: str,
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
