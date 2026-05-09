#!/usr/bin/env python3
"""Profile-structure diagnostics tests for task-layout response path."""

from __future__ import annotations

import json
from dataclasses import replace

from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.prepared_document_result import PreparedDocumentResult
from document_structure.enhanced_parse_trigger_evaluator import (
    EnhancedParseTriggerEvaluator,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from language.language_code import LanguageCode
from profile.document_profile import (
    DocumentProfile,
    DocumentStructureShape,
    LikelihoodLevel,
    ParserRelevantMetadata,
    PostStructureMetadata,
)
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import DocumentTaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _NoopSummaryService:
    pass


class _NoopQuizService:
    pass


class _FakeTaskUnitResolver:
    def __init__(self) -> None:
        self.split_mode = TaskUnitSplitMode.SEMANTIC_SAFE

    def resolve_with_options(self, **kwargs):  # pragma: no cover - not used in this test
        _ = kwargs
        return []


class _MissingProfileStore:
    @staticmethod
    def exists(config) -> bool:
        _ = config
        return False

    @staticmethod
    def load(config):  # pragma: no cover - should not be called
        _ = config
        raise RuntimeError("profile should be missing")


class _FakeRepository:
    @staticmethod
    def update_task_layout(doc_name, sections, task_layout_metadata):
        _ = (doc_name, sections, task_layout_metadata)
        raise RuntimeError("update_task_layout should not be called for cache hit")


class _FakePipeline:
    def __init__(self, document: StructuredDocument) -> None:
        self.document = document

    def prepare_and_load(self, doc_name: str, mode):  # noqa: ANN001
        _ = (doc_name, mode)
        assets = PreparedDocumentAssets(
            doc_name=doc_name,
            raw_text=self.document.raw_text,
            language=self.document.language,
            structured_document_ready=True,
            faiss_ready=False,
            profile_ready=False,
            bundle_ready=False,
            structured_document_path=None,
            faiss_namespace=None,
            errors=[],
        )
        return PreparedDocumentResult(
            assets=assets,
            structured_document=self.document,
            bundle=None,
        )


def _build_section(
    *,
    section_id: str,
    parent_chapter_id: str,
    title: str,
    with_task_units: bool,
) -> StructuredSection:
    return StructuredSection(
        section_id=section_id,
        section_index=0,
        title=title,
        level=1,
        content="Body content",
        char_start=0,
        char_end=12,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id=parent_chapter_id,
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=(
            []
            if not with_task_units
            else [
                TaskUnit(
                    unit_id=f"task-unit-{section_id}",
                    title=f"Unit {section_id}",
                    container_title=None,
                    content="Unit content",
                    source_section_ids=[section_id],
                    is_fallback_generated=False,
                    parent_section_id=section_id,
                )
            ]
        ),
    )


def _build_cached_document(
    *,
    sections_with_units: list[bool] | None = None,
) -> StructuredDocument:
    unit_flags = sections_with_units or [True]
    sections: list[StructuredSection] = []
    for index, has_units in enumerate(unit_flags, start=1):
        section_id = f"section-{index}"
        chapter_id = f"chapter-{index}"
        sections.append(
            _build_section(
                section_id=section_id,
                parent_chapter_id=chapter_id,
                title=f"Chapter {index}",
                with_task_units=has_units,
            )
        )

    chapters = [
        StructuredChapter(
            chapter_id=f"chapter-{index}",
            title=f"Chapter {index}",
            level=1,
            chapter_role="main_body",
            sections=[section],
        )
        for index, section in enumerate(sections, start=1)
    ]

    document = StructuredDocument(
        document_id="doc",
        title="Doc",
        source_path=None,
        language="en",
        raw_text="Body content",
        chapters=chapters,
        sections=[],
        structure_nodes=[],
    )
    source_hash = SectionTaskCoordinator._compute_source_hash(document)
    metadata = {
        "task_layout": {
            "source_hash": source_hash,
            "task_unit_split_mode": TaskUnitSplitMode.SEMANTIC_SAFE.value,
            "semantic_top_k_candidates": None,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return replace(
        document,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_coordinator(document: StructuredDocument) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=_FakePipeline(document),
        document_artifact_repository=_FakeRepository(),
        document_profile_store=_MissingProfileStore(),
        chapter_summary_service=_NoopSummaryService(),
        chapter_quiz_service=_NoopQuizService(),
        task_unit_resolver=_FakeTaskUnitResolver(),
        enhanced_parse_trigger_evaluator=EnhancedParseTriggerEvaluator(),
    )


def _profile(
    *,
    parser_shape: DocumentStructureShape | None,
    post_shape: DocumentStructureShape | None,
    title_risk: LikelihoodLevel | None,
    task_stats_available: bool,
    coverage: float | None,
) -> DocumentProfile:
    return DocumentProfile(
        topic="topic",
        summary="summary",
        document_language=LanguageCode.EN,
        parser_metadata=ParserRelevantMetadata(
            document_structure_shape=parser_shape,
        ),
        post_structure_metadata=PostStructureMetadata(
            actual_structure_shape=post_shape,
            title_uniqueness_risk=title_risk,
            task_unit_stats_available=task_stats_available,
            task_unit_section_coverage=coverage,
        ),
    )


def test_title_target_requires_id_and_shape_mismatch() -> None:
    coordinator = _build_coordinator(_build_cached_document())
    sections = [section for chapter in coordinator.document_preparation_pipeline.document.chapters for section in chapter.sections]
    diagnostics = coordinator._build_profile_structure_diagnostics(
        document_profile=_profile(
            parser_shape=DocumentStructureShape.CHAPTER_SECTION,
            post_shape=DocumentStructureShape.ESSAY_SECTIONS,
            title_risk=LikelihoodLevel.HIGH,
            task_stats_available=False,
            coverage=0.5,
        ),
        sections=sections,
    )
    _assert(diagnostics is not None, "diagnostics should exist")
    _assert(diagnostics.title_target_requires_id, "high title uniqueness risk should require id target")
    _assert(
        "chapter_or_section_titles_are_not_unique_use_id_targets" in diagnostics.warnings,
        "id-target warning should be present",
    )
    _assert(diagnostics.parser_post_shape_mismatch, "shape mismatch should be detected")
    _assert(
        "pre_structure_shape_differs_from_post_structure_shape" in diagnostics.warnings,
        "shape mismatch warning should be present",
    )
    _assert(
        "task_unit_stats_not_available_or_incomplete" not in diagnostics.warnings,
        "task-unit warning should follow current layout state (full coverage here)",
    )


def test_current_layout_overrides_stale_post_metadata() -> None:
    document = _build_cached_document(sections_with_units=[True, True, True])
    coordinator = _build_coordinator(document)
    sections = [section for chapter in document.chapters for section in chapter.sections]
    diagnostics = coordinator._build_profile_structure_diagnostics(
        document_profile=_profile(
            parser_shape=DocumentStructureShape.PART_CHAPTER,
            post_shape=DocumentStructureShape.PART_CHAPTER,
            title_risk=LikelihoodLevel.HIGH,
            task_stats_available=False,
            coverage=0.0,
        ),
        sections=sections,
    )
    _assert(diagnostics is not None, "diagnostics should exist")
    _assert(diagnostics.task_unit_stats_available is True, "current layout should override stale post metadata")
    _assert(diagnostics.task_unit_section_coverage == 1.0, "full current coverage should be 1.0")
    _assert(
        "task_unit_stats_not_available_or_incomplete" not in diagnostics.warnings,
        "incomplete warning should be absent for full current coverage",
    )


def test_current_layout_no_task_units() -> None:
    document = _build_cached_document(sections_with_units=[False, False, False])
    coordinator = _build_coordinator(document)
    sections = [section for chapter in document.chapters for section in chapter.sections]
    diagnostics = coordinator._build_profile_structure_diagnostics(
        document_profile=_profile(
            parser_shape=DocumentStructureShape.CHAPTER_ONLY,
            post_shape=DocumentStructureShape.CHAPTER_ONLY,
            title_risk=LikelihoodLevel.LOW,
            task_stats_available=True,
            coverage=1.0,
        ),
        sections=sections,
    )
    _assert(diagnostics is not None, "diagnostics should exist")
    _assert(diagnostics.task_unit_stats_available is False, "no task units should be unavailable")
    _assert(diagnostics.task_unit_section_coverage == 0.0, "coverage should be 0.0")
    _assert(
        "task_unit_stats_not_available_or_incomplete" in diagnostics.warnings,
        "incomplete warning should be present",
    )
    _assert(
        "task_unit_stats_partially_available" not in diagnostics.warnings,
        "partial warning should be absent for 0.0 coverage",
    )


def test_current_layout_partial_task_units() -> None:
    document = _build_cached_document(sections_with_units=[True, True, False, False])
    coordinator = _build_coordinator(document)
    sections = [section for chapter in document.chapters for section in chapter.sections]
    diagnostics = coordinator._build_profile_structure_diagnostics(
        document_profile=_profile(
            parser_shape=DocumentStructureShape.CHAPTER_SECTION,
            post_shape=DocumentStructureShape.CHAPTER_SECTION,
            title_risk=LikelihoodLevel.NONE,
            task_stats_available=True,
            coverage=1.0,
        ),
        sections=sections,
    )
    _assert(diagnostics is not None, "diagnostics should exist")
    _assert(diagnostics.task_unit_stats_available is False, "partial coverage should be unavailable")
    _assert(diagnostics.task_unit_section_coverage == 0.5, "partial coverage should be 0.5")
    _assert(
        "task_unit_stats_not_available_or_incomplete" in diagnostics.warnings,
        "incomplete warning should be present",
    )
    _assert(
        "task_unit_stats_partially_available" in diagnostics.warnings,
        "partial warning should be present",
    )


def test_missing_profile_non_blocking_task_layout() -> None:
    coordinator = _build_coordinator(_build_cached_document())
    layout = coordinator.get_document_task_layout(doc_name="doc")
    _assert(layout.profile_diagnostics is None, "missing profile should not block and diagnostics should be None")
    payload = layout.to_dict()
    serialized = json.dumps(payload, ensure_ascii=False)
    _assert("\"raw_text\":" not in serialized, "layout payload should not leak raw_text")
    _assert("\"content\":" not in serialized, "layout payload should not expose section/task content")
    _assert("\"items\":" not in serialized, "layout payload should not expose quiz items")


def main() -> None:
    test_title_target_requires_id_and_shape_mismatch()
    test_current_layout_overrides_stale_post_metadata()
    test_current_layout_no_task_units()
    test_current_layout_partial_task_units()
    test_missing_profile_non_blocking_task_layout()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "diagnostics_title_target_requires_id",
                    "diagnostics_shape_mismatch",
                    "diagnostics_current_layout_overrides_stale_post_metadata",
                    "diagnostics_current_layout_no_task_units",
                    "diagnostics_current_layout_partial_task_units",
                    "task_layout_missing_profile_non_blocking",
                    "task_layout_no_heavy_payload",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
