#!/usr/bin/env python3
"""Hierarchy-aware section/task-unit artifact write synchronization smoke tests."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.preparation_mode import PreparationMode
from document_structure.enhanced_parse_trigger_evaluator import (
    EnhancedParseTriggerDecision,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import QuizArtifact, SummaryArtifact, TaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_document(
    *,
    include_front_matter: bool = False,
    duplicate_hierarchy_section: bool = False,
) -> StructuredDocument:
    chapter_section = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="第一章",
        level=1,
        content="chapter body content",
        char_start=10,
        char_end=30,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-0",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-1",
                title="unit-1",
                container_title=None,
                content="chapter body content",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id="section-1",
            )
        ],
    )
    duplicate_section = chapter_section if duplicate_hierarchy_section else None
    chapters = [
        StructuredChapter(
            chapter_id="chapter-0",
            title="第一章",
            level=1,
            chapter_role="main_body",
            sections=[chapter_section] if duplicate_section is None else [chapter_section, duplicate_section],
        )
    ]
    legacy_sections: list[StructuredSection] = []
    if include_front_matter:
        legacy_sections.append(
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="前言",
                level=1,
                content="front matter content",
                char_start=0,
                char_end=9,
                section_role=SectionRole.FRONT_MATTER,
                task_artifacts=TaskArtifacts(
                    summary=SummaryArtifact(content="front-summary"),
                ),
            )
        )
    legacy_sections.append(
        StructuredSection(
            section_id="section-1",
            section_index=1,
            title="第一章",
            level=1,
            content="chapter body content",
            char_start=10,
            char_end=30,
            section_role=SectionRole.MAIN_BODY,
            task_units=[
                TaskUnit(
                    unit_id="task-unit-1",
                    title="unit-1",
                    container_title=None,
                    content="chapter body content",
                    source_section_ids=["section-1"],
                    is_fallback_generated=False,
                    parent_section_id="section-1",
                )
            ],
        )
    )
    return StructuredDocument(
        document_id="hier-artifact-doc",
        title="Hierarchy Artifact Doc",
        source_path=None,
        language="zh",
        raw_text="front matter content\nchapter body content",
        sections=legacy_sections,
        chapters=chapters,
    )


@dataclass(frozen=True)
class _FakeAssets:
    errors: list[str]


@dataclass(frozen=True)
class _FakePreparationResult:
    assets: _FakeAssets
    structured_document: StructuredDocument | None
    bundle: object | None = None


class _FakePreparationPipeline:
    def __init__(self, repository: StructuredDocumentArtifactRepository):
        self.repository = repository

    def prepare_and_load(
        self,
        doc_name: str,
        force_rebuild: bool = False,
        mode: PreparationMode | str = PreparationMode.BASE,
        structured_parser_mode: str = "common",
    ) -> _FakePreparationResult:
        _ = (force_rebuild, mode, structured_parser_mode)
        try:
            return _FakePreparationResult(
                assets=_FakeAssets(errors=[]),
                structured_document=self.repository.load_document(doc_name),
            )
        except Exception as error:
            return _FakePreparationResult(
                assets=_FakeAssets(errors=[str(error)]),
                structured_document=None,
            )


class _FakeTaskUnitResolver:
    split_mode = TaskUnitSplitMode.SEMANTIC_SAFE

    def resolve_with_options(
        self,
        *,
        document: StructuredDocument,
        split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> list[TaskUnit]:
        _ = (split_mode, semantic_top_k_candidates)
        units: list[TaskUnit] = []
        for section in document.sections:
            if section.content.strip():
                units.append(
                    TaskUnit(
                        unit_id=f"generated-{section.section_id}",
                        title=section.title,
                        container_title=section.container_title,
                        content=section.content,
                        source_section_ids=[section.section_id],
                        is_fallback_generated=False,
                        parent_section_id=section.section_id,
                    )
                )
        return units


class _FakeProfileStore:
    @staticmethod
    def exists(config) -> bool:
        _ = config
        return False

    @staticmethod
    def load(config):
        _ = config
        raise RuntimeError("profile not available")


class _FakeSectionTaskService:
    pass


class _FakeEnhancedParseEvaluator:
    @staticmethod
    def evaluate(
        *,
        structured_document: StructuredDocument,
        affected_section_ratio: float,
        fallback_task_unit_ratio: float,
        total_task_units: int,
    ) -> EnhancedParseTriggerDecision:
        _ = (
            structured_document,
            affected_section_ratio,
            fallback_task_unit_ratio,
            total_task_units,
        )
        return EnhancedParseTriggerDecision(
            should_recommend=False,
            score=0,
            reasons=[],
            metrics={},
        )


def _write_document(temp_dir: str, document: StructuredDocument) -> StructuredDocumentArtifactRepository:
    path = Path(temp_dir) / f"{document.document_id}.structured.json"
    path.write_text(document.to_json(), encoding="utf-8")
    return StructuredDocumentArtifactRepository(base_dir=temp_dir)


def test_section_summary_writes_hierarchy_and_legacy() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(temp_dir, _build_document())
        repository.update_section_summary_artifact(
            doc_name="hier-artifact-doc",
            section_id="section-1",
            summary=SummaryArtifact(content="section-summary"),
        )
        reloaded = repository.load_document("hier-artifact-doc")
        hierarchy_section = reloaded.chapters[0].sections[0]
        legacy_section = next(section for section in reloaded.sections if section.section_id == "section-1")
        _assert(
            hierarchy_section.task_artifacts is not None
            and hierarchy_section.task_artifacts.summary is not None
            and hierarchy_section.task_artifacts.summary.content == "section-summary",
            "hierarchy section summary should be updated",
        )
        _assert(
            legacy_section.task_artifacts is not None
            and legacy_section.task_artifacts.summary is not None
            and legacy_section.task_artifacts.summary.content == "section-summary",
            "legacy section summary should be updated",
        )


def test_section_quiz_writes_hierarchy_and_legacy_preserving_summary() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(temp_dir, _build_document())
        repository.update_section_summary_artifact(
            doc_name="hier-artifact-doc",
            section_id="section-1",
            summary=SummaryArtifact(content="section-summary"),
        )
        repository.update_section_quiz_artifact(
            doc_name="hier-artifact-doc",
            section_id="section-1",
            quiz=QuizArtifact(items=[{"question_id": "q1", "question_text": "Q", "answer_text": "A"}]),
        )
        reloaded = repository.load_document("hier-artifact-doc")
        section = reloaded.chapters[0].sections[0]
        _assert(section.task_artifacts is not None, "artifacts should exist")
        _assert(section.task_artifacts.summary is not None, "summary should be preserved")
        _assert(section.task_artifacts.quiz is not None, "quiz should exist")


def test_update_section_artifacts_hierarchy_and_legacy_and_front_matter_preserved() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(temp_dir, _build_document(include_front_matter=True))
        repository.update_section_artifacts(
            doc_name="hier-artifact-doc",
            section_id="section-1",
            artifacts=TaskArtifacts(summary=SummaryArtifact(content="replaced-summary")),
        )
        reloaded = repository.load_document("hier-artifact-doc")
        _assert(
            any(
                section.section_id == "section-0"
                and section.section_role == SectionRole.FRONT_MATTER
                for section in reloaded.sections
            ),
            "front matter section should be preserved in legacy sections",
        )
        _assert(
            reloaded.chapters[0].sections[0].task_artifacts is not None,
            "hierarchy section artifacts should be updated",
        )


def test_task_unit_artifacts_writes_hierarchy_and_legacy() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(temp_dir, _build_document())
        repository.update_task_unit_artifacts(
            doc_name="hier-artifact-doc",
            task_unit_id="task-unit-1",
            artifacts=TaskArtifacts(summary=SummaryArtifact(content="unit-summary")),
        )
        reloaded = repository.load_document("hier-artifact-doc")
        hierarchy_unit = reloaded.chapters[0].sections[0].task_units[0]
        legacy_section = next(section for section in reloaded.sections if section.section_id == "section-1")
        legacy_unit = legacy_section.task_units[0]
        _assert(
            hierarchy_unit.task_artifacts is not None
            and hierarchy_unit.task_artifacts.summary is not None
            and hierarchy_unit.task_artifacts.summary.content == "unit-summary",
            "hierarchy task-unit artifacts should be updated",
        )
        _assert(
            legacy_unit.task_artifacts is not None
            and legacy_unit.task_artifacts.summary is not None
            and legacy_unit.task_artifacts.summary.content == "unit-summary",
            "legacy task-unit artifacts should be updated",
        )
        _assert(hierarchy_unit.parent_section_id == "section-1", "parent_section_id should be preserved")
        _assert(legacy_unit.source_section_ids == ["section-1"], "source_section_ids should be preserved")


def test_duplicate_task_unit_id_rejected_hierarchy_first() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        document = _build_document()
        chapter_section = document.chapters[0].sections[0]
        duplicated = StructuredSection(
            section_id="section-2",
            section_index=2,
            title="第二章",
            level=1,
            content="chapter 2",
            char_start=31,
            char_end=40,
            section_role=SectionRole.MAIN_BODY,
            parent_chapter_id="chapter-0",
            section_kind="chapter_body",
            is_implicit_section=True,
            task_units=[
                TaskUnit(
                    unit_id=chapter_section.task_units[0].unit_id,
                    title="dup-unit",
                    container_title=None,
                    content="dup",
                    source_section_ids=["section-2"],
                    is_fallback_generated=False,
                    parent_section_id="section-2",
                )
            ],
        )
        document = StructuredDocument(
            document_id=document.document_id,
            title=document.title,
            source_path=document.source_path,
            language=document.language,
            raw_text=document.raw_text,
            sections=document.sections + [duplicated],
            chapters=[
                StructuredChapter(
                    chapter_id=document.chapters[0].chapter_id,
                    title=document.chapters[0].title,
                    level=document.chapters[0].level,
                    chapter_role=document.chapters[0].chapter_role,
                    sections=[chapter_section, duplicated],
                )
            ],
            document_task_artifacts=document.document_task_artifacts,
        )
        repository = _write_document(temp_dir, document)
        try:
            repository.update_task_unit_artifacts(
                doc_name="hier-artifact-doc",
                task_unit_id="task-unit-1",
                artifacts=TaskArtifacts(summary=SummaryArtifact(content="x")),
            )
        except ValueError as error:
            message = str(error)
            _assert("duplicate task_unit_id" in message, "error should mention duplicate id")
            _assert("match_count=2" in message, "error should include match_count")
            return
        raise AssertionError("duplicate task_unit_id should be rejected")


def test_legacy_fallback_without_chapters() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        document = StructuredDocument(
            document_id="legacy-only-doc",
            title="Legacy",
            source_path=None,
            language="en",
            raw_text="legacy content",
            sections=[
                StructuredSection(
                    section_id="section-0",
                    section_index=0,
                    title="Section",
                    level=1,
                    content="legacy content",
                    char_start=0,
                    char_end=14,
                    section_role=SectionRole.MAIN_BODY,
                    task_units=[
                        TaskUnit(
                            unit_id="legacy-unit-0",
                            title="legacy unit",
                            container_title=None,
                            content="legacy content",
                            source_section_ids=["section-0"],
                            is_fallback_generated=False,
                            parent_section_id="section-0",
                        )
                    ],
                )
            ],
        )
        repository = _write_document(temp_dir, document)
        repository.update_section_summary_artifact(
            doc_name="legacy-only-doc",
            section_id="section-0",
            summary=SummaryArtifact(content="legacy-summary"),
        )
        repository.update_task_unit_artifacts(
            doc_name="legacy-only-doc",
            task_unit_id="legacy-unit-0",
            artifacts=TaskArtifacts(summary=SummaryArtifact(content="legacy-unit-summary")),
        )
        reloaded = repository.load_document("legacy-only-doc")
        _assert(
            reloaded.sections[0].task_artifacts is not None
            and reloaded.sections[0].task_artifacts.summary is not None,
            "legacy section summary should be updated without chapters",
        )
        _assert(
            reloaded.sections[0].task_units[0].task_artifacts is not None,
            "legacy task-unit artifact should be updated without chapters",
        )


def test_hierarchy_severe_inconsistency_blocks_save() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(
            temp_dir,
            _build_document(duplicate_hierarchy_section=True),
        )
        try:
            repository.update_section_summary_artifact(
                doc_name="hier-artifact-doc",
                section_id="section-1",
                summary=SummaryArtifact(content="x"),
            )
        except ValueError:
            return
        raise AssertionError("severe hierarchy inconsistency should block save")


def test_task_layout_availability_sees_updated_section_artifact() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repository = _write_document(temp_dir, _build_document())
        repository.update_section_summary_artifact(
            doc_name="hier-artifact-doc",
            section_id="section-1",
            summary=SummaryArtifact(content="layout-visible-summary"),
        )
        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repository),
            document_artifact_repository=repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_FakeSectionTaskService(),
            chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=_FakeTaskUnitResolver(),
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
        )
        layout = coordinator.get_document_task_layout(doc_name="hier-artifact-doc")
        _assert(layout.chapters, "layout should expose hierarchy chapters")
        section_layout = next(
            section for section in layout.sections if section.section_id == "section-1"
        )
        _assert(section_layout.artifacts is not None, "section artifacts availability should exist")
        _assert(section_layout.artifacts.has_summary, "layout should observe updated section summary")


def main() -> None:
    test_section_summary_writes_hierarchy_and_legacy()
    test_section_quiz_writes_hierarchy_and_legacy_preserving_summary()
    test_update_section_artifacts_hierarchy_and_legacy_and_front_matter_preserved()
    test_task_unit_artifacts_writes_hierarchy_and_legacy()
    test_duplicate_task_unit_id_rejected_hierarchy_first()
    test_legacy_fallback_without_chapters()
    test_hierarchy_severe_inconsistency_blocks_save()
    test_task_layout_availability_sees_updated_section_artifact()

    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_summary_hierarchy_legacy_sync",
                    "section_quiz_hierarchy_legacy_sync",
                    "section_artifacts_replace_sync_and_front_matter_preserved",
                    "task_unit_artifacts_hierarchy_legacy_sync",
                    "duplicate_task_unit_id_rejected",
                    "legacy_fallback_without_chapters",
                    "severe_hierarchy_inconsistency_blocks_save",
                    "task_layout_availability_sees_updated_artifact",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
