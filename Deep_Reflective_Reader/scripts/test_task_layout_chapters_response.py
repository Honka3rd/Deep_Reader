#!/usr/bin/env python3
"""Hierarchy-first task-layout response smoke tests."""

from __future__ import annotations

import hashlib
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
from document_structure.structured_document_store import StructuredDocumentStore
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import (
    DocumentTaskArtifacts,
    QuizArtifact,
    SummaryArtifact,
    TaskArtifacts,
)
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
    def __init__(self) -> None:
        self.split_mode = TaskUnitSplitMode.SEMANTIC_SAFE
        self.calls = 0

    def resolve_with_options(
        self,
        *,
        document: StructuredDocument,
        split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> list[TaskUnit]:
        _ = (split_mode, semantic_top_k_candidates)
        self.calls += 1
        units: list[TaskUnit] = []
        for section_index, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            units.append(
                TaskUnit(
                    unit_id=f"resolver-{section_index}-{self.calls}",
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
        _ = structured_document
        return EnhancedParseTriggerDecision(
            should_recommend=(affected_section_ratio > 0.5),
            score=1 if total_task_units > 0 else 0,
            reasons=["smoke_reason"] if total_task_units > 0 else [],
            metrics={
                "affected_section_ratio": affected_section_ratio,
                "fallback_task_unit_ratio": fallback_task_unit_ratio,
                "total_task_units": total_task_units,
            },
        )


def _section_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_layout_ready_document(*, with_chapters: bool) -> StructuredDocument:
    raw_text = "第一章\n章一內容。\n\n第二章\n章二內容。"
    section_1 = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="第一章",
        level=1,
        content="第一章\n章一內容。",
        char_start=0,
        char_end=8,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-1",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-1",
                title="第一章 unit",
                container_title=None,
                content="第一章\n章一內容。",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id="section-1",
                task_artifacts=TaskArtifacts(
                    summary=SummaryArtifact(
                        content="tu-summary-1",
                        language="zh",
                        source_hash=_section_hash("第一章\n章一內容。"),
                        prompt_version=SectionTaskCoordinator._TASK_UNIT_SUMMARY_PROMPT_VERSION,
                        task_unit_split_mode="semantic_safe",
                        semantic_top_k_candidates=3,
                    )
                ),
            )
        ],
        task_artifacts=TaskArtifacts(
            summary=SummaryArtifact(
                content="section-summary-1",
                language="zh",
                source_hash=_section_hash("第一章\n章一內容。"),
                prompt_version=SectionTaskCoordinator._SECTION_SUMMARY_PROMPT_VERSION,
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={"summary_scope": "section", "section_id": "section-1"},
            ),
            quiz=QuizArtifact(
                items=[
                    {"question_id": "q1", "question_text": "Q1", "answer_text": "A1"}
                ],
                language="zh",
                source_hash=_section_hash("第一章\n章一內容。"),
                prompt_version=SectionTaskCoordinator._SECTION_QUIZ_PROMPT_VERSION,
                quiz_schema_version=SectionTaskCoordinator._QUIZ_SCHEMA_VERSION,
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={"quiz_scope": "section", "section_id": "section-1"},
            ),
        ),
    )
    section_2 = StructuredSection(
        section_id="section-2",
        section_index=2,
        title="第二章",
        level=1,
        content="第二章\n章二內容。",
        char_start=9,
        char_end=17,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-2",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-2",
                title="第二章 unit",
                container_title=None,
                content="第二章\n章二內容。",
                source_section_ids=["section-2"],
                is_fallback_generated=False,
                parent_section_id="section-2",
            )
        ],
    )
    sections = [section_1, section_2]

    chapters = []
    if with_chapters:
        chapters = [
            StructuredChapter(
                chapter_id="chapter-1",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[section_1],
                metadata={
                    "legacy_chapter_key": "chapter::第一章",
                },
            ),
            StructuredChapter(
                chapter_id="chapter-2",
                title="第二章",
                level=1,
                chapter_role="main_body",
                sections=[section_2],
            ),
        ]

    chapter_artifacts = {
        "chapter::第一章": TaskArtifacts(
            summary=SummaryArtifact(
                content="chapter-summary-1",
                language="zh",
                source_hash=_section_hash("第一章\n章一內容。"),
                prompt_version=SectionTaskCoordinator._CHAPTER_SUMMARY_PROMPT_VERSION,
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "summary_scope": "chapter",
                    "chapter_title": "第一章",
                    "chapter_key": "chapter::第一章",
                    "source_section_id": "section-1",
                },
            ),
            quiz=QuizArtifact(
                items=[
                    {"question_id": "cq1", "question_text": "CQ1", "answer_text": "CA1"}
                ],
                language="zh",
                source_hash=_section_hash("第一章\n章一內容。"),
                prompt_version=SectionTaskCoordinator._CHAPTER_QUIZ_PROMPT_VERSION,
                quiz_schema_version=SectionTaskCoordinator._QUIZ_SCHEMA_VERSION,
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "quiz_scope": "chapter",
                    "chapter_title": "第一章",
                    "chapter_key": "chapter::第一章",
                    "source_section_id": "section-1",
                },
            ),
        )
    }
    metadata = {
        "task_layout": {
            "source_hash": _section_hash(raw_text),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id="layout-doc",
        title="Layout Doc",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=sections,
        chapters=chapters,
        document_task_artifacts=DocumentTaskArtifacts(
            chapter_artifacts=chapter_artifacts,
            metadata=metadata,
        ),
    )


def _build_coordinator(repo: StructuredDocumentArtifactRepository) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(repo),
        document_artifact_repository=repo,
        document_profile_store=_FakeProfileStore(),
        chapter_summary_service=_FakeSectionTaskService(),
        chapter_quiz_service=_FakeSectionTaskService(),
        task_unit_resolver=_FakeTaskUnitResolver(),
        enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
        semantic_top_k_candidates_max=20,
    )


def test_hierarchy_response_shape() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_layout_ready_document(with_chapters=True)
        (Path(temp_dir) / "layout-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator = _build_coordinator(repo)
        layout = coordinator.get_document_task_layout(
            doc_name="layout-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(len(layout.chapters) == 2, "chapters tree should be primary response shape")
        _assert(
            layout.chapters[0].sections[0].task_units[0].unit_id == "task-unit-1",
            "chapter -> section -> task_unit nesting should preserve ordering",
        )


def test_chapter_only_novel_shape() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_layout_ready_document(with_chapters=True)
        (Path(temp_dir) / "layout-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator = _build_coordinator(repo)
        layout = coordinator.get_document_task_layout(doc_name="layout-doc")
        for chapter in layout.chapters:
            _assert(len(chapter.sections) == 1, "chapter-only novel should be chapter -> one section")
            _assert(
                chapter.sections[0].is_implicit_section is True,
                "implicit chapter section should be marked",
            )
            _assert(
                chapter.sections[0].section_kind == "chapter_body",
                "implicit chapter section kind should be chapter_body",
            )


def test_missing_chapters_hierarchy_required() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_layout_ready_document(with_chapters=False)
        (Path(temp_dir) / "layout-doc.structured.json").write_text(
            doc.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )
        coordinator = _build_coordinator(repo)
        try:
            coordinator.get_document_task_layout(doc_name="layout-doc")
            raise AssertionError(
                "expected failure when task-layout runtime receives legacy sections-only document"
            )
        except ValueError as error:
            _assert(
                "requires explicit migration" in str(error),
                "missing-chapters document should fail-fast with migration guidance",
            )


def test_artifact_availability_and_no_heavy_payload_and_recommendation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_layout_ready_document(with_chapters=True)
        (Path(temp_dir) / "layout-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator = _build_coordinator(repo)
        layout = coordinator.get_document_task_layout(
            doc_name="layout-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )

        chapter_av = layout.chapters[0].artifacts
        section_av = layout.chapters[0].sections[0].artifacts
        task_unit_av = layout.chapters[0].sections[0].task_units[0].artifacts
        _assert(chapter_av is not None and chapter_av.has_summary and chapter_av.has_quiz, "chapter artifacts should be visible")
        _assert(section_av is not None and section_av.has_summary and section_av.has_quiz, "section artifacts should be visible")
        _assert(task_unit_av is not None and task_unit_av.has_summary, "task-unit artifacts should be visible")

        payload = layout.to_dict()
        payload_str = json.dumps(payload, ensure_ascii=False)
        _assert("summary.content" not in payload_str, "layout should not include heavy summary content field name")
        _assert("question_text" not in payload_str, "layout should not include quiz question payload")
        _assert("answer_text" not in payload_str, "layout should not include quiz answer payload")
        _assert("raw_text" not in payload_str, "layout should not include document raw_text")
        _assert("content" not in payload["chapters"][0]["sections"][0], "section content should not be exposed in layout DTO")
        _assert(
            layout.enhanced_parse_recommendation is not None
            and layout.enhanced_parse_recommendation.metrics.get("total_task_units") == 2,
            "enhanced recommendation should still be produced with hierarchy-based metrics",
        )


def main() -> None:
    test_hierarchy_response_shape()
    test_chapter_only_novel_shape()
    test_missing_chapters_hierarchy_required()
    test_artifact_availability_and_no_heavy_payload_and_recommendation()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "hierarchy_response_shape",
                    "chapter_only_novel_shape",
                    "missing_chapters_hierarchy_required",
                    "artifact_availability_no_heavy_payload_recommendation",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
