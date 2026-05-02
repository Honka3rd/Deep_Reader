#!/usr/bin/env python3
"""Smoke test for task-layout artifact availability payload shape."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.preparation_mode import PreparationMode
from document_structure.enhanced_parse_trigger_evaluator import EnhancedParseTriggerDecision
from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.structured_document_store import StructuredDocumentStore
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
            document = self.repository.load_document(doc_name)
            return _FakePreparationResult(_FakeAssets(errors=[]), document)
        except Exception as error:
            return _FakePreparationResult(_FakeAssets(errors=[str(error)]), None)


class _NoopResolver:
    def resolve_with_options(self, *, document, split_mode=None, semantic_top_k_candidates=None):
        _ = (document, split_mode, semantic_top_k_candidates)
        raise AssertionError("resolver should not be called in this test")


class _FakeProfileStore:
    @staticmethod
    def exists(config) -> bool:
        _ = config
        return False

    @staticmethod
    def load(config):
        _ = config
        raise RuntimeError("profile not available")


class _FakeEnhancedParseEvaluator:
    @staticmethod
    def evaluate(*, structured_document, affected_section_ratio, fallback_task_unit_ratio, total_task_units):
        _ = (structured_document, affected_section_ratio, fallback_task_unit_ratio, total_task_units)
        return EnhancedParseTriggerDecision(should_recommend=False, score=0, reasons=[], metrics={})


class _NoopSummaryService:
    pass


class _NoopQuizService:
    pass


def _build_document() -> StructuredDocument:
    section_0 = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content="第一章\n內容 A",
        char_start=0,
        char_end=10,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="unit-0",
                title="第一章 (Part 1)",
                container_title=None,
                content="第一章\n內容 A",
                source_section_ids=["section-0"],
                is_fallback_generated=False,
                task_artifacts=TaskArtifacts(
                    summary=SummaryArtifact(content="unit-summary", generated_at="2026-01-01T00:00:00+00:00")
                ),
            )
        ],
        task_artifacts=TaskArtifacts(
            summary=SummaryArtifact(content="section-summary", generated_at="2026-01-02T00:00:00+00:00"),
            quiz=QuizArtifact(items=[{"question_id": "q1", "question_text": "Q", "answer_text": "A"}], generated_at="2026-01-03T00:00:00+00:00"),
        ),
    )
    raw_text = "第一章\n內容 A"
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return StructuredDocument(
        document_id="layout-doc",
        title="Layout Doc",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=[section_0],
        document_task_artifacts=DocumentTaskArtifacts(
            chapter_artifacts={
                "chapter::第一章": TaskArtifacts(
                    summary=SummaryArtifact(content="chapter-summary", source_hash=source_hash, generated_at="2026-01-04T00:00:00+00:00"),
                    quiz=QuizArtifact(items=[{"question_id": "cq1", "question_text": "CQ", "answer_text": "CA"}], source_hash=source_hash, generated_at="2026-01-05T00:00:00+00:00"),
                )
            },
            metadata={
                "task_layout": {
                    "source_hash": source_hash,
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 3,
                    "resolver_version": "task_unit_resolver_v2",
                }
            },
        ),
    )


def test_task_layout_artifact_availability() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(store=StructuredDocumentStore(), base_dir=temp_dir)
        doc = _build_document()
        Path(temp_dir, "layout-doc.structured.json").write_text(doc.to_json(), encoding="utf-8")

        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repo),
            document_artifact_repository=repo,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_NoopSummaryService(),
            chapter_quiz_service=_NoopQuizService(),
            task_unit_resolver=_NoopResolver(),
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
            semantic_top_k_candidates_max=20,
        )

        layout = coordinator.get_document_task_layout(
            doc_name="layout-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(layout.chapters, "layout should expose chapters tree")
        _assert(
            layout.chapters[0].sections[0].section_id == "section-0",
            "chapter section nesting should include target section",
        )

        _assert(layout.sections[0].artifacts is not None, "section artifacts availability should exist")
        _assert(layout.sections[0].artifacts.has_summary is True, "section has_summary should be true")
        _assert(layout.sections[0].artifacts.has_quiz is True, "section has_quiz should be true")

        _assert(layout.task_units[0].artifacts is not None, "task unit artifacts availability should exist")
        _assert(layout.task_units[0].artifacts.has_summary is True, "task unit has_summary should be true")
        _assert(layout.task_units[0].artifacts.has_quiz is False, "task unit has_quiz should be false")

        chapter_id = layout.chapters[0].chapter_id
        chapter_key = f"chapter_id::{chapter_id}"
        chapter_av = layout.chapter_artifacts.get(chapter_key)
        if chapter_av is None:
            chapter_av = layout.chapter_artifacts.get("chapter::第一章")
        _assert(chapter_av is not None, "chapter artifacts availability key should exist")
        _assert(chapter_av.has_summary is True, "chapter has_summary should be true")
        _assert(chapter_av.has_quiz is True, "chapter has_quiz should be true")

        payload = layout.to_dict()
        first_chapter_section = payload["chapters"][0]["sections"][0]
        first_section = payload["sections"][0]
        first_unit = payload["task_units"][0]
        _assert("content" not in first_chapter_section, "chapter section payload must not include section content")
        _assert("content" not in first_section, "section layout payload must not include summary content")
        _assert("content" not in first_unit, "task-unit layout payload must not include task-unit content")
        _assert("summary" not in first_section, "section layout payload must not include summary object")
        _assert("quiz" not in first_section, "section layout payload must not include quiz object")


if __name__ == "__main__":
    test_task_layout_artifact_availability()
    print(json.dumps({"status": "ok", "tests": ["task_layout_artifact_availability", "no_heavy_payload"]}, ensure_ascii=False, indent=2))
