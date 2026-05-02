#!/usr/bin/env python3
"""Smoke tests for chapter_id-based chapter summary/quiz target resolution."""

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
from section_tasks.quiz_question import QuizQuestion
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import DocumentTaskArtifacts, SummaryArtifact, TaskArtifacts
from shared.task_unit_model import TaskUnit


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        _ = (document, split_mode, semantic_top_k_candidates)
        self.calls += 1
        raise AssertionError("resolver should not be called in chapter_id target tests")


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


class _FakeSummaryService:
    def __init__(self) -> None:
        self.calls = 0
        self.last_task_unit_content: str | None = None

    def summarize_task_unit(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_unit_index: int = 0,
    ) -> SectionTaskResult[str]:
        _ = (document_title, document_profile, task_unit_index)
        self.calls += 1
        self.last_task_unit_content = task_unit.content
        return SectionTaskResult.ok(f"summary-{self.calls}")


class _FakeQuizService:
    def __init__(self) -> None:
        self.calls = 0
        self.last_task_unit_content: str | None = None

    def generate_task_unit_quiz(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_type: str = "chapter_quiz",
        task_unit_index: int = 0,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        _ = (document_title, document_profile, task_type, task_unit_index)
        self.calls += 1
        self.last_task_unit_content = task_unit.content
        return SectionTaskResult.ok(
            [
                QuizQuestion(
                    question_id=f"q-{self.calls}",
                    question_text="question",
                    answer_text="answer",
                )
            ]
        )


def _make_chapter(
    *,
    chapter_id: str,
    title: str,
    section_index: int,
    content: str,
) -> StructuredChapter:
    section_id = f"section-{chapter_id}"
    section = StructuredSection(
        section_id=section_id,
        section_index=section_index,
        title=title,
        level=1,
        content=content,
        char_start=section_index * 100,
        char_end=section_index * 100 + len(content),
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id=chapter_id,
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id=f"task-unit-{section_index}",
                title=f"{title} unit",
                container_title=title,
                content=content,
                source_section_ids=[section_id],
                is_fallback_generated=False,
                parent_section_id=section_id,
            )
        ],
    )
    return StructuredChapter(
        chapter_id=chapter_id,
        title=title,
        level=1,
        chapter_role="main_body",
        sections=[section],
    )


def _build_ambiguous_doc() -> StructuredDocument:
    chapters = [
        _make_chapter(chapter_id="chapter-0", title="Chapter One", section_index=0, content="part1-ch1"),
        _make_chapter(chapter_id="chapter-10", title="Chapter One", section_index=1, content="part2-ch1"),
        _make_chapter(chapter_id="chapter-24", title="Chapter One", section_index=2, content="part3-ch1"),
        _make_chapter(chapter_id="chapter-2", title="Unique Chapter", section_index=3, content="unique-ch"),
    ]
    raw_text = "\n".join(ch.sections[0].content for ch in chapters)
    metadata = {
        "task_layout": {
            "source_hash": _sha256(raw_text),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id="chapter-id-doc",
        title="Chapter ID Doc",
        source_path=None,
        language="en",
        raw_text=raw_text,
        sections=[],
        chapters=chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_coordinator(
    repository: StructuredDocumentArtifactRepository,
) -> tuple[SectionTaskCoordinator, _FakeSummaryService, _FakeQuizService, _FakeTaskUnitResolver]:
    summary = _FakeSummaryService()
    quiz = _FakeQuizService()
    resolver = _FakeTaskUnitResolver()
    coordinator = SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(repository),
        document_artifact_repository=repository,
        document_profile_store=_FakeProfileStore(),
        chapter_summary_service=summary,
        chapter_quiz_service=quiz,
        task_unit_resolver=resolver,
        enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
    )
    return coordinator, summary, quiz, resolver


def test_chapter_id_resolves_ambiguous_title() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(store=StructuredDocumentStore(), base_dir=temp_dir)
        doc = _build_ambiguous_doc()
        Path(temp_dir, "chapter-id-doc.structured.json").write_text(doc.to_json(), encoding="utf-8")
        coordinator, summary, quiz, resolver = _build_coordinator(repo)

        s1 = coordinator.summarize_chapter(
            doc_name="chapter-id-doc",
            chapter_id="chapter-0",
            chapter_title="Chapter One",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(s1.success, "chapter-0 summary should succeed")
        _assert(summary.last_task_unit_content == "part1-ch1", "chapter-0 should resolve part1 content")

        q1 = coordinator.generate_chapter_quiz(
            doc_name="chapter-id-doc",
            chapter_id="chapter-10",
            chapter_title="wrong title",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(q1.success, "chapter-10 quiz should succeed even with wrong title")
        _assert(quiz.last_task_unit_content == "part2-ch1", "chapter-10 should resolve part2 content")
        _assert(resolver.calls == 0, "persisted task units should avoid resolver calls")

        updated = repo.load_document("chapter-id-doc")
        artifacts = updated.document_task_artifacts.chapter_artifacts
        _assert("chapter_id::chapter-0" in artifacts, "chapter-0 summary should use id-based key")
        _assert("chapter_id::chapter-10" in artifacts, "chapter-10 quiz should use id-based key")
        _assert("chapter::Chapter One" not in artifacts, "new writes should not rely on title-based key")


def test_title_only_ambiguity_and_unique_title() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(store=StructuredDocumentStore(), base_dir=temp_dir)
        doc = _build_ambiguous_doc()
        Path(temp_dir, "chapter-id-doc.structured.json").write_text(doc.to_json(), encoding="utf-8")
        coordinator, summary, quiz, _ = _build_coordinator(repo)

        ambiguous_summary = coordinator.summarize_chapter(
            doc_name="chapter-id-doc",
            chapter_title="Chapter One",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(not ambiguous_summary.success, "title-only Chapter One should stay ambiguous")
        _assert("ambiguous chapter title" in (ambiguous_summary.reason or "").lower(), "ambiguity reason expected")

        ambiguous_quiz = coordinator.generate_chapter_quiz(
            doc_name="chapter-id-doc",
            chapter_title="Chapter One",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(not ambiguous_quiz.success, "title-only Chapter One quiz should stay ambiguous")
        _assert("ambiguous chapter title" in (ambiguous_quiz.reason or "").lower(), "ambiguity reason expected")

        unique_summary = coordinator.summarize_chapter(
            doc_name="chapter-id-doc",
            chapter_title="Unique Chapter",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(unique_summary.success, "title-only unique chapter should work")
        _assert(summary.last_task_unit_content == "unique-ch", "unique chapter content should resolve correctly")

        unique_quiz = coordinator.generate_chapter_quiz(
            doc_name="chapter-id-doc",
            chapter_title="Unique Chapter",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(unique_quiz.success, "title-only unique chapter quiz should work")
        _assert(quiz.last_task_unit_content == "unique-ch", "unique chapter quiz should resolve correctly")


def test_legacy_title_key_fallback_and_id_key_rewrite() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(store=StructuredDocumentStore(), base_dir=temp_dir)
        doc = _build_ambiguous_doc()
        legacy_summary = SummaryArtifact(
            content="legacy-summary",
            language="en",
            source_hash=_sha256("part1-ch1"),
            prompt_version=SectionTaskCoordinator._CHAPTER_SUMMARY_PROMPT_VERSION,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            metadata={
                "summary_scope": "chapter",
                "chapter_title": "Chapter One",
                "chapter_key": "chapter::Chapter One",
                "source_section_id": "section-chapter-0",
            },
        )
        doc = StructuredDocument(
            document_id=doc.document_id,
            title=doc.title,
            source_path=doc.source_path,
            language=doc.language,
            raw_text=doc.raw_text,
            sections=doc.sections,
            chapters=doc.chapters,
            document_task_artifacts=DocumentTaskArtifacts(
                chapter_artifacts={
                    "chapter::Chapter One": TaskArtifacts(summary=legacy_summary),
                },
                metadata=dict((doc.document_task_artifacts or DocumentTaskArtifacts()).metadata),
            ),
        )
        Path(temp_dir, "chapter-id-doc.structured.json").write_text(doc.to_json(), encoding="utf-8")
        coordinator, summary, _, _ = _build_coordinator(repo)

        hit = coordinator.summarize_chapter(
            doc_name="chapter-id-doc",
            chapter_id="chapter-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(hit.success, "legacy title-key fallback should still succeed")
        _assert(hit.cache_hit is True, "legacy title-key should be cache hit before refresh")
        _assert(summary.calls == 0, "legacy fallback hit should skip service call")

        rewrite = coordinator.summarize_chapter(
            doc_name="chapter-id-doc",
            chapter_id="chapter-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=True,
        )
        _assert(rewrite.success, "refresh should regenerate summary")
        updated = repo.load_document("chapter-id-doc")
        _assert(
            "chapter_id::chapter-0" in (updated.document_task_artifacts or DocumentTaskArtifacts()).chapter_artifacts,
            "refresh should write id-based chapter key",
        )


def main() -> None:
    test_chapter_id_resolves_ambiguous_title()
    test_title_only_ambiguity_and_unique_title()
    test_legacy_title_key_fallback_and_id_key_rewrite()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "chapter_id_resolves_ambiguous_title",
                    "title_only_ambiguous_still_fails",
                    "title_only_unique_still_works",
                    "id_title_both_prefers_id",
                    "legacy_title_key_fallback",
                    "id_key_rewrite_on_refresh",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
