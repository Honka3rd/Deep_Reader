#!/usr/bin/env python3
"""Hierarchy-first section/chapter task target resolution smoke tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.preparation_mode import PreparationMode
from document_structure.document_hierarchy_index import get_effective_sections
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
from shared.task_artifacts import DocumentTaskArtifacts
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
        _ = (split_mode, semantic_top_k_candidates)
        self.calls += 1
        units: list[TaskUnit] = []
        for index, section in enumerate(get_effective_sections(document)):
            if not section.content.strip():
                continue
            units.append(
                TaskUnit(
                    unit_id=f"resolver-{index}-{self.calls}",
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
        self.last_task_unit_index: int | None = None

    def summarize_task_unit(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_unit_index: int = 0,
    ) -> SectionTaskResult[str]:
        _ = (document_title, document_profile)
        self.calls += 1
        self.last_task_unit_content = task_unit.content
        self.last_task_unit_index = task_unit_index
        return SectionTaskResult.ok(f"summary-{self.calls}")


class _FakeQuizService:
    def __init__(self) -> None:
        self.calls = 0
        self.last_task_unit_content: str | None = None
        self.last_task_unit_index: int | None = None

    def generate_task_unit_quiz(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_type: str = "section_quiz",
        task_unit_index: int = 0,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        _ = (document_title, document_profile, task_type)
        self.calls += 1
        self.last_task_unit_content = task_unit.content
        self.last_task_unit_index = task_unit_index
        return SectionTaskResult.ok(
            [
                QuizQuestion(
                    question_id=f"q-{self.calls}",
                    question_text="question",
                    answer_text="answer",
                )
            ]
        )


def _build_doc_with_hierarchy_vs_legacy_drift() -> StructuredDocument:
    """Hierarchy sections are the only persisted source, including front matter."""
    section_a_hierarchy_content = "hierarchy-content-a"
    section_b_hierarchy_content = "hierarchy-content-b"

    section_a_hierarchy = StructuredSection(
        section_id="section-a",
        section_index=0,
        title="第一章",
        level=1,
        content=section_a_hierarchy_content,
        char_start=0,
        char_end=20,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-a",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-a",
                title="A",
                container_title=None,
                content=section_a_hierarchy_content,
                source_section_ids=["section-a"],
                is_fallback_generated=False,
                parent_section_id="section-a",
            )
        ],
    )
    section_b_hierarchy = StructuredSection(
        section_id="section-b",
        section_index=1,
        title="第二章",
        level=1,
        content=section_b_hierarchy_content,
        char_start=21,
        char_end=40,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-b",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-b",
                title="B",
                container_title=None,
                content=section_b_hierarchy_content,
                source_section_ids=["section-b"],
                is_fallback_generated=False,
                parent_section_id="section-b",
            )
        ],
    )

    front_section = StructuredSection(
        section_id="section-front",
        section_index=2,
        title="前言",
        level=1,
        content="front-matter-content",
        char_start=40,
        char_end=59,
        section_role=SectionRole.FRONT_MATTER,
        parent_chapter_id="front-matter-0",
        section_kind="front_matter",
        task_units=[
            TaskUnit(
                unit_id="task-unit-front",
                title="Front",
                container_title=None,
                content="front-matter-content",
                source_section_ids=["section-front"],
                is_fallback_generated=False,
                parent_section_id="section-front",
            )
        ],
    )

    raw_text = "\n".join(
        [
            "第一章",
            section_a_hierarchy_content,
            "第二章",
            section_b_hierarchy_content,
            "前言",
            "front-matter-content",
        ]
    )
    document = StructuredDocument(
        document_id="target-resolution-doc",
        title="Target Resolution",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=[],
        chapters=[
            StructuredChapter(
                chapter_id="front-matter-0",
                title="前言",
                level=1,
                chapter_role="front_matter",
                sections=[front_section],
            ),
            StructuredChapter(
                chapter_id="chapter-a",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[section_a_hierarchy],
            ),
            StructuredChapter(
                chapter_id="chapter-b",
                title="第二章",
                level=1,
                chapter_role="main_body",
                sections=[section_b_hierarchy],
            ),
        ],
    )
    metadata = {
        "task_layout": {
            "source_hash": _sha256(raw_text),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id=document.document_id,
        title=document.title,
        source_path=document.source_path,
        language=document.language,
        raw_text=document.raw_text,
        sections=document.sections,
        chapters=document.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_doc_with_duplicate_chapter_title() -> StructuredDocument:
    section_1 = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="Introduction",
        level=1,
        content="intro-1",
        char_start=0,
        char_end=6,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-1",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[],
    )
    section_2 = StructuredSection(
        section_id="section-2",
        section_index=1,
        title="Introduction",
        level=1,
        content="intro-2",
        char_start=7,
        char_end=13,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-2",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[],
    )
    return StructuredDocument(
        document_id="dup-chapter-doc",
        title="Duplicate Chapter",
        source_path=None,
        language="en",
        raw_text="intro-1\nintro-2",
        sections=[],
        chapters=[
            StructuredChapter("chapter-1", "Introduction", 1, "main_body", [section_1]),
            StructuredChapter("chapter-2", "Introduction", 1, "main_body", [section_2]),
        ],
        document_task_artifacts=DocumentTaskArtifacts(
            metadata={
                "task_layout": {
                    "source_hash": _sha256("intro-1\nintro-2"),
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 3,
                    "resolver_version": "task_unit_resolver_v2",
                }
            }
        ),
    )


def _build_coordinator(
    repository: StructuredDocumentArtifactRepository,
    *,
    resolver: _FakeTaskUnitResolver | None = None,
    summary_service: _FakeSummaryService | None = None,
    quiz_service: _FakeQuizService | None = None,
) -> tuple[SectionTaskCoordinator, _FakeTaskUnitResolver, _FakeSummaryService, _FakeQuizService]:
    fake_resolver = resolver or _FakeTaskUnitResolver()
    fake_summary = summary_service or _FakeSummaryService()
    fake_quiz = quiz_service or _FakeQuizService()
    coordinator = SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(repository),
        document_artifact_repository=repository,
        document_profile_store=_FakeProfileStore(),
        chapter_summary_service=fake_summary,
        chapter_quiz_service=fake_quiz,
        task_unit_resolver=fake_resolver,
        enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
    )
    return coordinator, fake_resolver, fake_summary, fake_quiz


def test_section_summary_and_quiz_use_hierarchy_section() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_doc_with_hierarchy_vs_legacy_drift()
        Path(temp_dir, "target-resolution-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator, resolver, summary_service, quiz_service = _build_coordinator(repo)

        summary_result = coordinator.summarize_section(
            doc_name="target-resolution-doc",
            section_id="section-a",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(summary_result.success, "section summary should succeed")
        _assert(resolver.calls == 0, "persisted layout should avoid resolver recompute")
        _assert(
            summary_service.last_task_unit_content == "hierarchy-content-a",
            "section summary should use hierarchy section task unit content",
        )
        _assert(
            summary_service.last_task_unit_index == 1,
            "section summary task_unit_index should follow hierarchy order including front matter",
        )

        updated = repo.load_document("target-resolution-doc")
        section_a = next(
            section for chapter in updated.chapters for section in chapter.sections if section.section_id == "section-a"
        )
        _assert(
            section_a.task_artifacts is not None and section_a.task_artifacts.summary is not None,
            "summary artifact should persist on hierarchy section",
        )
        _assert(
            section_a.task_artifacts.summary.source_hash == _sha256("hierarchy-content-a"),
            "summary source_hash should match hierarchy section content",
        )

        quiz_result = coordinator.generate_section_quiz(
            doc_name="target-resolution-doc",
            section_id="section-a",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(quiz_result.success, "section quiz should succeed")
        _assert(
            quiz_service.last_task_unit_content == "hierarchy-content-a",
            "section quiz should use hierarchy section task unit content",
        )
        _assert(
            quiz_service.last_task_unit_index == 1,
            "section quiz task_unit_index should follow hierarchy order including front matter",
        )
        updated_after_quiz = repo.load_document("target-resolution-doc")
        section_a_after_quiz = next(
            section for chapter in updated_after_quiz.chapters for section in chapter.sections if section.section_id == "section-a"
        )
        _assert(
            section_a_after_quiz.task_artifacts is not None
            and section_a_after_quiz.task_artifacts.quiz is not None,
            "quiz artifact should persist on hierarchy section",
        )
        _assert(
            section_a_after_quiz.task_artifacts.quiz.source_hash == _sha256("hierarchy-content-a"),
            "quiz source_hash should match hierarchy section content",
        )


def test_chapter_resolution_hierarchy_and_ambiguity() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_doc_with_hierarchy_vs_legacy_drift()
        Path(temp_dir, "target-resolution-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator, _, summary_service, quiz_service = _build_coordinator(repo)

        chapter_summary = coordinator.summarize_chapter(
            doc_name="target-resolution-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(chapter_summary.success, "chapter summary should succeed")
        _assert(
            summary_service.last_task_unit_content == "hierarchy-content-a",
            "chapter summary should resolve chapter_body hierarchy section",
        )

        chapter_quiz = coordinator.generate_chapter_quiz(
            doc_name="target-resolution-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(chapter_quiz.success, "chapter quiz should succeed")
        _assert(
            quiz_service.last_task_unit_content == "hierarchy-content-a",
            "chapter quiz should resolve chapter_body hierarchy section",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        dup_doc = _build_doc_with_duplicate_chapter_title()
        Path(temp_dir, "dup-chapter-doc.structured.json").write_text(
            dup_doc.to_json(),
            encoding="utf-8",
        )
        coordinator, _, _, _ = _build_coordinator(repo)
        ambiguous = coordinator.summarize_chapter(
            doc_name="dup-chapter-doc",
            chapter_title="Introduction",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(not ambiguous.success, "ambiguous chapter title should fail safely")
        _assert(
            "ambiguous chapter title" in (ambiguous.reason or "").lower(),
            "ambiguous chapter title failure reason should be explicit",
        )


def test_front_matter_hierarchy_resolution_and_order_index() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        doc = _build_doc_with_hierarchy_vs_legacy_drift()
        Path(temp_dir, "target-resolution-doc.structured.json").write_text(
            doc.to_json(),
            encoding="utf-8",
        )
        coordinator, resolver, summary_service, _ = _build_coordinator(repo)

        section_b_summary = coordinator.summarize_section(
            doc_name="target-resolution-doc",
            section_id="section-b",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(section_b_summary.success, "section-b summary should succeed")
        _assert(
            summary_service.last_task_unit_index == 2,
            "task unit index for section-b should follow hierarchy order (front matter, A, then B)",
        )

        front_summary = coordinator.summarize_section(
            doc_name="target-resolution-doc",
            section_id="section-front",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(front_summary.success, "front matter section summary should resolve from hierarchy and succeed")
        _assert(
            summary_service.last_task_unit_content == "front-matter-content",
            "front matter summary should resolve hierarchy front matter task unit",
        )
        _assert(
            resolver.calls == 0,
            "persisted layout resolution should remain hierarchy-first without resolver recompute",
        )


def test_chapter_title_missing_does_not_fallback_to_legacy_root_sections() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        hierarchy_section = StructuredSection(
            section_id="section-hierarchy",
            section_index=0,
            title="Hierarchy Chapter",
            level=1,
            content="hierarchy-content",
            char_start=0,
            char_end=20,
            section_role=SectionRole.MAIN_BODY,
            parent_chapter_id="chapter-0",
            section_kind="chapter_body",
            is_implicit_section=True,
            task_units=[
                TaskUnit(
                    unit_id="task-unit-hierarchy",
                    title="H",
                    container_title=None,
                    content="hierarchy-content",
                    source_section_ids=["section-hierarchy"],
                    is_fallback_generated=False,
                    parent_section_id="section-hierarchy",
                )
            ],
        )
        legacy_root_section = StructuredSection(
            section_id="section-legacy-root",
            section_index=1,
            title="Legacy Root Only",
            level=1,
            content="legacy-root-content",
            char_start=20,
            char_end=40,
            section_role=SectionRole.MAIN_BODY,
        )
        doc = StructuredDocument(
            document_id="chapter-title-no-legacy-fallback-doc",
            title="Chapter Title No Legacy Fallback",
            source_path=None,
            language="en",
            raw_text="Hierarchy Chapter\nhierarchy-content\nLegacy Root Only\nlegacy-root-content",
            sections=[legacy_root_section],
            chapters=[
                StructuredChapter(
                    chapter_id="chapter-0",
                    title="Hierarchy Chapter",
                    level=1,
                    chapter_role="main_body",
                    sections=[hierarchy_section],
                )
            ],
        )
        Path(temp_dir, "chapter-title-no-legacy-fallback-doc.structured.json").write_text(
            doc.to_json(include_legacy_sections=True),
            encoding="utf-8",
        )

        coordinator, _, _, _ = _build_coordinator(repo)
        result = coordinator.summarize_chapter(
            doc_name="chapter-title-no-legacy-fallback-doc",
            chapter_title="Legacy Root Only",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(not result.success, "chapter title not present in hierarchy should fail fast")
        _assert(
            "not found" in (result.reason or "").lower(),
            "failure reason should indicate hierarchy chapter title not found",
        )


def main() -> None:
    test_section_summary_and_quiz_use_hierarchy_section()
    test_chapter_resolution_hierarchy_and_ambiguity()
    test_front_matter_hierarchy_resolution_and_order_index()
    test_chapter_title_missing_does_not_fallback_to_legacy_root_sections()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_summary_hierarchy_first",
                    "section_quiz_hierarchy_first",
                    "chapter_summary_quiz_chapter_body",
                    "duplicate_chapter_title_ambiguous",
                    "front_matter_hierarchy_resolution",
                    "task_unit_index_hierarchy_order",
                    "persisted_layout_resolution_hierarchy_first",
                    "chapter_title_missing_no_legacy_root_fallback",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
