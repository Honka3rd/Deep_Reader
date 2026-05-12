#!/usr/bin/env python3
"""Task-layout hierarchy-first read path smoke tests."""

from __future__ import annotations

import json
import io
import tempfile
from contextlib import redirect_stdout
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
from shared.task_artifacts import DocumentTaskArtifacts
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
        task_units: list[TaskUnit] = []
        for idx, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            task_units.append(
                TaskUnit(
                    unit_id=f"resolver-u-{idx}",
                    title=section.title,
                    container_title=section.container_title,
                    content=section.content,
                    source_section_ids=[section.section_id],
                    is_fallback_generated=False,
                    parent_section_id=section.section_id,
                )
            )
        return task_units


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


def _build_hierarchy_first_doc() -> StructuredDocument:
    hierarchy_section = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="第一章",
        level=1,
        content="hierarchy content",
        char_start=0,
        char_end=16,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-0",
        section_kind="chapter_body",
        is_implicit_section=True,
        task_units=[
            TaskUnit(
                unit_id="task-unit-h1",
                title="h1",
                container_title=None,
                content="hierarchy content",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id="section-1",
            )
        ],
    )
    stale_legacy_section = StructuredSection(
        section_id="section-old",
        section_index=0,
        title="STALE",
        level=1,
        content="stale legacy content",
        char_start=0,
        char_end=18,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-old",
                title="old",
                container_title=None,
                content="stale legacy content",
                source_section_ids=["section-old"],
                is_fallback_generated=False,
                parent_section_id="section-old",
            )
        ],
    )
    doc = StructuredDocument(
        document_id="hierarchy-doc",
        title="Hierarchy Doc",
        source_path=None,
        language="zh",
        raw_text="hierarchy content",
        sections=[stale_legacy_section],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[hierarchy_section],
            )
        ],
    )
    metadata = {
        "task_layout": {
            "source_hash": SectionTaskCoordinator._compute_source_hash(doc),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=doc.sections,
        chapters=doc.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_legacy_only_doc() -> StructuredDocument:
    section = StructuredSection(
        section_id="section-legacy",
        section_index=0,
        title="Legacy",
        level=1,
        content="legacy content",
        char_start=0,
        char_end=13,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-legacy",
                title="legacy",
                container_title=None,
                content="legacy content",
                source_section_ids=["section-legacy"],
                is_fallback_generated=False,
                parent_section_id="section-legacy",
            )
        ],
    )
    doc = StructuredDocument(
        document_id="legacy-doc",
        title="Legacy Doc",
        source_path=None,
        language="zh",
        raw_text="legacy content",
        sections=[section],
        chapters=[],
    )
    metadata = {
        "task_layout": {
            "source_hash": SectionTaskCoordinator._compute_source_hash(doc),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=doc.sections,
        chapters=doc.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_hierarchy_cache_invalid_doc() -> StructuredDocument:
    hierarchy_section = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="第一章",
        level=1,
        content="hierarchy content",
        char_start=0,
        char_end=16,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-0",
        task_units=[],  # intentionally empty -> should invalidate cache
    )
    legacy_section = StructuredSection(
        section_id="section-legacy",
        section_index=0,
        title="Legacy",
        level=1,
        content="legacy content",
        char_start=0,
        char_end=13,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-legacy",
                title="legacy",
                container_title=None,
                content="legacy content",
                source_section_ids=["section-legacy"],
                is_fallback_generated=False,
                parent_section_id="section-legacy",
            )
        ],
    )
    doc = StructuredDocument(
        document_id="cache-invalid-doc",
        title="Cache Invalid Doc",
        source_path=None,
        language="zh",
        raw_text="hierarchy content",
        sections=[legacy_section],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[hierarchy_section],
            )
        ],
    )
    metadata = {
        "task_layout": {
            "source_hash": SectionTaskCoordinator._compute_source_hash(doc),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=doc.sections,
        chapters=doc.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_hierarchy_inconsistent_doc() -> StructuredDocument:
    dup_a = StructuredSection(
        section_id="section-dup",
        section_index=0,
        title="第一章",
        level=1,
        content="a",
        char_start=0,
        char_end=1,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-0",
        task_units=[],
    )
    dup_b = StructuredSection(
        section_id="section-dup",
        section_index=1,
        title="第二章",
        level=1,
        content="b",
        char_start=2,
        char_end=3,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-1",
        task_units=[],
    )
    doc = StructuredDocument(
        document_id="inconsistent-doc",
        title="Inconsistent",
        source_path=None,
        language="zh",
        raw_text="legacy",
        sections=[],
        chapters=[
            StructuredChapter("chapter-0", "第一章", 1, "main_body", [dup_a]),
            StructuredChapter("chapter-1", "第二章", 1, "main_body", [dup_b]),
        ],
    )
    metadata = {
        "task_layout": {
            "source_hash": SectionTaskCoordinator._compute_source_hash(doc),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=doc.sections,
        chapters=doc.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=metadata),
    )


def _build_legacy_ordering_fallback_doc() -> StructuredDocument:
    ordering_seed_section = StructuredSection(
        section_id="section-h1",
        section_index=0,
        title="Hierarchy One",
        level=1,
        content="hierarchy body",
        char_start=0,
        char_end=14,
        section_role=SectionRole.MAIN_BODY,
        parent_chapter_id="chapter-0",
        task_units=[
            TaskUnit(
                unit_id="task-unit-h1",
                title="h1",
                container_title=None,
                content="hierarchy body",
                source_section_ids=["section-h1"],
                is_fallback_generated=False,
                parent_section_id="section-h1",
            )
        ],
    )
    hierarchy_target_section = StructuredSection(
        section_id="section-target",
        section_index=1,
        title="Hierarchy Target",
        level=1,
        content="target body",
        char_start=15,
        char_end=26,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-target",
                title="target",
                container_title=None,
                content="target body",
                source_section_ids=["section-target"],
                is_fallback_generated=False,
                parent_section_id="section-target",
            )
        ],
        parent_chapter_id="chapter-0",
    )
    legacy_mirror_section = StructuredSection(
        section_id="section-target",
        section_index=0,
        title="Legacy Target",
        level=1,
        content="target body",
        char_start=0,
        char_end=11,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-target",
                title="target",
                container_title=None,
                content="target body",
                source_section_ids=["section-target"],
                is_fallback_generated=False,
                parent_section_id="section-target",
            )
        ],
    )
    return StructuredDocument(
        document_id="legacy-ordering-doc",
        title="Legacy Ordering Doc",
        source_path=None,
        language="en",
        raw_text="hierarchy body\ntarget body",
        sections=[legacy_mirror_section],
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="Chapter 1",
                level=1,
                chapter_role="main_body",
                sections=[ordering_seed_section, hierarchy_target_section],
            )
        ],
    )


def _build_coordinator(
    repository: StructuredDocumentArtifactRepository,
    resolver: _FakeTaskUnitResolver,
) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(repository),
        document_artifact_repository=repository,
        document_profile_store=_FakeProfileStore(),
        chapter_summary_service=_FakeSectionTaskService(),
        chapter_quiz_service=_FakeSectionTaskService(),
        task_unit_resolver=resolver,
        enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
        semantic_top_k_candidates_max=20,
    )


def test_task_layout_hierarchy_first_read() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(store=store, base_dir=temp_dir)
        resolver = _FakeTaskUnitResolver()
        coordinator = _build_coordinator(repository, resolver)

        # A. hierarchy-first read
        hierarchy_doc = _build_hierarchy_first_doc()
        Path(temp_dir, "hierarchy-doc.structured.json").write_text(
            hierarchy_doc.to_json(),
            encoding="utf-8",
        )
        hierarchy_layout = coordinator.get_document_task_layout(
            doc_name="hierarchy-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            [chapter.chapter_id for chapter in hierarchy_layout.chapters] == ["chapter-0"],
            "layout should expose hierarchy chapters when available",
        )
        _assert(
            [
                section.section_id
                for section in hierarchy_layout.chapters[0].sections
            ]
            == ["section-1"],
            "chapter sections should come from hierarchy source",
        )
        _assert(resolver.calls == 0, "valid hierarchy cache should not trigger resolver")
        _assert(
            "chapters" in hierarchy_layout.to_dict(),
            "task-layout schema should now expose chapters field",
        )

        # B. legacy fallback when chapters empty
        legacy_doc = _build_legacy_only_doc()
        Path(temp_dir, "legacy-doc.structured.json").write_text(
            legacy_doc.to_json(),
            encoding="utf-8",
        )
        legacy_layout = coordinator.get_document_task_layout(
            doc_name="legacy-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            [chapter.chapter_id for chapter in legacy_layout.chapters] == ["chapter-legacy-0"],
            "layout should provide synthetic chapter fallback when chapters are missing",
        )
        _assert(
            [section.section_id for section in legacy_layout.chapters[0].sections] == ["section-legacy"],
            "synthetic chapter should wrap legacy sections",
        )

        # C. cache validity checks effective(hierarchy) sections
        invalid_doc = _build_hierarchy_cache_invalid_doc()
        Path(temp_dir, "cache-invalid-doc.structured.json").write_text(
            invalid_doc.to_json(),
            encoding="utf-8",
        )
        before_calls = resolver.calls
        coordinator.get_document_task_layout(
            doc_name="cache-invalid-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            resolver.calls == before_calls + 1,
            "missing task_units on hierarchy sections should invalidate cache and trigger resolver",
        )

        # D. hierarchy inconsistency without legacy fallback fails fast (pure hierarchy)
        inconsistent_doc = _build_hierarchy_inconsistent_doc()
        Path(temp_dir, "inconsistent-doc.structured.json").write_text(
            inconsistent_doc.to_json(),
            encoding="utf-8",
        )
        try:
            coordinator.get_document_task_layout(
                doc_name="inconsistent-doc",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
            )
            raise AssertionError(
                "expected failure when hierarchy is inconsistent and no legacy sections exist"
            )
        except ValueError as error:
            _assert(
                "hierarchy is inconsistent and no legacy sections are available"
                in str(error),
                "should fail-fast with explicit pure-hierarchy error",
            )


def test_legacy_ordering_fallback_resolves_and_emits_diagnostic() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(store=store, base_dir=temp_dir)
        resolver = _FakeTaskUnitResolver()
        coordinator = _build_coordinator(repository, resolver)

        document = _build_legacy_ordering_fallback_doc()
        original_resolve_sections = coordinator._resolve_task_layout_sections

        def _stale_ordering_only_seed(
            self,
            *,
            document: StructuredDocument,
            context: str,
        ) -> list[StructuredSection]:
            _ = context
            return [document.chapters[0].sections[0]]

        coordinator._resolve_task_layout_sections = _stale_ordering_only_seed.__get__(
            coordinator,
            SectionTaskCoordinator,
        )
        log_buffer = io.StringIO()
        try:
            with redirect_stdout(log_buffer):
                resolved = coordinator._resolve_task_unit_for_section_id_from_persisted_layout(
                    document=document,
                    section_id="section-target",
                    allow_legacy_ordering_fallback=True,
                )
        finally:
            coordinator._resolve_task_layout_sections = original_resolve_sections
        logs = log_buffer.getvalue()

        _assert(
            resolved.task_unit.unit_id == "task-unit-target",
            "legacy ordering fallback should resolve selected task unit when hierarchy ordering misses it",
        )
        _assert(
            resolved.task_unit_index == 1,
            "legacy task unit should be appended after stale hierarchy ordering",
        )
        _assert(
            "SectionTaskCoordinator#legacy_section_ordering_fallback_hit" in logs,
            "legacy fallback diagnostic should be emitted when compatibility path is hit",
        )


def test_legacy_ordering_fallback_disabled_by_default() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(store=store, base_dir=temp_dir)
        resolver = _FakeTaskUnitResolver()
        coordinator = _build_coordinator(repository, resolver)

        document = _build_legacy_ordering_fallback_doc()
        original_resolve_sections = coordinator._resolve_task_layout_sections

        def _stale_ordering_only_seed(
            self,
            *,
            document: StructuredDocument,
            context: str,
        ) -> list[StructuredSection]:
            _ = context
            return [document.chapters[0].sections[0]]

        coordinator._resolve_task_layout_sections = _stale_ordering_only_seed.__get__(
            coordinator,
            SectionTaskCoordinator,
        )
        log_buffer = io.StringIO()
        try:
            with redirect_stdout(log_buffer):
                try:
                    coordinator._resolve_task_unit_for_section_id_from_persisted_layout(
                        document=document,
                        section_id="section-target",
                    )
                    raise AssertionError(
                        "expected no implicit legacy ordering fallback when flag is not enabled"
                    )
                except ValueError as error:
                    _assert(
                        "task units are not aligned with effective task-layout ordering" in str(error),
                        "default path should fail with ordering-misaligned error",
                    )
        finally:
            coordinator._resolve_task_layout_sections = original_resolve_sections
        logs = log_buffer.getvalue()

        _assert(
            "SectionTaskCoordinator#legacy_section_ordering_fallback_hit" not in logs,
            "legacy fallback diagnostic should not appear when fallback is disabled",
        )


def main() -> None:
    test_task_layout_hierarchy_first_read()
    test_legacy_ordering_fallback_resolves_and_emits_diagnostic()
    test_legacy_ordering_fallback_disabled_by_default()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "hierarchy_first_read",
                    "legacy_fallback",
                    "cache_validity_uses_effective_sections",
                    "inconsistent_hierarchy_without_legacy_fails_fast",
                    "legacy_ordering_fallback_resolves_and_emits_diagnostic",
                    "legacy_ordering_fallback_disabled_by_default",
                    "chapters_schema_exposed",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
