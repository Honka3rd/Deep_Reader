#!/usr/bin/env python3
"""Hierarchy-aware task-layout write/repair synchronization smoke tests."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.section_task_coordinator import SectionTaskCoordinator
from document_preparation.preparation_mode import PreparationMode
from document_structure.document_hierarchy_index import (
    is_severe_hierarchy_warning,
    validate_chapter_hierarchy_consistency,
)
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
from shared.task_artifacts import DocumentTaskArtifacts, SummaryArtifact, TaskArtifacts
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
        for idx, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            units.append(
                TaskUnit(
                    unit_id=f"resolver-{idx}-call-{self.calls}",
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


def _layout_metadata(document: StructuredDocument) -> dict[str, dict[str, str | int | None]]:
    return {
        "task_layout": {
            "source_hash": SectionTaskCoordinator._compute_source_hash(document),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
    }


def _build_hierarchy_doc(
    *,
    include_front_matter: bool = False,
    with_task_units: bool = False,
) -> StructuredDocument:
    chapter_section_units = (
        [
            TaskUnit(
                unit_id="task-unit-0",
                title="u0",
                container_title=None,
                content="chapter body content",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id="section-1",
            )
        ]
        if with_task_units
        else []
    )
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
        task_units=chapter_section_units,
    )
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
                task_units=[
                    TaskUnit(
                        unit_id="task-unit-front",
                        title="front",
                        container_title=None,
                        content="front matter content",
                        source_section_ids=["section-0"],
                        is_fallback_generated=False,
                        parent_section_id="section-0",
                        task_artifacts=TaskArtifacts(
                            summary=SummaryArtifact(content="front-summary"),
                        ),
                    )
                ]
                if with_task_units
                else [],
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
            task_units=list(chapter_section_units),
        )
    )
    doc = StructuredDocument(
        document_id="hier-sync-doc",
        title="Hierarchy Sync",
        source_path=None,
        language="zh",
        raw_text="front matter content\nchapter body content",
        sections=legacy_sections,
        chapters=[
            StructuredChapter(
                chapter_id="chapter-0",
                title="第一章",
                level=1,
                chapter_role="main_body",
                sections=[chapter_section],
            )
        ],
    )
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=doc.sections,
        chapters=doc.chapters,
        document_task_artifacts=DocumentTaskArtifacts(metadata=_layout_metadata(doc)),
    )


def _build_duplicate_id_doc() -> StructuredDocument:
    doc = _build_hierarchy_doc(include_front_matter=True, with_task_units=True)
    chapter = doc.chapters[0]
    chapter_dup_section = StructuredSection(
        section_id=chapter.sections[0].section_id,
        section_index=chapter.sections[0].section_index,
        title=chapter.sections[0].title,
        level=chapter.sections[0].level,
        content=chapter.sections[0].content,
        char_start=chapter.sections[0].char_start,
        char_end=chapter.sections[0].char_end,
        section_role=chapter.sections[0].section_role,
        parent_chapter_id=chapter.sections[0].parent_chapter_id,
        section_kind=chapter.sections[0].section_kind,
        is_implicit_section=chapter.sections[0].is_implicit_section,
        task_units=[
            TaskUnit(
                unit_id="task-unit-dup",
                title="dup-a",
                container_title=None,
                content="chapter body content",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
                parent_section_id="section-1",
            )
        ],
    )
    updated_chapter = StructuredChapter(
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        level=chapter.level,
        chapter_role=chapter.chapter_role,
        sections=[chapter_dup_section],
        metadata=dict(chapter.metadata),
    )

    updated_sections = list(doc.sections)
    updated_sections[0] = StructuredSection(
        section_id=updated_sections[0].section_id,
        section_index=updated_sections[0].section_index,
        title=updated_sections[0].title,
        level=updated_sections[0].level,
        content=updated_sections[0].content,
        char_start=updated_sections[0].char_start,
        char_end=updated_sections[0].char_end,
        section_role=updated_sections[0].section_role,
        task_units=[
            TaskUnit(
                unit_id="task-unit-dup",
                title="dup-front",
                container_title=None,
                content="front matter content",
                source_section_ids=["section-0"],
                is_fallback_generated=False,
                parent_section_id="section-0",
                task_artifacts=updated_sections[0].task_units[0].task_artifacts,
            )
        ],
    )
    updated_sections[1] = StructuredSection(
        section_id=updated_sections[1].section_id,
        section_index=updated_sections[1].section_index,
        title=updated_sections[1].title,
        level=updated_sections[1].level,
        content=updated_sections[1].content,
        char_start=updated_sections[1].char_start,
        char_end=updated_sections[1].char_end,
        section_role=updated_sections[1].section_role,
        task_units=list(chapter_dup_section.task_units),
    )
    return StructuredDocument(
        document_id=doc.document_id,
        title=doc.title,
        source_path=doc.source_path,
        language=doc.language,
        raw_text=doc.raw_text,
        sections=updated_sections,
        chapters=[updated_chapter],
        document_task_artifacts=doc.document_task_artifacts,
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


def test_hierarchy_write_sync() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        resolver = _FakeTaskUnitResolver()
        coordinator = _build_coordinator(repo, resolver)

        # A/B/C/F/G: cache miss write + refresh + hierarchy-first cache-hit + consistency + no schema change
        base_doc = _build_hierarchy_doc(include_front_matter=True, with_task_units=False)
        Path(temp_dir, "hier-sync-doc.structured.json").write_text(
            base_doc.to_json(),
            encoding="utf-8",
        )

        first_layout = coordinator.get_document_task_layout(
            doc_name="hier-sync-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(resolver.calls == 1, "cache miss should resolve once")
        after_first = repo.load_document("hier-sync-doc")
        chapter_units = after_first.chapters[0].sections[0].task_units
        legacy_chapter = next(section for section in after_first.sections if section.section_id == "section-1")
        _assert(chapter_units, "cache miss should write task units into hierarchy chapter section")
        _assert(legacy_chapter.task_units, "cache miss should write task units into legacy flat section")
        _assert(
            [unit.unit_id for unit in chapter_units] == [unit.unit_id for unit in legacy_chapter.task_units],
            "hierarchy and legacy chapter section task-unit ids should stay aligned",
        )
        _assert(
            "chapters" not in first_layout.to_dict(),
            "task-layout response schema should not add chapters field in this round",
        )

        # refresh should rewrite both
        refreshed_layout = coordinator.get_document_task_layout(
            doc_name="hier-sync-doc",
            refresh_task_units=True,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(resolver.calls == 2, "refresh=true should force resolver recompute")
        after_refresh = repo.load_document("hier-sync-doc")
        _assert(after_refresh.chapters[0].sections[0].task_units, "refresh should keep hierarchy task units populated")
        _assert(
            next(section for section in after_refresh.sections if section.section_id == "section-1").task_units,
            "refresh should keep legacy chapter section task units populated",
        )

        # cache hit should read hierarchy and skip resolver
        before_cache_hit_calls = resolver.calls
        cache_hit_layout = coordinator.get_document_task_layout(
            doc_name="hier-sync-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            resolver.calls == before_cache_hit_calls,
            "cache hit should not invoke resolver again",
        )
        _assert(
            [section.section_id for section in cache_hit_layout.sections] == ["section-1"],
            "hierarchy-first cache hit should read chapter sections",
        )

        # D: front matter preserved after write sync
        front_matter = next(
            section for section in after_refresh.sections if section.section_id == "section-0"
        )
        _assert(
            front_matter.section_role == SectionRole.FRONT_MATTER,
            "front matter section should be preserved in legacy sections",
        )
        _assert(
            len(front_matter.task_units) <= 1,
            "front matter task units/artifacts should not be dropped by hierarchy sync",
        )
        warnings = validate_chapter_hierarchy_consistency(after_refresh)
        severe = [warning for warning in warnings if is_severe_hierarchy_warning(warning)]
        _assert(not severe, f"post-save document should not have severe hierarchy warnings: {severe}")

        # E: cache-hit duplicate-id repair is hierarchy-aware
        duplicate_doc = _build_duplicate_id_doc()
        Path(temp_dir, "hier-sync-doc.structured.json").write_text(
            duplicate_doc.to_json(),
            encoding="utf-8",
        )
        before_repair_calls = resolver.calls
        coordinator.get_document_task_layout(
            doc_name="hier-sync-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            resolver.calls == before_repair_calls,
            "duplicate-id repair on cache-hit should not re-run resolver",
        )
        repaired = repo.load_document("hier-sync-doc")
        ids = [unit.unit_id for section in repaired.sections for unit in section.task_units]
        _assert(len(ids) == len(set(ids)), "repaired document should have globally unique task-unit ids")
        _assert(
            repaired.chapters[0].sections[0].task_units,
            "duplicate repair should also update hierarchy sections",
        )
        front_after_repair = next(section for section in repaired.sections if section.section_id == "section-0")
        _assert(
            front_after_repair.task_units[0].task_artifacts is not None,
            "duplicate repair should preserve task-unit artifacts by position",
        )


def main() -> None:
    test_hierarchy_write_sync()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "cache_miss_writes_hierarchy_and_legacy",
                    "refresh_rewrites_hierarchy_and_legacy",
                    "cache_hit_reads_hierarchy",
                    "front_matter_preserved",
                    "duplicate_id_repair_hierarchy_aware",
                    "consistency_after_save",
                    "no_schema_change",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
