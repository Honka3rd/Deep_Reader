#!/usr/bin/env python3
"""Coordinator-level smoke tests for task-layout persistence cache behavior."""

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
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.section_role import SectionRole
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
            document = self.repository.load_document(doc_name)
            return _FakePreparationResult(
                assets=_FakeAssets(errors=[]),
                structured_document=document,
                bundle=None,
            )
        except Exception as error:
            return _FakePreparationResult(
                assets=_FakeAssets(errors=[str(error)]),
                structured_document=None,
                bundle=None,
            )


class _FakeTaskUnitResolver:
    def __init__(self):
        self.split_mode = TaskUnitSplitMode.SEMANTIC_SAFE
        self.calls = 0
        self.call_history: list[dict[str, object]] = []

    def resolve_with_options(
        self,
        *,
        document: StructuredDocument,
        split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> list[TaskUnit]:
        self.calls += 1
        resolved_mode = TaskUnitSplitMode.resolve(split_mode or self.split_mode)
        self.call_history.append(
            {
                "call": self.calls,
                "split_mode": resolved_mode.value,
                "semantic_top_k_candidates": semantic_top_k_candidates,
            }
        )

        task_units: list[TaskUnit] = []
        for index, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            task_units.append(
                TaskUnit(
                    unit_id=f"{resolved_mode.value}-u{index}-call{self.calls}",
                    title=section.title or f"section-{index}",
                    container_title=section.container_title,
                    content=section.content.strip(),
                    source_section_ids=[section.section_id],
                    is_fallback_generated=False,
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


def _build_base_document() -> StructuredDocument:
    return StructuredDocument(
        document_id="cache-doc",
        title="Cache Doc",
        source_path=None,
        language="zh",
        raw_text="第一章\n內容 A\n\n第二章\n內容 B",
        sections=[
            StructuredSection(
                section_id="section-0",
                section_index=0,
                title="第一章",
                level=1,
                content="第一章\n內容 A",
                char_start=0,
                char_end=8,
                section_role=SectionRole.MAIN_BODY,
            ),
            StructuredSection(
                section_id="section-1",
                section_index=1,
                title="第二章",
                level=1,
                content="第二章\n內容 B",
                char_start=9,
                char_end=17,
                section_role=SectionRole.MAIN_BODY,
            ),
        ],
    )


def test_task_layout_cache_flow() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(
            store=store,
            base_dir=temp_dir,
        )
        base_document = _build_base_document()
        (temp_path / "cache-doc.structured.json").write_text(
            base_document.to_json(),
            encoding="utf-8",
        )

        fake_resolver = _FakeTaskUnitResolver()
        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repository),
            document_artifact_repository=repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_FakeSectionTaskService(),
            chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=fake_resolver,
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
            semantic_top_k_candidates_max=20,
        )

        # A. first call writes task_units
        first_layout = coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(fake_resolver.calls == 1, "first call should resolve task units")
        reloaded_after_first = repository.load_document("cache-doc")
        _assert(
            all(section.task_units for section in reloaded_after_first.sections),
            "first call should persist non-empty section.task_units",
        )

        # Add unit-level artifact manually and save back for cache-hit preservation check.
        section0_unit0 = reloaded_after_first.sections[0].task_units[0]
        updated_section0_units = list(reloaded_after_first.sections[0].task_units)
        updated_section0_units[0] = TaskUnit(
            unit_id=section0_unit0.unit_id,
            title=section0_unit0.title,
            container_title=section0_unit0.container_title,
            content=section0_unit0.content,
            source_section_ids=list(section0_unit0.source_section_ids),
            is_fallback_generated=section0_unit0.is_fallback_generated,
            task_artifacts=TaskArtifacts(
                summary=SummaryArtifact(content="manual persisted summary")
            ),
        )
        repository.update_section_task_units(
            doc_name="cache-doc",
            section_id="section-0",
            task_units=updated_section0_units,
        )

        # B. second call reuses cache
        second_layout = coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            fake_resolver.calls == 1,
            "second same-options call should hit cache and skip resolve",
        )
        persisted_after_second = repository.load_document("cache-doc")
        _assert(
            persisted_after_second.sections[0].task_units[0].task_artifacts is not None,
            "cache hit should not clear persisted task_unit.task_artifacts",
        )

        # C. refresh=true forces recompute
        refreshed_layout = coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=True,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            fake_resolver.calls == 2,
            "refresh_task_units=true should force resolver recompute",
        )

        # D. mode/top_k mismatch invalidates cache
        coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=False,
            task_unit_split_mode="progressive",
            semantic_top_k_candidates=3,
        )
        _assert(
            fake_resolver.calls == 3,
            "split mode change should invalidate cache and recompute",
        )
        coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
        )
        _assert(
            fake_resolver.calls == 4,
            "semantic_top_k change should invalidate cache and recompute",
        )

        # Sanity check: layout payloads are non-empty.
        _assert(first_layout.chapters, "first layout should return chapters")
        _assert(second_layout.chapters, "second layout should return chapters")
        _assert(refreshed_layout.chapters, "refreshed layout should return chapters")
        _assert(first_layout.sections, "first layout should return sections")
        _assert(second_layout.task_units, "second layout should return task units")
        _assert(refreshed_layout.task_units, "refreshed layout should return task units")


def test_task_layout_cache_duplicate_id_repair() -> None:
    """Cache-hit path should repair duplicated persisted task-unit ids without resolver recompute."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        base_document = _build_base_document()
        task_layout_metadata = {
            "source_hash": SectionTaskCoordinator._compute_source_hash(base_document),
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v2",
        }
        duplicate_document = StructuredDocument(
            document_id=base_document.document_id,
            title=base_document.title,
            source_path=base_document.source_path,
            language=base_document.language,
            raw_text=base_document.raw_text,
            sections=[
                StructuredSection(
                    section_id=base_document.sections[0].section_id,
                    section_index=base_document.sections[0].section_index,
                    title=base_document.sections[0].title,
                    level=base_document.sections[0].level,
                    content=base_document.sections[0].content,
                    char_start=base_document.sections[0].char_start,
                    char_end=base_document.sections[0].char_end,
                    section_role=base_document.sections[0].section_role,
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u0",
                            container_title=None,
                            content="chunk-a",
                            source_section_ids=[base_document.sections[0].section_id],
                            is_fallback_generated=False,
                            task_artifacts=TaskArtifacts(
                                summary=SummaryArtifact(content="artifact-a")
                            ),
                        )
                    ],
                ),
                StructuredSection(
                    section_id=base_document.sections[1].section_id,
                    section_index=base_document.sections[1].section_index,
                    title=base_document.sections[1].title,
                    level=base_document.sections[1].level,
                    content=base_document.sections[1].content,
                    char_start=base_document.sections[1].char_start,
                    char_end=base_document.sections[1].char_end,
                    section_role=base_document.sections[1].section_role,
                    task_units=[
                        TaskUnit(
                            unit_id="task-unit-0",
                            title="u1",
                            container_title=None,
                            content="chunk-b",
                            source_section_ids=[base_document.sections[1].section_id],
                            is_fallback_generated=False,
                            task_artifacts=TaskArtifacts(
                                summary=SummaryArtifact(content="artifact-b")
                            ),
                        )
                    ],
                ),
            ],
            document_task_artifacts=DocumentTaskArtifacts(
                metadata={"task_layout": task_layout_metadata}
            ),
        )
        (temp_path / "cache-doc.structured.json").write_text(
            duplicate_document.to_json(),
            encoding="utf-8",
        )

        repository = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        fake_resolver = _FakeTaskUnitResolver()
        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repository),
            document_artifact_repository=repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_FakeSectionTaskService(),
            chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=fake_resolver,
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
            semantic_top_k_candidates_max=20,
        )

        layout = coordinator.get_document_task_layout(
            doc_name="cache-doc",
            refresh_task_units=False,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        _assert(
            fake_resolver.calls == 0,
            "duplicate-id cache repair should not recompute resolver",
        )

        reloaded = repository.load_document("cache-doc")
        ids = [
            task_unit.unit_id
            for section in reloaded.sections
            for task_unit in section.task_units
        ]
        _assert(len(ids) == 2, "fixture should keep two persisted units")
        _assert(len(set(ids)) == 2, "cache repair should normalize ids to document-unique")
        _assert(ids == ["task-unit-0", "task-unit-1"], "normalized ids should follow reading order")
        _assert(
            reloaded.sections[0].task_units[0].task_artifacts is not None
            and reloaded.sections[0].task_units[0].task_artifacts.summary is not None
            and reloaded.sections[0].task_units[0].task_artifacts.summary.content == "artifact-a",
            "artifact-a should stay on original section/unit position after id repair",
        )
        _assert(
            reloaded.sections[1].task_units[0].task_artifacts is not None
            and reloaded.sections[1].task_units[0].task_artifacts.summary is not None
            and reloaded.sections[1].task_units[0].task_artifacts.summary.content == "artifact-b",
            "artifact-b should stay on original section/unit position after id repair",
        )
        _assert(
            len(layout.task_units)
            == sum(len(section.task_units) for section in reloaded.sections),
            "layout response should not silently drop units after duplicate-id repair",
        )


def main() -> None:
    test_task_layout_cache_flow()
    test_task_layout_cache_duplicate_id_repair()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "first_call_writes_task_units",
                    "second_call_cache_hit",
                    "refresh_forces_recompute",
                    "mode_topk_mismatch_invalidates_cache",
                    "task_unit_artifacts_preserved_on_cache_hit",
                    "cache_hit_duplicate_id_repair_without_recompute",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
