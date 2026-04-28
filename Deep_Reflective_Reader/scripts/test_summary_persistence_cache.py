#!/usr/bin/env python3
"""Coordinator-level smoke tests for section/chapter summary persistence cache."""

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
from document_structure.structured_document import StructuredDocument, StructuredSection
from document_structure.structured_document_artifact_repository import (
    StructuredDocumentArtifactRepository,
)
from document_structure.structured_document_store import StructuredDocumentStore
from section_tasks.section_task_result import SectionTaskResult
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

    def resolve_with_options(
        self,
        *,
        document: StructuredDocument,
        split_mode: TaskUnitSplitMode | str | None = None,
        semantic_top_k_candidates: int | None = None,
    ) -> list[TaskUnit]:
        self.calls += 1
        resolved_mode = TaskUnitSplitMode.resolve(split_mode or self.split_mode)
        units: list[TaskUnit] = []
        for index, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            units.append(
                TaskUnit(
                    unit_id=f"{resolved_mode.value}-u{index}-call{self.calls}",
                    title=section.title or f"section-{index}",
                    container_title=section.container_title,
                    content=section.content.strip(),
                    source_section_ids=[section.section_id],
                    is_fallback_generated=False,
                )
            )
        _ = semantic_top_k_candidates
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
    def __init__(self):
        self.calls = 0
        self.payloads: list[str] = []

    def summarize_task_unit(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_unit_index: int = 0,
    ) -> SectionTaskResult[str]:
        _ = (task_unit, document_title, document_profile, task_unit_index)
        self.calls += 1
        payload = f"summary-call-{self.calls}"
        self.payloads.append(payload)
        return SectionTaskResult.ok(payload)


class _FakeQuizService:
    pass


def _build_base_document() -> StructuredDocument:
    raw_text = "第一章\n內容 A\n\n第二章\n內容 B"
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    section0 = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content="第一章\n內容 A",
        char_start=0,
        char_end=8,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="unit-0",
                title="第一章",
                container_title=None,
                content="第一章\n內容 A",
                source_section_ids=["section-0"],
                is_fallback_generated=False,
            )
        ],
        task_artifacts=TaskArtifacts(
            quiz=QuizArtifact(
                items=[{"question_id": "q1", "question_text": "x", "answer_text": "y"}]
            )
        ),
    )
    section1 = StructuredSection(
        section_id="section-1",
        section_index=1,
        title="第二章",
        level=1,
        content="第二章\n內容 B",
        char_start=9,
        char_end=17,
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="unit-1",
                title="第二章",
                container_title=None,
                content="第二章\n內容 B",
                source_section_ids=["section-1"],
                is_fallback_generated=False,
            )
        ],
    )
    return StructuredDocument(
        document_id="summary-doc",
        title="Summary Doc",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=[section0, section1],
        document_task_artifacts=DocumentTaskArtifacts(
            metadata={
                "task_layout": {
                    "source_hash": source_hash,
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 3,
                    "resolver_version": "task_unit_resolver_v1",
                }
            }
        ),
    )


def test_summary_persistence_cache_flow() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(store=store, base_dir=temp_dir)
        base_document = _build_base_document()
        (temp_path / "summary-doc.structured.json").write_text(
            base_document.to_json(),
            encoding="utf-8",
        )

        fake_summary_service = _FakeSummaryService()
        fake_resolver = _FakeTaskUnitResolver()
        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repository),
            document_artifact_repository=repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=fake_summary_service,
            chapter_quiz_service=_FakeQuizService(),
            task_unit_resolver=fake_resolver,
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
            semantic_top_k_candidates_max=20,
        )

        # A. first call writes section summary cache
        result_1 = coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(result_1.success, "first summarize_section call should succeed")
        _assert(fake_summary_service.calls == 1, "first call should invoke summary service")
        _assert(fake_resolver.calls == 0, "persisted task_units should avoid resolver recompute")

        updated_1 = repository.load_document("summary-doc")
        section0_1 = next(s for s in updated_1.sections if s.section_id == "section-0")
        _assert(
            section0_1.task_artifacts is not None and section0_1.task_artifacts.summary is not None,
            "first call should persist section summary artifact",
        )

        # F. quiz artifact preserved after summary update
        _assert(
            section0_1.task_artifacts.quiz is not None,
            "summary update must preserve existing quiz artifact",
        )

        # B. second call cache hit
        result_2 = coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(result_2.success, "second summarize_section should succeed")
        _assert(result_2.payload == result_1.payload, "cache hit should return persisted summary")
        _assert(fake_summary_service.calls == 1, "cache hit should skip summary service")
        _assert(fake_resolver.calls == 0, "cache hit should still skip resolver")

        # C. refresh_summary=true forces regenerate
        result_3 = coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=True,
        )
        _assert(result_3.success, "refresh summarize_section should succeed")
        _assert(fake_summary_service.calls == 2, "refresh should invoke summary service again")
        _assert(result_3.payload != result_2.payload, "refresh should overwrite cached summary")

        # D. source_hash mismatch invalidates cache
        bad_hash_summary = SummaryArtifact(
            content="stale-summary",
            language="zh",
            generated_at="2026-01-01T00:00:00+00:00",
            source_hash="not-the-right-hash",
            prompt_version=coordinator._SECTION_SUMMARY_PROMPT_VERSION,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            metadata={
                "summary_scope": "section",
                "section_id": "section-0",
                "source_task_unit_id": "unit-0",
            },
        )
        repository.update_section_summary_artifact(
            doc_name="summary-doc",
            section_id="section-0",
            summary=bad_hash_summary,
        )
        result_4 = coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(result_4.success, "source_hash mismatch regenerate should succeed")
        _assert(fake_summary_service.calls == 3, "source_hash mismatch should invalidate cache")

        # E. split mode / top_k mismatch invalidates cache
        coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="progressive",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(fake_summary_service.calls == 4, "split mode mismatch should regenerate summary")
        coordinator.summarize_section(
            doc_name="summary-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_summary=False,
        )
        _assert(fake_summary_service.calls == 5, "semantic_top_k mismatch should regenerate summary")

        # H. chapter summary cache writes scope=chapter and then cache hit
        chapter_1 = coordinator.summarize_chapter(
            doc_name="summary-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_summary=False,
        )
        _assert(chapter_1.success, "first summarize_chapter should succeed")
        _assert(fake_summary_service.calls == 6, "first chapter call should invoke summary service")
        updated_chapter = repository.load_document("summary-doc")
        section0_chapter = next(
            s for s in updated_chapter.sections if s.section_id == "section-0"
        )
        _assert(
            section0_chapter.task_artifacts is not None
            and section0_chapter.task_artifacts.summary is not None,
            "chapter summary should persist section summary artifact",
        )
        chapter_meta = dict(section0_chapter.task_artifacts.summary.metadata or {})
        _assert(
            chapter_meta.get("summary_scope") == "chapter",
            "chapter summary metadata scope should be 'chapter'",
        )
        _assert(
            chapter_meta.get("chapter_title") == "第一章",
            "chapter summary metadata should include chapter_title",
        )

        chapter_2 = coordinator.summarize_chapter(
            doc_name="summary-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_summary=False,
        )
        _assert(chapter_2.success, "second summarize_chapter should succeed")
        _assert(chapter_2.payload == chapter_1.payload, "chapter cache hit should return persisted summary")
        _assert(fake_summary_service.calls == 6, "chapter cache hit should skip summary service")


def main() -> None:
    test_summary_persistence_cache_flow()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_summary_first_call_writes_cache",
                    "section_summary_second_call_cache_hit",
                    "refresh_summary_forces_regenerate",
                    "source_hash_mismatch_invalidates_cache",
                    "split_mode_topk_mismatch_invalidates_cache",
                    "quiz_artifact_preserved_on_summary_update",
                    "summary_uses_persisted_task_units",
                    "chapter_summary_cache_scope",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

