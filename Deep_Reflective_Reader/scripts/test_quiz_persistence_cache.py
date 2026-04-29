#!/usr/bin/env python3
"""Coordinator-level smoke tests for section/chapter quiz persistence cache."""

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
from section_tasks.quiz_question import QuizQuestion
from section_tasks.section_task_result import SectionTaskResult
from section_tasks.task_unit_split_mode import TaskUnitSplitMode
from shared.task_artifacts import QuizArtifact, SummaryArtifact, TaskArtifacts
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
    pass


class _FakeQuizService:
    def __init__(self):
        self.calls = 0

    def generate_task_unit_quiz(
        self,
        task_unit: TaskUnit,
        document_title: str | None = None,
        document_profile=None,
        *,
        task_type: str = "section_quiz",
        task_unit_index: int = 0,
    ) -> SectionTaskResult[list[QuizQuestion]]:
        _ = (task_unit, document_title, document_profile, task_type, task_unit_index)
        self.calls += 1
        return SectionTaskResult.ok(
            [
                QuizQuestion(
                    question_id=f"q-{self.calls}-1",
                    question_text=f"question-{self.calls}-1",
                    answer_text=f"answer-{self.calls}-1",
                ),
                QuizQuestion(
                    question_id=f"q-{self.calls}-2",
                    question_text=f"question-{self.calls}-2",
                    answer_text=f"answer-{self.calls}-2",
                ),
                QuizQuestion(
                    question_id=f"q-{self.calls}-3",
                    question_text=f"question-{self.calls}-3",
                    answer_text=f"answer-{self.calls}-3",
                ),
                QuizQuestion(
                    question_id=f"q-{self.calls}-4",
                    question_text=f"question-{self.calls}-4",
                    answer_text=f"answer-{self.calls}-4",
                ),
            ]
        )


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
            summary=SummaryArtifact(content="existing-section-summary", language="zh")
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
        document_id="quiz-doc",
        title="Quiz Doc",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=[section0, section1],
        document_task_artifacts=None,
    )


def _seed_valid_task_layout_metadata(
    *,
    repository: StructuredDocumentArtifactRepository,
    doc_name: str,
) -> None:
    document = repository.load_document(doc_name)
    source_hash = hashlib.sha256(document.raw_text.encode("utf-8")).hexdigest()
    metadata = {
        "task_layout": {
            "source_hash": source_hash,
            "task_unit_split_mode": "semantic_safe",
            "semantic_top_k_candidates": 3,
            "resolver_version": "task_unit_resolver_v1",
        }
    }
    from shared.task_artifacts import DocumentTaskArtifacts
    repository.update_document_artifacts(
        doc_name=doc_name,
        artifacts=DocumentTaskArtifacts(chapter_artifacts={}, metadata=metadata),
    )


def test_quiz_persistence_cache_flow() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        store = StructuredDocumentStore()
        repository = StructuredDocumentArtifactRepository(store=store, base_dir=temp_dir)
        base_document = _build_base_document()
        (temp_path / "quiz-doc.structured.json").write_text(
            base_document.to_json(),
            encoding="utf-8",
        )
        _seed_valid_task_layout_metadata(repository=repository, doc_name="quiz-doc")

        fake_quiz_service = _FakeQuizService()
        fake_resolver = _FakeTaskUnitResolver()
        coordinator = SectionTaskCoordinator(
            document_preparation_pipeline=_FakePreparationPipeline(repository),
            document_artifact_repository=repository,
            document_profile_store=_FakeProfileStore(),
            chapter_summary_service=_FakeSummaryService(),
            chapter_quiz_service=fake_quiz_service,
            task_unit_resolver=fake_resolver,
            enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
            semantic_top_k_candidates_max=20,
        )

        # A. first call writes section quiz cache
        quiz_1 = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(quiz_1.success, "first section quiz should succeed")
        _assert(fake_quiz_service.calls == 1, "first section quiz should call quiz service")
        _assert(fake_resolver.calls == 0, "persisted task units should avoid resolver call")
        updated_1 = repository.load_document("quiz-doc")
        section0_1 = next(s for s in updated_1.sections if s.section_id == "section-0")
        _assert(
            section0_1.task_artifacts is not None and section0_1.task_artifacts.quiz is not None,
            "first section quiz should persist section quiz artifact",
        )

        # F. summary preservation
        _assert(
            section0_1.task_artifacts.summary is not None,
            "quiz write must preserve existing summary artifact",
        )

        # B. second call cache hit
        quiz_2 = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(quiz_2.success, "second section quiz should succeed")
        _assert(fake_quiz_service.calls == 1, "second section quiz should hit cache")
        _assert(
            [q.question_id for q in quiz_2.payload] == [q.question_id for q in quiz_1.payload],
            "cache hit should return same quiz payload",
        )

        # C. refresh_quiz=true forces regenerate
        quiz_3 = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=True,
        )
        _assert(quiz_3.success, "refresh section quiz should succeed")
        _assert(fake_quiz_service.calls == 2, "refresh section quiz should regenerate")

        # D. source_hash mismatch invalidates cache
        bad_hash_quiz = QuizArtifact(
            items=[
                {"question_id": "bad-1", "question_text": "bad", "answer_text": "bad"},
                {"question_id": "bad-2", "question_text": "bad", "answer_text": "bad"},
                {"question_id": "bad-3", "question_text": "bad", "answer_text": "bad"},
                {"question_id": "bad-4", "question_text": "bad", "answer_text": "bad"},
            ],
            language="zh",
            generated_at="2026-01-01T00:00:00+00:00",
            source_hash="wrong-hash",
            prompt_version=coordinator._SECTION_QUIZ_PROMPT_VERSION,
            quiz_schema_version=coordinator._QUIZ_SCHEMA_VERSION,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            metadata={
                "quiz_scope": "section",
                "section_id": "section-0",
                "source_task_unit_id": "unit-0",
            },
        )
        repository.update_section_quiz_artifact(
            doc_name="quiz-doc",
            section_id="section-0",
            quiz=bad_hash_quiz,
        )
        quiz_4 = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(quiz_4.success, "source hash mismatch quiz regenerate should succeed")
        _assert(fake_quiz_service.calls == 3, "source hash mismatch should invalidate quiz cache")

        # E. split mode / top_k mismatch invalidates
        coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="progressive",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(fake_quiz_service.calls == 4, "split mode mismatch should regenerate quiz")
        coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(fake_quiz_service.calls == 5, "semantic top-k mismatch should regenerate quiz")

        # H. malformed cached quiz fallback
        malformed_quiz = QuizArtifact(
            items=[{"question_id": "x"}],
            language="zh",
            generated_at="2026-01-01T00:00:00+00:00",
            source_hash=hashlib.sha256(section0_1.content.encode("utf-8")).hexdigest(),
            prompt_version=coordinator._SECTION_QUIZ_PROMPT_VERSION,
            quiz_schema_version=coordinator._QUIZ_SCHEMA_VERSION,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            metadata={
                "quiz_scope": "section",
                "section_id": "section-0",
                "source_task_unit_id": "unit-0",
            },
        )
        repository.update_section_quiz_artifact(
            doc_name="quiz-doc",
            section_id="section-0",
            quiz=malformed_quiz,
        )
        before_malformed_calls = fake_quiz_service.calls
        malformed_result = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(malformed_result.success, "malformed cached quiz should fallback and regenerate")
        _assert(
            fake_quiz_service.calls == before_malformed_calls + 1,
            "malformed cached quiz should not cause 500; should regenerate",
        )

        # I. chapter quiz cache
        chapter_1 = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(chapter_1.success, "first chapter quiz should succeed")
        _assert(fake_quiz_service.calls == before_malformed_calls + 2, "chapter quiz first call should regenerate")
        chapter_doc = repository.load_document("quiz-doc")
        chapter_section = next(s for s in chapter_doc.sections if s.section_id == "section-0")
        chapter_quiz = chapter_section.task_artifacts.quiz
        _assert(chapter_quiz is not None, "chapter quiz should persist in section quiz slot")
        chapter_meta = dict(chapter_quiz.metadata or {})
        _assert(chapter_meta.get("quiz_scope") == "chapter", "chapter quiz scope metadata should be chapter")
        _assert(chapter_meta.get("chapter_title") == "第一章", "chapter quiz metadata should include chapter title")
        chapter_2 = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(chapter_2.success, "second chapter quiz should succeed")
        _assert(
            fake_quiz_service.calls == before_malformed_calls + 2,
            "second chapter quiz should hit cache",
        )


def main() -> None:
    test_quiz_persistence_cache_flow()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_quiz_first_call_writes_cache",
                    "section_quiz_second_call_cache_hit",
                    "refresh_quiz_forces_regenerate",
                    "source_hash_mismatch_invalidates_cache",
                    "split_mode_topk_mismatch_invalidates_cache",
                    "summary_artifact_preserved_on_quiz_update",
                    "quiz_uses_persisted_task_units",
                    "malformed_cached_quiz_fallback",
                    "chapter_quiz_cache",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

