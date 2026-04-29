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

        chapter_key = SectionTaskCoordinator._build_chapter_artifact_key("第一章")

        # A. section quiz remains section-level
        quiz_1 = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_quiz=False,
        )
        _assert(quiz_1.success, "first section quiz should succeed")
        _assert(quiz_1.cache_hit is False, "first section quiz should be cache miss")
        _assert(fake_quiz_service.calls == 1, "first section quiz should call quiz service")
        _assert(fake_resolver.calls == 0, "persisted task units should avoid resolver call")
        updated_1 = repository.load_document("quiz-doc")
        section0_1 = next(s for s in updated_1.sections if s.section_id == "section-0")
        _assert(
            section0_1.task_artifacts is not None and section0_1.task_artifacts.quiz is not None,
            "first section quiz should persist section quiz artifact",
        )
        _assert(
            not (
                updated_1.document_task_artifacts
                and updated_1.document_task_artifacts.chapter_artifacts
            ),
            "section quiz should not create chapter-level quiz artifact",
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
        _assert(quiz_2.cache_hit is True, "second section quiz should be cache hit")
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
        _assert(quiz_3.cache_hit is False, "refresh section quiz should regenerate")
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

        # B/C/D. chapter quiz writes document-level + does not overwrite section-level
        section_doc_before_chapter = repository.load_document("quiz-doc")
        section_quiz_before_chapter = next(
            s for s in section_doc_before_chapter.sections if s.section_id == "section-0"
        ).task_artifacts.quiz
        chapter_1 = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(chapter_1.success, "first chapter quiz should succeed")
        _assert(chapter_1.cache_hit is False, "first chapter quiz should be cache miss")
        _assert(fake_quiz_service.calls == before_malformed_calls + 2, "chapter quiz first call should regenerate")
        chapter_doc = repository.load_document("quiz-doc")
        chapter_section = next(s for s in chapter_doc.sections if s.section_id == "section-0")
        section_quiz_after_chapter = (
            None if chapter_section.task_artifacts is None else chapter_section.task_artifacts.quiz
        )
        _assert(
            section_quiz_after_chapter is not None and section_quiz_before_chapter is not None,
            "section quiz should still exist after chapter quiz write",
        )
        _assert(
            section_quiz_after_chapter.to_dict() == section_quiz_before_chapter.to_dict(),
            "chapter quiz write must not overwrite section-level quiz",
        )
        chapter_artifacts = (
            chapter_doc.document_task_artifacts.chapter_artifacts
            if chapter_doc.document_task_artifacts
            else {}
        )
        _assert(
            chapter_key in chapter_artifacts and chapter_artifacts[chapter_key].quiz is not None,
            "chapter quiz should persist at document-level chapter artifact",
        )
        chapter_quiz = chapter_artifacts[chapter_key].quiz
        assert chapter_quiz is not None
        chapter_meta = dict(chapter_quiz.metadata or {})
        _assert(chapter_meta.get("quiz_scope") == "chapter", "chapter quiz scope metadata should be chapter")
        _assert(chapter_meta.get("chapter_title") == "第一章", "chapter quiz metadata should include chapter title")
        _assert(chapter_meta.get("chapter_key") == chapter_key, "chapter quiz metadata should include chapter key")

        # E. chapter quiz cache hit (document-level)
        chapter_2 = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(chapter_2.success, "second chapter quiz should succeed")
        _assert(chapter_2.cache_hit is True, "second chapter quiz should be cache hit")
        _assert(
            fake_quiz_service.calls == before_malformed_calls + 2,
            "second chapter quiz should hit cache",
        )

        # F. chapter refresh updates chapter artifact only
        chapter_before_refresh_doc = repository.load_document("quiz-doc")
        chapter_before_refresh = (
            chapter_before_refresh_doc.document_task_artifacts.chapter_artifacts[chapter_key].quiz
        )
        section_before_refresh = next(
            s for s in chapter_before_refresh_doc.sections if s.section_id == "section-0"
        ).task_artifacts.quiz
        chapter_refresh = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=True,
        )
        _assert(chapter_refresh.success, "chapter refresh quiz should succeed")
        _assert(chapter_refresh.cache_hit is False, "chapter refresh quiz should regenerate")
        _assert(
            fake_quiz_service.calls == before_malformed_calls + 3,
            "chapter refresh should regenerate chapter quiz",
        )
        chapter_after_refresh_doc = repository.load_document("quiz-doc")
        chapter_after_refresh = (
            chapter_after_refresh_doc.document_task_artifacts.chapter_artifacts[chapter_key].quiz
        )
        section_after_refresh = next(
            s for s in chapter_after_refresh_doc.sections if s.section_id == "section-0"
        ).task_artifacts.quiz
        _assert(
            chapter_before_refresh is not None
            and chapter_after_refresh is not None
            and chapter_after_refresh.to_dict() != chapter_before_refresh.to_dict(),
            "chapter refresh should update document-level chapter quiz artifact",
        )
        _assert(
            section_before_refresh is not None
            and section_after_refresh is not None
            and section_after_refresh.to_dict() == section_before_refresh.to_dict(),
            "chapter refresh should not change section-level quiz artifact",
        )

        # G. legacy wrong chapter-in-section quiz should be ignored (no chapter cache hit)
        before_legacy_doc = repository.load_document("quiz-doc")
        before_legacy_artifacts = before_legacy_doc.document_task_artifacts or DocumentTaskArtifacts()
        before_legacy_chapters = dict(before_legacy_artifacts.chapter_artifacts)
        legacy_entry = before_legacy_chapters.get(chapter_key) or TaskArtifacts()
        before_legacy_chapters[chapter_key] = TaskArtifacts(
            summary=legacy_entry.summary,
            quiz=None,
        )
        repository.update_document_artifacts(
            doc_name="quiz-doc",
            artifacts=DocumentTaskArtifacts(
                chapter_artifacts=before_legacy_chapters,
                metadata=dict(before_legacy_artifacts.metadata),
            ),
        )

        legacy_wrong = QuizArtifact(
            items=[
                {"question_id": "legacy-1", "question_text": "legacy", "answer_text": "legacy"},
                {"question_id": "legacy-2", "question_text": "legacy", "answer_text": "legacy"},
                {"question_id": "legacy-3", "question_text": "legacy", "answer_text": "legacy"},
                {"question_id": "legacy-4", "question_text": "legacy", "answer_text": "legacy"},
            ],
            language="zh",
            generated_at="2026-01-01T00:00:00+00:00",
            source_hash=hashlib.sha256(chapter_section.content.encode("utf-8")).hexdigest(),
            prompt_version=coordinator._CHAPTER_QUIZ_PROMPT_VERSION,
            quiz_schema_version=coordinator._QUIZ_SCHEMA_VERSION,
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            metadata={
                "quiz_scope": "chapter",
                "chapter_title": "第一章",
                "chapter_key": chapter_key,
                "source_section_id": "section-0",
                "source_task_unit_id": "unit-0",
            },
        )
        repository.update_section_quiz_artifact(
            doc_name="quiz-doc",
            section_id="section-0",
            quiz=legacy_wrong,
        )
        calls_before_legacy = fake_quiz_service.calls
        legacy_result = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(legacy_result.success, "chapter quiz should still succeed with legacy wrong section artifact")
        _assert(
            fake_quiz_service.calls == calls_before_legacy + 1,
            "legacy section-level chapter quiz must not be treated as chapter cache hit",
        )
        legacy_doc = repository.load_document("quiz-doc")
        legacy_chapter_quiz = legacy_doc.document_task_artifacts.chapter_artifacts[chapter_key].quiz
        _assert(
            legacy_chapter_quiz is not None,
            "legacy case should regenerate and write document-level chapter quiz",
        )

        # H. chapter summary and chapter quiz coexist
        chapter_doc_before_summary = repository.load_document("quiz-doc")
        existing_doc_artifacts = chapter_doc_before_summary.document_task_artifacts or DocumentTaskArtifacts()
        chapter_map = dict(existing_doc_artifacts.chapter_artifacts)
        chapter_entry = chapter_map.get(chapter_key) or TaskArtifacts()
        chapter_map[chapter_key] = TaskArtifacts(
            summary=SummaryArtifact(
                content="chapter-summary",
                language="zh",
                generated_at="2026-01-01T00:00:00+00:00",
                source_hash=hashlib.sha256(chapter_section.content.encode("utf-8")).hexdigest(),
                prompt_version="chapter_summary_v1",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=5,
                metadata={
                    "summary_scope": "chapter",
                    "chapter_title": "第一章",
                    "chapter_key": chapter_key,
                    "source_section_id": "section-0",
                    "source_task_unit_id": "unit-0",
                },
            ),
            quiz=chapter_entry.quiz,
        )
        repository.update_document_artifacts(
            doc_name="quiz-doc",
            artifacts=DocumentTaskArtifacts(
                chapter_artifacts=chapter_map,
                metadata=dict(existing_doc_artifacts.metadata),
            ),
        )
        coexist_result = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第一章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=False,
        )
        _assert(coexist_result.success, "chapter quiz coexist check should succeed")
        coexist_doc = repository.load_document("quiz-doc")
        coexist_entry = coexist_doc.document_task_artifacts.chapter_artifacts[chapter_key]
        _assert(coexist_entry.summary is not None, "chapter summary should be preserved")
        _assert(coexist_entry.quiz is not None, "chapter quiz should remain present")

        # D (reverse order safe): chapter quiz first then section quiz should keep both scopes
        chapter_first = coordinator.generate_chapter_quiz(
            doc_name="quiz-doc",
            chapter_title="第二章",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=True,
        )
        _assert(chapter_first.success, "chapter-first quiz on 第二章 should succeed")
        section_second = coordinator.generate_section_quiz(
            doc_name="quiz-doc",
            section_id="section-1",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
            refresh_quiz=True,
        )
        _assert(section_second.success, "section-second quiz on section-1 should succeed")
        reverse_doc = repository.load_document("quiz-doc")
        chapter_key_2 = SectionTaskCoordinator._build_chapter_artifact_key("第二章")
        _assert(
            reverse_doc.document_task_artifacts.chapter_artifacts.get(chapter_key_2) is not None,
            "reverse order should keep chapter-level quiz for 第二章",
        )
        reverse_section_1 = next(s for s in reverse_doc.sections if s.section_id == "section-1")
        _assert(
            reverse_section_1.task_artifacts is not None
            and reverse_section_1.task_artifacts.quiz is not None,
            "reverse order should keep section-level quiz for section-1",
        )


def main() -> None:
    test_quiz_persistence_cache_flow()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_quiz_remains_section_level",
                    "section_quiz_second_call_cache_hit",
                    "refresh_quiz_forces_regenerate",
                    "source_hash_mismatch_invalidates_cache",
                    "split_mode_topk_mismatch_invalidates_cache",
                    "summary_artifact_preserved_on_quiz_update",
                    "quiz_uses_persisted_task_units",
                    "malformed_cached_quiz_fallback",
                    "chapter_quiz_document_level_cache",
                    "chapter_refresh_updates_chapter_artifact_only",
                    "legacy_section_level_chapter_quiz_ignored",
                    "chapter_summary_and_quiz_coexist",
                    "reverse_order_scope_safety",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
