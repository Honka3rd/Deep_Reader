#!/usr/bin/env python3
"""Artifact availability validity smoke tests for section/task-unit/chapter scopes."""

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
from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from section_tasks.section_task_result import SectionTaskResult
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


class _FakeProfileStore:
    def __init__(self, profile: DocumentProfile | None = None):
        self.profile = profile

    def exists(self, config) -> bool:
        _ = config
        return self.profile is not None

    def load(self, config):
        _ = config
        if self.profile is None:
            raise RuntimeError("profile missing")
        return self.profile


class _FakeEnhancedParseEvaluator:
    @staticmethod
    def evaluate(*, structured_document, affected_section_ratio, fallback_task_unit_ratio, total_task_units):
        _ = (structured_document, affected_section_ratio, fallback_task_unit_ratio, total_task_units)
        return EnhancedParseTriggerDecision(should_recommend=False, score=0, reasons=[], metrics={})


class _FakeResolver:
    def __init__(self):
        self.calls = 0

    def resolve_with_options(self, *, document, split_mode=None, semantic_top_k_candidates=None):
        _ = (split_mode, semantic_top_k_candidates)
        self.calls += 1
        units = []
        for index, section in enumerate(document.sections):
            if not section.content.strip():
                continue
            units.append(
                TaskUnit(
                    unit_id=f"resolved-{index}",
                    title=section.title,
                    container_title=section.container_title,
                    content=section.content.strip(),
                    source_section_ids=[section.section_id],
                    is_fallback_generated=False,
                )
            )
        return units


class _FakeSummaryService:
    def __init__(self):
        self.calls = 0

    def summarize_task_unit(self, *, task_unit, document_title, document_profile=None, task_unit_index=None):
        _ = (document_title, document_profile, task_unit_index)
        self.calls += 1
        return SectionTaskResult.ok(f"generated-summary-{self.calls}")


class _NoopQuizService:
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_document() -> StructuredDocument:
    section_content = "第一章\n這是一段正文。"
    section_hash = _sha(section_content)
    raw_text = section_content
    task_unit_content = section_content
    task_unit_hash = _sha(task_unit_content)
    section = StructuredSection(
        section_id="section-0",
        section_index=0,
        title="第一章",
        level=1,
        content=section_content,
        char_start=0,
        char_end=len(section_content),
        section_role=SectionRole.MAIN_BODY,
        task_units=[
            TaskUnit(
                unit_id="task-unit-0",
                title="第一章",
                container_title=None,
                content=task_unit_content,
                source_section_ids=["section-0"],
                is_fallback_generated=False,
                task_artifacts=TaskArtifacts(
                    summary=SummaryArtifact(
                        content="task-unit-summary",
                        source_hash=task_unit_hash,
                        language="zh",
                        prompt_version="task_unit_summary_v1",
                        task_unit_split_mode="semantic_safe",
                        semantic_top_k_candidates=3,
                    )
                ),
            )
        ],
        task_artifacts=TaskArtifacts(
            summary=SummaryArtifact(
                content="section-summary",
                source_hash=section_hash,
                language="zh",
                prompt_version="section_summary_v1",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "summary_scope": "section",
                    "source_section_id": "section-0",
                    "source_task_unit_id": "task-unit-0",
                },
            ),
            quiz=QuizArtifact(
                items=[{"question_id": "q1", "question_text": "Q", "answer_text": "A"}],
                source_hash=section_hash,
                language="zh",
                prompt_version="section_quiz_v1",
                quiz_schema_version="quiz_schema_v1",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "quiz_scope": "section",
                    "section_id": "section-0",
                    "source_section_id": "section-0",
                    "source_task_unit_id": "task-unit-0",
                },
            ),
        ),
    )
    chapter_artifacts = {
        "chapter::第一章": TaskArtifacts(
            summary=SummaryArtifact(
                content="chapter-summary",
                source_hash=section_hash,
                language="zh",
                prompt_version="chapter_summary_v1",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "summary_scope": "chapter",
                    "chapter_title": "第一章",
                    "chapter_key": "chapter::第一章",
                    "source_section_id": "section-0",
                    "source_task_unit_id": "task-unit-0",
                },
            ),
            quiz=QuizArtifact(
                items=[{"question_id": "cq1", "question_text": "CQ", "answer_text": "CA"}],
                source_hash=section_hash,
                language="zh",
                prompt_version="chapter_quiz_v1",
                quiz_schema_version="quiz_schema_v1",
                task_unit_split_mode="semantic_safe",
                semantic_top_k_candidates=3,
                metadata={
                    "quiz_scope": "chapter",
                    "chapter_title": "第一章",
                    "chapter_key": "chapter::第一章",
                    "source_section_id": "section-0",
                    "source_task_unit_id": "task-unit-0",
                },
            ),
        )
    }
    return StructuredDocument(
        document_id="validity-doc",
        title="Validity Doc",
        source_path=None,
        language="zh",
        raw_text=raw_text,
        sections=[section],
        document_task_artifacts=DocumentTaskArtifacts(
            chapter_artifacts=chapter_artifacts,
            metadata={
                "task_layout": {
                    "source_hash": _sha(raw_text),
                    "task_unit_split_mode": "semantic_safe",
                    "semantic_top_k_candidates": 3,
                    "resolver_version": "task_unit_resolver_v2",
                }
            },
        ),
    )


def _build_coordinator(
    *,
    repository: StructuredDocumentArtifactRepository,
    profile: DocumentProfile | None,
    summary_service: _FakeSummaryService,
    resolver: _FakeResolver,
) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=_FakePreparationPipeline(repository),
        document_artifact_repository=repository,
        document_profile_store=_FakeProfileStore(profile),
        chapter_summary_service=summary_service,
        chapter_quiz_service=_NoopQuizService(),
        task_unit_resolver=resolver,
        enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
        semantic_top_k_candidates_max=20,
    )


def test_artifact_availability_validity() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = StructuredDocumentArtifactRepository(
            store=StructuredDocumentStore(),
            base_dir=temp_dir,
        )
        def write_base_document() -> None:
            doc = _base_document()
            Path(temp_dir, "validity-doc.structured.json").write_text(
                doc.to_json(),
                encoding="utf-8",
            )

        write_base_document()
        profile = DocumentProfile(
            topic="topic",
            summary="summary",
            document_language=LanguageCode.ZH,
        )
        summary_service = _FakeSummaryService()
        resolver = _FakeResolver()
        coordinator = _build_coordinator(
            repository=repo,
            profile=profile,
            summary_service=summary_service,
            resolver=resolver,
        )
        def resolve_chapter_key() -> str:
            current = repo.load_document("validity-doc")
            current_artifacts = current.document_task_artifacts or DocumentTaskArtifacts()
            if current_artifacts.chapter_artifacts:
                if "chapter::第一章" in current_artifacts.chapter_artifacts:
                    return "chapter::第一章"
                return next(iter(current_artifacts.chapter_artifacts.keys()))
            chapter = coordinator._find_chapter_or_raise(
                document=current,
                chapter_id=None,
                chapter_title="第一章",
            )
            return coordinator._build_chapter_artifact_key(chapter)
        def resolve_doc_chapter_artifacts(chapter_key: str) -> TaskArtifacts:
            current = repo.load_document("validity-doc")
            chapter_map = (
                (current.document_task_artifacts or DocumentTaskArtifacts()).chapter_artifacts
            )
            if chapter_key in chapter_map:
                return chapter_map[chapter_key]
            if chapter_map:
                return chapter_map[next(iter(chapter_map.keys()))]
            raise AssertionError("chapter_artifacts should not be empty")
        def resolve_layout_chapter_availability(layout_obj, chapter_key: str):
            if chapter_key in layout_obj.chapter_artifacts:
                return layout_obj.chapter_artifacts[chapter_key]
            if layout_obj.chapter_artifacts:
                return layout_obj.chapter_artifacts[next(iter(layout_obj.chapter_artifacts.keys()))]
            raise AssertionError("layout.chapter_artifacts should not be empty")

        # A/E/G/H/J: valid baseline + no heavy payload.
        layout = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        section_av = layout.sections[0].artifacts
        _assert(section_av is not None and section_av.has_summary, "section summary should exist")
        _assert(section_av.summary_cache_valid is True, "section summary should be valid")
        _assert(section_av.quiz_cache_valid is True, "section quiz should be valid")
        unit_av = layout.task_units[0].artifacts
        _assert(unit_av is not None and unit_av.has_summary, "task unit summary should exist")
        _assert(unit_av.summary_cache_valid is True, "task unit summary should be valid")
        chapter_key = resolve_chapter_key()
        chapter_av = resolve_layout_chapter_availability(layout, chapter_key)
        _assert(chapter_av.summary_cache_valid is True, "chapter summary should be valid")
        _assert(chapter_av.quiz_cache_valid is True, "chapter quiz should be valid")
        payload = layout.to_dict()
        _assert("summary" not in payload["sections"][0], "section heavy summary payload must be absent")
        _assert("quiz" not in payload["sections"][0], "section heavy quiz payload must be absent")

        # B: section summary stale by source hash.
        write_base_document()
        stale = repo.load_document("validity-doc")
        stale_summary = stale.sections[0].task_artifacts.summary
        assert stale_summary is not None
        stale_summary = SummaryArtifact(
            content=stale_summary.content,
            source_hash="bad-hash",
            language=stale_summary.language,
            prompt_version=stale_summary.prompt_version,
            task_unit_split_mode=stale_summary.task_unit_split_mode,
            semantic_top_k_candidates=stale_summary.semantic_top_k_candidates,
            metadata=stale_summary.metadata,
        )
        repo.update_section_summary_artifact("validity-doc", "section-0", stale_summary)
        layout_hash_mismatch = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        av = layout_hash_mismatch.sections[0].artifacts
        _assert(av is not None and av.summary_cache_valid is False, "source-hash mismatch should be invalid")
        _assert(av.summary_invalid_reason == "source_hash_mismatch", "invalid reason should be source_hash_mismatch")

        # C/I: split mode / top-k mismatch should mark stale, not missing.
        write_base_document()
        layout_topk_mismatch = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=5,
        )
        av = layout_topk_mismatch.sections[0].artifacts
        _assert(av is not None and av.has_quiz is True, "artifact should still exist")
        _assert(av.quiz_cache_valid is False, "top-k mismatch should invalidate quiz cache")
        _assert(av.quiz_invalid_reason == "semantic_top_k_mismatch", "invalid reason should reflect top-k mismatch")

        layout_mode_mismatch = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="progressive",
        )
        av_mode = layout_mode_mismatch.sections[0].artifacts
        _assert(av_mode is not None and av_mode.summary_cache_valid is False, "mode mismatch should invalidate")
        _assert(av_mode.summary_invalid_reason == "split_mode_mismatch", "invalid reason should be split mode mismatch")

        # D: malformed section quiz items.
        write_base_document()
        malformed_doc = repo.load_document("validity-doc")
        quiz = malformed_doc.sections[0].task_artifacts.quiz
        assert quiz is not None
        malformed_quiz = QuizArtifact(
            items=[{"question_id": "q1", "question_text": "Q"}],
            source_hash=quiz.source_hash,
            language=quiz.language,
            prompt_version=quiz.prompt_version,
            quiz_schema_version=quiz.quiz_schema_version,
            task_unit_split_mode=quiz.task_unit_split_mode,
            semantic_top_k_candidates=quiz.semantic_top_k_candidates,
            metadata=quiz.metadata,
        )
        repo.update_section_quiz_artifact("validity-doc", "section-0", malformed_quiz)
        malformed_layout = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        malformed_av = malformed_layout.sections[0].artifacts
        _assert(malformed_av is not None and malformed_av.quiz_cache_valid is False, "malformed quiz must be invalid")
        _assert(malformed_av.quiz_invalid_reason == "malformed_quiz_items", "invalid reason should be malformed_quiz_items")

        # F: chapter summary source section not found.
        write_base_document()
        missing_source_doc = repo.load_document("validity-doc")
        chapter_key = resolve_chapter_key()
        chapter_summary = resolve_doc_chapter_artifacts(chapter_key).summary
        assert chapter_summary is not None
        metadata = dict(chapter_summary.metadata or {})
        metadata["source_section_id"] = "missing-section"
        repo.update_chapter_summary_artifact(
            "validity-doc",
            chapter_key,
            SummaryArtifact(
                content=chapter_summary.content,
                source_hash=chapter_summary.source_hash,
                language=chapter_summary.language,
                prompt_version=chapter_summary.prompt_version,
                task_unit_split_mode=chapter_summary.task_unit_split_mode,
                semantic_top_k_candidates=chapter_summary.semantic_top_k_candidates,
                metadata=metadata,
            ),
        )
        chapter_missing_layout = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        chapter_missing_av = resolve_layout_chapter_availability(chapter_missing_layout, chapter_key)
        _assert(chapter_missing_av.summary_cache_valid is False, "missing chapter source section should invalidate")
        _assert(chapter_missing_av.summary_invalid_reason == "source_section_not_found", "reason should be source_section_not_found")

        # G: chapter quiz schema mismatch.
        write_base_document()
        chapter_quiz_doc = repo.load_document("validity-doc")
        chapter_key = resolve_chapter_key()
        chapter_quiz = resolve_doc_chapter_artifacts(chapter_key).quiz
        assert chapter_quiz is not None
        repo.update_chapter_quiz_artifact(
            "validity-doc",
            chapter_key,
            QuizArtifact(
                items=list(chapter_quiz.items),
                source_hash=chapter_quiz.source_hash,
                language=chapter_quiz.language,
                prompt_version=chapter_quiz.prompt_version,
                quiz_schema_version="quiz_schema_old",
                task_unit_split_mode=chapter_quiz.task_unit_split_mode,
                semantic_top_k_candidates=chapter_quiz.semantic_top_k_candidates,
                metadata=chapter_quiz.metadata,
            ),
        )
        chapter_quiz_layout = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        chapter_quiz_av = resolve_layout_chapter_availability(chapter_quiz_layout, chapter_key)
        _assert(chapter_quiz_av.quiz_cache_valid is False, "chapter quiz schema mismatch should invalidate")
        _assert(chapter_quiz_av.quiz_invalid_reason == "quiz_schema_version_mismatch", "reason should be quiz_schema_version_mismatch")

        # H: task-unit summary stale by source hash.
        write_base_document()
        task_unit_doc = repo.load_document("validity-doc")
        task_unit = task_unit_doc.sections[0].task_units[0]
        tu_summary = task_unit.task_artifacts.summary
        assert tu_summary is not None
        bad_tu_summary = SummaryArtifact(
            content=tu_summary.content,
            source_hash="bad-task-unit-hash",
            language=tu_summary.language,
            prompt_version=tu_summary.prompt_version,
            task_unit_split_mode=tu_summary.task_unit_split_mode,
            semantic_top_k_candidates=tu_summary.semantic_top_k_candidates,
            metadata=tu_summary.metadata,
        )
        updated_units = [TaskUnit(
            unit_id=task_unit.unit_id,
            title=task_unit.title,
            container_title=task_unit.container_title,
            content=task_unit.content,
            source_section_ids=list(task_unit.source_section_ids),
            is_fallback_generated=task_unit.is_fallback_generated,
            task_artifacts=TaskArtifacts(
                summary=bad_tu_summary,
                quiz=(None if task_unit.task_artifacts is None else task_unit.task_artifacts.quiz),
            ),
        )]
        repo.update_section_task_units("validity-doc", "section-0", updated_units)
        task_unit_layout = coordinator.get_document_task_layout(
            doc_name="validity-doc",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
        )
        tu_av = task_unit_layout.task_units[0].artifacts
        _assert(tu_av is not None and tu_av.summary_cache_valid is False, "task-unit source hash mismatch should invalidate")
        _assert(tu_av.summary_invalid_reason == "source_hash_mismatch", "task-unit invalid reason should be source_hash_mismatch")

        # K: endpoint cache consistency with layout validity.
        # valid summary -> cache_hit true
        write_base_document()
        result_valid = coordinator.summarize_section(
            doc_name="validity-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(result_valid.success is True, "valid summary path should succeed")
        _assert(result_valid.cache_hit is True, "valid cached summary should hit cache")

        # stale summary -> cache_hit false and regenerate
        stale_doc = repo.load_document("validity-doc")
        stale_section_summary = stale_doc.sections[0].task_artifacts.summary
        assert stale_section_summary is not None
        repo.update_section_summary_artifact(
            "validity-doc",
            "section-0",
            SummaryArtifact(
                content=stale_section_summary.content,
                source_hash="stale-hash",
                language=stale_section_summary.language,
                prompt_version=stale_section_summary.prompt_version,
                task_unit_split_mode=stale_section_summary.task_unit_split_mode,
                semantic_top_k_candidates=stale_section_summary.semantic_top_k_candidates,
                metadata=stale_section_summary.metadata,
            ),
        )
        result_stale = coordinator.summarize_section(
            doc_name="validity-doc",
            section_id="section-0",
            task_unit_split_mode="semantic_safe",
            semantic_top_k_candidates=3,
            refresh_summary=False,
        )
        _assert(result_stale.success is True, "stale summary should regenerate successfully")
        _assert(result_stale.cache_hit is False, "stale summary should not hit cache")
        _assert(summary_service.calls >= 1, "stale summary should invoke summary generation service")


def main() -> None:
    test_artifact_availability_validity()
    print(
        json.dumps(
            {
                "status": "ok",
                "tests": [
                    "section_summary_valid_and_source_hash_mismatch",
                    "split_mode_and_topk_mismatch",
                    "section_quiz_malformed_items",
                    "chapter_source_section_not_found",
                    "chapter_quiz_schema_mismatch",
                    "task_unit_summary_source_hash_mismatch",
                    "no_heavy_payload",
                    "endpoint_layout_cache_consistency",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
