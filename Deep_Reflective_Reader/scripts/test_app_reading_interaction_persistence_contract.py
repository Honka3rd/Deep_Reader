#!/usr/bin/env python3
"""Regression tests for app-layer reading interaction persistence contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.section_task_coordinator import (  # noqa: E402
    CriticalThinkingSessionStoreResult,
    SectionTaskCoordinator,
)
from document_structure.structured_document import (  # noqa: E402
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)
from section_tasks.analysis_interaction_orchestrator import (  # noqa: E402
    AnalysisInteractionOrchestrator,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.artifact_validity import (  # noqa: E402
    ReadingInteractionTargetValidity,
)
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
)
from section_tasks.quiz_interaction_orchestrator import (  # noqa: E402
    QuizInteractionOrchestrator,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from shared.task_unit_model import TaskUnit  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _document(
    *,
    section_content: str = "This section explains how assumptions shape conclusions.",
    task_unit_content: str = "This task unit explains how assumptions shape conclusions.",
) -> StructuredDocument:
    task_unit = TaskUnit(
        unit_id="unit-1",
        title="Unit One",
        container_title="Section One",
        content=task_unit_content,
        source_section_ids=["section-1"],
        is_fallback_generated=False,
        parent_section_id="section-1",
    )
    section = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="Section One",
        level=2,
        content=section_content,
        char_start=0,
        char_end=len(section_content),
        parent_chapter_id="chapter-1",
        task_units=[task_unit],
    )
    chapter = StructuredChapter(
        chapter_id="chapter-1",
        title="Chapter One",
        level=1,
        chapter_role=None,
        sections=[section],
    )
    return StructuredDocument(
        document_id="doc-1",
        title="Document One",
        source_path=None,
        language="en",
        raw_text=section_content,
        chapters=[chapter],
    )


class FakePreparationPipeline:
    def __init__(
        self,
        *,
        section_content: str = "This section explains how assumptions shape conclusions.",
        task_unit_content: str = "This task unit explains how assumptions shape conclusions.",
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self._section_content = section_content
        self._task_unit_content = task_unit_content

    def prepare_and_load(self, *, doc_name: str, mode: object) -> SimpleNamespace:
        self.calls.append((doc_name, mode))
        return SimpleNamespace(
            structured_document=_document(
                section_content=self._section_content,
                task_unit_content=self._task_unit_content,
            ),
            assets=SimpleNamespace(errors=[]),
        )


class InMemoryCurrentArtifactStore:
    def __init__(self) -> None:
        self.artifacts: dict[tuple[str, str, str], ReadingInteractionArtifact] = {}
        self.get_calls = 0
        self.save_calls = 0

    def get_analysis_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self.get_calls += 1
        return self.artifacts.get(self._key_from_request(request))

    def save_analysis_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self.save_calls += 1
        self.artifacts[self._key_from_artifact(artifact)] = artifact
        return artifact

    def get_quiz_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self.get_calls += 1
        return self.artifacts.get(self._key_from_request(request))

    def save_quiz_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self.save_calls += 1
        self.artifacts[self._key_from_artifact(artifact)] = artifact
        return artifact

    @staticmethod
    def _key_from_request(
        request: ReadingInteractionRequest,
    ) -> tuple[str, str, str]:
        target = request.target
        return (target.document_id, target.target_level, target.target_id)

    @staticmethod
    def _key_from_artifact(
        artifact: ReadingInteractionArtifact,
    ) -> tuple[str, str, str]:
        return (artifact.document_id, artifact.target_level, artifact.target_id)


class InMemoryCriticalThinkingSessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, ReadingInteractionArtifact] = {}
        self.get_calls: list[tuple[ReadingInteractionRequest, str]] = []
        self.save_calls: list[ReadingInteractionArtifact] = []
        self._active_session_id: str | None = None
        self._next_id = 1

    def get_critical_thinking_session(
        self,
        request: ReadingInteractionRequest,
        session_id: str,
    ) -> ReadingInteractionArtifact | None:
        self.get_calls.append((request, session_id))
        self._active_session_id = session_id
        return self.sessions.get(session_id)

    def save_critical_thinking_session(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> CriticalThinkingSessionStoreResult:
        self.save_calls.append(artifact)
        session_id = self._active_session_id
        if session_id is None:
            session_id = f"session-{self._next_id}"
            self._next_id += 1
        self.sessions[session_id] = artifact
        self._active_session_id = session_id
        return CriticalThinkingSessionStoreResult(
            session_id=session_id,
            artifact=artifact,
        )


def _coordinator(
    *,
    preparation_pipeline: FakePreparationPipeline | None = None,
    analysis_orchestrator: AnalysisInteractionOrchestrator | None = None,
    quiz_orchestrator: QuizInteractionOrchestrator | None = None,
    critical_service: CriticalThinkingSessionService | None = None,
    critical_store: InMemoryCriticalThinkingSessionStore | None = None,
) -> SectionTaskCoordinator:
    return SectionTaskCoordinator(
        document_preparation_pipeline=preparation_pipeline or FakePreparationPipeline(),
        document_artifact_repository=object(),
        document_profile_store=object(),
        chapter_summary_service=object(),
        chapter_quiz_service=object(),
        task_unit_resolver=object(),
        enhanced_parse_trigger_evaluator=object(),
        analysis_interaction_orchestrator=analysis_orchestrator,
        quiz_interaction_orchestrator=quiz_orchestrator,
        critical_thinking_session_service=critical_service,
        critical_thinking_session_store=critical_store,
    )


def test_analysis_read_write_refresh_and_stale_contract() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": f"summary-{len(generator_calls)}",
            "reasoning": "reasoning",
            "explanation": "explanation",
        }

    store = InMemoryCurrentArtifactStore()
    coordinator = _coordinator(
        analysis_orchestrator=AnalysisInteractionOrchestrator(
            service=AnalysisInteractionService(generator),
            artifact_store=store,
        ),
    )

    missing = coordinator.read_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    generated = coordinator.generate_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    reused = coordinator.generate_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    refreshed = coordinator.refresh_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )

    _assert(missing.status == "not_generated", "read of missing analysis should be inert")
    _assert(generator_calls == ["section-1", "section-1"], "only explicit writes should generate")
    _assert(store.save_calls == 2, "generate and refresh should write validated results")
    _assert(generated.payload["summary"] == "summary-1", "first generate should persist")
    _assert(reused.payload["summary"] == "summary-1", "generate without refresh should reuse current artifact")
    _assert(refreshed.payload["summary"] == "summary-2", "refresh should replace current artifact")

    stale_store = InMemoryCurrentArtifactStore()
    stale_coordinator = _coordinator(
        analysis_orchestrator=AnalysisInteractionOrchestrator(
            service=AnalysisInteractionService(
                generator,
                target_validity_checker=lambda request: ReadingInteractionTargetValidity.stale(
                    "target source structure is stale"
                ),
            ),
            artifact_store=stale_store,
        ),
    )
    stale = stale_coordinator.generate_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )

    _assert(stale.status == "stale_target", "stale target should be explicit")
    _assert(stale.reason == "target source structure is stale", "stale reason should survive app mapping")
    _assert(stale_store.save_calls == 0, "stale target must not be persisted as current success")


def test_quiz_persists_insufficient_content_and_rejects_invalid_success() -> None:
    generator_calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        generator_calls.append(request.target.target_id)
        return {
            "items": [
                {
                    "type": "short_answer",
                    "question": "What matters?",
                    "answer": "Assumptions",
                }
            ]
        }

    insufficient_store = InMemoryCurrentArtifactStore()
    insufficient_coordinator = _coordinator(
        preparation_pipeline=FakePreparationPipeline(section_content="@@@ !!! ###"),
        quiz_orchestrator=QuizInteractionOrchestrator(
            service=QuizInteractionService(generator),
            artifact_store=insufficient_store,
        ),
    )
    insufficient = insufficient_coordinator.generate_quiz_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    read_insufficient = insufficient_coordinator.read_quiz_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )

    _assert(insufficient.status == "insufficient_content", "insufficient content should be terminal")
    _assert(generator_calls == [], "insufficient content should not call the quiz generator")
    _assert(insufficient_store.save_calls == 1, "insufficient content should be persisted")
    _assert(read_insufficient.status == "insufficient_content", "read should see persisted insufficient result")

    invalid_store = InMemoryCurrentArtifactStore()
    invalid_coordinator = _coordinator(
        quiz_orchestrator=QuizInteractionOrchestrator(
            service=QuizInteractionService(lambda request, max_items, valid_types: {"items": []}),
            artifact_store=invalid_store,
        ),
    )
    invalid = invalid_coordinator.generate_quiz_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )

    _assert(invalid.status == "generation_failed", "invalid quiz output should fail validation")
    _assert(invalid_store.save_calls == 0, "invalid generated output must not be persisted")


def test_critical_thinking_failed_evaluation_is_saved_with_answer_for_retry() -> None:
    attempts: list[int] = []

    def evaluation_generator(
        session: ReadingInteractionArtifact,
        instruction: str,
    ) -> dict[str, object]:
        attempts.append(1)
        if len(attempts) == 1:
            return {"feedback": ""}
        return {
            "feedback": "The answer identifies a real assumption.",
            "strengths": "It explains causal stakes.",
            "improvements": "It could cite the source more directly.",
        }

    store = InMemoryCriticalThinkingSessionStore()
    coordinator = _coordinator(
        critical_service=CriticalThinkingSessionService(
            lambda request, instruction: {
                "question": "Which hidden assumption changes the conclusion?"
            },
            evaluation_generator,
        ),
        critical_store=store,
    )

    generated = coordinator.generate_critical_thinking_question(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    failed = coordinator.submit_critical_thinking_answer(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id or "",
        answer="The hidden assumption is that incentives outweigh constraints.",
    )
    read_failed = coordinator.read_critical_thinking_session(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id,
    )
    retried = coordinator.retry_critical_thinking_evaluation(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id or "",
    )

    _assert(failed.status == "evaluation_failed", "failed evaluation should be returned")
    _assert(
        failed.payload["answer"] == "The hidden assumption is that incentives outweigh constraints.",
        "failed evaluation should preserve submitted answer",
    )
    _assert(read_failed.status == "evaluation_failed", "read should load saved failed evaluation")
    _assert(retried.status == "completed", "retry should complete after valid evaluation")
    _assert(retried.session_id == generated.session_id, "retry should preserve session identity")
    _assert(len(store.save_calls) == 3, "question, failed evaluation, and retry should be explicit writes")


def main() -> None:
    test_analysis_read_write_refresh_and_stale_contract()
    test_quiz_persists_insufficient_content_and_rejects_invalid_success()
    test_critical_thinking_failed_evaluation_is_saved_with_answer_for_retry()
    print("app reading interaction persistence contract tests passed")


if __name__ == "__main__":
    main()
