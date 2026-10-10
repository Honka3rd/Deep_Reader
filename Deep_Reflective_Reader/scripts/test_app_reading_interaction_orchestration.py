#!/usr/bin/env python3
"""Regression tests for app-layer reading interaction orchestration contracts."""

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
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from shared.task_unit_model import TaskUnit  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _document() -> StructuredDocument:
    task_unit = TaskUnit(
        unit_id="unit-1",
        title="Unit One",
        container_title="Section One",
        content="This task unit explains how assumptions shape conclusions.",
        source_section_ids=["section-1"],
        is_fallback_generated=False,
        parent_section_id="section-1",
    )
    section = StructuredSection(
        section_id="section-1",
        section_index=0,
        title="Section One",
        level=2,
        content="This section explains how assumptions shape conclusions.",
        char_start=0,
        char_end=60,
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
        raw_text="This document explains how assumptions shape conclusions.",
        chapters=[chapter],
    )


class FakePreparationPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def prepare_and_load(self, *, doc_name: str, mode: object) -> SimpleNamespace:
        self.calls.append((doc_name, mode))
        return SimpleNamespace(
            structured_document=_document(),
            assets=SimpleNamespace(errors=[]),
        )


class FakeInteractionOrchestrator:
    def __init__(self, interaction_type: str) -> None:
        self.interaction_type = interaction_type
        self.read_requests: list[ReadingInteractionRequest] = []
        self.generate_requests: list[ReadingInteractionRequest] = []

    def read(self, request: ReadingInteractionRequest) -> ReadingInteractionArtifact:
        self.read_requests.append(request)
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status="not_generated",
        )

    def generate(self, request: ReadingInteractionRequest) -> ReadingInteractionArtifact:
        self.generate_requests.append(request)
        payload: dict[str, object]
        if self.interaction_type == "analysis":
            payload = {
                "summary": "Summary",
                "reasoning": "Reasoning",
                "explanation": "Explanation",
            }
        else:
            payload = {
                "items": [
                    {
                        "type": "short_answer",
                        "question": "What matters?",
                        "answer": "Assumptions",
                    }
                ]
            }
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type=self.interaction_type,
            status="completed",
            payload=payload,
            metadata={"prompt_instruction_version": request.prompt_instruction_version},
        )


class FakeCriticalThinkingService:
    def __init__(self) -> None:
        self.generate_requests: list[ReadingInteractionRequest] = []
        self.submit_calls: list[tuple[ReadingInteractionArtifact, str]] = []
        self.evaluate_calls: list[ReadingInteractionArtifact] = []

    def generate_question(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact:
        self.generate_requests.append(request)
        return ReadingInteractionArtifact.from_target(
            target=request.target,
            interaction_type="critical_thinking_session",
            status="question_generated",
            payload={"question": "Which assumption matters most?"},
        )

    def submit_answer(
        self,
        session: ReadingInteractionArtifact,
        answer: str,
    ) -> ReadingInteractionArtifact:
        self.submit_calls.append((session, answer))
        payload = {
            "question": session.payload["question"],
            "answer": answer.strip(),
        }
        return ReadingInteractionArtifact(
            interaction_type="critical_thinking_session",
            status="answer_submitted",
            target_level=session.target_level,
            target_id=session.target_id,
            document_id=session.document_id,
            chapter_id=session.chapter_id,
            section_id=session.section_id,
            task_unit_id=session.task_unit_id,
            payload=payload,
            metadata=dict(session.metadata),
        )

    def evaluate_answer(
        self,
        session: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self.evaluate_calls.append(session)
        payload = {
            **session.payload,
            "evaluation": {
                "feedback": "Good assumption work.",
                "strengths": "Clear reasoning.",
                "improvements": "Use more evidence.",
                "score": 4,
            },
        }
        return ReadingInteractionArtifact(
            interaction_type="critical_thinking_session",
            status="completed",
            target_level=session.target_level,
            target_id=session.target_id,
            document_id=session.document_id,
            chapter_id=session.chapter_id,
            section_id=session.section_id,
            task_unit_id=session.task_unit_id,
            payload=payload,
            metadata=dict(session.metadata),
        )


class FakeCriticalThinkingSessionStore:
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
    analysis_orchestrator: FakeInteractionOrchestrator | None = None,
    quiz_orchestrator: FakeInteractionOrchestrator | None = None,
    critical_service: FakeCriticalThinkingService | None = None,
    critical_store: FakeCriticalThinkingSessionStore | None = None,
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


def test_analysis_methods_resolve_once_and_delegate() -> None:
    prep = FakePreparationPipeline()
    analysis = FakeInteractionOrchestrator("analysis")
    coordinator = _coordinator(
        preparation_pipeline=prep,
        analysis_orchestrator=analysis,
    )

    read_response = coordinator.read_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        chapter_id="chapter-1",
        section_id="section-1",
    )
    generate_response = coordinator.generate_analysis_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        prompt_instruction_version="analysis_prompt_v1",
    )
    refresh_response = coordinator.refresh_analysis_artifact(
        doc_name="book.json",
        target_level="task_unit",
        task_unit_id="unit-1",
    )

    _assert(len(prep.calls) == 3, "each analysis app method should resolve once")
    _assert(len(analysis.read_requests) == 1, "analysis read should delegate read")
    _assert(len(analysis.generate_requests) == 2, "generate/refresh should delegate generate")
    _assert(analysis.generate_requests[0].refresh is False, "generate should not refresh")
    _assert(analysis.generate_requests[1].refresh is True, "refresh should set refresh")
    _assert(read_response.status == "not_generated", "read response should map status")
    _assert(generate_response.status == "completed", "generate response should map status")
    _assert(refresh_response.target.task_unit_id == "unit-1", "target DTO should be safe")
    _assert(not hasattr(refresh_response.target, "content"), "target DTO must not expose raw content")


def test_quiz_methods_resolve_once_and_delegate() -> None:
    prep = FakePreparationPipeline()
    quiz = FakeInteractionOrchestrator("quiz")
    coordinator = _coordinator(
        preparation_pipeline=prep,
        quiz_orchestrator=quiz,
    )

    read_response = coordinator.read_quiz_artifact(
        doc_name="book.json",
        target_level="chapter",
        chapter_id="chapter-1",
    )
    generate_response = coordinator.generate_quiz_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    refresh_response = coordinator.refresh_quiz_artifact(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        prompt_instruction_version="quiz_prompt_v1",
    )

    _assert(len(prep.calls) == 3, "each quiz app method should resolve once")
    _assert(len(quiz.read_requests) == 1, "quiz read should delegate read")
    _assert(len(quiz.generate_requests) == 2, "quiz generate/refresh should delegate")
    _assert(quiz.generate_requests[1].refresh is True, "quiz refresh should set refresh")
    _assert(read_response.status == "not_generated", "quiz read response should map")
    _assert(generate_response.interaction_type == "quiz", "quiz response should map type")
    _assert(refresh_response.metadata["prompt_instruction_version"] == "quiz_prompt_v1", "metadata should pass through")


def test_critical_thinking_methods_resolve_once_and_delegate() -> None:
    prep = FakePreparationPipeline()
    service = FakeCriticalThinkingService()
    store = FakeCriticalThinkingSessionStore()
    coordinator = _coordinator(
        preparation_pipeline=prep,
        critical_service=service,
        critical_store=store,
    )

    missing = coordinator.read_critical_thinking_session(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
    )
    generated = coordinator.generate_critical_thinking_question(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        prompt_instruction_version="critical_prompt_v1",
    )
    read_existing = coordinator.read_critical_thinking_session(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id,
    )
    submitted = coordinator.submit_critical_thinking_answer(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id or "",
        answer="  Institutions matter.  ",
    )
    retried = coordinator.retry_critical_thinking_evaluation(
        doc_name="book.json",
        target_level="section",
        section_id="section-1",
        session_id=generated.session_id or "",
    )

    _assert(len(prep.calls) == 5, "each critical-thinking method should resolve once")
    _assert(missing.status == "not_generated", "session-less read should not generate")
    _assert(len(service.generate_requests) == 1, "question generation should delegate")
    _assert(len(store.save_calls) == 3, "generate, submit/evaluate, retry should save")
    _assert(read_existing.status == "question_generated", "read should load persisted session")
    _assert(len(service.submit_calls) == 1, "submit should call service submit")
    _assert(service.submit_calls[0][1] == "  Institutions matter.  ", "raw answer goes to service")
    _assert(len(service.evaluate_calls) == 2, "submit and retry should evaluate")
    _assert(submitted.status == "completed", "submit should return evaluated result")
    _assert(retried.status == "completed", "retry should return evaluation result")
    _assert(generated.session_id == submitted.session_id == retried.session_id, "session id should persist")


if __name__ == "__main__":
    test_analysis_methods_resolve_once_and_delegate()
    test_quiz_methods_resolve_once_and_delegate()
    test_critical_thinking_methods_resolve_once_and_delegate()
    print("app reading interaction orchestration tests passed")
