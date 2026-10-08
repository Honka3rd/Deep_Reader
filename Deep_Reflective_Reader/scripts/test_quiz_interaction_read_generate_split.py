#!/usr/bin/env python3
"""Regression tests for persisted quiz read/generate orchestration."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.quiz_interaction_orchestrator import (  # noqa: E402
    QuizInteractionOrchestrator,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class InMemoryQuizArtifactStore:
    def __init__(self) -> None:
        self.artifacts: dict[tuple[str, str, str], ReadingInteractionArtifact] = {}
        self.get_calls = 0
        self.save_calls = 0

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


def _target(content: str = "This section explains market incentives clearly.") -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="section",
        target_id="section-1",
        content=content,
        chapter_id="chapter-1",
        section_id="section-1",
    )


def _request(
    *,
    content: str = "This section explains market incentives clearly.",
    refresh: bool = False,
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content),
        interaction_type="quiz",
        refresh=refresh,
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="quiz_prompt_v1",
    )


def _generated_payload(question: str = "What drives behavior?") -> dict[str, object]:
    return {
        "items": [
            {
                "type": "short_answer",
                "question": question,
                "answer": "Incentives",
            }
        ]
    }


def _completed_artifact() -> ReadingInteractionArtifact:
    return ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="quiz",
        status="completed",
        payload=_generated_payload("Cached question?"),
    )


def test_read_missing_artifact_does_not_generate_or_persist() -> None:
    generator_calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        generator_calls.append(request.target.target_id)
        return _generated_payload()

    store = InMemoryQuizArtifactStore()
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.read(_request())

    _assert(artifact.status == "not_generated", "missing read should report not_generated")
    _assert(generator_calls == [], "read path must not call generator")
    _assert(store.save_calls == 0, "read path must not persist a placeholder")


def test_read_existing_artifact_returns_persisted_result() -> None:
    store = InMemoryQuizArtifactStore()
    existing = _completed_artifact()
    store.save_quiz_artifact(existing)
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(lambda request, max_items, valid_types: {}),
        artifact_store=store,
    )

    artifact = orchestrator.read(_request())

    _assert(artifact is existing, "read should return current persisted artifact")


def test_generate_missing_artifact_persists_completed_result() -> None:
    generator_calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        generator_calls.append(request.target.target_id)
        return _generated_payload("Generated question?")

    store = InMemoryQuizArtifactStore()
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request())

    _assert(generator_calls == ["section-1"], "generate should call generator once")
    _assert(artifact.status == "completed", "valid generated quiz should complete")
    _assert(store.save_calls == 1, "completed quiz should be persisted")
    _assert(
        orchestrator.read(_request()).payload == artifact.payload,
        "read should see saved artifact",
    )


def test_generate_persists_insufficient_content_without_generator_call() -> None:
    generator_calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        generator_calls.append(request.target.target_id)
        return _generated_payload()

    store = InMemoryQuizArtifactStore()
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request(content="@@@ !!! ###"))

    _assert(artifact.status == "insufficient_content", "noise should persist insufficient status")
    _assert(generator_calls == [], "insufficient content should skip generator")
    _assert(store.save_calls == 1, "insufficient-content status should be persisted")
    _assert(
        orchestrator.read(_request(content="@@@ !!! ###")).status == "insufficient_content",
        "read should see persisted insufficient-content artifact",
    )


def test_generate_does_not_persist_invalid_output_failure() -> None:
    store = InMemoryQuizArtifactStore()
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(lambda request, max_items, valid_types: "{not json"),
        artifact_store=store,
    )

    artifact = orchestrator.generate(_request())

    _assert(artifact.status == "generation_failed", "invalid JSON should fail generation")
    _assert(store.save_calls == 0, "generation failures should not be persisted")
    _assert(
        orchestrator.read(_request()).status == "not_generated",
        "failed generation should not create a readable artifact",
    )


def test_generate_reuses_existing_artifact_unless_refresh_is_requested() -> None:
    generator_calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        generator_calls.append(request.target.target_id)
        return _generated_payload("Refreshed question?")

    store = InMemoryQuizArtifactStore()
    existing = _completed_artifact()
    store.save_quiz_artifact(existing)
    store.save_calls = 0
    orchestrator = QuizInteractionOrchestrator(
        service=QuizInteractionService(generator),
        artifact_store=store,
    )

    reused = orchestrator.generate(_request())
    refreshed = orchestrator.generate(_request(refresh=True))

    _assert(reused is existing, "generate without refresh should reuse persisted artifact")
    _assert(generator_calls == ["section-1"], "only refresh should call generator")
    _assert(
        refreshed.payload["items"][0]["question"] == "Refreshed question?",
        "refresh should persist new output",
    )
    _assert(store.save_calls == 1, "only refresh should write a new artifact")


if __name__ == "__main__":
    test_read_missing_artifact_does_not_generate_or_persist()
    test_read_existing_artifact_returns_persisted_result()
    test_generate_missing_artifact_persists_completed_result()
    test_generate_persists_insufficient_content_without_generator_call()
    test_generate_does_not_persist_invalid_output_failure()
    test_generate_reuses_existing_artifact_unless_refresh_is_requested()
    print("quiz interaction read/generate split tests passed")
