#!/usr/bin/env python3
"""Regression tests for critical-thinking session service lifecycle."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CRITICAL_THINKING_EVALUATION_INSTRUCTION,
    CRITICAL_THINKING_OUTPUT_SCHEMA_VERSION,
    CRITICAL_THINKING_QUESTION_INSTRUCTION,
    CriticalThinkingSessionService,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target(
    content: str = (
        "The passage argues that incentives shape behavior, but also implies "
        "that institutions and assumptions affect the final outcome."
    ),
) -> ResolvedReadingTarget:
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
    content: str = (
        "The passage argues that incentives shape behavior, but also implies "
        "that institutions and assumptions affect the final outcome."
    ),
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content),
        interaction_type="critical_thinking_session",
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="critical_thinking_prompt_v1",
    )


def _service(
    question_output=None,
    evaluation_output=None,
) -> CriticalThinkingSessionService:
    return CriticalThinkingSessionService(
        lambda request, instruction: (
            {"question": "Which assumption would most change the argument?"}
            if question_output is None
            else question_output
        ),
        lambda session, instruction: (
            {
                "feedback": "The answer identifies a relevant assumption.",
                "strengths": "It connects incentives to institutions.",
                "improvements": "It should cite a sharper piece of evidence.",
                "score": 4,
            }
            if evaluation_output is None
            else evaluation_output
        ),
    )


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_generates_question_with_fixed_instruction_metadata() -> None:
    captured: dict[str, object] = {}

    def question_generator(
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> dict[str, str]:
        captured["target_id"] = request.target.target_id
        captured["instruction"] = instruction
        return {"question": "What assumption supports the author's conclusion?"}

    service = CriticalThinkingSessionService(
        question_generator,
        lambda session, instruction: {},
    )

    artifact = service.generate_question(_request())

    _assert(
        artifact.status == "question_generated",
        "valid question should enter question_generated status",
    )
    _assert(
        artifact.payload["question"]
        == "What assumption supports the author's conclusion?",
        "question should be normalized",
    )
    _assert(
        captured["instruction"] == CRITICAL_THINKING_QUESTION_INSTRUCTION,
        "fixed question instruction should be passed to generator",
    )
    _assert(
        artifact.metadata["output_schema_version"]
        == CRITICAL_THINKING_OUTPUT_SCHEMA_VERSION,
        "schema version should be recorded",
    )
    _assert(
        artifact.metadata["prompt_instruction_version"]
        == "critical_thinking_prompt_v1",
        "prompt version should be recorded",
    )
    _assert(
        artifact.metadata["context"] == {"context_mode": "full_target"},
        "context metadata should be recorded",
    )


def test_submit_answer_and_evaluate_completed_session() -> None:
    captured: dict[str, object] = {}

    def evaluation_generator(
        session: ReadingInteractionArtifact,
        instruction: str,
    ) -> dict[str, object]:
        captured["status"] = session.status
        captured["instruction"] = instruction
        return {
            "feedback": "The answer reasons from a clear assumption.",
            "strengths": "It names a concrete assumption.",
            "improvements": "It could compare an alternative assumption.",
            "score": 4.5,
        }

    service = CriticalThinkingSessionService(
        lambda request, instruction: {
            "question": "What alternative assumption would change the claim?"
        },
        evaluation_generator,
    )

    generated = service.generate(_request())
    answered = service.submit_answer(
        generated,
        "The claim changes if institutions are assumed to override incentives.",
    )
    completed = service.evaluate_answer(answered)

    _assert(answered.status == "answer_submitted", "answer should be submitted")
    _assert(
        completed.status == "completed",
        "valid evaluation should complete the session",
    )
    _assert(
        completed.payload["question"] == generated.payload["question"],
        "completed payload should preserve the question",
    )
    _assert(
        completed.payload["answer"] == answered.payload["answer"],
        "completed payload should preserve the answer",
    )
    _assert(
        completed.payload["evaluation"]["score"] == 4.5,
        "evaluation score should be preserved",
    )
    _assert(
        captured["instruction"] == CRITICAL_THINKING_EVALUATION_INSTRUCTION,
        "fixed evaluation instruction should be passed to generator",
    )


def test_evaluation_failure_preserves_answer_for_retry() -> None:
    attempts: list[int] = []

    def evaluation_generator(
        session: ReadingInteractionArtifact,
        instruction: str,
    ) -> dict[str, object]:
        attempts.append(1)
        if len(attempts) == 1:
            return {"feedback": ""}
        return {
            "feedback": "The answer improves by naming the assumption.",
            "strengths": "It gives a relevant causal explanation.",
            "improvements": "It can quote the source more directly.",
        }

    service = CriticalThinkingSessionService(
        lambda request, instruction: {
            "question": "Which hidden assumption matters most?"
        },
        evaluation_generator,
    )

    answered = service.submit_answer(
        service.generate_question(_request()),
        "The hidden assumption is that incentives outweigh constraints.",
    )
    failed = service.evaluate_answer(answered)
    completed = service.evaluate_answer(failed)

    _assert(
        failed.status == "evaluation_failed",
        "invalid evaluation should preserve retryable status",
    )
    _assert(
        failed.payload["answer"] == answered.payload["answer"],
        "evaluation failure should preserve submitted answer",
    )
    _assert(failed.reason is not None, "evaluation failure should include reason")
    _assert(completed.status == "completed", "retry should be able to complete")


def test_insufficient_content_and_invalid_question_do_not_complete() -> None:
    calls: list[str] = []

    def question_generator(
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {"question": "Should not be called"}

    insufficient = CriticalThinkingSessionService(
        question_generator,
        lambda session, instruction: {},
    ).generate_question(_request("@@@ !!! ###"))

    invalid_question = _service(question_output={"question": ""}).generate_question(
        _request()
    )

    _assert(
        insufficient.status == "insufficient_content",
        "symbol noise should fast path",
    )
    _assert(calls == [], "generator should not be called for insufficient content")
    _assert(
        invalid_question.status == "generation_failed",
        "invalid generated question should fail generation",
    )


def test_rejects_invalid_lifecycle_transitions() -> None:
    service = _service()
    question = service.generate_question(_request())
    answered = service.submit_answer(question, "An answer")
    completed = service.evaluate_answer(answered)
    non_critical = ReadingInteractionArtifact(
        interaction_type="analysis",
        status="completed",
        target_level="section",
        target_id="section-1",
        document_id="doc-1",
        payload={"summary": "s"},
    )

    _expect_value_error(
        lambda: service.submit_answer(question, " "),
        "answer must be non-empty",
    )
    _expect_value_error(
        lambda: service.evaluate_answer(question),
        "requires an answer_submitted",
    )
    _expect_value_error(
        lambda: service.submit_answer(completed, "Another answer"),
        "answer submission requires",
    )
    _expect_value_error(
        lambda: service.submit_answer(non_critical, "Answer"),
        "requires a critical-thinking session",
    )


def test_rejects_non_critical_thinking_requests() -> None:
    service = _service()
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="quiz",
    )

    _expect_value_error(
        lambda: service.generate_question(request),
        "requires interaction_type='critical_thinking_session'",
    )


if __name__ == "__main__":
    test_generates_question_with_fixed_instruction_metadata()
    test_submit_answer_and_evaluate_completed_session()
    test_evaluation_failure_preserves_answer_for_retry()
    test_insufficient_content_and_invalid_question_do_not_complete()
    test_rejects_invalid_lifecycle_transitions()
    test_rejects_non_critical_thinking_requests()
    print("critical-thinking session service tests passed")
