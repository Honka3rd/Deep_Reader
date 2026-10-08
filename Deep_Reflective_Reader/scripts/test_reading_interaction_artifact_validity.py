#!/usr/bin/env python3
"""Regression tests for reading interaction artifact validity semantics."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.artifact_validity import (  # noqa: E402
    ReadingInteractionTargetValidity,
    ReadingInteractionValidityPolicy,
)
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
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


def _target(
    content: str = (
        "This target explains a clear argument with assumptions, evidence, "
        "and consequences that can support learning interactions."
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


def _request(interaction_type: str, *, content: str | None = None) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content if content is not None else _target().content),
        interaction_type=interaction_type,
        context_metadata={"context_mode": "full_target"},
    )


def _stale_checker(
    request: ReadingInteractionRequest,
) -> ReadingInteractionTargetValidity:
    return ReadingInteractionTargetValidity.stale(
        "target context is stale after structure version change"
    )


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_shared_validity_policy_detects_stale_before_content_checks() -> None:
    request = _request("analysis", content="@@@")
    policy = ReadingInteractionValidityPolicy(
        interaction_label="analysis",
        min_content_chars=20,
        min_alnum_chars=8,
        target_validity_checker=_stale_checker,
    )

    result = policy.preflight(request)

    _assert(result is not None, "stale target should fail preflight")
    _assert(result.status == "stale_target", "stale should take priority")
    _assert("structure version" in result.reason, "stale reason should be preserved")


def test_analysis_service_reports_stale_target_without_generator_call() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }

    service = AnalysisInteractionService(
        generator,
        target_validity_checker=_stale_checker,
    )

    artifact = service.generate(_request("analysis"))

    _assert(artifact.status == "stale_target", "analysis should report stale target")
    _assert(calls == [], "analysis generator should not run for stale target")
    _assert(artifact.reason is not None, "stale target should include reason")


def test_quiz_service_reports_stale_target_without_generator_call() -> None:
    calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        calls.append(request.target.target_id)
        return {
            "items": [
                {
                    "type": "short_answer",
                    "question": "What is the idea?",
                    "answer": "The idea",
                }
            ]
        }

    service = QuizInteractionService(
        generator,
        target_validity_checker=_stale_checker,
    )

    artifact = service.generate(_request("quiz"))

    _assert(artifact.status == "stale_target", "quiz should report stale target")
    _assert(calls == [], "quiz generator should not run for stale target")
    _assert(artifact.reason is not None, "stale target should include reason")


def test_critical_thinking_reports_stale_target_without_generator_call() -> None:
    calls: list[str] = []

    def question_generator(
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {"question": "What assumption matters?"}

    service = CriticalThinkingSessionService(
        question_generator,
        lambda session, instruction: {},
        target_validity_checker=_stale_checker,
    )

    artifact = service.generate_question(_request("critical_thinking_session"))

    _assert(
        artifact.status == "stale_target",
        "critical-thinking should report stale target",
    )
    _assert(calls == [], "question generator should not run for stale target")
    _assert(artifact.reason is not None, "stale target should include reason")


def test_all_services_report_insufficient_content_with_reason() -> None:
    analysis = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }
    ).generate(_request("analysis", content="@@@ !!! ###"))
    quiz = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "short_answer",
                    "question": "Question?",
                    "answer": "Answer",
                }
            ]
        }
    ).generate(_request("quiz", content="@@@ !!! ###"))
    critical = CriticalThinkingSessionService(
        lambda request, instruction: {"question": "Question?"},
        lambda session, instruction: {},
    ).generate_question(
        _request("critical_thinking_session", content="@@@ !!! ###")
    )

    for artifact in (analysis, quiz, critical):
        _assert(
            artifact.status == "insufficient_content",
            "symbol noise should be insufficient content",
        )
        _assert(
            artifact.reason is not None and "target content" in artifact.reason,
            "insufficient content should include a target-content reason",
        )


def test_stale_and_insufficient_statuses_require_reasons() -> None:
    _expect_value_error(
        lambda: ReadingInteractionTargetValidity.stale(" "),
        "requires a reason",
    )
    _expect_value_error(
        lambda: ReadingInteractionArtifact(
            interaction_type="analysis",
            status="stale_target",
            target_level="section",
            target_id="section-1",
            document_id="doc-1",
        ),
        "requires a reason",
    )
    _expect_value_error(
        lambda: ReadingInteractionArtifact(
            interaction_type="analysis",
            status="insufficient_content",
            target_level="section",
            target_id="section-1",
            document_id="doc-1",
        ),
        "requires a reason",
    )


if __name__ == "__main__":
    test_shared_validity_policy_detects_stale_before_content_checks()
    test_analysis_service_reports_stale_target_without_generator_call()
    test_quiz_service_reports_stale_target_without_generator_call()
    test_critical_thinking_reports_stale_target_without_generator_call()
    test_all_services_report_insufficient_content_with_reason()
    test_stale_and_insufficient_statuses_require_reasons()
    print("reading interaction artifact validity tests passed")
