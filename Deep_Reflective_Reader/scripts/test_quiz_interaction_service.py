#!/usr/bin/env python3
"""Regression tests for quiz reading interaction service validation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.quiz_interaction_service import (  # noqa: E402
    DEFAULT_QUIZ_MAX_ITEMS_BY_TARGET_LEVEL,
    QUIZ_OUTPUT_SCHEMA_VERSION,
    QUIZ_TYPES,
    QuizInteractionService,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target(
    *,
    target_level: str = "task_unit",
    content: str = "This passage explains incentives, causes, and effects in detail.",
) -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level=target_level,
        target_id=f"{target_level}-1",
        content=content,
        chapter_id="chapter-1" if target_level != "document" else None,
        section_id=(
            "section-1"
            if target_level in {"section", "task_unit"}
            else None
        ),
        task_unit_id="task-unit-1" if target_level == "task_unit" else None,
    )


def _request(
    *,
    target_level: str = "task_unit",
    content: str = "This passage explains incentives, causes, and effects in detail.",
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(target_level=target_level, content=content),
        interaction_type="quiz",
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="quiz_prompt_v1",
    )


def _short_answer_item(index: int) -> dict[str, str]:
    return {
        "type": "short_answer",
        "question": f"What is concept {index}?",
        "answer": f"Concept {index}",
    }


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_generates_completed_quiz_from_valid_json() -> None:
    captured: dict[str, object] = {}

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> str:
        captured["target_id"] = request.target.target_id
        captured["max_items"] = max_items
        captured["valid_types"] = valid_types
        return """
        {
          "items": [
            {
              "type": "short_answer",
              "question": "What drives behavior?",
              "answer": "Incentives"
            },
            {
              "type": "multiple_choice",
              "question": "Which idea is central?",
              "choices": ["Incentives", "Weather", "Typography"],
              "answer": "Incentives",
              "explanation": "The passage centers on incentives."
            },
            {
              "type": "true_false",
              "question": "The passage discusses causes and effects.",
              "answer": true
            }
          ]
        }
        """

    artifact = QuizInteractionService(generator).generate(_request())

    _assert(artifact.status == "completed", "valid quiz JSON should complete")
    _assert(captured["max_items"] == 3, "task-unit max should be passed to generator")
    _assert(
        captured["valid_types"] == tuple(sorted(QUIZ_TYPES)),
        "valid quiz types should be passed to generator",
    )
    _assert(
        artifact.metadata["output_schema_version"] == QUIZ_OUTPUT_SCHEMA_VERSION,
        "schema version should be recorded",
    )
    _assert(
        artifact.metadata["prompt_instruction_version"] == "quiz_prompt_v1",
        "prompt version should be recorded",
    )
    _assert(
        artifact.metadata["context"] == {"context_mode": "full_target"},
        "context metadata should be recorded",
    )
    _assert(len(artifact.payload["items"]) == 3, "quiz items should be preserved")


def test_accepts_fewer_than_limit_for_section() -> None:
    service = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "true_false",
                    "question": "The section can support one question.",
                    "answer": True,
                }
            ]
        }
    )

    artifact = service.generate(_request(target_level="section"))

    _assert(artifact.status == "completed", "fewer-than-max quiz should complete")
    _assert(artifact.metadata["max_items"] == 5, "section max should default to 5")
    _assert(len(artifact.payload["items"]) == 1, "single item should be accepted")


def test_enforces_default_and_configured_target_level_limits() -> None:
    captured: dict[str, int] = {}

    def document_generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        captured["document_max_items"] = max_items
        return {"items": [_short_answer_item(index) for index in range(max_items)]}

    document_artifact = QuizInteractionService(document_generator).generate(
        _request(target_level="document")
    )

    _assert(document_artifact.status == "completed", "document default max should pass")
    _assert(
        captured["document_max_items"]
        == DEFAULT_QUIZ_MAX_ITEMS_BY_TARGET_LEVEL["document"],
        "document max should default to 25",
    )

    too_many_chapter_items = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [_short_answer_item(index) for index in range(max_items + 1)]
        },
        max_items_by_target_level={
            "task_unit": 1,
            "section": 2,
            "chapter": 3,
            "document": 4,
        },
    ).generate(_request(target_level="chapter"))

    _assert(
        too_many_chapter_items.status == "generation_failed",
        "quiz over configured max should fail generation",
    )
    _assert(
        "max is 3" in (too_many_chapter_items.reason or ""),
        "failure reason should include configured max",
    )


def test_rejects_invalid_type_and_answer_payloads() -> None:
    invalid_type_artifact = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "essay",
                    "question": "Unsupported?",
                    "answer": "Yes",
                }
            ]
        }
    ).generate(_request())

    bad_choice_answer_artifact = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "multiple_choice",
                    "question": "Pick one.",
                    "choices": ["A", "B"],
                    "answer": "C",
                }
            ]
        }
    ).generate(_request())

    bad_true_false_answer_artifact = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "true_false",
                    "question": "Is this boolean?",
                    "answer": "true",
                }
            ]
        }
    ).generate(_request())

    _assert(
        invalid_type_artifact.status == "generation_failed",
        "invalid quiz type should fail generation",
    )
    _assert(
        bad_choice_answer_artifact.status == "generation_failed",
        "multiple-choice answer outside choices should fail generation",
    )
    _assert(
        bad_true_false_answer_artifact.status == "generation_failed",
        "true/false non-boolean answer should fail generation",
    )


def test_insufficient_content_does_not_call_generator() -> None:
    calls: list[str] = []

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        calls.append(request.target.target_id)
        return {"items": [_short_answer_item(0)]}

    artifact = QuizInteractionService(generator).generate(
        _request(content="@@@ !!! ###")
    )

    _assert(artifact.status == "insufficient_content", "symbol noise should fast path")
    _assert(calls == [], "generator should not be called for insufficient content")
    _assert(artifact.reason is not None, "insufficient content should include reason")


def test_rejects_non_quiz_requests_and_bad_limit_config() -> None:
    service = QuizInteractionService(
        lambda request, max_items, valid_types: {"items": [_short_answer_item(0)]}
    )
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="analysis",
    )

    _expect_value_error(
        lambda: service.generate(request),
        "requires interaction_type='quiz'",
    )
    _expect_value_error(
        lambda: QuizInteractionService(
            lambda request, max_items, valid_types: {"items": [_short_answer_item(0)]},
            max_items_by_target_level={"task_unit": 0},
        ),
        "positive integers",
    )


if __name__ == "__main__":
    test_generates_completed_quiz_from_valid_json()
    test_accepts_fewer_than_limit_for_section()
    test_enforces_default_and_configured_target_level_limits()
    test_rejects_invalid_type_and_answer_payloads()
    test_insufficient_content_does_not_call_generator()
    test_rejects_non_quiz_requests_and_bad_limit_config()
    print("quiz interaction service tests passed")
