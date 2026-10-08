#!/usr/bin/env python3
"""Regression tests for quiz deduplication from lower-level artifact signals."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.artifact_aware_context import (  # noqa: E402
    ArtifactAwareContextBuilder,
    ArtifactAwareContextResult,
    ArtifactContextSummary,
)
from section_tasks.quiz_interaction_service import (  # noqa: E402
    QUIZ_DEDUPLICATION_INSTRUCTION,
    QUIZ_DEDUPLICATION_METADATA_VERSION,
    QuizInteractionService,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


PRIMARY_SOURCE_TEXT = (
    "This chapter compares incentives, institutions, and evidence quality across "
    "several sections. It asks the reader to connect local examples into a broader "
    "argument about decision making."
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _request() -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=ResolvedReadingTarget(
            document_id="doc-1",
            document_title="Document One",
            target_level="chapter",
            target_id="chapter-1",
            content=PRIMARY_SOURCE_TEXT,
            chapter_id="chapter-1",
        ),
        interaction_type="quiz",
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="quiz_prompt_v1",
    )


def _artifact_context_provider(
    request: ReadingInteractionRequest,
) -> ArtifactAwareContextResult:
    return ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="quiz-unit-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
                focus="What is an incentive?",
                concepts=("incentive definition", "local recall"),
            ),
            ArtifactContextSummary(
                artifact_id="quiz-section-1",
                artifact_type="quiz",
                target_level="section",
                target_id="section-1",
                focus="Which institution is mentioned in the section?",
                concepts=("institution example",),
            ),
        ],
        max_artifacts=5,
        max_context_chars=500,
    )


def test_higher_level_quiz_receives_deduplication_guidance_without_concatenation() -> None:
    captured: dict[str, object] = {}

    def generator(
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> dict[str, object]:
        captured["target_content"] = request.target.content
        captured["secondary_context"] = request.secondary_context
        return {
            "items": [
                {
                    "type": "short_answer",
                    "question": (
                        "How do incentives and institutions interact across "
                        "the chapter's examples?"
                    ),
                    "answer": (
                        "They combine local motivations and rules into a broader "
                        "decision-making pattern."
                    ),
                }
            ]
        }

    artifact = QuizInteractionService(
        generator,
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request())

    _assert(artifact.status == "completed", "synthesis quiz should complete")
    _assert(
        captured["target_content"] == PRIMARY_SOURCE_TEXT,
        "primary source context should remain unchanged",
    )
    secondary_context = captured["secondary_context"]
    _assert(
        isinstance(secondary_context, str)
        and QUIZ_DEDUPLICATION_INSTRUCTION in secondary_context,
        "generator should receive explicit quiz deduplication guidance",
    )
    _assert(
        "target=task_unit:unit-1" in secondary_context
        and "concepts=incentive definition, local recall" in secondary_context,
        "generator should receive lower-level quiz coverage signals",
    )
    questions = [item["question"] for item in artifact.payload["items"]]
    _assert(
        questions
        == ["How do incentives and institutions interact across the chapter's examples?"],
        "service should keep only higher-level generated quiz items",
    )
    _assert(
        "What is an incentive?" not in questions,
        "lower-level quiz focus should not be concatenated into payload items",
    )

    context_metadata = artifact.metadata["context"]
    _assert(
        context_metadata["quiz_deduplication"]
        == {
            "guidance_applied": True,
            "instruction_version": QUIZ_DEDUPLICATION_METADATA_VERSION,
        },
        "deduplication guidance metadata should be recorded without context text",
    )
    _assert(
        "context_text" not in context_metadata["artifact_context"],
        "artifact provenance metadata must not persist secondary context text",
    )


def test_exact_lower_level_focus_repeat_fails_generation() -> None:
    artifact = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "short_answer",
                    "question": "What is an incentive?",
                    "answer": "A motivation or reward that shapes behavior.",
                }
            ]
        },
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request())

    _assert(
        artifact.status == "generation_failed",
        "exact lower-level focus repeat should fail generation",
    )
    _assert(
        "duplicates lower-level artifact coverage" in (artifact.reason or ""),
        "failure reason should identify lower-level quiz duplication",
    )


if __name__ == "__main__":
    test_higher_level_quiz_receives_deduplication_guidance_without_concatenation()
    test_exact_lower_level_focus_repeat_fails_generation()
    print("quiz deduplication from lower artifacts tests passed")
