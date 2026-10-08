#!/usr/bin/env python3
"""Regression tests for persisted referenced-artifact metadata on generated artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.artifact_aware_context import (  # noqa: E402
    ArtifactAwareContextBuilder,
    ArtifactAwareContextResult,
    ArtifactContextSummary,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ARTIFACT_REFERENCE_METADATA_KEY,
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


PRIMARY_SOURCE_TEXT = (
    "This chapter compares incentives, institutions, and assumptions across "
    "several examples so that higher-level reading interactions can synthesize "
    "and critique the full argument."
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target(target_level: str = "chapter") -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level=target_level,
        target_id=f"{target_level}-1",
        content=PRIMARY_SOURCE_TEXT,
        chapter_id="chapter-1" if target_level != "document" else None,
    )


def _request(interaction_type: str) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(),
        interaction_type=interaction_type,
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version=f"{interaction_type}_prompt_v1",
    )


def _artifact_context_provider(
    request: ReadingInteractionRequest,
) -> ArtifactAwareContextResult:
    return ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="analysis-unit-1",
                artifact_type="analysis",
                target_level="task_unit",
                target_id="unit-1",
                focus="local incentive argument",
                outcome="incentives alone are not sufficient",
            ),
            ArtifactContextSummary(
                artifact_id="quiz-section-1",
                artifact_type="quiz",
                target_level="section",
                target_id="section-1",
                concepts=("incentives", "institutions"),
            ),
            ArtifactContextSummary(
                artifact_id="ct-section-2",
                artifact_type="critical_thinking_session",
                target_level="section",
                target_id="section-2",
                focus="challenge an institutional assumption",
            ),
        ],
        max_artifacts=5,
        max_context_chars=800,
    )


def _assert_reference_metadata(artifact: ReadingInteractionArtifact) -> None:
    reference_metadata = artifact.metadata[ARTIFACT_REFERENCE_METADATA_KEY]
    context_metadata = artifact.metadata["context"]
    artifact_context = context_metadata["artifact_context"]

    _assert(
        reference_metadata["artifact_context_mode"] == "referenced",
        "top-level artifact reference should record context mode",
    )
    _assert(
        reference_metadata["referenced_artifact_ids"]
        == ["analysis-unit-1", "quiz-section-1", "ct-section-2"],
        "top-level artifact reference should record artifact ids",
    )
    _assert(
        reference_metadata["referenced_artifact_types"]
        == ["analysis", "quiz", "critical_thinking_session"],
        "top-level artifact reference should record artifact types",
    )
    _assert(
        reference_metadata["referenced_artifact_target_levels"]
        == ["task_unit", "section", "section"],
        "top-level artifact reference should record target levels",
    )
    _assert(
        reference_metadata["coverage_counts"] == {"task_unit": 1, "section": 2},
        "top-level artifact reference should record coverage counts",
    )
    _assert(
        reference_metadata["deduplication_hint_applied"] is True,
        "deduplication hint should be persisted",
    )
    _assert(
        reference_metadata["abstraction_hint_applied"] is True,
        "abstraction hint should be persisted",
    )
    _assert(
        "context_text" not in reference_metadata,
        "top-level artifact reference must not persist secondary context text",
    )
    _assert(
        "payload" not in reference_metadata
        and "child_artifacts" not in reference_metadata,
        "top-level artifact reference must not expose child artifact payloads",
    )
    _assert(
        reference_metadata == artifact_context,
        "top-level artifact reference should mirror bounded provenance fields",
    )


def test_analysis_artifact_persists_referenced_artifact_metadata() -> None:
    artifact = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        },
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request("analysis"))

    _assert(artifact.status == "completed", "analysis should complete")
    _assert_reference_metadata(artifact)


def test_quiz_artifact_persists_referenced_artifact_metadata() -> None:
    artifact = QuizInteractionService(
        lambda request, max_items, valid_types: {
            "items": [
                {
                    "type": "short_answer",
                    "question": "How do incentives and institutions interact?",
                    "answer": "They jointly shape the chapter argument.",
                }
            ]
        },
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request("quiz"))

    _assert(artifact.status == "completed", "quiz should complete")
    _assert_reference_metadata(artifact)


def test_critical_thinking_session_persists_reference_metadata_through_lifecycle() -> None:
    service = CriticalThinkingSessionService(
        lambda request, instruction: {
            "question": "Which assumption should be tested across the chapter?"
        },
        lambda session, instruction: {
            "feedback": "The answer tests a relevant assumption.",
            "strengths": "It compares more than one section.",
            "improvements": "It could cite sharper evidence.",
        },
        artifact_context_provider=_artifact_context_provider,
    )

    generated = service.generate_question(_request("critical_thinking_session"))
    answered = service.submit_answer(
        generated,
        "The key assumption is that local incentives scale to chapter-level claims.",
    )
    completed = service.evaluate_answer(answered)

    _assert(
        generated.status == "question_generated",
        "question generation should preserve reference metadata",
    )
    _assert_reference_metadata(generated)
    _assert_reference_metadata(answered)
    _assert_reference_metadata(completed)


if __name__ == "__main__":
    test_analysis_artifact_persists_referenced_artifact_metadata()
    test_quiz_artifact_persists_referenced_artifact_metadata()
    test_critical_thinking_session_persists_reference_metadata_through_lifecycle()
    print("referenced artifact metadata persistence tests passed")
