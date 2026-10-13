#!/usr/bin/env python3
"""Regression tests for lower-level artifact reference policy in services."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.artifact_aware_context import (  # noqa: E402
    ArtifactAwareContextBuilder,
    ArtifactAwareContextResult,
    ArtifactContextTarget,
    ArtifactContextSummary,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


PRIMARY_SOURCE_TEXT = (
    "This chapter explains how incentives, institutions, and assumptions combine "
    "to shape decision making across several local examples."
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target(target_level: str = "chapter") -> ResolvedReadingTarget:
    if target_level == "content_block":
        return ResolvedReadingTarget(
            document_id="doc-1",
            document_title="Document One",
            target_level="content_block",
            target_id="unit-1:block:0",
            content="This block states a local claim about incentives.",
            chapter_id="chapter-1",
            section_id="section-1",
            task_unit_id="unit-1",
            content_block_id="unit-1:block:0",
        )
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level=target_level,
        target_id=f"{target_level}-1",
        content=PRIMARY_SOURCE_TEXT,
        chapter_id="chapter-1" if target_level != "document" else None,
    )


def _request(interaction_type: str, target_level: str = "chapter") -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(target_level),
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
                outcome="incentives are primary but not sufficient",
            ),
            ArtifactContextSummary(
                artifact_id="quiz-section-1",
                artifact_type="quiz",
                target_level="section",
                target_id="section-1",
                concepts=("incentives", "institutions"),
            ),
        ],
        max_artifacts=5,
        max_context_chars=500,
    )


def _content_block_artifact_context_provider(
    request: ReadingInteractionRequest,
) -> ArtifactAwareContextResult:
    return ArtifactAwareContextBuilder().build_for_target(
        target=ArtifactContextTarget(
            target_level=request.target.target_level,
            target_id=request.target.target_id,
            task_unit_id=request.target.task_unit_id,
            content_block_id=request.target.content_block_id,
        ),
        artifacts=[
            ArtifactContextSummary(
                artifact_id="analysis-unit-1",
                artifact_type="analysis",
                target_level="task_unit",
                target_id="unit-1",
                focus="parent unit argument",
                outcome="local incentive claim needs qualification",
                metadata={"task_unit_id": "unit-1"},
            ),
            ArtifactContextSummary(
                artifact_id="analysis-block-sibling",
                artifact_type="analysis",
                target_level="content_block",
                target_id="unit-1:block:1",
                focus="sibling block should stay excluded",
                metadata={
                    "task_unit_id": "unit-1",
                    "content_block_id": "unit-1:block:1",
                },
            ),
        ],
        max_artifacts=5,
        max_context_chars=500,
    )


def _assert_artifact_metadata(artifact: ReadingInteractionArtifact) -> None:
    context_metadata = artifact.metadata["context"]
    artifact_context = context_metadata["artifact_context"]

    _assert(
        context_metadata["context_mode"] == "full_target",
        "original context metadata should be preserved",
    )
    _assert(
        artifact_context["artifact_context_mode"] == "referenced",
        "artifact context should be recorded as referenced",
    )
    _assert(
        artifact_context["referenced_artifact_ids"]
        == ["analysis-unit-1", "quiz-section-1"],
        "referenced artifact ids should be recorded",
    )
    _assert(
        artifact_context["referenced_artifact_types"] == ["analysis", "quiz"],
        "referenced artifact types should be recorded",
    )
    _assert(
        artifact_context["coverage_counts"] == {"task_unit": 1, "section": 1},
        "coverage counts should be recorded",
    )
    _assert(
        artifact_context["abstraction_hint_applied"] is True,
        "analysis/focus should apply abstraction hint",
    )
    _assert(
        artifact_context["deduplication_hint_applied"] is True,
        "quiz/concepts should apply deduplication hint",
    )
    _assert(
        "context_text" not in artifact_context,
        "metadata must not persist secondary context text",
    )
    _assert(
        "payload" not in artifact_context,
        "metadata must not expose child artifact payload",
    )


def test_content_block_uses_parent_artifact_as_secondary_context_only() -> None:
    captured: dict[str, object] = {}

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        captured["target_content"] = request.target.content
        captured["secondary_context"] = request.secondary_context
        return {
            "summary": "The block makes a local claim about incentives.",
            "reasoning": "Uses block source with parent artifact context.",
            "explanation": "The parent artifact remains secondary.",
        }

    artifact = AnalysisInteractionService(
        generator,
        artifact_context_provider=_content_block_artifact_context_provider,
    ).generate(_request("analysis", target_level="content_block"))
    artifact_context = artifact.metadata["context"]["artifact_context"]
    artifact_reference = artifact.metadata["artifact_reference"]

    _assert(artifact.status == "completed", "content-block analysis should complete")
    _assert(
        captured["target_content"]
        == "This block states a local claim about incentives.",
        "content-block primary source text must remain the selected block",
    )
    _assert(
        "focus=parent unit argument" in captured["secondary_context"],
        "parent task-unit artifact should be available as secondary context",
    )
    _assert(
        "sibling block should stay excluded" not in captured["secondary_context"],
        "sibling content-block artifact should not be injected",
    )
    _assert(
        artifact_context["referenced_artifact_ids"] == ["analysis-unit-1"],
        "content-block provenance should only reference parent task-unit artifact",
    )
    _assert(
        artifact_reference == artifact_context,
        "top-level reference metadata should mirror bounded provenance",
    )
    _assert("context_text" not in artifact_reference, "metadata must omit context text")
    _assert("payload" not in artifact_reference, "metadata must omit child payloads")


def test_analysis_consumes_secondary_context_without_mutating_primary_source() -> None:
    captured: dict[str, object] = {}

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        captured["target_content"] = request.target.content
        captured["secondary_context"] = request.secondary_context
        return {
            "summary": "The chapter compares incentives and institutions.",
            "reasoning": "Uses primary source incentives with secondary abstraction hints.",
            "explanation": "Explains the chapter-level argument.",
        }

    artifact = AnalysisInteractionService(
        generator,
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request("analysis"))

    _assert(artifact.status == "completed", "analysis should complete")
    _assert(
        captured["target_content"] == PRIMARY_SOURCE_TEXT,
        "primary source text must remain unchanged",
    )
    _assert(
        "Secondary lower-level artifact context:" in captured["secondary_context"],
        "analysis generator should receive compact secondary context",
    )
    _assert_artifact_metadata(artifact)


def test_quiz_consumes_secondary_context_as_deduplication_signal() -> None:
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
                    "question": "How do incentives and institutions interact?",
                    "answer": "They jointly shape decisions.",
                }
            ]
        }

    artifact = QuizInteractionService(
        generator,
        artifact_context_provider=_artifact_context_provider,
    ).generate(_request("quiz"))

    _assert(artifact.status == "completed", "quiz should complete")
    _assert(
        captured["target_content"] == PRIMARY_SOURCE_TEXT,
        "primary source text must remain unchanged",
    )
    _assert(
        "target=section:section-1" in captured["secondary_context"]
        and "concepts=incentives, institutions" in captured["secondary_context"],
        "quiz generator should receive lower-level quiz coverage signal",
    )
    _assert_artifact_metadata(artifact)


def test_critical_thinking_consumes_secondary_context_for_abstraction() -> None:
    captured: dict[str, object] = {}

    def question_generator(
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> dict[str, str]:
        captured["target_content"] = request.target.content
        captured["secondary_context"] = request.secondary_context
        return {
            "question": "Which assumption from the chapter deserves challenge?"
        }

    artifact = CriticalThinkingSessionService(
        question_generator,
        lambda session, instruction: {},
        artifact_context_provider=_artifact_context_provider,
    ).generate_question(_request("critical_thinking_session"))

    _assert(
        artifact.status == "question_generated",
        "critical-thinking question should be generated",
    )
    _assert(
        captured["target_content"] == PRIMARY_SOURCE_TEXT,
        "primary source text must remain unchanged",
    )
    _assert(
        "focus=local incentive argument" in captured["secondary_context"]
        and "outcome=incentives are primary but not sufficient"
        in captured["secondary_context"],
        "critical-thinking generator should receive abstraction signal",
    )
    _assert_artifact_metadata(artifact)


def test_absent_child_artifacts_do_not_block_generation() -> None:
    def empty_provider(
        request: ReadingInteractionRequest,
    ) -> ArtifactAwareContextResult:
        return ArtifactAwareContextBuilder().build_secondary_context([])

    captured: dict[str, object] = {}

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        captured["secondary_context"] = request.secondary_context
        return {
            "summary": "The document argument turns on incentives and institutions.",
            "reasoning": "The source compares incentives, institutions, and assumptions.",
            "explanation": "Generation can proceed without child artifact context.",
        }

    artifact = AnalysisInteractionService(
        generator,
        artifact_context_provider=empty_provider,
    ).generate(_request("analysis", target_level="document"))

    _assert(artifact.status == "completed", "empty child artifacts should not block")
    _assert(
        captured["secondary_context"] == "",
        "empty child artifacts should produce empty secondary context",
    )
    _assert(
        artifact.metadata["context"]["artifact_context"]["artifact_context_mode"]
        == "none",
        "empty child artifact provenance should still be explicit",
    )


if __name__ == "__main__":
    test_content_block_uses_parent_artifact_as_secondary_context_only()
    test_analysis_consumes_secondary_context_without_mutating_primary_source()
    test_quiz_consumes_secondary_context_as_deduplication_signal()
    test_critical_thinking_consumes_secondary_context_for_abstraction()
    test_absent_child_artifacts_do_not_block_generation()
    print("lower-level artifact reference policy tests passed")
