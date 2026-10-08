#!/usr/bin/env python3
"""Regression tests for critical-thinking continuity from lower-level sessions."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.artifact_aware_context import (  # noqa: E402
    ArtifactAwareContextBuilder,
    ArtifactAwareContextResult,
    ArtifactContextSummary,
)
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CRITICAL_THINKING_CONTINUITY_GUIDANCE,
    CRITICAL_THINKING_CONTINUITY_METADATA_VERSION,
    CRITICAL_THINKING_QUESTION_INSTRUCTION,
    CriticalThinkingSessionService,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


PRIMARY_SOURCE_TEXT = (
    "This chapter compares local assumptions, conflicting evidence, and downstream "
    "implications across several sections. It asks the reader to test whether the "
    "argument still holds when the scope expands beyond one example."
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
        interaction_type="critical_thinking_session",
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="critical_thinking_prompt_v1",
    )


def _artifact_context_provider(
    request: ReadingInteractionRequest,
) -> ArtifactAwareContextResult:
    return ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="ct-unit-1",
                artifact_type="critical_thinking_session",
                target_level="task_unit",
                target_id="unit-1",
                focus="Which assumption makes the local claim vulnerable?",
                outcome="reader challenged a single example's causal assumption",
            ),
            ArtifactContextSummary(
                artifact_id="ct-section-1",
                artifact_type="critical_thinking_session",
                target_level="section",
                target_id="section-1",
                focus="What counterargument applies within this section?",
                outcome="reader compared one counterargument with local evidence",
            ),
        ],
        max_artifacts=5,
        max_context_chars=700,
    )


def test_higher_level_question_uses_learning_continuity_signals() -> None:
    captured: dict[str, object] = {}

    def question_generator(
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> dict[str, str]:
        captured["target_content"] = request.target.content
        captured["secondary_context"] = request.secondary_context
        captured["instruction"] = instruction
        return {
            "question": (
                "Which chapter-level assumption should be tested across multiple "
                "sections before accepting the broader argument?"
            )
        }

    artifact = CriticalThinkingSessionService(
        question_generator,
        lambda session, instruction: {},
        artifact_context_provider=_artifact_context_provider,
    ).generate_question(_request())

    _assert(
        artifact.status == "question_generated",
        "continuity-aware question should be generated",
    )
    _assert(
        captured["instruction"] == CRITICAL_THINKING_QUESTION_INSTRUCTION,
        "base question instruction should remain fixed and predictable",
    )
    _assert(
        captured["target_content"] == PRIMARY_SOURCE_TEXT,
        "primary target source should remain unchanged",
    )
    secondary_context = captured["secondary_context"]
    _assert(
        isinstance(secondary_context, str)
        and CRITICAL_THINKING_CONTINUITY_GUIDANCE in secondary_context,
        "generator should receive explicit learning-continuity guidance",
    )
    _assert(
        "target=task_unit:unit-1" in secondary_context
        and "focus=Which assumption makes the local claim vulnerable?"
        in secondary_context
        and "outcome=reader compared one counterargument with local evidence"
        in secondary_context,
        "generator should receive child session focus and outcome signals",
    )
    _assert(
        artifact.payload
        == {
            "question": (
                "Which chapter-level assumption should be tested across multiple "
                "sections before accepting the broader argument?"
            )
        },
        "service should preserve one higher-level generated question",
    )

    context_metadata = artifact.metadata["context"]
    _assert(
        context_metadata["critical_thinking_continuity"]
        == {
            "guidance_applied": True,
            "instruction_version": CRITICAL_THINKING_CONTINUITY_METADATA_VERSION,
        },
        "continuity metadata should be recorded without child payloads",
    )
    _assert(
        context_metadata["artifact_context"]["referenced_artifact_types"]
        == ["critical_thinking_session", "critical_thinking_session"],
        "critical-thinking child sessions should be referenced as provenance",
    )
    _assert(
        "context_text" not in context_metadata["artifact_context"],
        "artifact provenance metadata must not persist secondary context text",
    )
    _assert(
        "payload" not in context_metadata["artifact_context"],
        "artifact provenance metadata must not expose child session payloads",
    )


if __name__ == "__main__":
    test_higher_level_question_uses_learning_continuity_signals()
    print("critical-thinking abstraction from lower sessions tests passed")
