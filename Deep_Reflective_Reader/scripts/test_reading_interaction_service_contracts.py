#!/usr/bin/env python3
"""Regression tests for target-agnostic reading interaction service contracts."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="section",
        target_id="section-1",
        content="Useful section content.",
        chapter_id="chapter-1",
        section_id="section-1",
    )


def _content_block_target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="content_block",
        target_id="unit-1:content:0",
        content="Selected block source text.",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
    )


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_request_accepts_resolved_target_and_normalizes_metadata() -> None:
    metadata = {"context_mode": "full_target"}
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type=" analysis ",
        context_metadata=metadata,
        prompt_instruction_version=" analysis_v1 ",
    )

    metadata["context_mode"] = "mutated"

    _assert(request.interaction_type == "analysis", "interaction type should normalize")
    _assert(
        request.context_metadata == {"context_mode": "full_target"},
        "request metadata should be copied",
    )
    _assert(
        request.prompt_instruction_version == "analysis_v1",
        "prompt instruction version should normalize",
    )


def test_request_copy_preserves_ancestor_context_field() -> None:
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="analysis",
        ancestor_context=None,
    )
    copied = request.with_secondary_context("Secondary context.")

    _assert(
        copied.ancestor_context is request.ancestor_context,
        "request copy should preserve ancestor_context field",
    )


def test_artifact_from_resolved_target_serializes_without_raw_target() -> None:
    artifact = ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="analysis",
        status="completed",
        payload={
            "summary": "A concise summary.",
            "reasoning": "A reasoned interpretation.",
            "explanation": "A parsing explanation.",
        },
        metadata={"schema_version": "analysis_v1"},
    )

    payload = artifact.to_dict()

    _assert(payload["target_level"] == "section", "target level should be preserved")
    _assert(payload["target_id"] == "section-1", "target id should be preserved")
    _assert(payload["chapter_id"] == "chapter-1", "chapter id should be preserved")
    _assert(payload["section_id"] == "section-1", "section id should be preserved")
    _assert("target" not in payload, "serialized artifact should not include raw target object")
    _assert("content" not in payload, "serialized artifact should not duplicate source content")


def test_artifact_from_content_block_target_preserves_block_identity() -> None:
    artifact = ReadingInteractionArtifact.from_target(
        target=_content_block_target(),
        interaction_type="analysis",
        status="insufficient_content",
        reason="selected block source text is below the readable threshold",
    )

    payload = artifact.to_dict()

    _assert(payload["target_level"] == "content_block", "target level should preserve block")
    _assert(payload["target_id"] == "unit-1:content:0", "target id should preserve block id")
    _assert(payload["task_unit_id"] == "unit-1", "parent task unit should be preserved")
    _assert(
        payload["content_block_id"] == "unit-1:content:0",
        "content_block_id should be preserved for downstream artifact mapping",
    )
    _assert("content" not in payload, "serialized artifact should not duplicate block source text")


def test_insufficient_content_status_requires_reason_but_not_payload() -> None:
    artifact = ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="quiz",
        status="insufficient_content",
        reason="target contains only OCR noise",
    )

    _assert(artifact.payload == {}, "insufficient-content result can have empty payload")
    _assert(
        artifact.reason == "target contains only OCR noise",
        "insufficient-content reason should be preserved",
    )

    _expect_value_error(
        lambda: ReadingInteractionArtifact.from_target(
            target=_target(),
            interaction_type="quiz",
            status="insufficient_content",
        ),
        "requires a reason",
    )


def test_rejects_invalid_success_shape_and_invalid_type_or_status() -> None:
    _expect_value_error(
        lambda: ReadingInteractionRequest(
            target=_target(),
            interaction_type="summary",
        ),
        "unsupported reading interaction type",
    )
    _expect_value_error(
        lambda: ReadingInteractionArtifact.from_target(
            target=_target(),
            interaction_type="analysis",
            status="completed",
        ),
        "requires non-empty payload",
    )
    _expect_value_error(
        lambda: ReadingInteractionArtifact.from_target(
            target=_target(),
            interaction_type="analysis",
            status="cached",
        ),
        "unsupported reading interaction status",
    )


def test_critical_thinking_statuses_are_scoped_to_critical_thinking_sessions() -> None:
    artifact = ReadingInteractionArtifact.from_target(
        target=_target(),
        interaction_type="critical_thinking_session",
        status="question_generated",
        payload={"question": "What assumption should be challenged?"},
    )

    _assert(
        artifact.status == "question_generated",
        "critical-thinking question generation should be a valid contract status",
    )
    _expect_value_error(
        lambda: ReadingInteractionArtifact.from_target(
            target=_target(),
            interaction_type="analysis",
            status="question_generated",
            payload={"question": "Invalid for analysis"},
        ),
        "only valid for critical_thinking_session",
    )


if __name__ == "__main__":
    test_request_accepts_resolved_target_and_normalizes_metadata()
    test_request_copy_preserves_ancestor_context_field()
    test_artifact_from_resolved_target_serializes_without_raw_target()
    test_artifact_from_content_block_target_preserves_block_identity()
    test_insufficient_content_status_requires_reason_but_not_payload()
    test_rejects_invalid_success_shape_and_invalid_type_or_status()
    test_critical_thinking_statuses_are_scoped_to_critical_thinking_sessions()
    print("reading interaction service contract tests passed")
