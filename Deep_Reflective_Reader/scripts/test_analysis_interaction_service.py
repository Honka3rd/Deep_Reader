#!/usr/bin/env python3
"""Regression tests for analysis reading interaction service validation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import (  # noqa: E402
    ANALYSIS_OUTPUT_SCHEMA_VERSION,
    AnalysisInteractionService,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _target(content: str = "This section explains incentives and market behavior.") -> ResolvedReadingTarget:
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
    content: str = "This section explains incentives and market behavior.",
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content),
        interaction_type="analysis",
        context_metadata={"context_mode": "full_target"},
        prompt_instruction_version="analysis_prompt_v1",
    )


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def test_generates_completed_analysis_from_valid_json() -> None:
    service = AnalysisInteractionService(
        lambda request: """
        {
          "summary": "Markets respond to incentives.",
          "reasoning": "The passage connects decisions to incentives.",
          "explanation": "It parses the section as an economic argument."
        }
        """
    )

    artifact = service.generate(_request())

    _assert(artifact.status == "completed", "valid JSON should complete")
    _assert(
        artifact.payload == {
            "summary": "Markets respond to incentives.",
            "reasoning": "The passage connects decisions to incentives.",
            "explanation": "It parses the section as an economic argument.",
        },
        "analysis payload should be normalized",
    )
    _assert(
        artifact.metadata["output_schema_version"] == ANALYSIS_OUTPUT_SCHEMA_VERSION,
        "schema version should be recorded",
    )
    _assert(
        artifact.metadata["prompt_instruction_version"] == "analysis_prompt_v1",
        "prompt version should be recorded",
    )
    _assert(
        artifact.metadata["context"] == {"context_mode": "full_target"},
        "context metadata should be recorded",
    )


def test_accepts_valid_dict_output() -> None:
    service = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
            "extra": "ignored",
        }
    )

    artifact = service.generate(_request())

    _assert(artifact.status == "completed", "valid dict output should complete")
    _assert("extra" not in artifact.payload, "unexpected fields should not enter payload")


def test_insufficient_content_does_not_call_generator() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> str:
        calls.append(request.target.target_id)
        return "{}"

    service = AnalysisInteractionService(generator)
    artifact = service.generate(_request(content="@@@ !!! ###"))

    _assert(artifact.status == "insufficient_content", "symbol noise should fast path")
    _assert(calls == [], "generator should not be called for insufficient content")
    _assert(artifact.reason is not None, "insufficient content should include reason")


def test_invalid_json_and_missing_fields_are_generation_failures() -> None:
    invalid_json_artifact = AnalysisInteractionService(
        lambda request: "{not json"
    ).generate(_request())
    missing_field_artifact = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
        }
    ).generate(_request())

    _assert(
        invalid_json_artifact.status == "generation_failed",
        "invalid JSON should be a failed generation",
    )
    _assert(
        "invalid analysis JSON" in (invalid_json_artifact.reason or ""),
        "invalid JSON reason should be explicit",
    )
    _assert(
        missing_field_artifact.status == "generation_failed",
        "missing field should be a failed generation",
    )
    _assert(
        "explanation" in (missing_field_artifact.reason or ""),
        "missing-field reason should name the field",
    )


def test_rejects_non_analysis_requests() -> None:
    service = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }
    )
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="quiz",
    )

    _expect_value_error(
        lambda: service.generate(request),
        "requires interaction_type='analysis'",
    )


if __name__ == "__main__":
    test_generates_completed_analysis_from_valid_json()
    test_accepts_valid_dict_output()
    test_insufficient_content_does_not_call_generator()
    test_invalid_json_and_missing_fields_are_generation_failures()
    test_rejects_non_analysis_requests()
    print("analysis interaction service tests passed")
