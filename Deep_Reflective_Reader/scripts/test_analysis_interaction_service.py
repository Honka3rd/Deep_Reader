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


def _target(
    content: str = "This section explains incentives and market behavior.",
    *,
    target_level: str = "section",
) -> ResolvedReadingTarget:
    is_content_block = target_level == "content_block"
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level=target_level,
        target_id="unit-1:block:0" if is_content_block else "section-1",
        content=content,
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1" if is_content_block else None,
        content_block_id="unit-1:block:0" if is_content_block else None,
    )


def _request(
    *,
    content: str = "This section explains incentives and market behavior.",
    target_level: str = "section",
) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(content, target_level=target_level),
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
            "summary": "Incentives shape market behavior.",
            "reasoning": "The source connects incentives to market behavior.",
            "explanation": "The insight depends on the section's incentives detail.",
            "extra": "ignored",
        }
    )

    artifact = service.generate(_request())

    _assert(artifact.status == "completed", "valid dict output should complete")
    _assert("extra" not in artifact.payload, "unexpected fields should not enter payload")


def test_accepts_json_object_wrapped_by_model_text() -> None:
    service = AnalysisInteractionService(
        lambda request: (
            "```json\n"
            "{\"summary\":\"Incentives organize the section.\","
            "\"reasoning\":\"The source repeats incentives and market behavior as linked ideas.\","
            "\"explanation\":\"The insight is grounded in the section's incentives detail.\"}"
            "\n```"
        )
    )
    leading_text_artifact = AnalysisInteractionService(
        lambda request: (
            "Here is the insight:\n"
            "{\"summary\":\"Market behavior depends on incentives\","
            "\"reasoning\":\"The section names incentives as the driver of behavior.\","
            "\"explanation\":\"That source detail keeps the claim anchored.\"}"
        )
    ).generate(_request())

    artifact = service.generate(_request())

    _assert(artifact.status == "completed", "fenced analysis JSON should complete")
    _assert(
        artifact.payload["summary"] == "Incentives organize the section.",
        "analysis should parse JSON from fenced model output",
    )
    _assert(
        leading_text_artifact.status == "completed",
        "analysis JSON with leading text should complete",
    )


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


def test_vacuous_completed_output_becomes_generation_failure() -> None:
    service = AnalysisInteractionService(
        lambda request: {
            "summary": "This passage shows an important theme.",
            "reasoning": "The text shows a meaningful idea for the reader.",
            "explanation": "This helps the reader understand the passage.",
        }
    )

    artifact = service.generate(_request())

    _assert(
        artifact.status == "generation_failed",
        "generic unsupported output should not complete",
    )
    _assert(
        "primary-source detail" in (artifact.reason or ""),
        "vacuity failure should name missing source detail",
    )


def test_placeholder_analysis_output_becomes_generation_failure() -> None:
    service = AnalysisInteractionService(
        lambda request: {
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        }
    )

    artifact = service.generate(_request())

    _assert(
        artifact.status == "generation_failed",
        "placeholder output should not complete",
    )
    _assert(
        "placeholder-like" in (artifact.reason or ""),
        "placeholder failure should explain the problem",
    )


def test_meaningless_semantic_content_does_not_call_generator() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "Pseudo insight",
            "reasoning": "Unsupported reasoning",
            "explanation": "Unsupported explanation",
        }

    repeated_boilerplate = (
        "國富論 lite 國富論 lite 國富論 lite 國富論 lite 國富論 lite "
        "國富論 lite 國富論 lite 國富論 lite"
    )
    fragmented_index = "\n".join(
        [
            "第一章 1",
            "第二章 3",
            "第三章 5",
            "第四章 8",
            "第五章 13",
        ]
    )

    service = AnalysisInteractionService(generator)
    repeated_artifact = service.generate(_request(content=repeated_boilerplate))
    index_artifact = service.generate(_request(content=fragmented_index))

    _assert(
        repeated_artifact.status == "insufficient_content",
        "repeated boilerplate should be insufficient content",
    )
    _assert(
        "repetitive" in (repeated_artifact.reason or ""),
        "repeated boilerplate reason should explain semantic guard",
    )
    _assert(
        index_artifact.status == "insufficient_content",
        "index-like fragments should be insufficient content",
    )
    _assert(calls == [], "generator should not run for semantic-noise content")


def test_semantic_content_guard_allows_readable_cjk_source() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "水的異常感揭示了人物的警覺。",
            "reasoning": "人物觀察氣泡和沉澱物，並在不確定中選擇倒掉水。",
            "explanation": "這不是單純喝水，而是安全感被細節動搖的瞬間。",
        }

    readable_source = (
        "淑美喝水時覺得味道有點怪，便把杯子舉到燈下查看。"
        "她看見水裡有細小氣泡和沉澱物，無法確定來源，最後皺著眉把水倒掉。"
    )

    artifact = AnalysisInteractionService(generator).generate(
        _request(content=readable_source)
    )

    _assert(artifact.status == "completed", "readable CJK source should generate")
    _assert(calls == ["section-1"], "generator should run for readable source")


def test_content_block_preflight_allows_readable_and_rejects_tiny_blocks() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "The block identifies a local cause.",
            "reasoning": "It has enough readable text for local interpretation.",
            "explanation": "The source is the selected content block only.",
        }

    service = AnalysisInteractionService(generator)
    readable = service.generate(
        _request(
            target_level="content_block",
            content="This block explains a local cause and its consequence.",
        )
    )
    tiny = service.generate(_request(target_level="content_block", content="tiny"))

    _assert(readable.status == "completed", "readable content block should generate")
    _assert(
        readable.content_block_id == "unit-1:block:0",
        "content-block analysis should preserve block identity",
    )
    _assert(tiny.status == "insufficient_content", "tiny block should fast path")
    _assert(
        calls == ["unit-1:block:0"],
        "generator should be called only for readable content blocks",
    )


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
    test_accepts_json_object_wrapped_by_model_text()
    test_insufficient_content_does_not_call_generator()
    test_vacuous_completed_output_becomes_generation_failure()
    test_placeholder_analysis_output_becomes_generation_failure()
    test_meaningless_semantic_content_does_not_call_generator()
    test_semantic_content_guard_allows_readable_cjk_source()
    test_content_block_preflight_allows_readable_and_rejects_tiny_blocks()
    test_invalid_json_and_missing_fields_are_generation_failures()
    test_rejects_non_analysis_requests()
    print("analysis interaction service tests passed")
