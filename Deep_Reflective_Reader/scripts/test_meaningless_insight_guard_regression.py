#!/usr/bin/env python3
"""Regression coverage for meaningless-content analysis/insight guards."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _request(content: str) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=ResolvedReadingTarget(
            document_id="doc-1",
            document_title="Document One",
            target_level="section",
            target_id="section-1",
            content=content,
            chapter_id="chapter-1",
            section_id="section-1",
        ),
        interaction_type="analysis",
    )


def test_wealth_of_nations_lite_noise_fast_paths_without_generator_call() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "Unsupported insight.",
            "reasoning": "Unsupported reasoning.",
            "explanation": "Unsupported explanation.",
        }

    noisy_source = (
        "國富論 lite 國富論 lite 國富論 lite 國富論 lite 國富論 lite "
        "國富論 lite 國富論 lite 國富論 lite"
    )

    artifact = AnalysisInteractionService(generator).generate(_request(noisy_source))

    _assert(
        artifact.status == "insufficient_content",
        "國富論 lite-style boilerplate should not generate a fluent pseudo-insight",
    )
    _assert(
        "repetitive" in (artifact.reason or ""),
        "guarded boilerplate should report the semantic-content reason",
    )
    _assert(calls == [], "generator must not run for semantic-noise content")


def test_readable_cjk_source_still_generates_source_grounded_insight() -> None:
    calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        calls.append(request.target.target_id)
        return {
            "summary": "氣泡和沉澱物讓淑美的喝水動作變成警覺。",
            "reasoning": "她查看杯子、看見氣泡和沉澱物，最後把水倒掉。",
            "explanation": "這些細節讓安全感被水的異常動搖。",
        }

    readable_source = (
        "淑美喝水時覺得味道有點怪，便把杯子舉到燈下查看。"
        "她看見水裡有細小氣泡和沉澱物，無法確定來源，最後皺著眉把水倒掉。"
    )

    artifact = AnalysisInteractionService(generator).generate(_request(readable_source))

    _assert(
        artifact.status == "completed",
        "semantic guard should preserve generation for readable primary-source text",
    )
    _assert(calls == ["section-1"], "generator should run for meaningful content")
    _assert(
        artifact.payload["reasoning"].startswith("她查看杯子"),
        "completed insight should keep concrete source details",
    )


def test_generic_pseudo_insight_fails_after_generation() -> None:
    service = AnalysisInteractionService(
        lambda request: {
            "summary": "This passage shows an important theme.",
            "reasoning": "The text shows a meaningful idea for the reader.",
            "explanation": "This helps the reader understand the passage.",
        }
    )

    readable_source = (
        "淑美喝水時覺得味道有點怪，便把杯子舉到燈下查看。"
        "她看見水裡有細小氣泡和沉澱物，無法確定來源，最後皺著眉把水倒掉。"
    )

    artifact = service.generate(_request(readable_source))

    _assert(
        artifact.status == "generation_failed",
        "generic output without source detail should not persist as completed insight",
    )
    _assert(
        "primary-source detail" in (artifact.reason or ""),
        "post-generation failure should name missing source detail",
    )


if __name__ == "__main__":
    test_wealth_of_nations_lite_noise_fast_paths_without_generator_call()
    test_readable_cjk_source_still_generates_source_grounded_insight()
    test_generic_pseudo_insight_fails_after_generation()
    print("meaningless insight guard regression tests passed")
