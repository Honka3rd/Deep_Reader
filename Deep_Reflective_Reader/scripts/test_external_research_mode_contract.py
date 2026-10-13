#!/usr/bin/env python3
"""Regression tests for explicit external research mode provenance contracts."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    EXTERNAL_RESEARCH_METADATA_KEY,
    ExternalResearchContext,
    ExternalResearchSourceReference,
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
    build_external_research_metadata,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_value_error(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as error:
        _assert(expected in str(error), f"expected '{expected}' in '{error}'")
        return
    raise AssertionError(f"expected ValueError containing '{expected}'")


def _target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="content_block",
        target_id="unit-1:content:0",
        content="Selected block source text with enough readable content.",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
    )


def _external_context() -> ExternalResearchContext:
    return ExternalResearchContext(
        sources=(
            ExternalResearchSourceReference(
                source_id="source-1",
                label="Background source A",
                source_type="web_page",
                title="Historical Background",
                url="https://example.invalid/background",
                retrieved_at="2026-10-12T00:00:00Z",
            ),
        )
    )


def _request(interaction_type: str) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(),
        interaction_type=interaction_type,
        external_research_context=_external_context(),
    )


def _assert_external_metadata(artifact: ReadingInteractionArtifact) -> None:
    metadata = artifact.metadata[EXTERNAL_RESEARCH_METADATA_KEY]

    _assert(metadata["mode"] == "external_research", "mode should be explicit")
    _assert(metadata["usage_scope"] == "background_only", "scope should be bounded")
    _assert(metadata["provenance_required"] is True, "provenance required")
    _assert(metadata["source_count"] == 1, "source count should be recorded")
    _assert(
        metadata["sources"][0]["label"] == "Background source A",
        "source label should be recorded",
    )
    _assert(
        metadata["sources"][0]["url"] == "https://example.invalid/background",
        "source URL should be recorded as provenance",
    )
    _assert("payload" not in str(metadata), "external payloads must not be stored")
    _assert("external text body" not in str(metadata), "external content body forbidden")


def test_external_research_context_requires_explicit_labelled_sources() -> None:
    _expect_value_error(
        lambda: ExternalResearchContext(sources=()),
        "requires labelled sources",
    )
    _expect_value_error(
        lambda: ExternalResearchSourceReference(
            source_id=" ",
            label="Background source A",
            source_type="web_page",
        ),
        "requires source_id, label, and source_type",
    )
    _expect_value_error(
        lambda: ExternalResearchContext(
            sources=(_external_context().sources[0],),
            usage_scope="answer_authority",
        ),
        "usage_scope must be",
    )


def test_external_research_metadata_builder_is_explicit_opt_in() -> None:
    request_without_external = ReadingInteractionRequest(
        target=_target(),
        interaction_type="analysis",
    )

    _assert(
        build_external_research_metadata(request_without_external) is None,
        "external metadata should be absent without explicit context",
    )
    metadata = build_external_research_metadata(_request("analysis"))
    _assert(metadata is not None, "explicit external mode should produce metadata")
    _assert(metadata["mode"] == "external_research", "metadata mode")


def test_analysis_persists_external_research_provenance() -> None:
    service = AnalysisInteractionService(
        lambda _request: {
            "summary": "The selected block remains the primary source.",
            "reasoning": "The output cites selected block source text before external context.",
            "explanation": "External research provenance stays subordinate to the readable block.",
        }
    )

    artifact = service.generate(_request("analysis"))

    _assert(artifact.status == "completed", "analysis should complete")
    _assert_external_metadata(artifact)


def test_quiz_persists_external_research_provenance() -> None:
    service = QuizInteractionService(
        lambda _request, _max_items, _valid_types: {
            "items": [
                {
                    "type": "short_answer",
                    "question": "What is directly supported?",
                    "answer": "The selected block.",
                    "explanation": "Primary source remains authority.",
                }
            ]
        }
    )

    artifact = service.generate(_request("quiz"))

    _assert(artifact.status == "completed", "quiz should complete")
    _assert_external_metadata(artifact)


def test_critical_thinking_preserves_external_research_provenance() -> None:
    service = CriticalThinkingSessionService(
        question_generator=lambda _request, _instruction: {
            "question": "Which assumption should be tested against the passage?"
        },
        evaluation_generator=lambda _session, _instruction: {
            "feedback": "Good.",
            "strengths": "Uses source evidence.",
            "improvements": "Keep external context labelled.",
        },
    )

    generated = service.generate_question(_request("critical_thinking_session"))
    answered = service.submit_answer(generated, "It assumes context explains motive.")
    completed = service.evaluate_answer(answered)

    _assert(generated.status == "question_generated", "question should generate")
    _assert(completed.status == "completed", "evaluation should complete")
    _assert_external_metadata(generated)
    _assert_external_metadata(answered)
    _assert_external_metadata(completed)


if __name__ == "__main__":
    test_external_research_context_requires_explicit_labelled_sources()
    test_external_research_metadata_builder_is_explicit_opt_in()
    test_analysis_persists_external_research_provenance()
    test_quiz_persists_external_research_provenance()
    test_critical_thinking_preserves_external_research_provenance()
    print("external research mode contract tests passed")
