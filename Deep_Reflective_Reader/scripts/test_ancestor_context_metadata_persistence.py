#!/usr/bin/env python3
"""Regression tests for bounded ancestor context metadata on generated artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.ancestor_summary_context import (  # noqa: E402
    ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY,
    AncestorSummaryArtifact,
    AncestorSummaryPromptContext,
    AncestorSummaryTarget,
)
from section_tasks.critical_thinking_session_service import (  # noqa: E402
    CriticalThinkingSessionService,
)
from section_tasks.quiz_interaction_service import QuizInteractionService  # noqa: E402
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
        target_level="content_block",
        target_id="unit-1:content:0",
        content="Selected block source text with enough readable content.",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
    )


def _ancestor_context() -> AncestorSummaryPromptContext:
    return AncestorSummaryPromptContext(
        summaries=(
            AncestorSummaryArtifact(
                artifact_id="ancestor-summary::doc-1::section::section-1",
                target=AncestorSummaryTarget(
                    target_level="section",
                    target_id="section-1",
                    document_id="doc-1",
                    chapter_id="chapter-1",
                    section_id="section-1",
                    title="Section One",
                ),
                summary="Safe ancestor summary text should not be persisted here.",
                source_hash="section-source-hash",
                generated_at="2026-10-12T00:00:00Z",
                metadata={"raw_source_text": "RAW ANCESTOR SHOULD NOT APPEAR"},
            ),
        ),
        context_status="partial",
        missing_target_levels=("chapter",),
        stale_target_levels=("task_unit",),
        pruned_reason="test pruning",
    )


def _request(interaction_type: str) -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(),
        interaction_type=interaction_type,
        ancestor_context=_ancestor_context(),
        context_metadata={"document_language": "en"},
    )


def _assert_ancestor_metadata(artifact: ReadingInteractionArtifact) -> None:
    metadata = artifact.metadata[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY]

    _assert(
        metadata["context_mode"] == "ancestor_summary_context",
        "ancestor context mode should be recorded",
    )
    _assert(metadata["context_status"] == "partial", "status should be preserved")
    _assert(metadata["summary_count"] == 1, "summary count should be recorded")
    _assert(
        metadata["summary_artifact_ids"]
        == ["ancestor-summary::doc-1::section::section-1"],
        "summary artifact ids should be recorded",
    )
    _assert(
        metadata["summary_target_levels"] == ["section"],
        "summary target levels should be recorded",
    )
    _assert(
        metadata["summary_target_ids"] == ["section-1"],
        "summary target ids should be recorded",
    )
    _assert(
        metadata["missing_target_levels"] == ["chapter"],
        "missing levels should be recorded",
    )
    _assert(
        metadata["stale_target_levels"] == ["task_unit"],
        "stale levels should be recorded",
    )
    _assert(
        metadata["primary_source_target_level"] == "content_block",
        "primary target level should be recorded",
    )
    _assert(
        metadata["primary_source_content_block_id"] == "unit-1:content:0",
        "primary content block id should be recorded",
    )
    _assert(
        "Safe ancestor summary text should not be persisted here." not in str(metadata),
        "ancestor summary text must not be persisted in artifact metadata",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in str(metadata),
        "raw ancestor source metadata must not be persisted",
    )


def test_analysis_persists_bounded_ancestor_context_metadata() -> None:
    service = AnalysisInteractionService(
        lambda _request: {
            "summary": "The selected block is treated as the primary source.",
            "reasoning": "The analysis cites the selected block source text directly.",
            "explanation": "This keeps ancestor metadata subordinate to the selected block.",
        }
    )

    artifact = service.generate(_request("analysis"))

    _assert(artifact.status == "completed", "analysis should complete")
    _assert_ancestor_metadata(artifact)


def test_quiz_persists_bounded_ancestor_context_metadata() -> None:
    service = QuizInteractionService(
        lambda _request, _max_items, _valid_types: {
            "items": [
                {
                    "type": "short_answer",
                    "question": "What matters here?",
                    "answer": "The selected block.",
                    "explanation": "It is grounded in the selected block.",
                }
            ]
        }
    )

    artifact = service.generate(_request("quiz"))

    _assert(artifact.status == "completed", "quiz should complete")
    _assert_ancestor_metadata(artifact)


def test_critical_thinking_preserves_ancestor_context_metadata_through_lifecycle() -> None:
    service = CriticalThinkingSessionService(
        question_generator=lambda _request, _instruction: {
            "question": "What assumption should be tested?"
        },
        evaluation_generator=lambda _session, _instruction: {
            "feedback": "Good.",
            "strengths": "Uses evidence.",
            "improvements": "Name the assumption more directly.",
        },
    )

    generated = service.generate_question(_request("critical_thinking_session"))
    answered = service.submit_answer(generated, "It assumes the block is decisive.")
    completed = service.evaluate_answer(answered)

    _assert(generated.status == "question_generated", "question should generate")
    _assert(completed.status == "completed", "evaluation should complete")
    _assert_ancestor_metadata(generated)
    _assert_ancestor_metadata(answered)
    _assert_ancestor_metadata(completed)


if __name__ == "__main__":
    test_analysis_persists_bounded_ancestor_context_metadata()
    test_quiz_persists_bounded_ancestor_context_metadata()
    test_critical_thinking_preserves_ancestor_context_metadata_through_lifecycle()
    print("ancestor context metadata persistence tests passed")
