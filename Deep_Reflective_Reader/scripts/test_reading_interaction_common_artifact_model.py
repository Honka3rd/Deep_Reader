#!/usr/bin/env python3
"""Regression tests for mapping reading interactions onto one common artifact entity."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.reading_interaction_common_artifact import (  # noqa: E402
    common_artifact_to_reading_interaction_artifact,
    reading_interaction_artifact_to_common_artifact,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ReadingInteractionArtifact,
)
from shared.common_artifact_model import CommonArtifact, CommonArtifactTarget  # noqa: E402


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


def _reading_artifact(
    interaction_type: str,
    *,
    status: str = "completed",
    payload: dict[str, object] | None = None,
    reason: str | None = None,
) -> ReadingInteractionArtifact:
    return ReadingInteractionArtifact(
        interaction_type=interaction_type,
        status=status,
        target_level="section",
        target_id="section-1",
        document_id="doc-1",
        chapter_id="chapter-1",
        section_id="section-1",
        payload={} if payload is None else payload,
        metadata={"schema_version": f"{interaction_type}_v1"},
        reason=reason,
    )


def test_analysis_quiz_and_critical_thinking_share_common_entity_shape() -> None:
    artifacts = [
        _reading_artifact(
            "analysis",
            payload={
                "summary": "Summary",
                "reasoning": "Reasoning",
                "explanation": "Explanation",
            },
        ),
        _reading_artifact(
            "quiz",
            payload={"items": [{"type": "true_false", "question": "Q", "answer": True}]},
        ),
        _reading_artifact(
            "critical_thinking_session",
            status="question_generated",
            payload={"question": "What assumption should be tested?"},
        ),
    ]

    common_artifacts = [
        reading_interaction_artifact_to_common_artifact(
            artifact,
            artifact_id=f"artifact-{index}",
            source_structure_version=3,
            source_hash="source-hash",
        )
        for index, artifact in enumerate(artifacts, start=1)
    ]

    _assert(
        {type(artifact) for artifact in common_artifacts} == {CommonArtifact},
        "all reading interaction artifacts should map to the same CommonArtifact class",
    )
    _assert(
        [artifact.artifact_type for artifact in common_artifacts]
        == ["analysis", "quiz", "critical_thinking_session"],
        "artifact_type should distinguish interaction categories on one entity",
    )
    for artifact in common_artifacts:
        payload = artifact.to_dict()
        _assert(
            "analysis" not in payload
            and "quiz" not in payload
            and "critical_thinking_session" not in payload,
            "common artifact serialization should not create category-specific roots",
        )
        _assert(
            payload["target"]["chapter_id"] == "chapter-1"
            and payload["target"]["section_id"] == "section-1",
            "common artifact target should preserve hierarchy-aware ids",
        )
        _assert(
            payload["source_structure_version"] == 3,
            "source structure version should be preserved as provenance",
        )


def test_common_artifact_round_trip_preserves_service_contract() -> None:
    source = _reading_artifact(
        "analysis",
        payload={
            "summary": "Summary",
            "reasoning": "Reasoning",
            "explanation": "Explanation",
        },
    )
    common = reading_interaction_artifact_to_common_artifact(
        source,
        artifact_id="artifact-1",
        source_structure_version=1,
        source_hash="hash-1",
    )
    round_tripped_common = CommonArtifact.from_dict(common.to_dict())
    restored = common_artifact_to_reading_interaction_artifact(round_tripped_common)

    _assert(restored.to_dict() == source.to_dict(), "service artifact should round-trip")
    _assert(
        round_tripped_common.artifact_id == "artifact-1",
        "common artifact id should round-trip",
    )
    _assert(
        round_tripped_common.source_hash == "hash-1",
        "source hash should round-trip",
    )


def test_common_artifact_uses_defensive_payload_and_metadata_copies() -> None:
    payload = {"items": [{"type": "short_answer", "question": "Q", "answer": "A"}]}
    metadata = {"schema_version": "quiz_v1"}
    common = CommonArtifact(
        artifact_type="quiz",
        status="completed",
        target=CommonArtifactTarget(
            target_level="section",
            target_id="section-1",
            document_id="doc-1",
            chapter_id="chapter-1",
            section_id="section-1",
        ),
        payload=payload,
        metadata=metadata,
    )

    payload["items"] = []
    metadata["schema_version"] = "mutated"

    _assert(
        common.payload["items"],
        "common artifact should defensively copy payload dict",
    )
    _assert(
        common.metadata["schema_version"] == "quiz_v1",
        "common artifact should defensively copy metadata dict",
    )


def test_common_artifact_validation_rejects_unsupported_types_and_invalid_targets() -> None:
    target = CommonArtifactTarget(
        target_level="section",
        target_id="section-1",
        document_id="doc-1",
        chapter_id="chapter-1",
        section_id="section-1",
    )
    _expect_value_error(
        lambda: CommonArtifact(
            artifact_type="flashcard",
            status="completed",
            target=target,
            payload={"items": []},
        ),
        "unsupported common artifact type",
    )
    _expect_value_error(
        lambda: CommonArtifactTarget(
            target_level="task_unit",
            target_id="unit-1",
            document_id="doc-1",
            section_id="section-1",
        ),
        "task_unit artifact target requires task_unit_id",
    )
    _expect_value_error(
        lambda: CommonArtifact(
            artifact_type="analysis",
            status="completed",
            target=target,
        ),
        "completed common artifact requires non-empty payload",
    )
    _expect_value_error(
        lambda: CommonArtifact(
            artifact_type="analysis",
            status="question_generated",
            target=target,
            payload={"question": "Invalid for analysis"},
        ),
        "only valid for critical_thinking_session",
    )
    _expect_value_error(
        lambda: CommonArtifact(
            artifact_type="quiz",
            status="generation_failed",
            target=target,
        ),
        "requires a reason",
    )


def test_book_target_alias_normalizes_to_document_target() -> None:
    common = CommonArtifact(
        artifact_type="analysis",
        status="completed",
        target=CommonArtifactTarget(
            target_level="book",
            target_id="doc-1",
            document_id="doc-1",
        ),
        payload={
            "summary": "Book summary",
            "reasoning": "Book reasoning",
            "explanation": "Book explanation",
        },
    )

    _assert(
        common.target.target_level == "document",
        "book target aliases should normalize to document for shared target levels",
    )


if __name__ == "__main__":
    test_analysis_quiz_and_critical_thinking_share_common_entity_shape()
    test_common_artifact_round_trip_preserves_service_contract()
    test_common_artifact_uses_defensive_payload_and_metadata_copies()
    test_common_artifact_validation_rejects_unsupported_types_and_invalid_targets()
    test_book_target_alias_normalizes_to_document_target()
    print("reading interaction common artifact model tests passed")
