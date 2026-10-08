#!/usr/bin/env python3
"""Schema regression tests for artifact-aware interaction metadata."""

from __future__ import annotations

from pathlib import Path
import sys

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_schemas import ArtifactAwareInteractionMetadataResponse  # noqa: E402
from shared.artifact_target_model import ArtifactTargetLevel  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_validation_error(payload: dict, expected_message: str) -> None:
    try:
        ArtifactAwareInteractionMetadataResponse.model_validate(payload)
    except ValidationError as error:
        _assert(
            expected_message in str(error),
            f"expected {expected_message!r} in validation error: {error}",
        )
        return
    raise AssertionError(f"expected validation error containing {expected_message!r}")


def test_accepts_empty_artifact_context_metadata() -> None:
    metadata = ArtifactAwareInteractionMetadataResponse.model_validate({})

    _assert(metadata.artifact_context_mode == "none", "default mode should be none")
    _assert(metadata.primary_source_evidence_ids == [], "source evidence ids should default empty")
    _assert(metadata.referenced_artifact_ids == [], "artifact ids should default empty")
    _assert(metadata.coverage_counts == {}, "coverage counts should default empty")


def test_accepts_referenced_artifact_context_metadata() -> None:
    metadata = ArtifactAwareInteractionMetadataResponse.model_validate(
        {
            "artifact_context_mode": " Referenced ",
            "primary_source_evidence_ids": [" source-chapter-1 "],
            "referenced_artifact_ids": [" unit-quiz-1 ", " unit-critical-1 "],
            "referenced_artifact_types": [" quiz ", " critical_thinking_session "],
            "referenced_artifact_target_levels": ["task_unit", ArtifactTargetLevel.SECTION],
            "coverage_counts": {" task_unit ": 2, "section": 1},
            "deduplication_hint_applied": True,
            "abstraction_hint_applied": True,
        }
    )

    _assert(metadata.artifact_context_mode == "referenced", "mode should normalize")
    _assert(
        metadata.primary_source_evidence_ids == ["source-chapter-1"],
        "source evidence ids should trim",
    )
    _assert(
        metadata.referenced_artifact_ids == ["unit-quiz-1", "unit-critical-1"],
        "referenced artifact ids should trim",
    )
    _assert(
        metadata.referenced_artifact_types == ["quiz", "critical_thinking_session"],
        "artifact types should trim",
    )
    _assert(
        metadata.referenced_artifact_target_levels
        == [ArtifactTargetLevel.TASK_UNIT, ArtifactTargetLevel.SECTION],
        "target levels should validate through shared enum",
    )
    _assert(metadata.coverage_counts == {"task_unit": 2, "section": 1}, "counts should trim keys")
    _assert(metadata.deduplication_hint_applied is True, "deduplication flag should preserve")
    _assert(metadata.abstraction_hint_applied is True, "abstraction flag should preserve")


def test_rejects_invalid_artifact_context_metadata() -> None:
    _assert_validation_error(
        {"artifact_context_mode": "raw_payload"},
        "artifact_context_mode must be one of",
    )
    _assert_validation_error(
        {
            "artifact_context_mode": "none",
            "referenced_artifact_ids": ["artifact-1"],
        },
        "artifact_context_mode=none cannot include referenced_artifact_ids",
    )
    _assert_validation_error(
        {
            "artifact_context_mode": "referenced",
            "referenced_artifact_ids": ["artifact-1", "  "],
        },
        "referenced_artifact_ids cannot contain empty values",
    )
    _assert_validation_error(
        {
            "artifact_context_mode": "referenced",
            "coverage_counts": {"task_unit": -1},
        },
        "coverage_counts cannot contain negative values",
    )


def test_rejects_nested_child_artifact_payload_expansion() -> None:
    _assert_validation_error(
        {
            "artifact_context_mode": "referenced",
            "referenced_artifact_ids": ["artifact-1"],
            "child_artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "payload": {"question": "Should not be embedded"},
                }
            ],
        },
        "Extra inputs are not permitted",
    )
    _assert_validation_error(
        {
            "artifact_context_mode": "referenced",
            "referenced_artifact_ids": ["artifact-1"],
            "referenced_artifact_payloads": {
                "artifact-1": {"question": "Should not be embedded"}
            },
        },
        "Extra inputs are not permitted",
    )


if __name__ == "__main__":
    test_accepts_empty_artifact_context_metadata()
    test_accepts_referenced_artifact_context_metadata()
    test_rejects_invalid_artifact_context_metadata()
    test_rejects_nested_child_artifact_payload_expansion()
    print("artifact-aware interaction metadata schema tests passed")
