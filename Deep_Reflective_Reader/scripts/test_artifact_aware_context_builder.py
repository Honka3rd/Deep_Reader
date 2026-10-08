#!/usr/bin/env python3
"""Regression tests for artifact-aware secondary context assembly."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context.artifact_aware_context import (  # noqa: E402
    ArtifactAwareContextBuilder,
    ArtifactContextSummary,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_builds_compact_secondary_artifact_context() -> None:
    result = ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="quiz-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
                focus="local concept recall",
                concepts=("market", "incentive"),
            ),
            ArtifactContextSummary(
                artifact_id="ct-1",
                artifact_type="critical_thinking_session",
                target_level="task_unit",
                target_id="unit-2",
                focus="compare two claims",
                outcome="needs stronger evidence",
            ),
        ]
    )

    _assert(result.artifact_context_mode == "referenced", "mode should be referenced")
    _assert(
        result.referenced_artifact_ids == ["quiz-1", "ct-1"],
        "artifact ids should preserve order",
    )
    _assert(
        result.coverage_counts == {"task_unit": 2},
        "coverage counts should track target levels",
    )
    _assert(result.deduplication_hint_applied is True, "quiz/concepts should trigger dedup hint")
    _assert(result.abstraction_hint_applied is True, "critical/focus should trigger abstraction hint")
    _assert("Secondary lower-level artifact context:" in result.context_text, "should label context")
    _assert("payload" not in result.context_text, "should not embed raw payload fields")


def test_skips_empty_and_insufficient_artifacts_without_blocking() -> None:
    result = ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id=" ",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
            ),
            ArtifactContextSummary(
                artifact_id="noise-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-2",
                status="insufficient_content",
            ),
        ]
    )

    _assert(result.artifact_context_mode == "none", "all skipped artifacts should produce none")
    _assert(result.context_text == "", "empty artifact context should not block generation")
    _assert(result.referenced_artifact_ids == [], "no artifacts should be referenced")
    _assert(result.artifact_context_pruned_reason == "skipped=2", "skip reason should be recorded")


def test_prunes_duplicates_and_max_artifacts() -> None:
    result = ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="quiz-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
            ),
            ArtifactContextSummary(
                artifact_id="quiz-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
            ),
            ArtifactContextSummary(
                artifact_id="quiz-2",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-2",
            ),
        ],
        max_artifacts=1,
    )

    _assert(result.artifact_context_mode == "pruned", "duplicates/max should mark pruned")
    _assert(result.referenced_artifact_ids == ["quiz-1"], "only first artifact should remain")
    _assert(
        result.artifact_context_pruned_reason == "skipped=1, duplicates=1",
        "prune reason should track skipped and duplicate artifacts",
    )


def test_applies_artifact_context_char_budget_after_primary_context_priority() -> None:
    result = ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="quiz-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
                focus="short",
            ),
            ArtifactContextSummary(
                artifact_id="quiz-2",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-2",
                focus="this focus is intentionally too long for the artifact budget",
            ),
        ],
        max_context_chars=130,
    )

    _assert(result.artifact_context_mode == "pruned", "budget pruning should mark pruned")
    _assert(
        result.referenced_artifact_ids == ["quiz-1"],
        "artifact budget should retain only entries that fit after primary context",
    )
    _assert(
        result.coverage_counts == {"task_unit": 1},
        "coverage counts should reflect budget-kept artifacts",
    )
    _assert(
        result.artifact_context_pruned_reason == "budget_pruned=1",
        "budget pruning reason should be recorded",
    )
    _assert("quiz-2" not in result.context_text, "pruned artifact should not appear in context text")


def test_exports_provenance_metadata_without_context_payload() -> None:
    result = ArtifactAwareContextBuilder().build_secondary_context(
        [
            ArtifactContextSummary(
                artifact_id="quiz-1",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-1",
                concepts=("incentive",),
            ),
            ArtifactContextSummary(
                artifact_id="quiz-2",
                artifact_type="quiz",
                target_level="task_unit",
                target_id="unit-2",
            ),
        ],
        max_artifacts=1,
    )

    metadata = result.to_metadata()

    _assert(
        metadata == {
            "artifact_context_mode": "pruned",
            "referenced_artifact_ids": ["quiz-1"],
            "referenced_artifact_types": ["quiz"],
            "referenced_artifact_target_levels": ["task_unit"],
            "coverage_counts": {"task_unit": 1},
            "deduplication_hint_applied": True,
            "abstraction_hint_applied": False,
            "artifact_context_pruned_reason": "skipped=1",
        },
        "metadata should record artifact context provenance",
    )
    _assert("context_text" not in metadata, "metadata should not include prompt context text")
    _assert("payload" not in metadata, "metadata should not expose raw child payloads")

    result.referenced_artifact_ids.append("mutated-result")
    result.coverage_counts["task_unit"] = 99

    _assert(
        metadata["referenced_artifact_ids"] == ["quiz-1"],
        "metadata should copy referenced artifact ids defensively",
    )
    _assert(
        metadata["coverage_counts"] == {"task_unit": 1},
        "metadata should copy coverage counts defensively",
    )


def test_omits_empty_pruning_reason_from_provenance_metadata() -> None:
    metadata = ArtifactAwareContextBuilder().build_secondary_context([]).to_metadata()

    _assert(metadata["artifact_context_mode"] == "none", "empty context should remain none")
    _assert(
        "artifact_context_pruned_reason" not in metadata,
        "empty pruning reason should be omitted",
    )


def test_rejects_negative_artifact_context_char_budget() -> None:
    try:
        ArtifactAwareContextBuilder().build_secondary_context([], max_context_chars=-1)
    except ValueError as error:
        _assert("max_context_chars must be >= 0" in str(error), "expected char budget error")
        return
    raise AssertionError("expected ValueError for negative max_context_chars")


def test_rejects_negative_max_artifacts() -> None:
    try:
        ArtifactAwareContextBuilder().build_secondary_context([], max_artifacts=-1)
    except ValueError as error:
        _assert("max_artifacts must be >= 0" in str(error), "expected max_artifacts error")
        return
    raise AssertionError("expected ValueError for negative max_artifacts")


if __name__ == "__main__":
    test_builds_compact_secondary_artifact_context()
    test_skips_empty_and_insufficient_artifacts_without_blocking()
    test_prunes_duplicates_and_max_artifacts()
    test_applies_artifact_context_char_budget_after_primary_context_priority()
    test_exports_provenance_metadata_without_context_payload()
    test_omits_empty_pruning_reason_from_provenance_metadata()
    test_rejects_negative_artifact_context_char_budget()
    test_rejects_negative_max_artifacts()
    print("artifact-aware context builder tests passed")
