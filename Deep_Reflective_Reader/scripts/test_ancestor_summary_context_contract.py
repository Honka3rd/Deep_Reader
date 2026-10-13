#!/usr/bin/env python3
"""Regression tests for ancestor summary prompt-context contracts."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.ancestor_summary_context import (  # noqa: E402
    ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY,
    ANCESTOR_SUMMARY_SCHEMA_VERSION,
    AncestorSummaryArtifact,
    AncestorSummaryPreparationError,
    AncestorSummaryPreparationFailure,
    AncestorSummaryPreparationService,
    AncestorSummaryPromptContext,
    AncestorSummaryReadinessResult,
    AncestorSummaryRequirement,
    AncestorSummaryTarget,
    evaluate_ancestor_summary_readiness,
)


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


def _task_unit_summary() -> AncestorSummaryArtifact:
    return AncestorSummaryArtifact(
        artifact_id="ancestor-summary::doc-1::task_unit::unit-1",
        target=AncestorSummaryTarget(
            target_level="task_unit",
            target_id="unit-1",
            document_id="doc-1",
            chapter_id="chapter-1",
            section_id="section-1",
            task_unit_id="unit-1",
            title="Unit One",
        ),
        summary="The unit frames the selected passage as a choice with consequences.",
        prompt_instruction_version="ancestor_summary_prompt_v1",
        source_structure_version=3,
        source_hash="hash-unit-1",
        document_language="en",
        generated_at="2026-10-12T10:00:00+00:00",
        metadata={"token_estimate": 14},
    )


def _task_unit_requirement() -> AncestorSummaryRequirement:
    summary = _task_unit_summary()
    return AncestorSummaryRequirement(
        target=summary.target,
        prompt_instruction_version="ancestor_summary_prompt_v1",
        source_structure_version=3,
        source_hash="hash-unit-1",
        document_language="en",
    )


class _FakeAncestorSummaryReader:
    def __init__(
        self,
        summaries: dict[str, AncestorSummaryArtifact] | None = None,
    ) -> None:
        self.summaries = {} if summaries is None else dict(summaries)
        self.calls: list[AncestorSummaryRequirement] = []

    def get_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact | None:
        self.calls.append(requirement)
        return self.summaries.get(requirement.target.target_id)


class _FakeAncestorSummaryGenerator:
    def __init__(self) -> None:
        self.calls: list[AncestorSummaryRequirement] = []

    def generate_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact:
        self.calls.append(requirement)
        return AncestorSummaryArtifact(
            artifact_id=(
                "ancestor-summary::"
                f"{requirement.target.document_id}::{requirement.target.target_level}::"
                f"{requirement.target.target_id}"
            ),
            target=requirement.target,
            summary=f"Generated summary for {requirement.target.target_id}.",
            schema_version=requirement.schema_version,
            prompt_instruction_version=requirement.prompt_instruction_version,
            source_structure_version=requirement.source_structure_version,
            source_hash=requirement.source_hash,
            document_language=requirement.document_language,
            generated_at="2026-10-12T11:00:00+00:00",
        )


def test_ancestor_summary_artifact_serializes_contract_metadata() -> None:
    artifact = _task_unit_summary()
    payload = artifact.to_dict()

    _assert(payload["schema_version"] == ANCESTOR_SUMMARY_SCHEMA_VERSION, "schema version")
    _assert(payload["target"]["target_level"] == "task_unit", "target level")
    _assert(payload["target"]["task_unit_id"] == "unit-1", "task unit identity")
    _assert(payload["source_structure_version"] == 3, "structure version")
    _assert(payload["source_hash"] == "hash-unit-1", "source hash")
    _assert(payload["document_language"] == "en", "language metadata")
    _assert("source_text" not in payload, "raw source text should not be serialized")
    _assert("content" not in payload, "raw ancestor content should not be serialized")


def test_ancestor_summary_prompt_reference_is_bounded() -> None:
    reference = _task_unit_summary().to_prompt_reference()

    _assert(reference["summary"].startswith("The unit frames"), "summary included")
    _assert(reference["target_level"] == "task_unit", "level included")
    _assert(reference["target_id"] == "unit-1", "id included")
    _assert(reference["source_hash"] == "hash-unit-1", "source hash included")
    _assert("metadata" not in reference, "metadata bag should stay out of prompt ref")
    _assert("source_text" not in reference, "raw source text forbidden")
    _assert("content" not in reference, "raw content forbidden")


def test_prompt_context_metadata_records_provenance_without_summary_text() -> None:
    context = AncestorSummaryPromptContext(summaries=(_task_unit_summary(),))
    metadata = context.to_metadata()[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY]

    _assert(metadata["context_status"] == "complete", "status")
    _assert(metadata["summary_count"] == 1, "count")
    _assert(
        metadata["summary_artifact_ids"]
        == ["ancestor-summary::doc-1::task_unit::unit-1"],
        "artifact ids",
    )
    _assert(metadata["summary_target_levels"] == ["task_unit"], "levels")
    _assert("summary" not in metadata, "summary text should not persist in metadata")


def test_prompt_context_allows_explicit_partial_context() -> None:
    context = AncestorSummaryPromptContext(
        summaries=(_task_unit_summary(),),
        context_status="partial",
        missing_target_levels=("section",),
        stale_target_levels=("chapter",),
        pruned_reason="token_budget",
    )
    metadata = context.to_metadata()[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY]

    _assert(metadata["context_status"] == "partial", "partial status")
    _assert(metadata["missing_target_levels"] == ["section"], "missing levels")
    _assert(metadata["stale_target_levels"] == ["chapter"], "stale levels")
    _assert(metadata["pruned_reason"] == "token_budget", "pruned reason")


def test_contract_rejects_invalid_target_or_empty_summary() -> None:
    _expect_value_error(
        lambda: AncestorSummaryTarget(
            target_level="content_block",
            target_id="unit-1:content:0",
            document_id="doc-1",
            task_unit_id="unit-1",
        ),
        "unsupported ancestor summary target level",
    )
    _expect_value_error(
        lambda: AncestorSummaryArtifact(
            target=AncestorSummaryTarget(
                target_level="section",
                target_id="section-1",
                document_id="doc-1",
                section_id="section-1",
            ),
            summary=" ",
        ),
        "requires non-empty summary",
    )


def test_contract_rejects_inconsistent_context_status() -> None:
    _expect_value_error(
        lambda: AncestorSummaryPromptContext(
            summaries=(_task_unit_summary(),),
            context_status="complete",
            missing_target_levels=("section",),
        ),
        "complete context cannot declare missing or stale levels",
    )
    _expect_value_error(
        lambda: AncestorSummaryPromptContext(
            summaries=(_task_unit_summary(),),
            context_status="empty",
        ),
        "empty context cannot include ancestor summaries",
    )


def test_readiness_policy_reuses_matching_summary() -> None:
    summary = _task_unit_summary()
    requirement = AncestorSummaryRequirement(
        target=summary.target,
        prompt_instruction_version="ancestor_summary_prompt_v1",
        source_structure_version=3,
        source_hash="hash-unit-1",
        document_language="en",
    )

    result = evaluate_ancestor_summary_readiness(requirement, summary)

    _assert(result.status == "reusable", "matching summary should be reusable")
    _assert(result.is_reusable, "reusable property")
    _assert(result.summary is summary, "summary should be preserved")
    _assert(result.to_metadata()["stale_fields"] == [], "no stale fields")


def test_readiness_policy_reports_missing_without_generation() -> None:
    requirement = AncestorSummaryRequirement(target=_task_unit_summary().target)

    result = evaluate_ancestor_summary_readiness(requirement, None)

    _assert(result.status == "missing", "missing summary should be explicit")
    _assert(not result.is_reusable, "missing is not reusable")
    _assert(result.summary is None, "missing should not invent a summary")
    _assert("missing" in (result.reason or ""), "missing reason")


def test_readiness_policy_reports_explicit_refresh_required() -> None:
    summary = _task_unit_summary()
    requirement = AncestorSummaryRequirement(target=summary.target)

    result = evaluate_ancestor_summary_readiness(
        requirement,
        summary,
        refresh=True,
    )

    _assert(result.status == "refresh_required", "refresh status")
    _assert(result.summary is summary, "existing summary is preserved for provenance")
    _assert(result.reason == "explicit refresh requested", "refresh reason")


def test_readiness_policy_detects_stale_source_and_prompt_fields() -> None:
    summary = _task_unit_summary()
    requirement = AncestorSummaryRequirement(
        target=summary.target,
        prompt_instruction_version="ancestor_summary_prompt_v2",
        source_structure_version=4,
        source_hash="hash-unit-2",
        document_language="fr",
    )

    result = evaluate_ancestor_summary_readiness(requirement, summary)

    _assert(result.status == "stale", "mismatched requirement should be stale")
    _assert(
        result.stale_fields
        == (
            "prompt_instruction_version",
            "source_structure_version",
            "source_hash",
            "document_language",
        ),
        "stale fields should identify changed evidence",
    )


def test_readiness_policy_ignores_unavailable_optional_source_evidence() -> None:
    summary = _task_unit_summary()
    requirement = AncestorSummaryRequirement(target=summary.target)

    result = evaluate_ancestor_summary_readiness(requirement, summary)

    _assert(result.status == "reusable", "missing optional requirement evidence ignored")


def test_readiness_result_rejects_invalid_shapes() -> None:
    requirement = AncestorSummaryRequirement(target=_task_unit_summary().target)
    _expect_value_error(
        lambda: AncestorSummaryReadinessResult(
            status="reusable",
            requirement=requirement,
            reason="should not exist",
        ),
        "requires a summary",
    )
    _expect_value_error(
        lambda: AncestorSummaryReadinessResult(
            status="stale",
            requirement=requirement,
            summary=_task_unit_summary(),
            reason="stale",
        ),
        "requires stale_fields",
    )


def test_preparation_rejects_read_like_operation_before_generation() -> None:
    reader = _FakeAncestorSummaryReader()
    generator = _FakeAncestorSummaryGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)

    _expect_value_error(
        lambda: service.prepare((_task_unit_requirement(),), operation="read"),
        "only allowed for explicit generate or refresh operations",
    )
    _assert(reader.calls == [], "read-like operation should not query reader")
    _assert(generator.calls == [], "read-like operation should not call generator")


def test_preparation_reuses_existing_summary_for_generate_operation() -> None:
    summary = _task_unit_summary()
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader({summary.target.target_id: summary})
    generator = _FakeAncestorSummaryGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)

    result = service.prepare((requirement,), operation="generate")

    _assert(len(reader.calls) == 1, "reader should be queried")
    _assert(generator.calls == [], "matching summary should not regenerate")
    _assert(result.prompt_context.summaries == (summary,), "summary should be reused")
    _assert(result.generated_summaries == (), "no generated summaries")
    _assert(result.readiness_results[0].status == "reusable", "reusable readiness")


def test_preparation_generates_missing_summary_for_generate_operation() -> None:
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader()
    generator = _FakeAncestorSummaryGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)

    result = service.prepare((requirement,), operation="generate")

    _assert(len(reader.calls) == 1, "reader should be queried")
    _assert(generator.calls == [requirement], "missing summary should generate")
    _assert(result.readiness_results[0].status == "missing", "missing readiness")
    _assert(len(result.generated_summaries) == 1, "generated summary recorded")
    _assert(
        result.prompt_context.summaries == result.generated_summaries,
        "generated summary should become prompt context",
    )


def test_preparation_refresh_forces_regeneration_even_when_existing_is_reusable() -> None:
    summary = _task_unit_summary()
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader({summary.target.target_id: summary})
    generator = _FakeAncestorSummaryGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)

    result = service.prepare((requirement,), operation="refresh")

    _assert(generator.calls == [requirement], "refresh should generate")
    _assert(result.readiness_results[0].status == "refresh_required", "refresh readiness")
    _assert(result.prompt_context.summaries != (summary,), "old summary should not be used")


def test_preparation_rejects_generated_summary_that_is_still_stale() -> None:
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader()

    class _BadGenerator:
        def generate_ancestor_summary(
            self,
            requirement: AncestorSummaryRequirement,
        ) -> AncestorSummaryArtifact:
            return AncestorSummaryArtifact(
                target=requirement.target,
                summary="Generated with wrong source hash.",
                schema_version=requirement.schema_version,
                prompt_instruction_version=requirement.prompt_instruction_version,
                source_structure_version=requirement.source_structure_version,
                source_hash="wrong-hash",
                document_language=requirement.document_language,
            )

    service = AncestorSummaryPreparationService(
        reader=reader,
        generator=_BadGenerator(),
    )

    _expect_value_error(
        lambda: service.prepare((requirement,), operation="generate"),
        "does not satisfy current requirement",
    )


def test_preparation_fail_fast_reports_generation_failure_without_partial_context() -> None:
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader()

    class _FailingGenerator:
        def generate_ancestor_summary(
            self,
            requirement: AncestorSummaryRequirement,
        ) -> AncestorSummaryArtifact:
            raise RuntimeError("upstream summary service unavailable")

    service = AncestorSummaryPreparationService(
        reader=reader,
        generator=_FailingGenerator(),
    )

    try:
        service.prepare((requirement,), operation="generate")
    except AncestorSummaryPreparationError as error:
        _assert(
            "upstream summary service unavailable" in str(error),
            "failure reason should be surfaced",
        )
        _assert(error.requirement == requirement, "requirement should be attached")
        _assert(error.readiness.status == "missing", "readiness should be attached")
        return
    raise AssertionError("expected AncestorSummaryPreparationError")


def test_preparation_degraded_mode_returns_explicit_partial_context() -> None:
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader()

    class _FailingGenerator:
        def generate_ancestor_summary(
            self,
            requirement: AncestorSummaryRequirement,
        ) -> AncestorSummaryArtifact:
            raise RuntimeError("upstream summary service unavailable")

    service = AncestorSummaryPreparationService(
        reader=reader,
        generator=_FailingGenerator(),
    )

    result = service.prepare(
        (requirement,),
        operation="generate",
        failure_policy="degraded",
    )
    metadata = result.to_metadata()[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY]

    _assert(
        result.prompt_context.context_status == "partial",
        "degraded preparation should return partial context",
    )
    _assert(result.prompt_context.summaries == (), "failed summaries should not be used")
    _assert(
        result.prompt_context.missing_target_levels == ("task_unit",),
        "failed missing levels should be recorded",
    )
    _assert(
        result.prompt_context.pruned_reason == "ancestor_summary_preparation_failed",
        "partial context should name degraded reason",
    )
    _assert(
        len(result.preparation_failures) == 1,
        "preparation failure should be recorded",
    )
    _assert(metadata["context_status"] == "partial", "metadata partial status")
    _assert(
        metadata["preparation_failure_count"] == 1,
        "metadata should record failure count",
    )
    _assert(
        metadata["preparation_failures"][0]["error_type"] == "RuntimeError",
        "metadata should record bounded error type",
    )
    _assert(
        "upstream summary service unavailable" in metadata["preparation_failures"][0]["reason"],
        "metadata should record failure reason",
    )
    _assert(
        "The unit frames the selected passage" not in str(metadata),
        "metadata should not store summary text",
    )


def test_preparation_degraded_mode_records_generated_stale_context_as_partial() -> None:
    requirement = _task_unit_requirement()
    reader = _FakeAncestorSummaryReader()

    class _BadGenerator:
        def generate_ancestor_summary(
            self,
            requirement: AncestorSummaryRequirement,
        ) -> AncestorSummaryArtifact:
            return AncestorSummaryArtifact(
                target=requirement.target,
                summary="Generated with wrong source hash.",
                schema_version=requirement.schema_version,
                prompt_instruction_version=requirement.prompt_instruction_version,
                source_structure_version=requirement.source_structure_version,
                source_hash="wrong-hash",
                document_language=requirement.document_language,
            )

    service = AncestorSummaryPreparationService(
        reader=reader,
        generator=_BadGenerator(),
    )

    result = service.prepare(
        (requirement,),
        operation="generate",
        failure_policy="degraded",
    )

    _assert(
        result.prompt_context.context_status == "partial",
        "stale generated summary should degrade explicitly",
    )
    _assert(
        result.prompt_context.stale_target_levels == ("task_unit",),
        "stale failed levels should be recorded",
    )
    _assert(result.generated_summaries == (), "stale summary should not be admitted")
    _assert(
        result.preparation_failures[0].readiness_status == "stale",
        "failure readiness should reflect generated stale result",
    )


def test_preparation_rejects_unknown_failure_policy_before_generation() -> None:
    reader = _FakeAncestorSummaryReader()
    generator = _FakeAncestorSummaryGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)

    _expect_value_error(
        lambda: service.prepare(
            (_task_unit_requirement(),),
            operation="generate",
            failure_policy="silent",
        ),
        "failure_policy must be",
    )
    _assert(reader.calls == [], "invalid policy should not query reader")
    _assert(generator.calls == [], "invalid policy should not call generator")


def test_preparation_failure_rejects_empty_reason() -> None:
    _expect_value_error(
        lambda: AncestorSummaryPreparationFailure(
            requirement=_task_unit_requirement(),
            readiness_status="missing",
            reason=" ",
        ),
        "requires a reason",
    )


if __name__ == "__main__":
    test_ancestor_summary_artifact_serializes_contract_metadata()
    test_ancestor_summary_prompt_reference_is_bounded()
    test_prompt_context_metadata_records_provenance_without_summary_text()
    test_prompt_context_allows_explicit_partial_context()
    test_contract_rejects_invalid_target_or_empty_summary()
    test_contract_rejects_inconsistent_context_status()
    test_readiness_policy_reuses_matching_summary()
    test_readiness_policy_reports_missing_without_generation()
    test_readiness_policy_reports_explicit_refresh_required()
    test_readiness_policy_detects_stale_source_and_prompt_fields()
    test_readiness_policy_ignores_unavailable_optional_source_evidence()
    test_readiness_result_rejects_invalid_shapes()
    test_preparation_rejects_read_like_operation_before_generation()
    test_preparation_reuses_existing_summary_for_generate_operation()
    test_preparation_generates_missing_summary_for_generate_operation()
    test_preparation_refresh_forces_regeneration_even_when_existing_is_reusable()
    test_preparation_rejects_generated_summary_that_is_still_stale()
    test_preparation_fail_fast_reports_generation_failure_without_partial_context()
    test_preparation_degraded_mode_returns_explicit_partial_context()
    test_preparation_degraded_mode_records_generated_stale_context_as_partial()
    test_preparation_rejects_unknown_failure_policy_before_generation()
    test_preparation_failure_rejects_empty_reason()
    print("ancestor summary context contract tests passed")
