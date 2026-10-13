#!/usr/bin/env python3
"""Regression coverage for ancestor context and read/no-generate boundaries."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_orchestrator import (  # noqa: E402
    AnalysisInteractionOrchestrator,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService  # noqa: E402
from section_tasks.ancestor_summary_context import (  # noqa: E402
    ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY,
    AncestorSummaryArtifact,
    AncestorSummaryPreparationService,
    AncestorSummaryPromptContext,
    AncestorSummaryRequirement,
    AncestorSummaryTarget,
)
from section_tasks.reading_interaction_prompt_context import (  # noqa: E402
    NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION,
    build_evidence_priority_context_text,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ExternalResearchContext,
    ExternalResearchSourceReference,
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _AnalysisStore:
    def __init__(self) -> None:
        self.get_calls = 0
        self.save_calls = 0

    def get_analysis_artifact(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionArtifact | None:
        self.get_calls += 1
        return None

    def save_analysis_artifact(
        self,
        artifact: ReadingInteractionArtifact,
    ) -> ReadingInteractionArtifact:
        self.save_calls += 1
        return artifact


class _AncestorReader:
    def __init__(self) -> None:
        self.calls = 0

    def get_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact | None:
        self.calls += 1
        return None


class _AncestorGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate_ancestor_summary(
        self,
        requirement: AncestorSummaryRequirement,
    ) -> AncestorSummaryArtifact:
        self.calls += 1
        return AncestorSummaryArtifact(
            target=requirement.target,
            summary="Generated ancestor summary.",
            schema_version=requirement.schema_version,
            prompt_instruction_version=requirement.prompt_instruction_version,
            source_structure_version=requirement.source_structure_version,
            source_hash=requirement.source_hash,
            document_language=requirement.document_language,
        )


def _target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="content_block",
        target_id="unit-1:content:0",
        content="Selected block source text with enough readable content.",
        document_language="en",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
        content_block_id="unit-1:content:0",
    )


def _ancestor_summary() -> AncestorSummaryArtifact:
    return AncestorSummaryArtifact(
        artifact_id="ancestor-summary::doc-1::section::section-1",
        target=AncestorSummaryTarget(
            target_level="section",
            target_id="section-1",
            document_id="doc-1",
            chapter_id="chapter-1",
            section_id="section-1",
            title="Section One",
        ),
        summary="Safe ancestor summary.",
        source_hash="section-source-hash",
        generated_at="2026-10-12T00:00:00Z",
        metadata={"raw_source_text": "RAW ANCESTOR SHOULD NOT APPEAR"},
    )


def _ancestor_context() -> AncestorSummaryPromptContext:
    return AncestorSummaryPromptContext(summaries=(_ancestor_summary(),))


def _external_context() -> ExternalResearchContext:
    return ExternalResearchContext(
        sources=(
            ExternalResearchSourceReference(
                source_id="source-1",
                label="Background source A",
                source_type="web_page",
                title="Historical Background",
                url="https://example.invalid/background",
            ),
        )
    )


def _request() -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_target(),
        interaction_type="analysis",
        ancestor_context=_ancestor_context(),
        context_metadata={
            "document_profile_summary": "Document profile summary.",
            "document_language": "en",
        },
    )


def test_read_path_with_ancestor_context_does_not_generate_or_persist() -> None:
    generator_calls: list[str] = []

    def generator(request: ReadingInteractionRequest) -> dict[str, str]:
        generator_calls.append(request.target.target_id)
        return {
            "summary": "Generated summary.",
            "reasoning": "Generated reasoning.",
            "explanation": "Generated explanation.",
        }

    store = _AnalysisStore()
    orchestrator = AnalysisInteractionOrchestrator(
        service=AnalysisInteractionService(generator),
        artifact_store=store,
    )

    artifact = orchestrator.read(_request())

    _assert(artifact.status == "not_generated", "read should report not_generated")
    _assert(generator_calls == [], "read path must not call interaction generator")
    _assert(store.save_calls == 0, "read path must not persist placeholders")
    _assert(
        ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY not in artifact.metadata,
        "read placeholder should not claim ancestor context metadata",
    )


def test_read_like_ancestor_preparation_is_rejected_before_side_effects() -> None:
    reader = _AncestorReader()
    generator = _AncestorGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)
    requirement = AncestorSummaryRequirement(target=_ancestor_summary().target)

    try:
        service.prepare((requirement,), operation="read")
    except ValueError as error:
        _assert("only allowed for explicit generate or refresh" in str(error), "reason")
    else:
        raise AssertionError("read-like operation should be rejected")

    _assert(reader.calls == 0, "read-like preparation should not query reader")
    _assert(generator.calls == 0, "read-like preparation should not call generator")


def test_explicit_generate_preparation_records_context_without_raw_text() -> None:
    reader = _AncestorReader()
    generator = _AncestorGenerator()
    service = AncestorSummaryPreparationService(reader=reader, generator=generator)
    requirement = AncestorSummaryRequirement(target=_ancestor_summary().target)

    result = service.prepare((requirement,), operation="generate")
    metadata = result.to_metadata()[ANCESTOR_SUMMARY_CONTEXT_METADATA_KEY]

    _assert(generator.calls == 1, "explicit generate should call ancestor generator")
    _assert(metadata["context_status"] == "complete", "context should be complete")
    _assert(metadata["generated_summary_count"] == 1, "generated summary count")
    _assert(
        "Generated ancestor summary." not in str(metadata),
        "metadata must not persist ancestor summary text",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in str(metadata),
        "metadata must not persist raw ancestor text",
    )


def test_prompt_context_excludes_raw_ancestor_text_and_external_mode_by_default() -> None:
    rendered = build_evidence_priority_context_text(_request())

    _assert("Safe ancestor summary." in rendered, "bounded summary should be in prompt")
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in rendered,
        "raw ancestor metadata should not be in prompt",
    )
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in rendered,
        "default-off external research instruction should be present",
    )
    _assert(
        "External Research Mode:" not in rendered,
        "external research section should be absent without explicit context",
    )


def test_explicit_external_context_is_labelled_and_subordinate() -> None:
    request = ReadingInteractionRequest(
        target=_target(),
        interaction_type="analysis",
        external_research_context=_external_context(),
    )

    rendered = build_evidence_priority_context_text(request)

    _assert("External Research Mode:" in rendered, "external mode should be explicit")
    _assert("[Background source A] web_page" in rendered, "source should be labelled")
    _assert(
        rendered.index("Primary Source") < rendered.index("External Research Mode"),
        "primary source should remain first evidence",
    )


if __name__ == "__main__":
    test_read_path_with_ancestor_context_does_not_generate_or_persist()
    test_read_like_ancestor_preparation_is_rejected_before_side_effects()
    test_explicit_generate_preparation_records_context_without_raw_text()
    test_prompt_context_excludes_raw_ancestor_text_and_external_mode_by_default()
    test_explicit_external_context_is_labelled_and_subordinate()
    print("ancestor context regression boundary tests passed")
