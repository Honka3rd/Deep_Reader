#!/usr/bin/env python3
"""Regression tests for reading-interaction evidence-priority prompt context."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_llm_generator import (  # noqa: E402
    AnalysisInteractionLLMGenerator,
)
from section_tasks.ancestor_summary_context import (  # noqa: E402
    AncestorSummaryArtifact,
    AncestorSummaryPromptContext,
    AncestorSummaryTarget,
)
from section_tasks.reading_interaction_prompt_context import (  # noqa: E402
    NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION,
    build_evidence_priority_context_text,
)
from section_tasks.reading_interaction_service_contracts import (  # noqa: E402
    ExternalResearchContext,
    ExternalResearchSourceReference,
    ReadingInteractionRequest,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget  # noqa: E402
from section_tasks.quiz_interaction_llm_generator import (  # noqa: E402
    QuizInteractionLLMGenerator,
)
from section_tasks.critical_thinking_llm_generator import (  # noqa: E402
    CriticalThinkingLLMGenerator,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeLLMProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "{}"


def _content_block_target() -> ResolvedReadingTarget:
    return ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Document One",
        target_level="content_block",
        target_id="unit-1:content:0",
        content="Selected block source text.",
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
        source_structure_version=3,
        source_hash="section-source-hash",
        generated_at="2026-10-12T00:00:00Z",
        metadata={"raw_source_text": "RAW ANCESTOR SHOULD NOT APPEAR"},
    )


def _request() -> ReadingInteractionRequest:
    return ReadingInteractionRequest(
        target=_content_block_target(),
        interaction_type="analysis",
        secondary_context="Previously covered concept.",
        ancestor_context=AncestorSummaryPromptContext(summaries=(_ancestor_summary(),)),
        context_metadata={
            "document_profile_summary": "Document-level profile summary.",
            "document_language": "en",
        },
    )


def _external_research_context() -> ExternalResearchContext:
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


def test_prompt_context_labels_evidence_priority_sections() -> None:
    rendered = build_evidence_priority_context_text(
        _request(),
        secondary_context_label="Secondary Artifact Signals",
    )

    _assert("Evidence priority rules:" in rendered, "prompt should include priority rules")
    _assert("Primary Source (selected content_block text):" in rendered, "block source should be labelled")
    _assert("Selected block source text." in rendered, "primary source should be included")
    _assert("Ancestor Summaries:" in rendered, "ancestor summaries section should be included")
    _assert("Safe ancestor summary." in rendered, "summary text should be included")
    _assert("Document Profile:" in rendered, "profile section should be included")
    _assert("Document-level profile summary." in rendered, "profile summary should be included")
    _assert("Document language: en" in rendered, "document language should be included")
    _assert("Secondary Artifact Signals:" in rendered, "secondary section should be included")
    _assert("Previously covered concept." in rendered, "secondary context should be included")
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in rendered,
        "prompt should forbid external research by default",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in rendered,
        "ancestor raw source metadata must not be concatenated into prompt",
    )
    _assert(
        rendered.index("Primary Source") < rendered.index("Ancestor Summaries"),
        "primary source should precede ancestor summaries",
    )
    _assert(
        rendered.index("Ancestor Summaries") < rendered.index("Document Profile"),
        "ancestor summaries should precede document profile",
    )


def test_prompt_context_labels_explicit_external_research_mode() -> None:
    request = ReadingInteractionRequest(
        target=_content_block_target(),
        interaction_type="analysis",
        external_research_context=_external_research_context(),
    )

    rendered = build_evidence_priority_context_text(request)

    _assert("External Research Mode:" in rendered, "external mode section should render")
    _assert(
        "Explicit external_research mode is enabled." in rendered,
        "external mode should be explicit",
    )
    _assert(
        "[Background source A] web_page: Historical Background" in rendered,
        "external source should be labelled",
    )
    _assert(
        "must not override Primary Source evidence" in rendered,
        "external context should stay subordinate to primary source",
    )
    _assert(
        rendered.index("Primary Source") < rendered.index("External Research Mode"),
        "primary source should precede external research",
    )


def test_request_copy_preserves_ancestor_context() -> None:
    request = _request()
    copied = request.with_secondary_context(
        "Updated secondary context.",
        context_metadata_updates={"context_mode": "artifact_aware"},
    )

    _assert(
        copied.ancestor_context is request.ancestor_context,
        "copying request should preserve ancestor prompt context",
    )
    _assert(
        copied.secondary_context == "Updated secondary context.",
        "secondary context should update",
    )


def test_analysis_generator_uses_evidence_priority_prompt_context() -> None:
    llm_provider = FakeLLMProvider()
    generator = AnalysisInteractionLLMGenerator(llm_provider)

    generator(_request())

    prompt = llm_provider.prompts[0]
    _assert(
        "Insight is not a summary action" in prompt,
        "analysis prompt should separate insight from summary behavior",
    )
    _assert(
        "summary for a short insight thesis" in prompt,
        "analysis prompt should define summary field as insight thesis",
    )
    _assert(
        "1-3 concrete textual details from the Primary Source" in prompt,
        "analysis prompt should require concrete textual details",
    )
    _assert(
        "action, object, bodily condition, metaphor, diction, narrative angle, or ambiguity"
        in prompt,
        "analysis prompt should name examples of textual details",
    )
    _assert(
        "complicate or correct a simple reading" in prompt,
        "analysis prompt should require a complexity move",
    )
    _assert(
        "identify a tension or contrast" in prompt,
        "analysis prompt should ask for supported tension or contrast",
    )
    _assert(
        "plot recap, chapter overview" in prompt,
        "analysis prompt should guard against recap-dominated output",
    )
    _assert(
        "'this section mainly says' framing" in prompt,
        "analysis prompt should reject section-summary framing",
    )
    _assert(
        "brief orientation when needed to anchor the insight" in prompt,
        "analysis prompt should allow only minimal orientation",
    )
    _assert("Evidence priority rules:" in prompt, "generator prompt should include priority rules")
    _assert("Primary Source (selected content_block text):" in prompt, "generator should label source")
    _assert("Selected block source text." in prompt, "generator should include selected block text")
    _assert("Ancestor Summaries:" in prompt, "generator should include ancestor section")
    _assert("Safe ancestor summary." in prompt, "generator should include safe summary")
    _assert("Document language: en" in prompt, "analysis prompt should include language guidance")
    _assert(
        prompt.index("Primary Source") < prompt.index("Ancestor Summaries"),
        "analysis prompt should keep primary source before ancestor summaries",
    )
    _assert(
        prompt.index("Ancestor Summaries") < prompt.index("Document Profile"),
        "analysis prompt should keep ancestor summaries before profile guidance",
    )
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in prompt,
        "analysis prompt should forbid external research by default",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in prompt,
        "generator prompt should not include raw ancestor source metadata",
    )


def test_quiz_generator_uses_evidence_priority_prompt_context() -> None:
    llm_provider = FakeLLMProvider()
    generator = QuizInteractionLLMGenerator(llm_provider)

    generator(_request(), max_items=3, valid_types=("short_answer", "multiple_choice"))

    prompt = llm_provider.prompts[0]
    _assert("Evidence priority rules:" in prompt, "quiz prompt should include priority rules")
    _assert("Primary Source (selected content_block text):" in prompt, "quiz should label source")
    _assert("Ancestor Summaries:" in prompt, "quiz should include ancestor section")
    _assert("Safe ancestor summary." in prompt, "quiz should include safe summary")
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in prompt,
        "quiz prompt should forbid external research by default",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in prompt,
        "quiz prompt should not include raw ancestor source metadata",
    )


def test_critical_thinking_generator_uses_evidence_priority_prompt_context() -> None:
    llm_provider = FakeLLMProvider()
    generator = CriticalThinkingLLMGenerator(llm_provider)
    request = ReadingInteractionRequest(
        target=_content_block_target(),
        interaction_type="critical_thinking_session",
        secondary_context="Previously covered concept.",
        ancestor_context=AncestorSummaryPromptContext(summaries=(_ancestor_summary(),)),
        context_metadata={
            "document_profile_summary": "Document-level profile summary.",
            "document_language": "en",
        },
    )

    generator.generate_question(request, "Generate one question.")

    prompt = llm_provider.prompts[0]
    _assert("Evidence priority rules:" in prompt, "critical prompt should include priority rules")
    _assert("Primary Source (selected content_block text):" in prompt, "critical should label source")
    _assert("Ancestor Summaries:" in prompt, "critical should include ancestor section")
    _assert("Safe ancestor summary." in prompt, "critical should include safe summary")
    _assert(
        NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION in prompt,
        "critical prompt should forbid external research by default",
    )
    _assert(
        "RAW ANCESTOR SHOULD NOT APPEAR" not in prompt,
        "critical prompt should not include raw ancestor source metadata",
    )


if __name__ == "__main__":
    test_prompt_context_labels_evidence_priority_sections()
    test_prompt_context_labels_explicit_external_research_mode()
    test_request_copy_preserves_ancestor_context()
    test_analysis_generator_uses_evidence_priority_prompt_context()
    test_quiz_generator_uses_evidence_priority_prompt_context()
    test_critical_thinking_generator_uses_evidence_priority_prompt_context()
    print("reading interaction prompt context tests passed")
