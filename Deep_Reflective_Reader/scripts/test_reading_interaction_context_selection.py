from __future__ import annotations

from llm.llm_model_capabilities import (
    ENDPOINT_KIND_RESPONSES,
    LLMModelCapabilities,
)
from context.reading_interaction_context import (
    COMPACTION_REASON_TARGET_EXCEEDS_BUDGET,
    CONTEXT_MODE_FULL_TARGET,
    CONTEXT_MODE_SEMANTIC_COMPACT,
    ReadingInteractionContextBuilder,
)
from section_tasks.reading_target_resolver import ResolvedReadingTarget


def _capabilities(max_input_tokens: int) -> LLMModelCapabilities:
    return LLMModelCapabilities(
        model_name="test-model",
        endpoint_kind=ENDPOINT_KIND_RESPONSES,
        max_input_tokens=max_input_tokens,
        max_output_tokens=256,
    )


def test_full_target_context_when_within_model_capability() -> None:
    target = ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Example",
        target_level="chapter",
        target_id="chapter-1",
        content="A short chapter about product strategy and customer focus.",
        chapter_id="chapter-1",
    )
    builder = ReadingInteractionContextBuilder(default_max_context_tokens=200)

    result = builder.build_context(
        target,
        fixed_prompt_instruction="Summarize this target.",
        model_capabilities=_capabilities(500),
        reserved_output_tokens=40,
    )

    assert result.context_mode == CONTEXT_MODE_FULL_TARGET
    assert result.context_text == target.content
    assert result.evidence_ids == ["chapter:chapter-1"]
    assert result.used_context_tokens == result.token_estimate
    assert result.effective_context_budget <= 200

    metadata = result.to_metadata()
    assert metadata["context_mode"] == CONTEXT_MODE_FULL_TARGET
    assert metadata["target_level"] == "chapter"
    assert metadata["target_id"] == "chapter-1"
    assert metadata["model_capability_source"] == "test-model"
    assert metadata["evidence_ids"] == ["chapter:chapter-1"]
    assert "context_text" not in metadata


def test_semantic_compact_context_when_target_exceeds_capability() -> None:
    target = ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Example",
        target_level="section",
        target_id="section-1",
        content=(
            "Opening thesis: customer retention drives durable growth. "
            "This paragraph explains why recurring feedback loops matter.\n\n"
            "Middle evidence: churn falls when onboarding and support are connected. "
            "This paragraph connects operational practice with financial impact.\n\n"
            "Closing synthesis: leaders should compare acquisition cost with retention "
            "quality before scaling spend. This paragraph frames the final implication."
        ),
        chapter_id="chapter-1",
        section_id="section-1",
    )
    builder = ReadingInteractionContextBuilder(default_max_context_tokens=45)

    result = builder.build_context(
        target,
        fixed_prompt_instruction="Generate quiz items from this target.",
        model_capabilities=_capabilities(80),
        reserved_output_tokens=20,
    )

    assert result.context_mode == CONTEXT_MODE_SEMANTIC_COMPACT
    assert result.context_text != target.content
    assert "[section:section-1#chunk-" in result.context_text
    assert result.used_context_tokens <= result.effective_context_budget
    assert result.evidence_ids
    assert all(
        evidence_id.startswith("section:section-1#chunk-")
        for evidence_id in result.evidence_ids
    )
    assert result.compaction_reason == COMPACTION_REASON_TARGET_EXCEEDS_BUDGET
    assert result.truncated is True

    metadata = result.to_metadata()
    assert metadata["context_mode"] == CONTEXT_MODE_SEMANTIC_COMPACT
    assert metadata["compaction_reason"] == COMPACTION_REASON_TARGET_EXCEEDS_BUDGET
    assert metadata["effective_context_budget"] == result.effective_context_budget
    assert metadata["evidence_ids"] == result.evidence_ids
    assert "context_text" not in metadata


def test_empty_target_content_rejects_without_title_fallback() -> None:
    target = ResolvedReadingTarget(
        document_id="doc-1",
        document_title="Fallback Title Must Not Become Context",
        target_level="task_unit",
        target_id="unit-1",
        content="   ",
        chapter_id="chapter-1",
        section_id="section-1",
        task_unit_id="unit-1",
    )
    builder = ReadingInteractionContextBuilder(default_max_context_tokens=200)

    try:
        builder.build_context(target, model_capabilities=_capabilities(500))
    except ValueError as exc:
        assert "resolved target content is required" in str(exc)
    else:
        raise AssertionError("empty target content should be rejected")


if __name__ == "__main__":
    test_full_target_context_when_within_model_capability()
    test_semantic_compact_context_when_target_exceeds_capability()
    test_empty_target_content_rejects_without_title_fallback()
    print("reading interaction context selection tests passed")
