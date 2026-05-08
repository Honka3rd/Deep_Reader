#!/usr/bin/env python3
"""Semantic hardening tests for parser metadata + profile classification prompt."""

from __future__ import annotations

from dataclasses import dataclass

from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile import DialogueDensity, DocumentStructureShape, HeadingStyle, LikelihoodLevel
from profile.document_profile_builder import DocumentProfileBuilder
from profile.parser_metadata_extractor import ParserMetadataExtractor


@dataclass(frozen=True)
class _PromptTextNormalizationConfig:
    default_excerpt_chars: int
    min_excerpt_chars: int
    max_excerpt_chars_hard_cap: int
    chars_per_token: int


@dataclass(frozen=True)
class _ProfilePromptPolicyConfig:
    topic_prompt_overhead_tokens: int
    summary_prompt_overhead_tokens: int
    structure_prompt_overhead_tokens: int
    topic_excerpt_token_utilization_ratio: float
    summary_excerpt_token_utilization_ratio: float
    structure_excerpt_token_utilization_ratio: float
    evidence_head_chars: int
    evidence_tail_chars: int
    evidence_middle_chars: int
    evidence_heading_lines_limit: int


class _FakeLLMProvider(LLMProvider):
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._outputs:
            raise RuntimeError("missing_fake_output")
        return self._outputs.pop(0)

    def get_model_capabilities(self) -> LLMModelCapabilities:
        return LLMModelCapabilities(
            model_name="fake-model",
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=64_000,
            max_output_tokens=2_000,
        )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_builder(provider: LLMProvider) -> DocumentProfileBuilder:
    return DocumentProfileBuilder(
        llm_provider=provider,
        prompt_text_normalization=_PromptTextNormalizationConfig(
            default_excerpt_chars=16_000,
            min_excerpt_chars=2_000,
            max_excerpt_chars_hard_cap=400_000,
            chars_per_token=4,
        ),
        profile_prompt_policy=_ProfilePromptPolicyConfig(
            topic_prompt_overhead_tokens=700,
            summary_prompt_overhead_tokens=900,
            structure_prompt_overhead_tokens=1_100,
            topic_excerpt_token_utilization_ratio=0.55,
            summary_excerpt_token_utilization_ratio=0.70,
            structure_excerpt_token_utilization_ratio=0.72,
            evidence_head_chars=4_000,
            evidence_tail_chars=2_500,
            evidence_middle_chars=2_200,
            evidence_heading_lines_limit=60,
        ),
    )


def test_dialogue_density_quotation_only_essay_not_high() -> None:
    extractor = ParserMetadataExtractor()
    text = "\n".join(
        [
            "本文讨论“理性”“经验”“制度”“秩序”等概念，强调“分析”与“比较”的方法。",
            "作者指出“现代性”的形成并非线性过程，而是多变量共同作用。",
            "在“传统”与“变革”的张力中，关键在于“制度能力”的累积。",
            "这些“关键词”用于说明概念，不构成人物对话。",
        ]
    )
    density = extractor.extract(text=text, document_language="zh").dialogue_density
    _assert(
        density != DialogueDensity.HIGH,
        "quotation-heavy essay should not be classified as high dialogue density",
    )


def test_dialogue_density_real_dialogue_is_medium_or_high() -> None:
    extractor = ParserMetadataExtractor()
    text = "\n".join(
        [
            "“你今天还去吗？”他说。",
            "“去。”她回答。",
            "— 我们现在就出发。",
            "— 好，马上走。",
            "“别忘了带钥匙。”",
            "“知道了。”",
        ]
    )
    density = extractor.extract(text=text, document_language="zh").dialogue_density
    _assert(
        density in {DialogueDensity.MEDIUM, DialogueDensity.HIGH},
        "real dialogue should be medium/high",
    )


def test_prompt_contains_title_uniqueness_semantics() -> None:
    provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "literary fiction",
              "summary": "A novel.",
              "document_language": "en",
              "text_form": "novel",
              "discourse_mode": "narrative",
              "document_structure_shape": "part_chapter",
              "likely_heading_style": "english_chapter_words",
              "title_uniqueness_risk": "high",
              "confidence": 0.9,
              "notes": []
            }
            """
        ]
    )
    builder = _build_builder(provider)
    builder.build(
        text="PART I\nChapter One\n...\nPART II\nChapter One\n...",
        document_language="en",
    )
    prompt = provider.prompts[0]
    _assert(
        "not about whether the document/book title is unique" in prompt,
        "prompt should clarify not about document title",
    )
    _assert(
        "structural node titles inside the document" in prompt,
        "prompt should define structural title semantics",
    )
    _assert(
        "parent grouping units indicate high risk" in prompt,
        "prompt should use parent-grouping semantic definition",
    )


def test_madame_bovary_style_classification_round_trip() -> None:
    provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "literary fiction",
              "summary": "A classic novel with repeated chapter titles across parts.",
              "document_language": "en",
              "text_form": "novel",
              "discourse_mode": "narrative",
              "document_structure_shape": "part_chapter",
              "likely_heading_style": "english_chapter_words",
              "title_uniqueness_risk": "high",
              "confidence": 0.88,
              "notes": ["Chapter labels repeat under different parts."]
            }
            """
        ]
    )
    builder = _build_builder(provider)
    profile = builder.build(
        text="PART I\nChapter One\n...\nPART II\nChapter One\n...",
        document_language="en",
    )
    metadata = profile.parser_metadata
    _assert(metadata is not None, "parser metadata should exist")
    _assert(
        metadata.document_structure_shape == DocumentStructureShape.PART_CHAPTER,
        "structure shape should be part_chapter",
    )
    _assert(
        metadata.likely_heading_style == HeadingStyle.ENGLISH_CHAPTER_WORDS,
        "heading style should be english_chapter_words",
    )
    _assert(
        metadata.title_uniqueness_risk == LikelihoodLevel.HIGH,
        "title uniqueness risk should be high",
    )


if __name__ == "__main__":
    test_dialogue_density_quotation_only_essay_not_high()
    test_dialogue_density_real_dialogue_is_medium_or_high()
    test_prompt_contains_title_uniqueness_semantics()
    test_madame_bovary_style_classification_round_trip()
    print("test_profile_metadata_semantic_hardening: ok")
