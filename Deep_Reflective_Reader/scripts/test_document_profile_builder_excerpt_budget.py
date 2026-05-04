#!/usr/bin/env python3
"""Regression tests for DocumentProfileBuilder excerpt budgeting."""

from __future__ import annotations

from dataclasses import dataclass
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile_builder import DocumentProfileBuilder


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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        max_input_tokens: int = 8_000,
        capability_should_raise: bool = False,
    ):
        self.prompts: list[str] = []
        self._max_input_tokens = max_input_tokens
        self._capability_should_raise = capability_should_raise
        self._call_count = 0

    def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._call_count == 0:
            self._call_count += 1
            return "{invalid-json}"
        if self._call_count == 1:
            self._call_count += 1
            return "essay"
        self._call_count += 1
        return "summary"

    def get_model_capabilities(self) -> LLMModelCapabilities:
        if self._capability_should_raise:
            raise RuntimeError("capability_unavailable")
        return LLMModelCapabilities(
            model_name="fake-model",
            endpoint_kind=ENDPOINT_KIND_RESPONSES,
            max_input_tokens=self._max_input_tokens,
            max_output_tokens=2_000,
        )


def _extract_document_excerpt(prompt: str) -> str:
    marker = "Document excerpt:\n"
    start = prompt.rfind(marker)
    _assert(start >= 0, "Document excerpt marker missing in prompt")
    return prompt[start + len(marker) :].rstrip("\n")


def _build_prompt_text_normalization() -> _PromptTextNormalizationConfig:
    return _PromptTextNormalizationConfig(
        default_excerpt_chars=16_000,
        min_excerpt_chars=2_000,
        max_excerpt_chars_hard_cap=400_000,
        chars_per_token=4,
    )


def _build_profile_prompt_policy() -> _ProfilePromptPolicyConfig:
    return _ProfilePromptPolicyConfig(
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
    )


def test_summary_excerpt_uses_model_capability_budget() -> None:
    text = "".join(str(index % 10) for index in range(30_000))
    provider = _FakeLLMProvider(max_input_tokens=8_000)
    builder = DocumentProfileBuilder(
        provider,
        prompt_text_normalization=_build_prompt_text_normalization(),
        profile_prompt_policy=_build_profile_prompt_policy(),
    )

    profile = builder.build(text=text, document_language="en")
    _assert(profile.topic == "essay", "topic should be generated")
    _assert(profile.summary == "summary", "summary should be generated")
    _assert(len(provider.prompts) == 3, "builder should issue three LLM calls on fallback path")

    summary_excerpt = _extract_document_excerpt(provider.prompts[2])
    _assert(
        len(summary_excerpt) > 5_000,
        "summary excerpt should exceed legacy 5000 hard cap when model allows",
    )
    _assert(
        text.startswith(summary_excerpt),
        "summary excerpt should be a prefix slice of input text",
    )


def test_excerpt_fallback_when_capability_unavailable() -> None:
    text = "x" * 30_000
    provider = _FakeLLMProvider(capability_should_raise=True)
    builder = DocumentProfileBuilder(
        provider,
        prompt_text_normalization=_build_prompt_text_normalization(),
        profile_prompt_policy=_build_profile_prompt_policy(),
    )
    builder.build(text=text, document_language="zh")

    topic_excerpt = _extract_document_excerpt(provider.prompts[1])
    summary_excerpt = _extract_document_excerpt(provider.prompts[2])
    _assert(
        len(topic_excerpt) == 16_000,
        "topic excerpt should fall back to default excerpt length",
    )
    _assert(
        len(summary_excerpt) == 16_000,
        "summary excerpt should fall back to default excerpt length",
    )


if __name__ == "__main__":
    test_summary_excerpt_uses_model_capability_budget()
    test_excerpt_fallback_when_capability_unavailable()
    print("test_document_profile_builder_excerpt_budget: ok")
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
