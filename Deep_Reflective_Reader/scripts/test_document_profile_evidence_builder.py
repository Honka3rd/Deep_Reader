#!/usr/bin/env python3
"""Tests for DocumentProfileEvidenceBuilder using shared structure language registry."""

from __future__ import annotations

from dataclasses import dataclass

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile_evidence_builder import DocumentProfileEvidenceBuilder


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
    def complete_text(self, prompt: str) -> str:
        _ = prompt
        return ""

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


def test_evidence_builder_outputs_candidate_lines() -> None:
    builder = DocumentProfileEvidenceBuilder(
        structure_language_registry=DocumentStructureLanguageRegistry(),
    )
    text = "\n".join(
        [
            "PART I",
            "Chapter One",
            "第一章",
            "序",
            "Appendix",
            "普通正文普通正文普通正文普通正文普通正文普通正文普通正文普通正文普通正文普通正文",
        ]
    )
    evidence = builder.build(
        text=text,
        document_language="zh",
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
            evidence_heading_lines_limit=3,
        ),
        llm_provider=_FakeLLMProvider(),
        prompt_text_normalization=_PromptTextNormalizationConfig(
            default_excerpt_chars=16_000,
            min_excerpt_chars=2_000,
            max_excerpt_chars_hard_cap=400_000,
            chars_per_token=4,
        ),
    )

    _assert("candidate_lines" in evidence, "evidence should include candidate_lines")
    _assert("candidate_heading_lines" not in evidence, "legacy candidate_heading_lines should not exist")
    _assert(len(evidence["candidate_lines"]) <= 3, "candidate_lines should honor limit")
    _assert(len(set(evidence["candidate_lines"])) == len(evidence["candidate_lines"]), "candidate_lines should be deduplicated")

    # No parser-specific internals should leak.
    _assert("pattern" not in evidence, "evidence should not expose pattern internals")
    _assert("regex" not in evidence, "evidence should not expose regex internals")
    _assert("confidence" not in evidence, "evidence should not expose confidence")


if __name__ == "__main__":
    test_evidence_builder_outputs_candidate_lines()
    print("test_document_profile_evidence_builder: ok")
