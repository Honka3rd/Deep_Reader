#!/usr/bin/env python3
"""Boundary tests for DocumentProfile language string access and evidence-only helpers."""

from __future__ import annotations

from dataclasses import dataclass

from language.language_code import LanguageCode
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
import profile.document_profile_builder as profile_builder_module
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


def _build_builder(fake_provider: _FakeLLMProvider) -> DocumentProfileBuilder:
    return DocumentProfileBuilder(
        llm_provider=fake_provider,
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


def test_document_language_code_property() -> None:
    profile = DocumentProfile(
        topic="essay",
        summary="...",
        document_language=LanguageCode.ZH,
    )
    _assert(profile.document_language_code == "zh", "document_language_code should be 'zh'")


def test_profile_json_compatibility() -> None:
    payload = {
        "topic": "essay",
        "summary": "test",
        "document_language": "zh",
    }
    profile = DocumentProfile.from_dict(payload)
    _assert(profile.document_language == LanguageCode.ZH, "from_dict should resolve LanguageCode.ZH")
    _assert(profile.document_language_code == "zh", "language code should stay canonical")
    restored = profile.to_dict()
    _assert(restored["document_language"] == "zh", "to_dict should output string code")
    _assert("LanguageCode" not in restored["document_language"], "to_dict should not leak enum repr")


def test_prompt_uses_language_code_string() -> None:
    fake_provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "essay",
              "summary": "summary",
              "document_language": "zh",
              "text_form": "essay",
              "discourse_mode": "expository",
              "document_structure_shape": "essay_sections",
              "likely_heading_style": "plain_title_headings",
              "title_uniqueness_risk": "low",
              "confidence": 0.8,
              "notes": []
            }
            """
        ]
    )
    builder = _build_builder(fake_provider)
    builder.build(text="第一章\n正文", document_language="zh")
    _assert(len(fake_provider.prompts) >= 1, "classification prompt should be sent")
    prompt = fake_provider.prompts[0]
    _assert('"document_language": "zh"' in prompt, "prompt evidence should contain string language code")
    _assert("LanguageCode." not in prompt, "prompt should not leak enum representation")
    _assert('"candidate_lines":' in prompt, "prompt evidence should use candidate_lines")
    _assert("candidate_heading_lines" not in prompt, "prompt should not use legacy candidate_heading_lines")
    _assert(
        "not authoritative headings or parser boundaries" in prompt,
        "prompt should state evidence-only boundary",
    )
    _assert("regex_candidates" not in prompt, "prompt evidence should not expose regex candidates")
    _assert("\"pattern\":" not in prompt, "prompt evidence should not expose pattern internals")


def test_builder_no_longer_owns_profile_regex_patterns() -> None:
    _assert(
        not hasattr(profile_builder_module, "_PROFILE_HEADING_EVIDENCE_PATTERNS"),
        "builder module should not own profile evidence regex patterns",
    )
    _assert(
        not hasattr(DocumentProfileBuilder, "_extract_profile_heading_evidence_lines"),
        "builder should not own heading evidence extraction helper",
    )


if __name__ == "__main__":
    test_document_language_code_property()
    test_profile_json_compatibility()
    test_prompt_uses_language_code_string()
    test_builder_no_longer_owns_profile_regex_patterns()
    print("test_document_profile_language_and_evidence_boundaries: ok")
