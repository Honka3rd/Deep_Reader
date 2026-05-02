#!/usr/bin/env python3
"""Regression tests for DocumentProfile structure_profile schema and builder fallback."""

from __future__ import annotations

import json

from config.app_DI_config import ProfilePromptPolicyConfig, PromptTextNormalizationConfig
from config.faiss_storage_config import FaissStorageConfig
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeLLMProvider(LLMProvider):
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    def complete_text(self, prompt: str) -> str:
        self.calls.append(prompt)
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


def _build_prompt_text_normalization() -> PromptTextNormalizationConfig:
    return PromptTextNormalizationConfig(
        default_excerpt_chars=16_000,
        min_excerpt_chars=2_000,
        max_excerpt_chars_hard_cap=400_000,
        chars_per_token=4,
    )


def _build_profile_prompt_policy() -> ProfilePromptPolicyConfig:
    return ProfilePromptPolicyConfig(
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


def _build_builder(fake_provider: _FakeLLMProvider) -> DocumentProfileBuilder:
    return DocumentProfileBuilder(
        llm_provider=fake_provider,
        prompt_text_normalization=_build_prompt_text_normalization(),
        profile_prompt_policy=_build_profile_prompt_policy(),
    )


def _full_structure_profile_payload() -> dict[str, object]:
    return {
        "topic": "essay",
        "summary": "A compact essay about structure and cognition.",
        "document_language": "zh",
        "structure_profile": {
            "profile_version": "structure_profile_v1",
            "generated_by": "llm_profile_builder",
            "document_structure_type": "essay",
            "confidence": 0.86,
            "heading_patterns": {
                "chapter": {
                    "exists": True,
                    "pattern_type": "numbered_conceptual_point",
                    "description": "Chinese ordinal point headings.",
                    "examples": ["第一点：形式主义与象征主义"],
                    "confidence": 0.9,
                    "suggested_regex": None,
                },
                "section": {
                    "exists": False,
                    "pattern_type": None,
                    "description": None,
                    "examples": [],
                    "confidence": 0.6,
                    "suggested_regex": None,
                },
                "front_matter": {
                    "exists": True,
                    "pattern_type": "essay_title",
                    "description": "Opening essay title.",
                    "examples": ["为什么中式思维既不产生科学，也不产生民主"],
                    "confidence": 0.75,
                    "suggested_regex": None,
                },
                "appendix": {
                    "exists": False,
                    "pattern_type": None,
                    "description": None,
                    "examples": [],
                    "confidence": 0.7,
                    "suggested_regex": None,
                },
                "back_matter": {
                    "exists": False,
                    "pattern_type": None,
                    "description": None,
                    "examples": [],
                    "confidence": 0.7,
                    "suggested_regex": None,
                },
            },
            "special_regions": {
                "toc": {
                    "exists": False,
                    "count_estimate": 0,
                    "examples": [],
                    "confidence": 0.7,
                    "notes": None,
                },
                "front_matter": {
                    "exists": True,
                    "count_estimate": 1,
                    "examples": ["为什么中式思维既不产生科学，也不产生民主"],
                    "confidence": 0.75,
                    "notes": "Opening title-like region before numbered conceptual sections.",
                },
                "appendix": {
                    "exists": False,
                    "count_estimate": 0,
                    "examples": [],
                    "confidence": 0.7,
                    "notes": None,
                },
                "back_matter": {
                    "exists": False,
                    "count_estimate": 0,
                    "examples": [],
                    "confidence": 0.7,
                    "notes": None,
                },
            },
            "quality_hints": {
                "quality_label": "usable_with_warnings",
                "likely_single_blob": False,
                "likely_over_fragmented": False,
                "likely_chapter_only": False,
                "likely_chapter_section": False,
                "likely_essay": True,
                "likely_ocr_noisy": False,
                "confidence": 0.85,
            },
            "recommended_strategy": {
                "structured_parser_mode": "llm_enhanced",
                "task_unit_split_mode": "semantic_safe",
                "semantic_top_k_candidates": 3,
                "needs_enhanced_parse": True,
                "needs_manual_review": False,
                "reason": "Essay-like conceptual headings likely missed by common parser.",
            },
            "risks": ["non_standard_headings"],
            "evidence": ["Found conceptual headings using 第一点/第二点 pattern."],
        },
    }


def test_old_profile_compatibility() -> None:
    payload = {
        "topic": "Novel",
        "summary": "A novel.",
        "document_language": "en",
    }
    profile = DocumentProfile.from_dict(payload)
    _assert(profile.structure_profile is None, "old profile should load without structure_profile")
    restored_payload = profile.to_dict()
    _assert("structure_profile" not in restored_payload, "legacy payload should not include structure_profile")


def test_new_profile_round_trip() -> None:
    profile = DocumentProfile.from_dict(_full_structure_profile_payload())
    payload = profile.to_dict()
    restored = DocumentProfile.from_dict(payload)
    _assert(restored.structure_profile is not None, "round-trip should keep structure_profile")
    _assert(
        restored.structure_profile.heading_patterns is not None,
        "heading_patterns should be preserved",
    )
    _assert(
        restored.structure_profile.risks == ["non_standard_headings"],
        "risks should be preserved",
    )
    _assert(
        restored.structure_profile.evidence,
        "evidence should be preserved",
    )


def test_builder_parses_structured_profile_json() -> None:
    fake_provider = _FakeLLMProvider(
        outputs=[json.dumps(_full_structure_profile_payload(), ensure_ascii=False)]
    )
    builder = _build_builder(fake_provider)
    profile = builder.build(text="第一点：结构", document_language="zh")
    _assert(profile.topic == "essay", "topic should come from structured payload")
    _assert(profile.structure_profile is not None, "structure_profile should be parsed")
    _assert(
        profile.structure_profile.document_structure_type == "essay",
        "document_structure_type should be parsed",
    )
    _assert(
        profile.structure_profile.recommended_strategy is not None,
        "recommended_strategy should be parsed",
    )


def test_builder_malformed_structured_output_fallback() -> None:
    fake_provider = _FakeLLMProvider(
        outputs=[
            "{not-json}",
            "literary fiction",
            "A valid fallback summary.",
        ]
    )
    builder = _build_builder(fake_provider)
    profile = builder.build(text="Chapter One\nText", document_language="en")
    _assert(profile.topic == "literary fiction", "fallback topic should be used")
    _assert(profile.summary == "A valid fallback summary.", "fallback summary should be used")
    _assert(profile.structure_profile is None, "malformed structure payload should fallback to None")
    _assert(len(fake_provider.calls) == 3, "fallback path should invoke topic+summary calls")


def test_profile_store_round_trip_with_structure_profile() -> None:
    profile = DocumentProfile.from_dict(_full_structure_profile_payload())
    namespace = "profile-store-structure-test"
    config = FaissStorageConfig(namespace=namespace)
    try:
        DocumentProfileStore.save(profile, config)
        restored = DocumentProfileStore.load(config)
        _assert(
            restored.structure_profile is not None,
            "profile store should persist structure_profile",
        )
        _assert(
            restored.structure_profile.profile_version == "structure_profile_v1",
            "profile version should round-trip",
        )
    finally:
        DocumentProfileStore.clear(config)


if __name__ == "__main__":
    test_old_profile_compatibility()
    test_new_profile_round_trip()
    test_builder_parses_structured_profile_json()
    test_builder_malformed_structured_output_fallback()
    test_profile_store_round_trip_with_structure_profile()
    print("test_document_profile_structure_profile: ok")
