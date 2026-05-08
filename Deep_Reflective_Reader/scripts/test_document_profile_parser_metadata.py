#!/usr/bin/env python3
"""Regression tests for parser_metadata-focused DocumentProfile builder."""

from __future__ import annotations

from dataclasses import dataclass

from config.faiss_storage_config import FaissStorageConfig
from language.language_code import LanguageCode
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile import DocumentProfile, ParserRelevantMetadata
from profile.document_profile import (
    DialogueDensity,
    DiscourseMode,
    DocumentStructureShape,
    HeadingStyle,
    LikelihoodLevel,
    LineBreakQuality,
    OCRNoiseLevel,
    ScriptSystem,
    TextForm,
)
from profile.document_profile_builder import DocumentProfileBuilder
from profile.document_profile_store import DocumentProfileStore
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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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


class _BrokenCapabilityProvider(_FakeLLMProvider):
    def get_model_capabilities(self) -> LLMModelCapabilities:
        raise RuntimeError("capability_unavailable")


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


def _build_builder(provider: LLMProvider) -> DocumentProfileBuilder:
    return DocumentProfileBuilder(
        llm_provider=provider,
        prompt_text_normalization=_build_prompt_text_normalization(),
        profile_prompt_policy=_build_profile_prompt_policy(),
    )


def test_old_profile_compatibility() -> None:
    payload = {
        "topic": "literary fiction",
        "summary": "A novel.",
        "document_language": "en",
    }
    profile = DocumentProfile.from_dict(payload)
    _assert(profile.parser_metadata is None, "old profile should load parser_metadata=None")
    _assert(profile.structure_profile is None, "old profile should load structure_profile=None")


def test_parser_metadata_round_trip() -> None:
    parser_metadata = ParserRelevantMetadata(
        script_system=ScriptSystem.LATIN,
        text_form=TextForm.NOVEL,
        discourse_mode=DiscourseMode.NARRATIVE,
        line_break_quality=LineBreakQuality.PARAGRAPH_LIKE,
        ocr_noise_level=OCRNoiseLevel.LOW,
        dialogue_density=DialogueDensity.HIGH,
        toc_likelihood=LikelihoodLevel.NONE,
        front_matter_likelihood=LikelihoodLevel.LOW,
        terminal_region_likelihood=LikelihoodLevel.MEDIUM,
        document_structure_shape=DocumentStructureShape.PART_CHAPTER,
        likely_heading_style=HeadingStyle.ENGLISH_CHAPTER_WORDS,
        title_uniqueness_risk=LikelihoodLevel.HIGH,
        confidence=0.91,
        notes=["repeated chapter titles likely across parts"],
    )
    profile = DocumentProfile(
        topic="literary fiction",
        summary="A classic novel.",
        document_language=LanguageCode.EN,
        parser_metadata=parser_metadata,
    )
    payload = profile.to_dict()
    restored = DocumentProfile.from_dict(payload)
    _assert(restored.parser_metadata is not None, "parser_metadata should round-trip")
    _assert(
        restored.parser_metadata.document_structure_shape == DocumentStructureShape.PART_CHAPTER,
        "document_structure_shape should persist",
    )
    _assert(
        restored.parser_metadata.notes == ["repeated chapter titles likely across parts"],
        "notes should persist",
    )

    namespace = "profile-parser-metadata-roundtrip"
    config = FaissStorageConfig(namespace=namespace)
    try:
        DocumentProfileStore.save(profile, config)
        loaded = DocumentProfileStore.load(config)
        _assert(loaded.parser_metadata is not None, "store round-trip should keep parser_metadata")
        _assert(loaded.structure_profile is None, "new profile should not require structure_profile")
    finally:
        DocumentProfileStore.clear(config)


def test_deterministic_metadata_extractor() -> None:
    extractor = ParserMetadataExtractor()
    english_text = "\n".join(
        [
            "PART I",
            "Chapter One",
            '"Hello," she said.',
            "This is a long paragraph with narrative prose and punctuation.",
        ]
    )
    zh_text = "为什么中式思维既不产生科学，也不产生民主。\n\n第一点：形式主义。\n第二点：概念误读。"
    noisy_text = "\ufffd\ufffd\ufffd  \u0007\u0008 abc \ufffd\ufffd\ufffd\n\ufffd\ufffd\ufffd\n"

    en_meta = extractor.extract(text=english_text, document_language="en")
    zh_meta = extractor.extract(text=zh_text, document_language="zh")
    noisy_meta = extractor.extract(text=noisy_text, document_language="zh")

    _assert(
        en_meta.script_system in {ScriptSystem.LATIN, ScriptSystem.MIXED},
        "english script should be latin/mixed",
    )
    _assert(
        zh_meta.script_system in {ScriptSystem.SIMPLIFIED_CHINESE, ScriptSystem.MIXED},
        "zh script should be chinese/mixed",
    )
    _assert(
        noisy_meta.ocr_noise_level in {OCRNoiseLevel.MEDIUM, OCRNoiseLevel.HIGH},
        "noisy text should have medium/high ocr noise",
    )


def test_fake_llm_classification_merge() -> None:
    provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "literary fiction",
              "summary": "A novel about social change.",
              "document_language": "en",
              "text_form": "novel",
              "discourse_mode": "narrative",
              "document_structure_shape": "part_chapter",
              "likely_heading_style": "english_chapter_words",
              "title_uniqueness_risk": "high",
              "confidence": 0.9,
              "notes": ["Chapter titles may repeat across parts."]
            }
            """
        ]
    )
    builder = _build_builder(provider)
    profile = builder.build(text="PART I\nChapter One\nText", document_language="en")
    _assert(profile.topic == "literary fiction", "topic should come from classification")
    _assert(profile.parser_metadata is not None, "parser_metadata should exist")
    _assert(
        profile.parser_metadata.text_form == TextForm.NOVEL,
        "LLM classification should merge into parser_metadata",
    )
    _assert(
        profile.parser_metadata.document_structure_shape == DocumentStructureShape.PART_CHAPTER,
        "structure shape should merge",
    )


def test_malformed_llm_fallback_preserves_deterministic_metadata() -> None:
    provider = _BrokenCapabilityProvider(
        outputs=[
            "{not-json}",
            "essay",
            "A fallback summary.",
        ]
    )
    builder = _build_builder(provider)
    profile = builder.build(
        text="为什么中式思维既不产生科学，也不产生民主。",
        document_language="zh",
    )
    _assert(profile.topic == "essay", "fallback topic should be used")
    _assert(profile.summary == "A fallback summary.", "fallback summary should be used")
    _assert(profile.parser_metadata is not None, "deterministic parser_metadata should remain")
    _assert(profile.structure_profile is None, "builder should not generate structure_profile")
    _assert(
        profile.parser_metadata.script_system in {
            ScriptSystem.SIMPLIFIED_CHINESE,
            ScriptSystem.MIXED,
        },
        "deterministic script_system should be present",
    )


def test_builder_no_structure_profile_generated() -> None:
    provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "essay",
              "summary": "A concise essay.",
              "document_language": "zh",
              "text_form": "essay",
              "discourse_mode": "argumentative",
              "document_structure_shape": "essay_sections",
              "likely_heading_style": "numbered_chinese_points",
              "title_uniqueness_risk": "low",
              "confidence": 0.82,
              "notes": ["Conceptual numbered points observed."]
            }
            """
        ]
    )
    builder = _build_builder(provider)
    profile = builder.build(text="第一点：...\n第二点：...", document_language="zh")
    _assert(profile.structure_profile is None, "new builder should not generate structure_profile")
    _assert(profile.parser_metadata is not None, "new builder should generate parser_metadata")


if __name__ == "__main__":
    test_old_profile_compatibility()
    test_parser_metadata_round_trip()
    test_deterministic_metadata_extractor()
    test_fake_llm_classification_merge()
    test_malformed_llm_fallback_preserves_deterministic_metadata()
    test_builder_no_structure_profile_generated()
    print("test_document_profile_parser_metadata: ok")
