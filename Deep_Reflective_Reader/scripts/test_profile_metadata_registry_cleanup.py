#!/usr/bin/env python3
"""Architecture cleanup tests for parser metadata registries and semantic prompt wording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from language.language_code import LanguageCode
from language.language_discourse_registry import LanguageDiscourseRegistry
from llm.llm_model_capabilities import ENDPOINT_KIND_RESPONSES, LLMModelCapabilities
from llm.llm_provider import LLMProvider
from profile.document_profile_builder import DocumentProfileBuilder
from profile.parser_metadata_extractor import ParserMetadataExtractor
from profile.document_profile import DialogueDensity, LikelihoodLevel


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


class _FakeStructureRegistry:
    def get_toc_markers(self, language: LanguageCode | str) -> tuple[str, ...]:
        return ("__toc_marker__",)

    def get_front_matter_markers(self, language: LanguageCode | str) -> tuple[str, ...]:
        return ("__front_marker__",)

    def get_appendix_markers(self, language: LanguageCode | str) -> tuple[str, ...]:
        return ("__appendix_marker__",)

    def get_back_matter_markers(self, language: LanguageCode | str) -> tuple[str, ...]:
        return ("__back_marker__",)


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


def test_parser_metadata_extractor_no_local_language_scatter() -> None:
    source = Path("profile/parser_metadata_extractor.py").read_text(encoding="utf-8")
    forbidden_tokens = (
        "_SPEECH_VERB_HINTS",
        '"table of contents"',
        '"目录"',
        '"目錄"',
        '"preface"',
        '"foreword"',
        '"afterword"',
        '"bibliography"',
        '"参考文献"',
        '"參考文獻"',
        '"他说"',
        '"he said"',
    )
    for token in forbidden_tokens:
        _assert(
            token not in source,
            f"parser_metadata_extractor should not hardcode language marker token: {token}",
        )


def test_language_discourse_registry_basics() -> None:
    registry = LanguageDiscourseRegistry()
    en_cues = registry.get_dialogue_cues("en")
    zh_cues = registry.get_dialogue_cues("zh")
    unknown_cues = registry.get_dialogue_cues(LanguageCode.UNKNOWN)

    _assert(len(en_cues.speech_verb_hints) > 0, "EN should include speech verb hints")
    _assert(len(zh_cues.speech_verb_hints) > 0, "ZH should include speech verb hints")
    _assert(len(unknown_cues.dialogue_dash_prefixes) > 0, "UNKNOWN should keep common dash prefixes")
    _assert(len(unknown_cues.quote_chars) > 0, "UNKNOWN should keep common quote chars")


def test_dialogue_density_regression_cases() -> None:
    extractor = ParserMetadataExtractor()
    quotation_only_essay = "\n".join(
        [
            "本文讨论“理性”“经验”“制度”等概念，属于“分析”框架。",
            "这一段强调“结构性条件”与“历史动力”的关系。",
            "引号用于术语强调，不构成人物对话。",
        ]
    )
    real_dialogue = "\n".join(
        [
            "— 我们现在出发。",
            "— 好，马上走。",
            "“你准备好了吗？”",
            "“准备好了。”",
        ]
    )
    speech_verbs_without_dialogue = "\n".join(
        [
            "The report said the policy changed in 2010 and asked for review.",
            "He said this model is robust in theory, then continued exposition.",
            "This paragraph is long and explanatory rather than dialogue.",
        ]
    )

    essay_density = extractor.extract(
        text=quotation_only_essay,
        document_language="zh",
    ).dialogue_density
    dialogue_density = extractor.extract(
        text=real_dialogue,
        document_language="zh",
    ).dialogue_density
    speech_only_density = extractor.extract(
        text=speech_verbs_without_dialogue,
        document_language="en",
    ).dialogue_density

    _assert(essay_density != DialogueDensity.HIGH, "quotation-only essay should not be HIGH")
    _assert(
        dialogue_density in {DialogueDensity.MEDIUM, DialogueDensity.HIGH},
        "real dialogue should be MEDIUM/HIGH",
    )
    _assert(
        speech_only_density != DialogueDensity.HIGH,
        "speech verbs alone should not escalate to HIGH",
    )


def test_region_likelihood_uses_structure_registry() -> None:
    extractor = ParserMetadataExtractor(structure_registry=_FakeStructureRegistry())
    toc = extractor._detect_toc_likelihood(
        text="header\n__toc_marker__\nitems",
        document_language="en",
    )
    front = extractor._detect_front_matter_likelihood(
        text="__front_marker__\nrest",
        document_language="en",
    )
    terminal = extractor._detect_terminal_region_likelihood(
        text="body\n__back_marker__",
        document_language="en",
    )

    _assert(toc == LikelihoodLevel.MEDIUM, "toc likelihood should use injected registry marker")
    _assert(front == LikelihoodLevel.LOW, "front likelihood should use injected registry marker")
    _assert(
        terminal == LikelihoodLevel.LOW,
        "terminal likelihood should use injected registry markers",
    )


def test_prompt_semantic_wording() -> None:
    provider = _FakeLLMProvider(
        outputs=[
            """
            {
              "topic": "novel",
              "summary": "summary",
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
    builder.build(text="Part I\n...\nPart II\n...", document_language="en")
    prompt = provider.prompts[0]

    _assert(
        "not about whether the document/book title is unique" in prompt,
        "prompt should define title uniqueness semantically",
    )
    _assert(
        "structural node titles inside the document" in prompt,
        "prompt should target structural node titles",
    )
    _assert("parent grouping units" in prompt, "prompt should mention parent grouping semantics")
    _assert(
        "dominant visible heading convention" in prompt,
        "prompt should define likely_heading_style semantically",
    )
    _assert(
        "candidate_lines are sampled evidence for classification only" in prompt,
        "prompt should keep candidate_lines evidence boundary",
    )
    _assert(
        'If evidence contains "Chapter One"' not in prompt,
        "prompt should not rely on hardcoded language-specific trigger sentence",
    )
    _assert(
        'If evidence contains "第一章"' not in prompt,
        "prompt should not rely on hardcoded language-specific trigger sentence",
    )


if __name__ == "__main__":
    test_parser_metadata_extractor_no_local_language_scatter()
    test_language_discourse_registry_basics()
    test_dialogue_density_regression_cases()
    test_region_likelihood_uses_structure_registry()
    test_prompt_semantic_wording()
    print("test_profile_metadata_registry_cleanup: ok")
