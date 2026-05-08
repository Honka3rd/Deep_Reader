#!/usr/bin/env python3
"""Tests for language discourse registry coverage and conservative cue behavior."""

from __future__ import annotations

from language.language_code import LanguageCode
from language.language_discourse_registry import LanguageDiscourseRegistry
from profile.document_profile import DialogueDensity
from profile.parser_metadata_extractor import ParserMetadataExtractor


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_all_language_entries_have_dialogue_cues() -> None:
    registry = LanguageDiscourseRegistry()
    for language in LanguageCode:
        entry = registry.get_entry(language)
        cues = registry.get_dialogue_cues(language)
        _assert(entry is not None, f"{language} entry should exist")
        _assert(cues is not None, f"{language} dialogue cues should exist")
        _assert(len(cues.quote_chars) > 0, f"{language} quote chars should not be empty")
        _assert(
            len(cues.dialogue_dash_prefixes) > 0,
            f"{language} dialogue dash prefixes should not be empty",
        )


def test_explicit_entry_coverage() -> None:
    supported = set(LanguageDiscourseRegistry.supported_languages())
    for language in LanguageCode:
        _assert(
            language in supported,
            f"{language} should be explicitly covered by LanguageDiscourseRegistry",
        )


def test_known_language_hints() -> None:
    registry = LanguageDiscourseRegistry()
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.EN).speech_verb_hints) > 0,
        "EN should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.ZH).speech_verb_hints) > 0,
        "ZH should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.JA).speech_verb_hints) > 0,
        "JA should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.KO).speech_verb_hints) > 0,
        "KO should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.FR).speech_verb_hints) > 0,
        "FR should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.DE).speech_verb_hints) > 0,
        "DE should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.ES).speech_verb_hints) > 0,
        "ES should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.RU).speech_verb_hints) > 0,
        "RU should have speech verb hints",
    )
    _assert(
        len(registry.get_dialogue_cues(LanguageCode.UNKNOWN).speech_verb_hints) == 0,
        "UNKNOWN should keep conservative empty speech hints",
    )


def test_unknown_and_sparse_languages_conservative_fallback() -> None:
    registry = LanguageDiscourseRegistry()
    ar_cues = registry.get_dialogue_cues(LanguageCode.AR)
    unknown_cues = registry.get_dialogue_cues(LanguageCode.UNKNOWN)
    _assert(ar_cues is not None and unknown_cues is not None, "fallback cues should exist")
    _assert(len(ar_cues.quote_chars) > 0, "AR should still have quote chars")
    _assert(len(ar_cues.dialogue_dash_prefixes) > 0, "AR should still have dash prefixes")


def test_parser_metadata_extractor_integration() -> None:
    extractor = ParserMetadataExtractor()
    en_dialogue = "\n".join(
        [
            "He said, \"Let's go.\"",
            "\"Okay,\" she replied.",
            "- We leave now.",
            "- Yes.",
        ]
    )
    zh_quotation_essay = "\n".join(
        [
            "本文讨论“理性”“制度”“结构”等概念。",
            "这些引号用于术语强调，不代表人物对话。",
        ]
    )
    ja_dialogue = "\n".join(
        [
            "「行こう」と言った。",
            "「はい」と答えた。",
        ]
    )
    ko_dialogue = "\n".join(
        [
            "\"지금 가자\"라고 말했다.",
            "\"좋아\"라고 대답했다.",
        ]
    )
    unknown_text = "sample text without known language-specific cues"

    en_density = extractor.extract(text=en_dialogue, document_language="en").dialogue_density
    zh_density = extractor.extract(text=zh_quotation_essay, document_language="zh").dialogue_density
    ja_density = extractor.extract(text=ja_dialogue, document_language="ja").dialogue_density
    ko_density = extractor.extract(text=ko_dialogue, document_language="ko").dialogue_density
    unknown_density = extractor.extract(text=unknown_text, document_language="unknown").dialogue_density

    _assert(
        en_density in {DialogueDensity.MEDIUM, DialogueDensity.HIGH},
        "EN dialogue sample should be medium/high",
    )
    _assert(
        zh_density != DialogueDensity.HIGH,
        "ZH quotation-only essay should not be high",
    )
    _assert(ja_density is not None, "JA integration should not crash")
    _assert(ko_density is not None, "KO integration should not crash")
    _assert(unknown_density is not None, "UNKNOWN integration should not crash")


if __name__ == "__main__":
    test_all_language_entries_have_dialogue_cues()
    test_explicit_entry_coverage()
    test_known_language_hints()
    test_unknown_and_sparse_languages_conservative_fallback()
    test_parser_metadata_extractor_integration()
    print("test_language_discourse_registry: ok")

