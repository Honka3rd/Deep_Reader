#!/usr/bin/env python3
"""Tests for language-aware script system detection registry."""

from __future__ import annotations

from language.language_code import LanguageCode
from language.language_script_registry import LanguageScriptRegistry
from profile.document_profile import ScriptSystem
import profile.parser_metadata_extractor as parser_metadata_extractor_module
from profile.parser_metadata_extractor import ParserMetadataExtractor


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_default_script_mapping() -> None:
    registry = LanguageScriptRegistry()
    _assert(
        registry.detect_script_system(text="Hello world", language=LanguageCode.EN)
        == ScriptSystem.LATIN,
        "EN should default/detect latin",
    )
    _assert(
        registry.detect_script_system(text="Bonjour le monde", language=LanguageCode.FR)
        == ScriptSystem.LATIN,
        "FR should default/detect latin",
    )
    _assert(
        registry.detect_script_system(text="あいうえおカナ", language=LanguageCode.JA)
        == ScriptSystem.JAPANESE,
        "JA should detect japanese",
    )
    _assert(
        registry.detect_script_system(text="한국어 텍스트", language=LanguageCode.KO)
        == ScriptSystem.KOREAN,
        "KO should detect korean",
    )
    _assert(
        registry.detect_script_system(text="Привет мир", language=LanguageCode.RU)
        == ScriptSystem.CYRILLIC,
        "RU should detect cyrillic",
    )
    _assert(
        registry.detect_script_system(text="Привіт світ", language=LanguageCode.UK)
        == ScriptSystem.CYRILLIC,
        "UK should detect cyrillic",
    )
    _assert(
        registry.detect_script_system(text="مرحبا بالعالم", language=LanguageCode.AR)
        == ScriptSystem.ARABIC,
        "AR should detect arabic",
    )
    _assert(
        registry.detect_script_system(text="नमस्ते दुनिया", language=LanguageCode.HI)
        == ScriptSystem.DEVANAGARI,
        "HI should detect devanagari",
    )
    _assert(
        registry.detect_script_system(text="สวัสดีชาวโลก", language=LanguageCode.TH)
        == ScriptSystem.THAI,
        "TH should detect thai",
    )
    _assert(
        registry.detect_script_system(text="", language=LanguageCode.UNKNOWN)
        == ScriptSystem.UNKNOWN,
        "UNKNOWN empty text should resolve to unknown",
    )
    _assert(
        registry.detect_script_system(text="中文文本", language=LanguageCode.ZH)
        == ScriptSystem.MIXED,
        "ZH neutral text should conservatively be mixed",
    )


def test_chinese_simplified_traditional_mixed_hints() -> None:
    registry = LanguageScriptRegistry()
    simplified_text = "这个问题需要通过组织和发展来解决。"
    traditional_text = "這個問題需要透過組織與發展來解決。"
    mixed_text = "這个问题需要通过組織来解决。"

    _assert(
        registry.detect_script_system(text=simplified_text, language="zh")
        == ScriptSystem.SIMPLIFIED_CHINESE,
        "simplified fixture should detect simplified_chinese",
    )
    _assert(
        registry.detect_script_system(text=traditional_text, language="zh")
        == ScriptSystem.TRADITIONAL_CHINESE,
        "traditional fixture should detect traditional_chinese",
    )
    _assert(
        registry.detect_script_system(text=mixed_text, language="zh")
        == ScriptSystem.MIXED,
        "mixed fixture should detect mixed",
    )


def test_all_language_code_entries_do_not_crash() -> None:
    registry = LanguageScriptRegistry()
    for language in LanguageCode:
        script = registry.detect_script_system(text="sample text", language=language)
        _assert(isinstance(script, ScriptSystem), f"{language} should return ScriptSystem")


def test_parser_metadata_extractor_no_local_hint_constants() -> None:
    _assert(
        not hasattr(parser_metadata_extractor_module, "_TRADITIONAL_HINT_CHARS"),
        "ParserMetadataExtractor module should not own traditional hint chars",
    )
    _assert(
        not hasattr(parser_metadata_extractor_module, "_SIMPLIFIED_HINT_CHARS"),
        "ParserMetadataExtractor module should not own simplified hint chars",
    )


def test_parser_metadata_extractor_integration() -> None:
    extractor = ParserMetadataExtractor()
    _assert(
        extractor.extract(text="Hello world", document_language="en").script_system
        == ScriptSystem.LATIN,
        "extractor should use latin detection",
    )
    _assert(
        extractor.extract(
            text="这个问题需要通过组织和发展来解决。",
            document_language="zh",
        ).script_system
        == ScriptSystem.SIMPLIFIED_CHINESE,
        "extractor should use simplified chinese detection",
    )
    _assert(
        extractor.extract(
            text="這個問題需要透過組織與發展來解決。",
            document_language="zh",
        ).script_system
        == ScriptSystem.TRADITIONAL_CHINESE,
        "extractor should use traditional chinese detection",
    )
    _assert(
        extractor.extract(text="あいうえおカナ", document_language="ja").script_system
        == ScriptSystem.JAPANESE,
        "extractor should use japanese detection",
    )
    _assert(
        extractor.extract(text="한국어 텍스트", document_language="ko").script_system
        == ScriptSystem.KOREAN,
        "extractor should use korean detection",
    )


if __name__ == "__main__":
    test_default_script_mapping()
    test_chinese_simplified_traditional_mixed_hints()
    test_all_language_code_entries_do_not_crash()
    test_parser_metadata_extractor_no_local_hint_constants()
    test_parser_metadata_extractor_integration()
    print("test_language_script_registry: ok")

