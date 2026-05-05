#!/usr/bin/env python3
"""Tests for profile evidence helpers in DocumentStructureLanguageRegistry."""

from __future__ import annotations

from document_structure.document_structure_language_registry import (
    DocumentStructureLanguageRegistry,
)
from language.language_code import LanguageCode


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_profile_evidence_patterns_multilingual() -> None:
    en_patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(LanguageCode.EN)
    zh_patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(LanguageCode.ZH)
    ja_patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(LanguageCode.JA)
    ko_patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(LanguageCode.KO)
    unknown_patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(LanguageCode.UNKNOWN)

    _assert(any("chapter" in p.name for p in en_patterns), "EN should include chapter-like evidence pattern")
    _assert(any("zh_" in p.name for p in zh_patterns), "ZH should include zh-specific evidence patterns")
    _assert(len(ja_patterns) > 0, "JA evidence patterns should not be empty")
    _assert(len(ko_patterns) > 0, "KO evidence patterns should not be empty")
    _assert(len(unknown_patterns) > 0, "UNKNOWN should fallback to common evidence patterns")


def test_profile_evidence_keywords_multilingual() -> None:
    en_keywords = DocumentStructureLanguageRegistry.get_profile_evidence_keywords(LanguageCode.EN)
    zh_keywords = DocumentStructureLanguageRegistry.get_profile_evidence_keywords(LanguageCode.ZH)
    unknown_keywords = DocumentStructureLanguageRegistry.get_profile_evidence_keywords(LanguageCode.UNKNOWN)

    _assert("chapter" in en_keywords, "EN keywords should include chapter")
    _assert("第一章" not in en_keywords, "EN keywords should not hardcode zh heading sample")
    _assert("章" in zh_keywords, "ZH keywords should include chapter marker")
    _assert(len(unknown_keywords) > 0, "UNKNOWN should fallback to common keywords")


def test_all_language_codes_safe() -> None:
    for language in LanguageCode:
        patterns = DocumentStructureLanguageRegistry.get_profile_evidence_patterns(language)
        keywords = DocumentStructureLanguageRegistry.get_profile_evidence_keywords(language)
        _assert(isinstance(patterns, tuple), f"patterns should be tuple for {language}")
        _assert(isinstance(keywords, tuple), f"keywords should be tuple for {language}")


if __name__ == "__main__":
    test_profile_evidence_patterns_multilingual()
    test_profile_evidence_keywords_multilingual()
    test_all_language_codes_safe()
    print("test_document_structure_language_registry_profile_evidence: ok")
