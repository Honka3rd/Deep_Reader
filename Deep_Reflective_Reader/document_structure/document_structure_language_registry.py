import re
from dataclasses import dataclass
from typing import Pattern

from language.language_code import LanguageCode


def _compile_numeric_heading_pattern(
    aliases: tuple[str, ...],
    number_pattern: str,
) -> Pattern[str]:
    alias_part = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        rf"^\s*(?:{alias_part})\s+(?:{number_pattern})\b(?:[\s:：.\-–—].*)?\s*$",
        re.IGNORECASE,
    )


def _compile_single_heading_pattern(aliases: tuple[str, ...]) -> Pattern[str]:
    alias_part = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        rf"^\s*(?:{alias_part})\b(?:[\s:：.\-–—].*)?\s*$",
        re.IGNORECASE,
    )


@dataclass(frozen=True)
class DocumentStructureLanguageRules:
    """Language-scoped section heading rules for document structure parsing."""

    strong_heading_patterns: tuple[Pattern[str], ...]
    weak_heading_aliases: tuple[str, ...]
    weak_heading_signals: tuple[str, ...]


class DocumentStructureLanguageRegistry:
    """Registry for document-structure heading rules keyed by LanguageCode."""

    _ROMAN_OR_DIGIT_NUMBER = r"\d+|[ivxlcdm]+"
    _CJK_NUMBER = r"[0-9一二三四五六七八九十百千零〇两兩]+"
    _ARABIC_NUMBER = r"[0-9٠-٩]+"
    _DEVANAGARI_NUMBER = r"[0-9०-९]+"

    _EN_ALIASES_NUMERIC = ("chapter", "part", "section", "book", "lesson", "unit")
    _EN_ALIASES_SINGLE = ("prologue", "epilogue", "introduction", "appendix")
    _EN_ALIASES_WEAK = _EN_ALIASES_NUMERIC + _EN_ALIASES_SINGLE
    _EN_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(_EN_ALIASES_NUMERIC, _ROMAN_OR_DIGIT_NUMBER),
            re.compile(
                r"^\s*appendix(?:\s+([a-z0-9]+|[ivxlcdm]+))?(?:[\s:：.\-–—].*)?\s*$",
                re.IGNORECASE,
            ),
            _compile_single_heading_pattern(("prologue", "epilogue", "introduction")),
        ),
        weak_heading_aliases=_EN_ALIASES_WEAK,
        weak_heading_signals=_EN_ALIASES_WEAK,
    )

    _ZH_NUMERIC_SECTION_ALIASES = ("章", "节", "節")
    _ZH_STRONG_SINGLE_ALIASES = ("序章", "终章", "終章", "附录", "附錄")
    _ZH_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:章)(?:[\s:：.\-–—].*)?\s*$"
            ),
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:节|節)(?:[\s:：.\-–—].*)?\s*$"
            ),
            re.compile(
                rf"^\s*卷\s*(?:{_CJK_NUMBER})(?:[\s:：.\-–—].*)?\s*$"
            ),
            _compile_single_heading_pattern(_ZH_STRONG_SINGLE_ALIASES),
        ),
        weak_heading_aliases=(
            "章",
            "节",
            "節",
            "卷",
            "序章",
            "终章",
            "終章",
            "附录",
            "附錄",
        ),
        weak_heading_signals=(
            "章",
            "节",
            "節",
            "卷",
            "序章",
            "终章",
            "終章",
            "附录",
            "附錄",
        ),
    )

    _JA_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(
                rf"^\s*第\s*(?:{_CJK_NUMBER})\s*(?:章|節)(?:[\s:：.\-–—].*)?\s*$"
            ),
            _compile_single_heading_pattern(("序章", "終章", "付録", "附録", "はじめに", "あとがき")),
        ),
        weak_heading_aliases=("章", "節", "序章", "終章", "付録", "附録", "はじめに", "あとがき"),
        weak_heading_signals=("章", "節", "序章", "終章", "付録", "附録", "はじめに", "あとがき"),
    )

    _KO_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            re.compile(r"^\s*제\s*(?:\d+)\s*(?:장|절)(?:[\s:：.\-–—].*)?\s*$"),
            _compile_single_heading_pattern(("프롤로그", "에필로그", "서문", "부록")),
        ),
        weak_heading_aliases=("장", "절", "프롤로그", "에필로그", "서문", "부록"),
        weak_heading_signals=("장", "절", "프롤로그", "에필로그", "서문", "부록"),
    )

    _RU_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("глава", "раздел", "книга"), _ROMAN_OR_DIGIT_NUMBER),
            _compile_single_heading_pattern(("пролог", "эпилог", "введение", "приложение")),
        ),
        weak_heading_aliases=("глава", "раздел", "книга", "пролог", "эпилог", "введение", "приложение"),
        weak_heading_signals=("глава", "раздел", "книга", "пролог", "эпилог", "введение", "приложение"),
    )

    _UK_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("глава", "розділ", "книга"), _ROMAN_OR_DIGIT_NUMBER),
            _compile_single_heading_pattern(("пролог", "епілог", "вступ", "додаток")),
        ),
        weak_heading_aliases=("глава", "розділ", "книга", "пролог", "епілог", "вступ", "додаток"),
        weak_heading_signals=("глава", "розділ", "книга", "пролог", "епілог", "вступ", "додаток"),
    )

    _AR_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("الفصل", "القسم", "الكتاب"), _ARABIC_NUMBER),
            _compile_single_heading_pattern(("مقدمة", "خاتمة", "ملحق")),
        ),
        weak_heading_aliases=("الفصل", "القسم", "الكتاب", "مقدمة", "خاتمة", "ملحق"),
        weak_heading_signals=("الفصل", "القسم", "الكتاب", "مقدمة", "خاتمة", "ملحق"),
    )

    _HI_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("अध्याय", "भाग", "खंड"), _DEVANAGARI_NUMBER),
            _compile_single_heading_pattern(("प्रस्तावना", "उपसंहार", "परिशिष्ट")),
        ),
        weak_heading_aliases=("अध्याय", "भाग", "खंड", "प्रस्तावना", "उपसंहार", "परिशिष्ट"),
        weak_heading_signals=("अध्याय", "भाग", "खंड", "प्रस्तावना", "उपसंहार", "परिशिष्ट"),
    )

    _TH_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=(
            _compile_numeric_heading_pattern(("บทที่", "ภาค", "ตอน"), r"\d+"),
            _compile_single_heading_pattern(("บทนำ", "บทส่งท้าย", "ภาคผนวก")),
        ),
        weak_heading_aliases=("บทที่", "ภาค", "ตอน", "บทนำ", "บทส่งท้าย", "ภาคผนวก"),
        weak_heading_signals=("บทที่", "ภาค", "ตอน", "บทนำ", "บทส่งท้าย", "ภาคผนวก"),
    )

    _LATIN_FAMILY_RULES = _EN_RULES
    _EMPTY_RULES = DocumentStructureLanguageRules(
        strong_heading_patterns=tuple(),
        weak_heading_aliases=tuple(),
        weak_heading_signals=tuple(),
    )

    _RULES_BY_LANGUAGE: dict[LanguageCode, DocumentStructureLanguageRules] = {
        LanguageCode.EN: _EN_RULES,
        LanguageCode.ZH: _ZH_RULES,
        LanguageCode.JA: _JA_RULES,
        LanguageCode.KO: _KO_RULES,
        LanguageCode.FR: _LATIN_FAMILY_RULES,
        LanguageCode.DE: _LATIN_FAMILY_RULES,
        LanguageCode.ES: _LATIN_FAMILY_RULES,
        LanguageCode.PT: _LATIN_FAMILY_RULES,
        LanguageCode.IT: _LATIN_FAMILY_RULES,
        LanguageCode.RU: _RU_RULES,
        LanguageCode.AR: _AR_RULES,
        LanguageCode.HI: _HI_RULES,
        LanguageCode.TR: _LATIN_FAMILY_RULES,
        LanguageCode.NL: _LATIN_FAMILY_RULES,
        LanguageCode.PL: _LATIN_FAMILY_RULES,
        LanguageCode.UK: _UK_RULES,
        LanguageCode.ID: _LATIN_FAMILY_RULES,
        LanguageCode.VI: _LATIN_FAMILY_RULES,
        LanguageCode.TH: _TH_RULES,
    }

    @classmethod
    def is_language_supported(cls, language: LanguageCode) -> bool:
        """Return whether registry has explicit rules for this language."""
        return language in cls._RULES_BY_LANGUAGE

    @classmethod
    def get_rules(cls, language: LanguageCode) -> DocumentStructureLanguageRules:
        """Return language-specific rules, or safe empty fallback rules."""
        return cls._RULES_BY_LANGUAGE.get(language, cls._EMPTY_RULES)

    @classmethod
    def get_strong_heading_patterns(
        cls,
        language: LanguageCode,
    ) -> tuple[Pattern[str], ...]:
        """Return strong heading regex patterns for one language."""
        return cls.get_rules(language).strong_heading_patterns

    @classmethod
    def get_weak_heading_signals(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return weak heading aliases/signals for one language."""
        return cls.get_rules(language).weak_heading_signals

    @classmethod
    def get_weak_heading_aliases(cls, language: LanguageCode) -> tuple[str, ...]:
        """Return weak heading aliases for one language."""
        return cls.get_rules(language).weak_heading_aliases
