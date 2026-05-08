from __future__ import annotations

import re
from dataclasses import dataclass

from language.language_code import LanguageCode, LanguageCodeResolver
from profile.document_profile import ScriptSystem


_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")


@dataclass(frozen=True)
class ChineseScriptHints:
    simplified_hint_chars: frozenset[str]
    traditional_hint_chars: frozenset[str]


@dataclass(frozen=True)
class LanguageScriptRegistryEntry:
    language: LanguageCode
    default_script_system: ScriptSystem
    chinese_hints: ChineseScriptHints | None = None


class LanguageScriptRegistry:
    """Language-aware script detection defaults and hints."""

    _ZH_HINTS = ChineseScriptHints(
        simplified_hint_chars=frozenset("学体国说与为这来开会后台万点应变气实数据语录"),
        traditional_hint_chars=frozenset("學體國說與為這來開會後臺萬點應變氣實數據語錄"),
    )

    _ENTRIES: dict[LanguageCode, LanguageScriptRegistryEntry] = {
        LanguageCode.EN: LanguageScriptRegistryEntry(LanguageCode.EN, ScriptSystem.LATIN),
        LanguageCode.FR: LanguageScriptRegistryEntry(LanguageCode.FR, ScriptSystem.LATIN),
        LanguageCode.DE: LanguageScriptRegistryEntry(LanguageCode.DE, ScriptSystem.LATIN),
        LanguageCode.ES: LanguageScriptRegistryEntry(LanguageCode.ES, ScriptSystem.LATIN),
        LanguageCode.PT: LanguageScriptRegistryEntry(LanguageCode.PT, ScriptSystem.LATIN),
        LanguageCode.IT: LanguageScriptRegistryEntry(LanguageCode.IT, ScriptSystem.LATIN),
        LanguageCode.TR: LanguageScriptRegistryEntry(LanguageCode.TR, ScriptSystem.LATIN),
        LanguageCode.NL: LanguageScriptRegistryEntry(LanguageCode.NL, ScriptSystem.LATIN),
        LanguageCode.PL: LanguageScriptRegistryEntry(LanguageCode.PL, ScriptSystem.LATIN),
        LanguageCode.ID: LanguageScriptRegistryEntry(LanguageCode.ID, ScriptSystem.LATIN),
        LanguageCode.VI: LanguageScriptRegistryEntry(LanguageCode.VI, ScriptSystem.LATIN),
        LanguageCode.ZH: LanguageScriptRegistryEntry(
            LanguageCode.ZH,
            ScriptSystem.MIXED,
            chinese_hints=_ZH_HINTS,
        ),
        LanguageCode.JA: LanguageScriptRegistryEntry(LanguageCode.JA, ScriptSystem.JAPANESE),
        LanguageCode.KO: LanguageScriptRegistryEntry(LanguageCode.KO, ScriptSystem.KOREAN),
        LanguageCode.RU: LanguageScriptRegistryEntry(LanguageCode.RU, ScriptSystem.CYRILLIC),
        LanguageCode.UK: LanguageScriptRegistryEntry(LanguageCode.UK, ScriptSystem.CYRILLIC),
        LanguageCode.AR: LanguageScriptRegistryEntry(LanguageCode.AR, ScriptSystem.ARABIC),
        LanguageCode.HI: LanguageScriptRegistryEntry(
            LanguageCode.HI,
            ScriptSystem.DEVANAGARI,
        ),
        LanguageCode.TH: LanguageScriptRegistryEntry(LanguageCode.TH, ScriptSystem.THAI),
        LanguageCode.UNKNOWN: LanguageScriptRegistryEntry(
            LanguageCode.UNKNOWN,
            ScriptSystem.UNKNOWN,
        ),
    }

    _SCRIPT_BY_NAME: dict[str, ScriptSystem] = {
        "latin": ScriptSystem.LATIN,
        "cjk": ScriptSystem.MIXED,
        "japanese": ScriptSystem.JAPANESE,
        "korean": ScriptSystem.KOREAN,
        "cyrillic": ScriptSystem.CYRILLIC,
        "arabic": ScriptSystem.ARABIC,
        "devanagari": ScriptSystem.DEVANAGARI,
        "thai": ScriptSystem.THAI,
    }

    def get_entry(self, language: LanguageCode | str) -> LanguageScriptRegistryEntry:
        resolved = LanguageCodeResolver.resolve(language)
        return self._ENTRIES.get(resolved, self._ENTRIES[LanguageCode.UNKNOWN])

    def detect_script_system(
        self,
        *,
        text: str,
        language: LanguageCode | str,
    ) -> ScriptSystem:
        resolved = LanguageCodeResolver.resolve(language)
        entry = self.get_entry(resolved)
        normalized_text = text or ""
        if not normalized_text.strip():
            return entry.default_script_system

        counts = self._count_scripts(normalized_text)
        total = sum(counts.values())
        if total == 0:
            return entry.default_script_system

        if resolved == LanguageCode.ZH:
            return self._detect_chinese_script(
                text=normalized_text,
                entry=entry,
                counts=counts,
            )

        dominant_name, dominant_count = max(counts.items(), key=lambda item: item[1])
        dominant_ratio = dominant_count / total
        second_ratio = sorted(counts.values(), reverse=True)[1] / total
        if dominant_ratio < 0.45 or second_ratio > 0.28:
            return ScriptSystem.MIXED

        detected = self._SCRIPT_BY_NAME.get(dominant_name)
        if detected is None:
            return entry.default_script_system
        return detected

    def _detect_chinese_script(
        self,
        *,
        text: str,
        entry: LanguageScriptRegistryEntry,
        counts: dict[str, int],
    ) -> ScriptSystem:
        hints = entry.chinese_hints
        if hints is None:
            return entry.default_script_system

        sample = text[:5000]
        trad_hits = sum(char in hints.traditional_hint_chars for char in sample)
        simp_hits = sum(char in hints.simplified_hint_chars for char in sample)

        if trad_hits >= 2 and trad_hits > simp_hits * 1.2:
            return ScriptSystem.TRADITIONAL_CHINESE
        if simp_hits >= 2 and simp_hits > trad_hits * 1.2:
            return ScriptSystem.SIMPLIFIED_CHINESE
        if trad_hits > 0 and simp_hits > 0:
            return ScriptSystem.MIXED

        total = sum(counts.values())
        cjk_share = counts["cjk"] / total if total else 0.0
        if cjk_share >= 0.45:
            return entry.default_script_system
        if cjk_share >= 0.15:
            return ScriptSystem.MIXED
        return entry.default_script_system

    def _count_scripts(self, text: str) -> dict[str, int]:
        return {
            "latin": len(_LATIN_CHAR_RE.findall(text)),
            "cjk": len(_CJK_CHAR_RE.findall(text)),
            "japanese": len(_HIRAGANA_KATAKANA_RE.findall(text)),
            "korean": len(_HANGUL_RE.findall(text)),
            "cyrillic": len(_CYRILLIC_RE.findall(text)),
            "arabic": len(_ARABIC_RE.findall(text)),
            "devanagari": len(_DEVANAGARI_RE.findall(text)),
            "thai": len(_THAI_RE.findall(text)),
        }

