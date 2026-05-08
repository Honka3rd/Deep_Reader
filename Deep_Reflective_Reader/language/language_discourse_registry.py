from __future__ import annotations

from dataclasses import dataclass

from language.language_code import LanguageCode, LanguageCodeResolver


@dataclass(frozen=True)
class DialogueCueConfig:
    speech_verb_hints: tuple[str, ...] = tuple()
    dialogue_dash_prefixes: tuple[str, ...] = ("-", "—", "–")
    quote_chars: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class LanguageDiscourseRegistryEntry:
    language: LanguageCode
    dialogue_cues: DialogueCueConfig


class LanguageDiscourseRegistry:
    """Registry for language-scoped discourse cues used by profile metadata only."""

    _COMMON_DIALOGUE_CUES = DialogueCueConfig(
        speech_verb_hints=tuple(),
        dialogue_dash_prefixes=("-", "—", "–"),
        quote_chars=(
            '"',
            "“",
            "”",
            "「",
            "」",
            "『",
            "』",
            "‘",
            "’",
        ),
    )

    _ENTRIES: dict[LanguageCode, LanguageDiscourseRegistryEntry] = {
        LanguageCode.EN: LanguageDiscourseRegistryEntry(
            language=LanguageCode.EN,
            dialogue_cues=DialogueCueConfig(
                speech_verb_hints=(
                    "he said",
                    "she said",
                    "i said",
                    "they said",
                    "he asked",
                    "she asked",
                    "i asked",
                    "they asked",
                    "he replied",
                    "she replied",
                    "i replied",
                    "they replied",
                ),
                dialogue_dash_prefixes=_COMMON_DIALOGUE_CUES.dialogue_dash_prefixes,
                quote_chars=_COMMON_DIALOGUE_CUES.quote_chars,
            ),
        ),
        LanguageCode.ZH: LanguageDiscourseRegistryEntry(
            language=LanguageCode.ZH,
            dialogue_cues=DialogueCueConfig(
                speech_verb_hints=(
                    "他说",
                    "她说",
                    "我说",
                    "他们说",
                    "问道",
                    "答道",
                    "说道",
                    "答曰",
                ),
                dialogue_dash_prefixes=_COMMON_DIALOGUE_CUES.dialogue_dash_prefixes,
                quote_chars=_COMMON_DIALOGUE_CUES.quote_chars,
            ),
        ),
    }

    def get_entry(self, language: LanguageCode | str) -> LanguageDiscourseRegistryEntry:
        resolved = LanguageCodeResolver.resolve(language)
        entry = self._ENTRIES.get(resolved)
        if entry is not None:
            return entry
        return LanguageDiscourseRegistryEntry(
            language=resolved,
            dialogue_cues=self._COMMON_DIALOGUE_CUES,
        )

    def get_dialogue_cues(self, language: LanguageCode | str) -> DialogueCueConfig:
        return self.get_entry(language).dialogue_cues

