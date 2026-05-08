from __future__ import annotations

from dataclasses import dataclass

from language.language_code import LanguageCode, LanguageCodeResolver

_COMMON_DIALOGUE_DASH_PREFIXES: tuple[str, ...] = ("-", "—", "–")
_COMMON_QUOTE_CHARS: tuple[str, ...] = (
    '"',
    "“",
    "”",
    "「",
    "」",
    "『",
    "』",
    "‘",
    "’",
)
_EURO_QUOTE_CHARS: tuple[str, ...] = _COMMON_QUOTE_CHARS + ("«", "»")
_DE_QUOTE_CHARS: tuple[str, ...] = _EURO_QUOTE_CHARS + ("„",)


@dataclass(frozen=True)
class DialogueCueConfig:
    speech_verb_hints: tuple[str, ...] = tuple()
    dialogue_dash_prefixes: tuple[str, ...] = ("-", "—", "–")
    quote_chars: tuple[str, ...] = tuple()


@dataclass(frozen=True)
class LanguageDiscourseRegistryEntry:
    language: LanguageCode
    dialogue_cues: DialogueCueConfig


def _entry(
    language: LanguageCode,
    *,
    speech_verb_hints: tuple[str, ...] = tuple(),
    quote_chars: tuple[str, ...] | None = None,
    dialogue_dash_prefixes: tuple[str, ...] | None = None,
) -> LanguageDiscourseRegistryEntry:
    return LanguageDiscourseRegistryEntry(
        language=language,
        dialogue_cues=DialogueCueConfig(
            speech_verb_hints=speech_verb_hints,
            quote_chars=quote_chars or _COMMON_QUOTE_CHARS,
            dialogue_dash_prefixes=(
                dialogue_dash_prefixes
                or _COMMON_DIALOGUE_DASH_PREFIXES
            ),
        ),
    )


class LanguageDiscourseRegistry:
    """Registry for language-scoped discourse cues used by profile metadata only.

    This registry intentionally keeps speech verb hints sparse.
    Missing speech verb hints for a language means conservative fallback,
    not unsupported language.
    """

    _COMMON_DIALOGUE_CUES = DialogueCueConfig(
        speech_verb_hints=tuple(),
        dialogue_dash_prefixes=_COMMON_DIALOGUE_DASH_PREFIXES,
        quote_chars=_COMMON_QUOTE_CHARS,
    )

    _ENTRIES: dict[LanguageCode, LanguageDiscourseRegistryEntry] = {
        LanguageCode.EN: _entry(
            LanguageCode.EN,
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
        ),
        LanguageCode.ZH: _entry(
            LanguageCode.ZH,
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
        ),
        LanguageCode.JA: _entry(
            LanguageCode.JA,
            speech_verb_hints=(
                "と言った",
                "尋ねた",
                "答えた",
            ),
        ),
        LanguageCode.KO: _entry(
            LanguageCode.KO,
            speech_verb_hints=(
                "말했다",
                "물었다",
                "대답했다",
            ),
        ),
        LanguageCode.FR: _entry(
            LanguageCode.FR,
            speech_verb_hints=(
                "dit-il",
                "dit-elle",
                "demanda",
                "répondit",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.DE: _entry(
            LanguageCode.DE,
            speech_verb_hints=(
                "sagte er",
                "sagte sie",
                "fragte",
                "antwortete",
            ),
            quote_chars=_DE_QUOTE_CHARS,
        ),
        LanguageCode.ES: _entry(
            LanguageCode.ES,
            speech_verb_hints=(
                "dijo",
                "preguntó",
                "respondió",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.PT: _entry(
            LanguageCode.PT,
            speech_verb_hints=(
                "disse",
                "perguntou",
                "respondeu",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.IT: _entry(
            LanguageCode.IT,
            speech_verb_hints=(
                "disse",
                "chiese",
                "rispose",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.RU: _entry(
            LanguageCode.RU,
            speech_verb_hints=(
                "сказал",
                "сказала",
                "спросил",
                "ответил",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.UK: _entry(
            LanguageCode.UK,
            speech_verb_hints=(
                "сказав",
                "сказала",
                "запитав",
                "відповів",
            ),
            quote_chars=_EURO_QUOTE_CHARS,
        ),
        LanguageCode.AR: _entry(LanguageCode.AR),
        LanguageCode.HI: _entry(LanguageCode.HI),
        LanguageCode.TR: _entry(LanguageCode.TR),
        LanguageCode.NL: _entry(LanguageCode.NL),
        LanguageCode.PL: _entry(LanguageCode.PL),
        LanguageCode.ID: _entry(LanguageCode.ID),
        LanguageCode.VI: _entry(LanguageCode.VI),
        LanguageCode.TH: _entry(LanguageCode.TH),
        LanguageCode.UNKNOWN: _entry(LanguageCode.UNKNOWN),
    }

    @classmethod
    def supported_languages(cls) -> tuple[LanguageCode, ...]:
        return tuple(cls._ENTRIES.keys())

    def get_entry(self, language: LanguageCode | str) -> LanguageDiscourseRegistryEntry:
        resolved = LanguageCodeResolver.resolve(language)
        return self._ENTRIES.get(
            resolved,
            self._ENTRIES[LanguageCode.UNKNOWN],
        )

    def get_dialogue_cues(self, language: LanguageCode | str) -> DialogueCueConfig:
        return self.get_entry(language).dialogue_cues
