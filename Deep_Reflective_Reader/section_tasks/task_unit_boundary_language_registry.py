from __future__ import annotations

from dataclasses import dataclass

from language.language_code import LanguageCode, LanguageCodeResolver


@dataclass(frozen=True)
class TaskUnitBoundaryLanguageConfig:
    """Language-aware punctuation config used by task-unit boundary normalization."""

    sentence_terminal_chars: frozenset[str]
    strong_punctuation_chars: frozenset[str]
    weak_clause_chars: frozenset[str]
    closing_quote_chars: frozenset[str]
    closing_bracket_chars: frozenset[str]
    paragraph_separators: tuple[str, ...]


class TaskUnitBoundaryLanguageRegistry:
    """Resolve boundary punctuation config for all LanguageCode values."""

    _COMMON_CONFIG = TaskUnitBoundaryLanguageConfig(
        sentence_terminal_chars=frozenset(".!?。！？؛؟"),
        strong_punctuation_chars=frozenset(";；"),
        weak_clause_chars=frozenset(",，:：、"),
        closing_quote_chars=frozenset("\"'”’»›」』》"),
        closing_bracket_chars=frozenset(")]}）】〕〉》"),
        paragraph_separators=("\n\n",),
    )
    @classmethod
    def get_config(
        cls,
        language: LanguageCode | str | None,
    ) -> TaskUnitBoundaryLanguageConfig:
        """Return config for one language; fallback to common defaults."""
        resolved_language: LanguageCode = (
            language
            if isinstance(language, LanguageCode)
            else LanguageCodeResolver.resolve(language)
        )
        return _TASK_UNIT_BOUNDARY_CONFIG_BY_LANGUAGE.get(
            resolved_language,
            cls._COMMON_CONFIG,
        )

    @classmethod
    def get_supported_languages(cls) -> tuple[LanguageCode, ...]:
        """Return all language enums to support coverage tests."""
        return tuple(LanguageCode)


_TASK_UNIT_BOUNDARY_CONFIG_BY_LANGUAGE: dict[LanguageCode, TaskUnitBoundaryLanguageConfig] = {
    language_code: TaskUnitBoundaryLanguageRegistry._COMMON_CONFIG
    for language_code in LanguageCode
}
