"""Text-only translation contracts; no document or persistence dependencies."""

from dataclasses import dataclass
import re
from typing import Literal

from language.language_code import LanguageCode


_BASE_LANGUAGES = {code.value for code in LanguageCode if code != LanguageCode.UNKNOWN}


class TranslationError(ValueError):
    """Stable, source-free failure suitable for a future transport adapter."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TranslationResult:
    source_language: str
    target_language: str
    status: Literal["completed", "partial", "untranslatable"]
    translated_text: str | None


def normalize_translation_language(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TranslationError(f"translation_invalid_{field}")
    match = re.fullmatch(r"([a-z]{2})(?:-([a-z]{4}))?(?:-([a-z]{2}|[0-9]{3}))?",
                         value.strip().lower().replace("_", "-"))
    if not match or match[1] not in _BASE_LANGUAGES:
        raise TranslationError(f"translation_invalid_{field}")
    base, script, region = match.groups()
    return "-".join(part for part in (base, script.title() if script else None,
                                     region.upper() if region else None) if part)


def ensure_distinct_translation_languages(source: str | None, target: str) -> None:
    """Compare validated/normalized codes before generation or saved-result reuse."""
    if source is not None and source == target:
        raise TranslationError("translation_same_language")
