import re
import unicodedata

from language.language_code import LanguageCode


class CommonHeadingTypographyNormalizationPlugin:
    """Conservative, language-agnostic heading typography normalization."""

    name = "common_heading_typography_normalization"

    _WHITESPACE_COLLAPSE_PATTERN = re.compile(r"\s+")
    _DASH_TRANSLATION_TABLE = str.maketrans(
        {
            "—": "-",
            "–": "-",
            "―": "-",
            "−": "-",
            "‑": "-",
            "‒": "-",
            "﹣": "-",
            "－": "-",
            "─": "-",
        }
    )

    def normalize(self, heading: str, _language: LanguageCode) -> str:
        """Normalize Unicode form, dash variants, and redundant whitespace."""
        normalized = unicodedata.normalize("NFKC", heading)
        normalized = normalized.translate(self._DASH_TRANSLATION_TABLE)
        normalized = normalized.replace("\u3000", " ")
        normalized = self._WHITESPACE_COLLAPSE_PATTERN.sub(" ", normalized)
        return normalized.strip()
