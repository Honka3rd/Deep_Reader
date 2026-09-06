import unicodedata

from document_structure.text_normalization import normalize_ocr_whitespace
from language.language_code import LanguageCode


class CommonHeadingTypographyNormalizationPlugin:
    """Conservative, language-agnostic heading typography normalization."""

    name = "common_heading_typography_normalization"

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
        return normalize_ocr_whitespace(normalized)
