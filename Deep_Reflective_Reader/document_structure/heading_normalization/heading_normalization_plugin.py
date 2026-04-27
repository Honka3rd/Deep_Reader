from typing import Protocol

from language.language_code import LanguageCode


class HeadingNormalizationPlugin(Protocol):
    """Plugin contract for heading-text normalization before heading detection."""

    name: str

    def normalize(self, heading: str, language: LanguageCode) -> str:
        """Return normalized heading text."""
        ...
