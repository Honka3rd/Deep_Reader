from dataclasses import dataclass

from language.language_code import LanguageCode


@dataclass
class StandardizedQuestion:
    """Normalized user-question payload used in retrieval and prompting."""
    original_query: str
    standardized_query: str
    user_language: LanguageCode
    document_language: LanguageCode
