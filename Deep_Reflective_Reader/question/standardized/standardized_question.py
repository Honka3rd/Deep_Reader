from dataclasses import dataclass, field

from language.language_code import LanguageCode


@dataclass
class StandardizedQuestion:
    """Normalized user-question payload used in retrieval and prompting."""
    # Consumers log this DTO; retain text for retrieval, omit it from diagnostics.
    original_query: str = field(repr=False)
    standardized_query: str = field(repr=False)
    user_language: LanguageCode
    document_language: LanguageCode
