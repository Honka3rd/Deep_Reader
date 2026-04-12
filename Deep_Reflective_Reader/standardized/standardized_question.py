from dataclasses import dataclass

@dataclass
class StandardizedQuestion:
    """Normalized user-question payload used in retrieval and prompting."""
    original_query: str
    standardized_query: str
    user_language: str
    document_language: str
    def __init__(self, original_query, standardized_query, user_language, document_language):
        """Initialize object state and injected dependencies.

Args:
    original_query: Original query.
    standardized_query: Standardized query.
    user_language: User language.
    document_language: Primary document language code (e.g. en/zh).
"""
        self.original_query = original_query
        self.standardized_query = standardized_query
        self.user_language = user_language
        self.document_language = document_language
