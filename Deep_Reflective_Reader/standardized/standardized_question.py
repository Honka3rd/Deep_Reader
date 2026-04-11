from dataclasses import dataclass

@dataclass
class StandardizedQuestion:
    original_query: str
    standardized_query: str
    user_language: str
    document_language: str
    def __init__(self, original_query, standardized_query, user_language, document_language):
        self.original_query = original_query
        self.standardized_query = standardized_query
        self.user_language = user_language
        self.document_language = document_language