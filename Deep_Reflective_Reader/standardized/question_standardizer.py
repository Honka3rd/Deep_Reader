import json

from standardized.standardized_question import StandardizedQuestion
from llm_provider import LLMProvider

class QuestionStandardizer:
    """Detect and align question language to document language."""
    llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider):
        """Initialize object state and injected dependencies.

Args:
    llm_provider: Llm provider.
"""
        self.llm_provider = llm_provider

    def standardize(
        self,
        query: str,
        document_language: str,
    ) -> StandardizedQuestion:
        """Normalize/translate question to match target document language.

Args:
    query: Question text used in retrieval/answering flow.
    document_language: Primary document language code (e.g. en/zh).

Returns:
    Language-aligned question payload for retrieval and prompting."""
        prompt = f"""
You are a language processor.

Your job:
1. Detect the language of the user's query.
2. If the user's query language is different from the target document language,
   translate the query into the target document language.
3. Preserve the original meaning exactly.
4. Return ONLY valid JSON.

Return JSON in this exact format:
{{
  "user_language": "...",
  "standardized_query": "..."
}}

Target document language: {document_language}
User query: {query}
"""

        raw_response: str = self.llm_provider.complete_text(prompt).strip()

        try:
            payload: dict = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"QuestionStandardizer#standardize: failed to parse JSON response: {raw_response}"
            ) from e

        user_language = payload.get("user_language")
        standardized_query = payload.get("standardized_query")

        if not user_language or not standardized_query:
            raise RuntimeError(
                f"QuestionStandardizer#standardize: invalid payload: {payload}"
            )

        return StandardizedQuestion(
            original_query=query,
            standardized_query=standardized_query,
            user_language=user_language,
            document_language=document_language,
        )
