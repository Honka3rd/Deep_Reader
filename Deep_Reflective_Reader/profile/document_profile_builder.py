from llm_provider import LLMProvider
from profile.document_profile import DocumentProfile


class DocumentProfileBuilder:
    llm_provider: LLMProvider

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def build(
        self,
        text: str,
        document_language: str,
    ) -> DocumentProfile:
        topic: str = self._detect_topic(text, document_language)
        summary: str = self._generate_summary(text, document_language, topic)

        return DocumentProfile(
            topic=topic,
            summary=summary,
            document_language=document_language,
        )

    def _detect_topic(self, text: str, document_language: str) -> str:
        prompt = f"""
You are a document classifier.

Determine the primary topic/category of the following document.

Possible examples include:
- literary fiction
- science fiction
- biography
- history
- philosophy
- financial report
- news
- technical documentation
- essay
- academic paper
- other

Return ONLY a short topic label in English.

Document language: {document_language}

Document excerpt:
{text}
"""
        return self.llm_provider.complete_text(prompt).strip().lower()

    def _generate_summary(
        self,
        text: str,
        document_language: str,
        topic: str,
    ) -> str:
        prompt = f"""
You are a document profiling assistant.

The document topic is: {topic}
The document language is: {document_language}

Generate a concise background summary for prompt conditioning.

Rules:
1. The summary must be useful for later question answering.
2. Keep it factual and high-level.
3. Do not include unnecessary details.
4. Prefer stable background information over narrow local details.
5. Keep it within 120-180 English words.
6. If the document is literary, emphasize main characters, relationships, and disambiguation notes.
7. If the document is a biography, emphasize subject identity, timeline, and major roles.
8. If the document is a financial report, emphasize company/entity, reporting period, business scope, and major financial themes.
9. Return only the summary text.

Document excerpt:
{text[:5000]}
"""
        return self.llm_provider.complete_text(prompt).strip()