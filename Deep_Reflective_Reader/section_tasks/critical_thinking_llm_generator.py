from __future__ import annotations

from llm.llm_provider import LLMProvider
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_interaction_prompt_context import (
    NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION,
    build_evidence_priority_context_text,
)


CRITICAL_THINKING_PROMPT_VERSION = "critical_thinking_prompt_v1"
LANGUAGE_LABELS_BY_CODE = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class CriticalThinkingLLMGenerator:
    """Generate strict JSON critical-thinking questions and evaluations."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def generate_question(
        self,
        request: ReadingInteractionRequest,
        instruction: str,
    ) -> str:
        context_text = build_evidence_priority_context_text(
            request,
            secondary_context_label="Secondary artifact context for learning continuity",
        )
        prompt = (
            f"{instruction}\n\n"
            "Return only the JSON object. Do not include markdown, code fences, "
            "headings, explanations, or any text before or after the JSON.\n\n"
            f"{self._language_instruction_from_context(request.context_metadata)}"
            f"{context_text}\n"
        )
        return self._llm_provider.complete_text(prompt)

    def evaluate_answer(
        self,
        session: ReadingInteractionArtifact,
        instruction: str,
    ) -> str:
        question = session.payload.get("question")
        answer = session.payload.get("answer")
        prompt = (
            f"{instruction}\n\n"
            "Return only the JSON object. Do not include markdown, code fences, "
            "headings, explanations, or any text before or after the JSON.\n\n"
            f"{NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION}\n"
            f"{self._language_instruction_from_session(session)}"
            f"Document id: {session.document_id}\n"
            f"Target level: {session.target_level}\n"
            f"Target id: {session.target_id}\n\n"
            f"Question:\n{question}\n\n"
            f"User answer:\n{answer}\n"
        )
        return self._llm_provider.complete_text(prompt)

    @classmethod
    def _language_instruction_from_session(
        cls,
        session: ReadingInteractionArtifact,
    ) -> str:
        context = session.metadata.get("context")
        if not isinstance(context, dict):
            return ""
        return cls._language_instruction_from_context(context)

    @staticmethod
    def _language_instruction_from_context(context_metadata: dict[str, object]) -> str:
        raw_language = context_metadata.get("document_language")
        if not isinstance(raw_language, str):
            return ""
        language = raw_language.strip()
        if not language or language.lower() == "unknown":
            return ""
        language_label = LANGUAGE_LABELS_BY_CODE.get(language.lower(), language)
        return (
            "Output language: Write every user-facing JSON string in "
            f"{language_label} ({language}).\n"
        )
