from __future__ import annotations

from llm.llm_provider import LLMProvider
from section_tasks.interaction_source_sufficiency import SOURCE_SUFFICIENCY_INSTRUCTION
from section_tasks.interaction_output_language import (
    language_instruction, request_language, session_language,
)
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionArtifact,
    ReadingInteractionRequest,
)
from section_tasks.reading_interaction_prompt_context import (
    NO_EXTERNAL_RESEARCH_DEFAULT_INSTRUCTION,
    build_evidence_priority_context_text,
)


CRITICAL_THINKING_PROMPT_VERSION = "critical_thinking_prompt_v3"


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
            f"{SOURCE_SUFFICIENCY_INSTRUCTION}\n"
            f"{context_text}\n"
            f"{language_instruction(request_language(request))}"
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
            f"Document id: {session.document_id}\n"
            f"Target level: {session.target_level}\n"
            f"Target id: {session.target_id}\n\n"
            f"Question:\n{question}\n\n"
            f"User answer:\n{answer}\n"
            f"{language_instruction(session_language(session), evaluation=True)}"
        )
        return self._llm_provider.complete_text(prompt)
