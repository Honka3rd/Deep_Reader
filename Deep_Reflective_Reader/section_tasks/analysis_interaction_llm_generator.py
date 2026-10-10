from __future__ import annotations

from llm.llm_provider import LLMProvider
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)


ANALYSIS_INTERACTION_PROMPT_VERSION = "analysis_interaction_prompt_v1"


class AnalysisInteractionLLMGenerator:
    """Generate strict JSON analysis artifacts for one resolved reading target."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def __call__(self, request: ReadingInteractionRequest) -> str:
        target = request.target
        prompt = (
            "You are generating a compact reading insight for an inline reading UI.\n"
            "Return only a strict JSON object with non-empty string fields: "
            "summary, reasoning, explanation.\n"
            "Do not include markdown, code fences, citations, or extra keys.\n\n"
            f"Document title: {target.document_title}\n"
            f"Target level: {target.target_level}\n"
            f"Target id: {target.target_id}\n\n"
            "Primary source context:\n"
            f"{target.content}\n"
        )
        if request.secondary_context:
            prompt += (
                "\nSecondary artifact context for abstraction/deduplication:\n"
                f"{request.secondary_context}\n"
            )
        return self._llm_provider.complete_text(prompt)
