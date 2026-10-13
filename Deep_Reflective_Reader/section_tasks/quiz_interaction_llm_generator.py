from __future__ import annotations

from llm.llm_provider import LLMProvider
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)
from section_tasks.reading_interaction_prompt_context import (
    build_evidence_priority_context_text,
)


QUIZ_INTERACTION_PROMPT_VERSION = "quiz_interaction_prompt_v1"


class QuizInteractionLLMGenerator:
    """Generate strict JSON quiz artifacts for one resolved reading target."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def __call__(
        self,
        request: ReadingInteractionRequest,
        max_items: int,
        valid_types: tuple[str, ...],
    ) -> str:
        context_text = build_evidence_priority_context_text(
            request,
            secondary_context_label="Secondary artifact context for coverage/deduplication",
        )
        prompt = (
            "You are generating a reading-comprehension quiz for a drawer UI.\n"
            "Return only a strict JSON object with an items array.\n"
            f"Generate at most {max_items} items. Valid item types are: "
            f"{', '.join(valid_types)}.\n"
            "Each item must include a non-empty question, type, answer, and "
            "an explanation or rationale. Multiple-choice items must use a "
            "choices array with at least two strings, and answer must exactly "
            "match one choice. Do not use an options field in the model output. "
            "True/false answers must be boolean. Do not include markdown, "
            "code fences, or extra prose.\n\n"
            f"{context_text}\n"
        )
        return self._llm_provider.complete_text(prompt)
