from __future__ import annotations

from llm.llm_provider import LLMProvider
from section_tasks.interaction_source_sufficiency import SOURCE_SUFFICIENCY_INSTRUCTION
from section_tasks.interaction_output_language import language_instruction, request_language
from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)
from section_tasks.reading_interaction_prompt_context import (
    build_evidence_priority_context_text,
)


ANALYSIS_INTERACTION_PROMPT_VERSION = "analysis_interaction_prompt_v7"


class AnalysisInteractionLLMGenerator:
    """Generate strict JSON analysis artifacts for one resolved reading target."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def __call__(self, request: ReadingInteractionRequest) -> str:
        context_text = build_evidence_priority_context_text(
            request,
            secondary_context_label=(
                "Secondary artifact context for abstraction/deduplication"
            ),
        )
        prompt = (
            "You are generating an interpretive reading insight for an inline reading UI.\n"
            "Insight is not a summary action: do not primarily recap what happens.\n"
            "Use the JSON field named summary for a short insight thesis, not for plot "
            "summary or section overview.\n"
            "Ground the insight in 1-3 concrete textual details from the Primary Source, "
            "such as an action, object, bodily condition, metaphor, diction, narrative "
            "angle, or ambiguity.\n"
            "Explain how those details complicate or correct a simple reading of the "
            "character, event, theme, relation, or narrative method.\n"
            "When supported by the Primary Source, identify a tension or contrast "
            "inside the passage.\n"
            "Do not let the output be dominated by plot recap, chapter overview, "
            "'this section mainly says' framing, or generic thematic summary; use only "
            "brief orientation when needed to anchor the insight.\n"
            "Return only a strict JSON object with non-empty string fields: "
            "summary, reasoning, explanation.\n"
            "Do not include markdown, code fences, citations, or extra keys.\n\n"
            f"{SOURCE_SUFFICIENCY_INSTRUCTION}\n"
            f"{context_text}\n"
            f"{language_instruction(request_language(request))}"
        )
        return self._llm_provider.complete_text(prompt)
