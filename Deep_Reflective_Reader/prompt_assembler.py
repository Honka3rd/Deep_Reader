from profile.document_profile import DocumentProfile
from standardized.standardized_question import StandardizedQuestion
from evaluated_answer.answer_mode import AnswerMode

class PromptAssembler:
    """Build final LLM prompts from rules, context mode, and profile."""

    @staticmethod
    def render_profile(profile: DocumentProfile) -> str:
        """Render document profile into a stable prompt section.

        Args:
            profile: Topic/language/summary metadata generated for the document.

        Returns:
            Multiline profile block injected into the final answer prompt.
        """
        topic: str = profile.topic.strip() if profile.topic else "unknown"
        summary: str = profile.summary.strip() if profile.summary else "No summary available."
        document_language: str = (
            profile.document_language.strip()
            if profile.document_language
            else "unknown"
        )

        return f"""Document topic:
{topic}

Document language:
{document_language}

Document summary:
{summary}
"""
    @staticmethod
    def _render_rules(
                answer_mode: AnswerMode,
                question: StandardizedQuestion,
        ) -> str:
        """Render answer constraints based on retrieval confidence.

        Args:
            answer_mode: Strict/cautious/reject mode from relevance evaluator.
            question: Standardized question object with user/doc language fields.

        Returns:
            Rule text used in the prompt's instruction section.
        """
        if answer_mode.level == "strict":
            return f"""Rules:
    1. Use ONLY the provided retrieved context.
    2. Do NOT use outside knowledge.
    3. If the evidence is insufficient, answer exactly: Not found.
    4. Do not infer beyond explicit evidence.
    5. Distinguish carefully between different people or entities with similar names.
    6. The retrieved context is written in {question.document_language}.
    7. The user's original question language is {question.user_language}.
    8. Answer in {question.user_language}.
    """

        if answer_mode.level == "cautious":
            return f"""Rules:
    1. Use the retrieved context as primary evidence.
    2. You may use the document summary as supporting background.
    3. Do NOT use outside knowledge.
    4. If evidence is incomplete, use cautious wording such as "可能", "看起來", "may", "might", or "suggests".
    5. Do not present uncertain conclusions as facts.
    6. Distinguish carefully between different people or entities with similar names.
    7. The retrieved context is written in {question.document_language}.
    8. The user's original question language is {question.user_language}.
    9. Answer in {question.user_language}.
    """

        return f"""Rules:
    1. The current question appears weakly related to the document.
    2. Answer exactly: Not found.
    3. The user's original question language is {question.user_language}.
    4. Answer in {question.user_language}.
    """

    @staticmethod
    def _render_prompt_mode_guidance(prompt_mode: str) -> str:
        """Render mode-specific guidance for context interpretation.

        Args:
            prompt_mode: ``local_reading_mode`` / ``retrieval_mode`` / ``full_text_mode``.

        Returns:
            Additional guidance that tells the model how to use provided context.
        """
        if prompt_mode == "local_reading_mode":
            return """Context mode:
local_reading_mode

Reading guidance:
1. You are helping the user read around the currently active passage.
2. Treat the provided context as a locally continuous window near where the user is reading.
3. Prefer continuity and nearby textual evidence when describing people, events, tone, or intent.
4. If needed, refer to immediate neighboring sentences in this local window before broad generalization.
"""

        if prompt_mode == "full_text_mode":
            return """Context mode:
full_text_mode

Reading guidance:
1. The provided context covers the full document (possibly budget-trimmed).
2. Prefer globally consistent interpretation across sections and avoid local overfitting.
3. When evidence conflicts, state uncertainty explicitly and cite the stronger evidence in context.
"""

        return """Context mode:
retrieval_mode

Reading guidance:
1. The provided context is assembled from retrieval results across the document.
2. Use the retrieved evidence directly and avoid assumptions about nearby continuity unless explicitly shown.
"""

    def build_answer_prompt(
        self,
        context: str,
        question: StandardizedQuestion,
        profile: DocumentProfile,
        answer_mode: AnswerMode,
        prompt_mode: str = "retrieval_mode",
    ) -> str:
        """Assemble complete answer prompt passed to the LLM backend.

        Args:
            context: Evidence context selected by retrieval/local-window logic.
            question: Standardized question payload used in current QA turn.
            profile: Document profile metadata for global disambiguation.
            answer_mode: Strictness mode controlling answer constraints.
            prompt_mode: Context mode flag controlling local vs global wording.

        Returns:
            Final prompt string for model completion.
        """
        rendered_profile: str = self.render_profile(profile)
        rules: str = self._render_rules(answer_mode, question)
        mode_guidance: str = self._render_prompt_mode_guidance(prompt_mode)
        return f"""
You are a literary and document analysis assistant.

Rules:
{rules}

Mode guidance:
{mode_guidance}

profile:
{rendered_profile}

Retrieved context:
{context}

User question:
{question.original_query}

Standardized question for retrieval:
{question.standardized_query}

Answer:
"""
