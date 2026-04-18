import re
from typing import List

from evaluated_answer.answer_mode import AnswerMode
from profile.document_profile import DocumentProfile
from prompts.prompt_assembler import PromptAssembler
from question.qa_enums import PromptMode
from question.standardized.standardized_question import StandardizedQuestion


class TokenBudgetManager:
    """Stateless token/budget utility service for context construction."""

    prompt_assembler: PromptAssembler

    def __init__(self, prompt_assembler: PromptAssembler):
        """Initialize manager with prompt assembler dependency."""
        self.prompt_assembler = prompt_assembler

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count with a mixed-language heuristic."""
        if not text:
            return 0

        cjk_and_kana = len(
            re.findall(
                r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]",
                text,
            )
        )
        hangul = len(re.findall(r"[\uac00-\ud7af]", text))

        ascii_chars = 0
        other_chars = 0
        for ch in text:
            if ch.isspace():
                continue
            if ord(ch) < 128:
                ascii_chars += 1
            else:
                other_chars += 1

        # Remove CJK/Hangul/Kana already counted separately.
        other_non_cjk = max(0, other_chars - cjk_and_kana - hangul)
        ascii_tokens = (ascii_chars + 3) // 4
        return max(1, cjk_and_kana + hangul + ascii_tokens + other_non_cjk)

    def truncate_text_to_token_budget(self, text: str, budget_tokens: int) -> str:
        """Truncate one text segment to fit within token budget."""
        if budget_tokens <= 0:
            return ""
        if self.estimate_tokens(text) <= budget_tokens:
            return text

        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            if self.estimate_tokens(candidate) <= budget_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best.rstrip()

    def join_texts_with_budget(
        self,
        texts: List[str],
        default_max_context_tokens: int,
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Join ordered texts while respecting context token budget."""
        budget = max_context_tokens or default_max_context_tokens
        kept: List[str] = []
        used_tokens = 0
        truncated = False

        for i, text in enumerate(texts):
            text_tokens = self.estimate_tokens(text)
            if used_tokens + text_tokens > budget:
                # Keep first evidence chunk in truncated form if nothing has been added yet.
                if i == 0 and not kept and budget > 0:
                    clipped = self.truncate_text_to_token_budget(text, budget)
                    if clipped:
                        kept.append(clipped)
                        used_tokens = self.estimate_tokens(clipped)
                truncated = True
                break
            kept.append(text)
            used_tokens += text_tokens

        return "\n".join(kept), used_tokens, truncated

    def estimate_non_context_prompt_tokens(
        self,
        question: StandardizedQuestion,
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
        profile: DocumentProfile,
    ) -> int:
        """Estimate token usage of prompt parts excluding retrieved context."""
        base_prompt = self.prompt_assembler.build_answer_prompt(
            context="",
            question=question,
            profile=profile,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )
        return self.estimate_tokens(base_prompt)

    def compute_available_context_budget(
        self,
        question: StandardizedQuestion,
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
        profile: DocumentProfile,
        default_max_context_tokens: int,
        default_max_prompt_tokens: int,
        default_reserved_output_tokens: int,
        max_context_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
    ) -> int:
        """Compute effective context budget under total prompt token constraint."""
        non_context_tokens = self.estimate_non_context_prompt_tokens(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
            profile=profile,
        )
        effective_context_limit = (
            default_max_context_tokens if max_context_tokens is None else max_context_tokens
        )
        effective_prompt_limit = (
            default_max_prompt_tokens if max_prompt_tokens is None else max_prompt_tokens
        )
        effective_output_reserve = (
            default_reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        available_by_total = (
            effective_prompt_limit
            - non_context_tokens
            - effective_output_reserve
        )
        return max(0, min(effective_context_limit, available_by_total))

