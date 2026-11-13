"""Profile-owned complete request accounting; no provider or persistence IO."""
from __future__ import annotations

import math
from collections.abc import Callable

from llm.llm_provider import LLMProvider


class ProfilePromptBudgetError(ValueError):
    """A profile request cannot be safely dispatched with known capacity."""


class ProfilePromptBudget:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        try:
            capability = provider.get_model_capabilities()
            reserve = getattr(provider, "effective_output_budget", capability.max_output_tokens)
            available = capability.input_token_limit(reserve)
        except Exception:
            raise ProfilePromptBudgetError("profile_prompt_budget:capacity_unavailable") from None
        # Provider counts cover prompt text, not transport framing. Reserve output
        # exactly once, then leave space for framing and counting differences.
        self.limit = available - max(256, math.ceil(available * 0.05))

    def fits(self, prompt: str) -> bool:
        try:
            count = self.provider.count_input_tokens(prompt)
        except Exception:
            count = None
        if type(count) is not int or count <= 0:
            # A UTF-8 byte bound avoids chars/4 undercounting for CJK/numeric text.
            count = len(prompt.encode("utf-8"))
        return count <= self.limit

    def fit_excerpt(self, render: Callable[[str], str], excerpt: str) -> str:
        prompt = render(excerpt)
        if self.fits(prompt):
            return prompt
        best = render("")
        if not self.fits(best):
            raise ProfilePromptBudgetError("profile_prompt_budget:mandatory_content_exceeds_capacity")
        # Keep only measured fitting candidates; correctness does not assume token
        # count is strictly monotonic with character length.
        low, high = 0, len(excerpt)
        while low + 1 < high:
            middle = (low + high) // 2
            candidate = render(excerpt[:middle])
            if self.fits(candidate):
                best, low = candidate, middle
            else:
                high = middle
        return best
