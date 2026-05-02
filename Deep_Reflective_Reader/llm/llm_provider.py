from abc import ABC, abstractmethod

from llm.llm_model_capabilities import LLMModelCapabilities


class LLMProvider(ABC):
    """Abstract interface for text completion backends."""

    @abstractmethod
    def complete_text(self, prompt: str) -> str:
        """Call underlying completion backend and return generated text.

Args:
    prompt: Prompt text sent to completion model.

Returns:
    Model-generated plain text response."""
        raise NotImplementedError

    @abstractmethod
    def get_model_capabilities(self) -> LLMModelCapabilities:
        """Return static model capability metadata used for runtime budgeting."""
        raise NotImplementedError

    def normalize_input_text_for_prompt(
        self,
        text: str,
        *,
        prompt_overhead_tokens: int,
        token_utilization_ratio: float,
        default_excerpt_chars: int,
        min_excerpt_chars: int,
        max_excerpt_chars_hard_cap: int,
        chars_per_token: int,
    ) -> str:
        """Return prompt excerpt normalized by model capability with safe fallback."""
        if not text:
            return ""
        if prompt_overhead_tokens < 0:
            raise ValueError("prompt_overhead_tokens must be >= 0")
        if not (0.0 < token_utilization_ratio <= 1.0):
            raise ValueError("token_utilization_ratio must be in (0, 1]")
        if default_excerpt_chars <= 0:
            raise ValueError("default_excerpt_chars must be > 0")
        if min_excerpt_chars <= 0:
            raise ValueError("min_excerpt_chars must be > 0")
        if max_excerpt_chars_hard_cap <= 0:
            raise ValueError("max_excerpt_chars_hard_cap must be > 0")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")

        fallback_chars = min(len(text), default_excerpt_chars)
        try:
            capability = self.get_model_capabilities()
        except Exception as error:
            print(
                "LLMProvider#normalize_input_text_for_prompt_fallback:",
                f"reason=capability_unavailable error={error}",
            )
            return text[:fallback_chars]

        available_prompt_tokens = int(capability.max_input_tokens) - int(prompt_overhead_tokens)
        if available_prompt_tokens <= 0:
            return text[:fallback_chars]

        excerpt_tokens = int(available_prompt_tokens * token_utilization_ratio)
        if excerpt_tokens <= 0:
            return text[:fallback_chars]

        excerpt_chars = excerpt_tokens * chars_per_token
        excerpt_chars = max(min_excerpt_chars, excerpt_chars)
        excerpt_chars = min(
            len(text),
            excerpt_chars,
            max_excerpt_chars_hard_cap,
        )
        return text[: max(1, excerpt_chars)]
