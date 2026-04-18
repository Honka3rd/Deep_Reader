from abc import ABC, abstractmethod

from llm_model_capabilities import LLMModelCapabilities


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
