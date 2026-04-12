from abc import ABC, abstractmethod


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
