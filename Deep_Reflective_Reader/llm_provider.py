from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete_text(self, prompt: str) -> str:
        raise NotImplementedError