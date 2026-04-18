from abc import ABC, abstractmethod


class APIKeyProvider(ABC):
    """Abstract provider interface for reading API keys."""

    @abstractmethod
    def get(self) -> str:
        """Return API key value from concrete provider implementation."""
        raise NotImplementedError

