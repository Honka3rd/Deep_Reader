from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """Abstract interface for embedding generation backends."""

    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for one text input.

Args:
    text: Input text content.

Returns:
    Dense embedding vector for ``text``."""
        raise NotImplementedError

    @abstractmethod
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts.

Args:
    texts: Texts.

Returns:
    One embedding vector per input text, preserving input order."""
        raise NotImplementedError

    @abstractmethod
    def probe_vector_dimension(self) -> int:
        """Probe embedding dimension used by current model backend.

Returns:
    Embedding vector size used by the backend model."""
        raise NotImplementedError
