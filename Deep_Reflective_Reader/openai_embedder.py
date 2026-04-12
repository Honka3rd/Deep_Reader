from typing import List

from llama_index.embeddings.openai import OpenAIEmbedding

from api_key_provider import APIKeyProvider
from embedder import Embedder


class OpenAIEmbedder(Embedder):
    """Embedder implementation backed by OpenAI via llama-index."""
    embed_model: OpenAIEmbedding

    def __init__(
        self,
        api_key_provider: APIKeyProvider,
        model: str = "text-embedding-3-small",
    ):
        """Initialize object state and injected dependencies.

Args:
    api_key_provider: Api key provider.
    model: Model.
"""
        api_key = api_key_provider.get()

        self.embed_model = OpenAIEmbedding(
            model=model,
            api_key=api_key,
        )

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for one text input.

Args:
    text: Input text content.

Returns:
    Embedding vector for the input text."""
        return self.embed_model.get_text_embedding(text)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts.

Args:
    texts: Texts.

Returns:
    Embedding vectors for the input texts in original order."""
        return self.embed_model.get_text_embedding_batch(texts)

    def probe_vector_dimension(self) -> int:
        """Probe embedding dimension used by current model backend.

Returns:
    Embedding dimension of the configured model."""
        return len(self.get_text_embedding("dimension probe"))
