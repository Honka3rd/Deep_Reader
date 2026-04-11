from typing import List

from llama_index.embeddings.openai import OpenAIEmbedding

from api_key_provider import APIKeyProvider
from embedder import Embedder


class OpenAIEmbedder(Embedder):
    embed_model: OpenAIEmbedding

    def __init__(
        self,
        api_key_provider: APIKeyProvider,
        model: str = "text-embedding-3-small",
    ):
        api_key = api_key_provider.get()

        self.embed_model = OpenAIEmbedding(
            model=model,
            api_key=api_key,
        )

    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)

    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embed_model.get_text_embedding_batch(texts)

    def probe_vector_dimension(self) -> int:
        return len(self.get_text_embedding("dimension probe"))