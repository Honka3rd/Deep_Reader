from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    @abstractmethod
    def get_text_embedding(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def probe_vector_dimension(self) -> int:
        raise NotImplementedError