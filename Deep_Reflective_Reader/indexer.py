from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode

from embedder import Embedder


class Indexer:
    def __init__(self, embedder: Embedder):
        # Embedding strategy is injected so callers can swap it when needed.
        self.embedder = embedder

    def build(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        index: VectorStoreIndex = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embedder.get(),
        )
        return index
