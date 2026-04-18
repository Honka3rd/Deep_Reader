from typing import Dict, List, Set

import faiss
import numpy as np

from embeddings.embedder import Embedder
from llm.llm_model_capabilities import LLMModelCapabilities
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata
from profile.document_profile import DocumentProfile

class FaissIndexBundle:
    """Document-scoped runtime retrieval bundle."""
    faiss_index: faiss.IndexIDMap
    id_to_record: Dict[int, NodeRecord]
    dimension: int
    embedder: Embedder
    profile: DocumentProfile
    node_key_to_faiss_id: Dict[str, int]
    chunk_index_to_faiss_id: Dict[int, int]
    max_context_tokens: int
    max_prompt_tokens: int
    reserved_output_tokens: int
    model_capabilities: LLMModelCapabilities | None

    def set_profile(self, profile: DocumentProfile):
        """Set profile.

Args:
    profile: Document profile with topic/language/summary fields.

Returns:
    Current bundle instance (for fluent-style chaining)."""
        self.profile = profile
        return self

    def __init__(
        self,
        faiss_index: faiss.IndexIDMap,
        embedder: Embedder,
        model_capabilities: LLMModelCapabilities | None,
        id_to_record: Dict[int, NodeRecord],
        dimension: int,
        document_language: str,
        max_context_tokens: int = 1500,
        max_prompt_tokens: int = 3200,
        reserved_output_tokens: int = 500,
    ):
        """Initialize object state and injected dependencies.

Args:
    faiss_index: Faiss index.
    embedder: Embedder.
    model_capabilities: Optional static model capability metadata for this runtime.
    id_to_record: Id to record.
    dimension: Dimension.
    document_language: Primary document language code (e.g. en/zh).
    max_context_tokens: Soft token budget for context text passed into prompt.
    max_prompt_tokens: Soft total prompt budget (context + instructions + profile).
    reserved_output_tokens: Reserved completion budget not available to prompt context.
"""
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.model_capabilities = model_capabilities
        self.id_to_record = id_to_record
        self.dimension = dimension
        self.document_language = document_language
        self.max_context_tokens = max_context_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.node_key_to_faiss_id = {
            record.node_key(): faiss_id
            for faiss_id, record in self.id_to_record.items()
            if record.node_key()
        }
        self.chunk_index_to_faiss_id = {
            chunk_index: faiss_id
            for faiss_id, record in self.id_to_record.items()
            for chunk_index in [record.chunk_index()]
            if isinstance(chunk_index, int)
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Internal helper for normalize text.

Args:
    text: Input text content.

Returns:
    Text with collapsed whitespace for deduplication/comparison."""
        return " ".join(text.strip().split())

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchMetadata]:
        """Retrieve top-k nearest chunks from the vector index.

Args:
    query: Question text used in retrieval/answering flow.
    top_k: Maximum number of retrieval hits to use.

Returns:
    Ranked retrieval hits with score and node metadata."""
        query_vector_list: List[float] = self.embedder.get_text_embedding(query)
        query_vector_np: np.ndarray = np.array(
            query_vector_list,
            dtype=np.float32
        ).reshape(1, -1)

        distances, indices = self.faiss_index.search(query_vector_np, top_k)

        results: List[SearchMetadata] = []
        seen_texts: Set[str] = set()

        for score, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue

            record: NodeRecord | None = self.id_to_record.get(int(faiss_id))
            if record is None:
                continue

            text: str = record.text()
            normalized_text: str = self._normalize_text(text)

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            results.append(
                SearchMetadata(
                    faiss_id=int(faiss_id),
                    node_key=record.node_key(),
                    text=text,
                    score=float(score),
                    source=record.source(),
                    chapter=record.chapter(),
                    position=record.position(),
                )
            )

        print("FaissIndexBundle#search:", results)
        return results

    def _record_from_faiss_id(self, faiss_id: int) -> NodeRecord | None:
        """Internal helper for record from faiss id.

Args:
    faiss_id: FAISS index id for a stored vector.

Returns:
    Matching node record, or ``None`` if id is not present."""
        return self.id_to_record.get(int(faiss_id))

    def get_node_by_id(self, faiss_id: int) -> NodeRecord | None:
        """Return node record by FAISS id."""
        return self._record_from_faiss_id(faiss_id)
