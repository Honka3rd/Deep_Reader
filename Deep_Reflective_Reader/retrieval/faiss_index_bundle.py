from typing import Dict, List, Set

import faiss
import numpy as np

from context.document_context_builder import DocumentContextBuilder
from context.token_budget_manager import TokenBudgetManager
from embeddings.embedder import Embedder
from llm.llm_model_capabilities import LLMModelCapabilities
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata
from profile.document_profile import DocumentProfile
from question.qa_enums import PromptMode
from evaluated_answer.answer_mode import AnswerMode

class FaissIndexBundle:
    """Runtime retrieval bundle that supports search and context utility methods."""
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
    token_budget_manager: TokenBudgetManager
    document_context_builder: DocumentContextBuilder

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
        token_budget_manager: TokenBudgetManager,
        document_context_builder: DocumentContextBuilder,
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
        self.token_budget_manager = token_budget_manager
        self.document_context_builder = document_context_builder
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

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count with a mixed-language heuristic."""
        return self.token_budget_manager.estimate_tokens(text)

    def _truncate_text_to_token_budget(self, text: str, budget_tokens: int) -> str:
        """Truncate one text segment to fit within token budget."""
        return self.token_budget_manager.truncate_text_to_token_budget(
            text=text,
            budget_tokens=budget_tokens,
        )

    def _join_texts_with_budget(
        self,
        texts: List[str],
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Join ordered texts while respecting context token budget."""
        return self.token_budget_manager.join_texts_with_budget(
            texts=texts,
            default_max_context_tokens=self.max_context_tokens,
            max_context_tokens=max_context_tokens,
        )

    def _estimate_non_context_prompt_tokens(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
    ) -> int:
        """Estimate token usage of prompt parts excluding retrieved context."""
        return self.token_budget_manager.estimate_non_context_prompt_tokens(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
            profile=self.profile,
        )

    def _compute_available_context_budget(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
        max_context_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
    ) -> int:
        """Compute effective context budget under total prompt token constraint."""
        return self.token_budget_manager.compute_available_context_budget(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
            profile=self.profile,
            default_max_context_tokens=self.max_context_tokens,
            default_max_prompt_tokens=self.max_prompt_tokens,
            default_reserved_output_tokens=self.reserved_output_tokens,
            max_context_tokens=max_context_tokens,
            max_prompt_tokens=max_prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

    def compute_available_context_budget(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
    ) -> int:
        """Public wrapper for computing effective context token budget."""
        return self._compute_available_context_budget(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )

    def compute_available_context_budget_with_override(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
        max_context_tokens: int,
        max_prompt_tokens: int,
        reserved_output_tokens: int,
    ) -> int:
        """Compute context budget with explicit limits overriding bundle defaults."""
        return self._compute_available_context_budget(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
            max_context_tokens=max_context_tokens,
            max_prompt_tokens=max_prompt_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

    def join_texts_with_budget(
        self,
        texts: List[str],
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Public wrapper for joining text under context budget."""
        return self._join_texts_with_budget(
            texts=texts,
            max_context_tokens=max_context_tokens,
        )

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

    def build_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """Build plain context by joining retrieved chunk text.

Args:
    query: Question text used in retrieval/answering flow.
    top_k: Maximum number of retrieval hits to use.

Returns:
    Newline-joined retrieval context text."""
        results: List[SearchMetadata] = self.search(query, top_k)
        texts: List[str] = [result.text for result in results]
        context, used_tokens, truncated = self._join_texts_with_budget(texts)
        print(
            "FaissIndexBundle#build_context:",
            f"token_used={used_tokens}",
            f"budget={self.max_context_tokens}",
            f"truncated={truncated}",
        )
        return context

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

    def build_full_text_context(
        self,
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Build full-document context (chunk-ordered) within token budget."""
        return self.document_context_builder.build_full_text_context(
            bundle=self,
            max_context_tokens=max_context_tokens,
        )

    def estimate_full_text_tokens(self) -> int:
        """Estimate token usage for full document text in chunk order."""
        return self.document_context_builder.estimate_full_text_tokens(self)

    def _resolve_neighbor_by_link(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Compatibility wrapper delegated to document context builder."""
        return self.document_context_builder._resolve_neighbor_by_link(
            bundle=self,
            record=record,
            is_prev=is_prev,
        )

    def _resolve_neighbor_by_chunk_index(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Compatibility wrapper delegated to document context builder."""
        return self.document_context_builder._resolve_neighbor_by_chunk_index(
            bundle=self,
            record=record,
            is_prev=is_prev,
        )

    def _resolve_neighbor(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Compatibility wrapper delegated to document context builder."""
        return self.document_context_builder._resolve_neighbor(
            bundle=self,
            record=record,
            is_prev=is_prev,
        )

    def _expand_side(
        self,
        center: NodeRecord,
        radius: int,
        is_prev: bool,
    ) -> List[NodeRecord]:
        """Compatibility wrapper delegated to document context builder."""
        return self.document_context_builder._expand_side(
            bundle=self,
            center=center,
            radius=radius,
            is_prev=is_prev,
        )

    def build_local_window(
        self,
        center: SearchMetadata | int,
        radius: int = 1,
    ) -> List[str]:
        """Build local context window around a center hit.

Args:
    center: Center hit (SearchMetadata or FAISS id) for local window build.
    radius: Neighbor expansion radius on each side of center chunk.

        Returns:
    Ordered text chunks around center (left->center->right)."""
        return self.document_context_builder.build_local_window(
            bundle=self,
            center=center,
            radius=radius,
        )

    def build_local_window_dynamic(
        self,
        center: SearchMetadata | int,
        max_context_tokens: int | None = None,
    ) -> tuple[List[str], int, bool, int]:
        """Build local window by dynamic expansion under token budget.

        Args:
            center: Center hit (SearchMetadata or FAISS id).
            max_context_tokens: Optional override for context budget.

        Returns:
            Tuple ``(window_texts, used_tokens, truncated, used_radius)``.
        """
        return self.document_context_builder.build_local_window_dynamic(
            bundle=self,
            center=center,
            max_context_tokens=max_context_tokens,
        )

    def build_context_with_window(
        self,
        query: str,
        top_k: int = 3,
        radius: int = 1,
    ) -> str:
        """Build merged context by expanding hits into local windows.

Args:
    query: Question text used in retrieval/answering flow.
    top_k: Maximum number of retrieval hits to use.
    radius: Neighbor expansion radius on each side of center chunk.

Returns:
    Merged context text from window-expanded hits."""
        return self.document_context_builder.build_context_with_window(
            bundle=self,
            query=query,
            top_k=top_k,
            radius=radius,
        )
