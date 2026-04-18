import re
from typing import Dict, List, Set

import faiss
import numpy as np

from embeddings.embedder import Embedder
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata
from llm.llm_provider import LLMProvider
from question.standardized.question_standardizer import QuestionStandardizer
from profile.document_profile import DocumentProfile
from prompts.prompt_assembler import PromptAssembler
from question.qa_enums import PromptMode
from evaluated_answer.question_relevance import QuestionRelevanceEvaluator, AnswerMode

class FaissIndexBundle:
    """Runtime retrieval bundle that supports search and context utility methods."""
    faiss_index: faiss.IndexIDMap
    id_to_record: Dict[int, NodeRecord]
    dimension: int
    embedder: Embedder
    relevance_evaluator: QuestionRelevanceEvaluator
    profile: DocumentProfile
    node_key_to_faiss_id: Dict[str, int]
    chunk_index_to_faiss_id: Dict[int, int]
    max_context_tokens: int
    max_prompt_tokens: int
    reserved_output_tokens: int

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
        llm_provider: LLMProvider,
        prompt_assembler: PromptAssembler,
        question_standardizer: QuestionStandardizer,
        relevance_evaluator: QuestionRelevanceEvaluator,
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
    llm_provider: Llm provider.
    prompt_assembler: Prompt assembler.
    question_standardizer: Question standardizer.
    relevance_evaluator: Relevance evaluator.
    id_to_record: Id to record.
    dimension: Dimension.
    document_language: Primary document language code (e.g. en/zh).
    max_context_tokens: Soft token budget for context text passed into prompt.
    max_prompt_tokens: Soft total prompt budget (context + instructions + profile).
    reserved_output_tokens: Reserved completion budget not available to prompt context.
"""
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.prompt_assembler = prompt_assembler
        self.question_standardizer = question_standardizer
        self.relevance_evaluator = relevance_evaluator
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count with a mixed-language heuristic.

Args:
    text: Input text content.

Returns:
    Rough token estimate used for context budget control.
        """
        if not text:
            return 0

        cjk_and_kana = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]", text))
        hangul = len(re.findall(r"[\uac00-\ud7af]", text))

        ascii_chars = 0
        other_chars = 0
        for ch in text:
            if ch.isspace():
                continue
            if ord(ch) < 128:
                ascii_chars += 1
            else:
                other_chars += 1

        # Remove CJK/Hangul/Kana already counted separately.
        other_non_cjk = max(0, other_chars - cjk_and_kana - hangul)
        ascii_tokens = (ascii_chars + 3) // 4
        return max(1, cjk_and_kana + hangul + ascii_tokens + other_non_cjk)

    def _truncate_text_to_token_budget(self, text: str, budget_tokens: int) -> str:
        """Truncate one text segment to fit within token budget.

Args:
    text: Input text content.
    budget_tokens: Max allowed token estimate for returned text.

Returns:
    Truncated text that does not exceed ``budget_tokens`` under estimator.
        """
        if budget_tokens <= 0:
            return ""
        if self._estimate_tokens(text) <= budget_tokens:
            return text

        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            if self._estimate_tokens(candidate) <= budget_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best.rstrip()

    def _join_texts_with_budget(
        self,
        texts: List[str],
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Join ordered texts while respecting context token budget.

        Args:
            texts: Candidate context chunks in order.
            max_context_tokens: Optional override of bundle budget.

        Returns:
            Tuple ``(context_text, used_tokens, truncated)``.
        """
        budget = max_context_tokens or self.max_context_tokens
        kept: List[str] = []
        used_tokens = 0
        truncated = False

        for i, text in enumerate(texts):
            text_tokens = self._estimate_tokens(text)
            if used_tokens + text_tokens > budget:
                # Keep first evidence chunk in truncated form if nothing has been added yet.
                if i == 0 and not kept and budget > 0:
                    clipped = self._truncate_text_to_token_budget(text, budget)
                    if clipped:
                        kept.append(clipped)
                        used_tokens = self._estimate_tokens(clipped)
                truncated = True
                break
            kept.append(text)
            used_tokens += text_tokens

        return "\n".join(kept), used_tokens, truncated

    def _estimate_non_context_prompt_tokens(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
    ) -> int:
        """Estimate token usage of prompt parts excluding retrieved context.

Args:
    question: Standardized question payload used in current QA turn.
    answer_mode: Strictness mode controlling answer rules.
    prompt_mode: Context source mode for prompt wording.

Returns:
    Estimated tokens consumed by prompt template/profile/question sections.
        """
        base_prompt = self.prompt_assembler.build_answer_prompt(
            context="",
            question=question,
            profile=self.profile,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )
        return self._estimate_tokens(base_prompt)

    def _compute_available_context_budget(
        self,
        question: "StandardizedQuestion",
        answer_mode: AnswerMode,
        prompt_mode: PromptMode | str,
        max_context_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
    ) -> int:
        """Compute effective context budget under total prompt token constraint.

Args:
    question: Standardized question payload used in current QA turn.
    answer_mode: Strictness mode controlling answer rules.
    prompt_mode: Context source mode for prompt wording.

    Returns:
    Context-token budget after subtracting non-context prompt and output reserve.
    Optional overrides allow mode-specific budget policy (e.g. global full-text).
        """
        non_context_tokens = self._estimate_non_context_prompt_tokens(
            question=question,
            answer_mode=answer_mode,
            prompt_mode=prompt_mode,
        )
        effective_context_limit = (
            self.max_context_tokens if max_context_tokens is None else max_context_tokens
        )
        effective_prompt_limit = (
            self.max_prompt_tokens if max_prompt_tokens is None else max_prompt_tokens
        )
        effective_output_reserve = (
            self.reserved_output_tokens
            if reserved_output_tokens is None
            else reserved_output_tokens
        )
        available_by_total = (
            effective_prompt_limit
            - non_context_tokens
            - effective_output_reserve
        )
        return max(0, min(effective_context_limit, available_by_total))

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
        ordered_records = sorted(
            self.id_to_record.values(),
            key=lambda record: (
                record.chunk_index() if isinstance(record.chunk_index(), int) else 10**9,
                record.node_id(),
            ),
        )
        texts = [record.text() for record in ordered_records]
        return self._join_texts_with_budget(
            texts=texts,
            max_context_tokens=max_context_tokens,
        )

    def estimate_full_text_tokens(self) -> int:
        """Estimate token usage for full document text in chunk order."""
        ordered_records = sorted(
            self.id_to_record.values(),
            key=lambda record: (
                record.chunk_index() if isinstance(record.chunk_index(), int) else 10**9,
                record.node_id(),
            ),
        )
        return sum(self._estimate_tokens(record.text()) for record in ordered_records)

    def _resolve_neighbor_by_link(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Internal helper for resolve neighbor by link.

Args:
    record: Record.
    is_prev: Direction flag for neighbor traversal (previous when True).

Returns:
    Linked neighbor record by node-id relation, or ``None`` if unavailable."""
        linked_node_key = record.prev_node_id() if is_prev else record.next_node_id()
        if linked_node_key is None:
            return None
        linked_faiss_id = self.node_key_to_faiss_id.get(str(linked_node_key))
        if linked_faiss_id is None:
            return None
        return self._record_from_faiss_id(linked_faiss_id)

    def _resolve_neighbor_by_chunk_index(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Internal helper for resolve neighbor by chunk index.

Args:
    record: Record.
    is_prev: Direction flag for neighbor traversal (previous when True).

Returns:
    Neighbor record by adjacent chunk index, or ``None`` if unavailable."""
        center_chunk_index = record.chunk_index()
        if not isinstance(center_chunk_index, int):
            return None
        target_chunk_index = center_chunk_index - 1 if is_prev else center_chunk_index + 1
        target_faiss_id = self.chunk_index_to_faiss_id.get(target_chunk_index)
        if target_faiss_id is None:
            return None
        return self._record_from_faiss_id(target_faiss_id)

    def _resolve_neighbor(
        self,
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Internal helper for resolve neighbor.

Args:
    record: Record.
    is_prev: Direction flag for neighbor traversal (previous when True).

Returns:
    Best-effort neighbor record resolved by link first, then chunk index."""
        neighbor = self._resolve_neighbor_by_link(record, is_prev=is_prev)
        if neighbor is not None:
            return neighbor
        return self._resolve_neighbor_by_chunk_index(record, is_prev=is_prev)

    def _expand_side(
        self,
        center: NodeRecord,
        radius: int,
        is_prev: bool,
    ) -> List[NodeRecord]:
        """Internal helper for expand side.

Args:
    center: Center hit (SearchMetadata or FAISS id) for local window build.
    radius: Neighbor expansion radius on each side of center chunk.
    is_prev: Direction flag for neighbor traversal (previous when True).

Returns:
    Ordered neighbor records collected on one side of the center node."""
        side_nodes: List[NodeRecord] = []
        cursor: NodeRecord = center
        visited_node_ids: Set[int] = {center.node_id()}

        for _ in range(radius):
            neighbor = self._resolve_neighbor(cursor, is_prev=is_prev)
            if neighbor is None:
                break
            if neighbor.node_id() in visited_node_ids:
                break
            visited_node_ids.add(neighbor.node_id())
            side_nodes.append(neighbor)
            cursor = neighbor

        return side_nodes

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
        if radius < 0:
            raise ValueError("radius must be >= 0")

        center_faiss_id = center.faiss_id if isinstance(center, SearchMetadata) else int(center)
        center_record = self._record_from_faiss_id(center_faiss_id)
        if center_record is None:
            return []

        left_nodes = self._expand_side(center_record, radius=radius, is_prev=True)
        right_nodes = self._expand_side(center_record, radius=radius, is_prev=False)

        ordered_nodes: List[NodeRecord] = list(reversed(left_nodes)) + [center_record] + right_nodes
        return [node.text() for node in ordered_nodes]

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
        budget = max_context_tokens or self.max_context_tokens
        center_faiss_id = center.faiss_id if isinstance(center, SearchMetadata) else int(center)
        center_record = self._record_from_faiss_id(center_faiss_id)
        if center_record is None:
            return [], 0, False, 0

        center_text = center_record.text()
        center_tokens = self._estimate_tokens(center_text)

        if budget <= 0:
            return [], 0, True, 0

        if center_tokens > budget:
            clipped_center = self._truncate_text_to_token_budget(center_text, budget)
            used_tokens = self._estimate_tokens(clipped_center) if clipped_center else 0
            return ([clipped_center] if clipped_center else []), used_tokens, True, 0

        left_nodes: List[NodeRecord] = []
        right_nodes: List[NodeRecord] = []
        used_tokens = center_tokens
        truncated = False
        used_radius = 0

        # Always keep center chunk for semantic anchor.
        left_cursor = center_record
        right_cursor = center_record
        visited_node_ids: Set[int] = {center_record.node_id()}

        while True:
            progressed = False
            used_this_step = False

            left_neighbor = self._resolve_neighbor(left_cursor, is_prev=True)
            if left_neighbor is not None and left_neighbor.node_id() not in visited_node_ids:
                left_tokens = self._estimate_tokens(left_neighbor.text())
                if used_tokens + left_tokens <= budget:
                    left_nodes.append(left_neighbor)
                    left_cursor = left_neighbor
                    visited_node_ids.add(left_neighbor.node_id())
                    used_tokens += left_tokens
                    progressed = True
                    used_this_step = True
                else:
                    truncated = True

            right_neighbor = self._resolve_neighbor(right_cursor, is_prev=False)
            if right_neighbor is not None and right_neighbor.node_id() not in visited_node_ids:
                right_tokens = self._estimate_tokens(right_neighbor.text())
                if used_tokens + right_tokens <= budget:
                    right_nodes.append(right_neighbor)
                    right_cursor = right_neighbor
                    visited_node_ids.add(right_neighbor.node_id())
                    used_tokens += right_tokens
                    progressed = True
                    used_this_step = True
                else:
                    truncated = True

            if used_this_step:
                used_radius += 1
            if not progressed:
                break

        ordered_nodes: List[NodeRecord] = list(reversed(left_nodes)) + [center_record] + right_nodes
        return [node.text() for node in ordered_nodes], used_tokens, truncated, used_radius

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
        results: List[SearchMetadata] = self.search(query, top_k)
        merged_texts: List[str] = []
        seen_texts: Set[str] = set()

        for result in results:
            window_texts = self.build_local_window(result, radius=radius)
            for text in window_texts:
                normalized_text = self._normalize_text(text)
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                merged_texts.append(text)

        context, used_tokens, truncated = self._join_texts_with_budget(merged_texts)
        print(
            "FaissIndexBundle#build_context_with_window:",
            f"token_used={used_tokens}",
            f"budget={self.max_context_tokens}",
            f"truncated={truncated}",
        )
        return context
