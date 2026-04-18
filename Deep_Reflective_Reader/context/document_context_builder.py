from __future__ import annotations

from typing import TYPE_CHECKING, List, Set

from context.token_budget_manager import TokenBudgetManager
from retrieval.node_record import NodeRecord
from retrieval.search_metadata import SearchMetadata

if TYPE_CHECKING:
    from retrieval.faiss_index_bundle import FaissIndexBundle


class DocumentContextBuilder:
    """Build document contexts (local/full) from bundle runtime data."""

    token_budget_manager: TokenBudgetManager

    def __init__(
        self,
        token_budget_manager: TokenBudgetManager,
    ):
        """Initialize builder with bundle runtime and token budget service."""
        self.token_budget_manager = token_budget_manager

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace for deduplication/comparison."""
        return " ".join(text.strip().split())

    def _record_from_faiss_id(
        self,
        bundle: "FaissIndexBundle",
        faiss_id: int,
    ) -> NodeRecord | None:
        """Resolve record by FAISS id via bundle lookup."""
        return bundle.get_node_by_id(faiss_id)

    def _resolve_neighbor_by_link(
        self,
        bundle: "FaissIndexBundle",
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Resolve neighbor by linked node key relation."""
        linked_node_key = record.prev_node_id() if is_prev else record.next_node_id()
        if linked_node_key is None:
            return None
        linked_faiss_id = bundle.node_key_to_faiss_id.get(str(linked_node_key))
        if linked_faiss_id is None:
            return None
        return self._record_from_faiss_id(bundle, linked_faiss_id)

    def _resolve_neighbor_by_chunk_index(
        self,
        bundle: "FaissIndexBundle",
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Resolve neighbor by adjacent chunk index."""
        center_chunk_index = record.chunk_index()
        if not isinstance(center_chunk_index, int):
            return None
        target_chunk_index = center_chunk_index - 1 if is_prev else center_chunk_index + 1
        target_faiss_id = bundle.chunk_index_to_faiss_id.get(target_chunk_index)
        if target_faiss_id is None:
            return None
        return self._record_from_faiss_id(bundle, target_faiss_id)

    def _resolve_neighbor(
        self,
        bundle: "FaissIndexBundle",
        record: NodeRecord,
        is_prev: bool,
    ) -> NodeRecord | None:
        """Resolve neighbor by link first, then chunk-index fallback."""
        neighbor = self._resolve_neighbor_by_link(
            bundle,
            record,
            is_prev=is_prev,
        )
        if neighbor is not None:
            return neighbor
        return self._resolve_neighbor_by_chunk_index(
            bundle,
            record,
            is_prev=is_prev,
        )

    def _expand_side(
        self,
        bundle: "FaissIndexBundle",
        center: NodeRecord,
        radius: int,
        is_prev: bool,
    ) -> List[NodeRecord]:
        """Expand one side of the center node by radius."""
        side_nodes: List[NodeRecord] = []
        cursor: NodeRecord = center
        visited_node_ids: Set[int] = {center.node_id()}

        for _ in range(radius):
            neighbor = self._resolve_neighbor(bundle, cursor, is_prev=is_prev)
            if neighbor is None:
                break
            if neighbor.node_id() in visited_node_ids:
                break
            visited_node_ids.add(neighbor.node_id())
            side_nodes.append(neighbor)
            cursor = neighbor

        return side_nodes

    def build_full_text_context(
        self,
        bundle: "FaissIndexBundle",
        max_context_tokens: int | None = None,
    ) -> tuple[str, int, bool]:
        """Build full-document context (chunk-ordered) within token budget."""
        ordered_records = sorted(
            bundle.id_to_record.values(),
            key=lambda record: (
                record.chunk_index() if isinstance(record.chunk_index(), int) else 10**9,
                record.node_id(),
            ),
        )
        texts = [record.text() for record in ordered_records]
        return self.token_budget_manager.join_texts_with_budget(
            texts=texts,
            default_max_context_tokens=bundle.max_context_tokens,
            max_context_tokens=max_context_tokens,
        )

    def estimate_full_text_tokens(self, bundle: "FaissIndexBundle") -> int:
        """Estimate token usage for full document text in chunk order."""
        ordered_records = sorted(
            bundle.id_to_record.values(),
            key=lambda record: (
                record.chunk_index() if isinstance(record.chunk_index(), int) else 10**9,
                record.node_id(),
            ),
        )
        return sum(
            self.token_budget_manager.estimate_tokens(record.text())
            for record in ordered_records
        )

    def build_local_window(
        self,
        bundle: "FaissIndexBundle",
        center: SearchMetadata | int,
        radius: int = 1,
    ) -> List[str]:
        """Build local context window around a center hit."""
        if radius < 0:
            raise ValueError("radius must be >= 0")

        center_faiss_id = center.faiss_id if isinstance(center, SearchMetadata) else int(center)
        center_record = self._record_from_faiss_id(bundle, center_faiss_id)
        if center_record is None:
            return []

        left_nodes = self._expand_side(bundle, center_record, radius=radius, is_prev=True)
        right_nodes = self._expand_side(bundle, center_record, radius=radius, is_prev=False)

        ordered_nodes: List[NodeRecord] = list(reversed(left_nodes)) + [center_record] + right_nodes
        return [node.text() for node in ordered_nodes]

    def build_local_window_dynamic(
        self,
        bundle: "FaissIndexBundle",
        center: SearchMetadata | int,
        max_context_tokens: int | None = None,
    ) -> tuple[List[str], int, bool, int]:
        """Build local window by dynamic expansion under token budget."""
        budget = max_context_tokens or bundle.max_context_tokens
        center_faiss_id = center.faiss_id if isinstance(center, SearchMetadata) else int(center)
        center_record = self._record_from_faiss_id(bundle, center_faiss_id)
        if center_record is None:
            return [], 0, False, 0

        center_text = center_record.text()
        center_tokens = self.token_budget_manager.estimate_tokens(center_text)

        if budget <= 0:
            return [], 0, True, 0

        if center_tokens > budget:
            clipped_center = self.token_budget_manager.truncate_text_to_token_budget(
                center_text,
                budget,
            )
            used_tokens = (
                self.token_budget_manager.estimate_tokens(clipped_center)
                if clipped_center
                else 0
            )
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

            left_neighbor = self._resolve_neighbor(bundle, left_cursor, is_prev=True)
            if left_neighbor is not None and left_neighbor.node_id() not in visited_node_ids:
                left_tokens = self.token_budget_manager.estimate_tokens(left_neighbor.text())
                if used_tokens + left_tokens <= budget:
                    left_nodes.append(left_neighbor)
                    left_cursor = left_neighbor
                    visited_node_ids.add(left_neighbor.node_id())
                    used_tokens += left_tokens
                    progressed = True
                    used_this_step = True
                else:
                    truncated = True

            right_neighbor = self._resolve_neighbor(bundle, right_cursor, is_prev=False)
            if right_neighbor is not None and right_neighbor.node_id() not in visited_node_ids:
                right_tokens = self.token_budget_manager.estimate_tokens(right_neighbor.text())
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
        bundle: "FaissIndexBundle",
        query: str,
        top_k: int = 3,
        radius: int = 1,
    ) -> str:
        """Build merged context by expanding hits into local windows."""
        results: List[SearchMetadata] = bundle.search(query, top_k)
        merged_texts: List[str] = []
        seen_texts: Set[str] = set()

        for result in results:
            window_texts = self.build_local_window(
                bundle=bundle,
                center=result,
                radius=radius,
            )
            for text in window_texts:
                normalized_text = self._normalize_text(text)
                if normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                merged_texts.append(text)

        context, used_tokens, truncated = self.token_budget_manager.join_texts_with_budget(
            texts=merged_texts,
            default_max_context_tokens=bundle.max_context_tokens,
        )
        print(
            "FaissIndexBundle#build_context_with_window:",
            f"token_used={used_tokens}",
            f"budget={bundle.max_context_tokens}",
            f"truncated={truncated}",
        )
        return context
