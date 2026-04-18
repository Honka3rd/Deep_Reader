from dataclasses import dataclass

from retrieval.faiss_index_bundle import FaissIndexBundle
from retrieval.search_metadata import SearchMetadata


@dataclass(frozen=True)
class CoverageSelection:
    """Coverage-oriented selection output for global retrieval context."""
    selected_results: list[SearchMetadata]
    raw_chunk_indices: list[int | None]
    selected_chunk_indices: list[int | None]


class CoverageOrientedContextBuilder:
    """Prefer broader chunk coverage instead of dense same-area evidence."""
    nearby_chunk_distance: int

    def __init__(self, nearby_chunk_distance: int = 2):
        """Initialize builder with chunk-neighborhood dedup distance."""
        self.nearby_chunk_distance = max(0, nearby_chunk_distance)

    @staticmethod
    def _extract_chunk_index(bundle: FaissIndexBundle, result: SearchMetadata) -> int | None:
        """Resolve chunk index for one retrieval hit."""
        record = bundle.get_node_by_id(result.faiss_id)
        if record is None:
            return None
        chunk_index = record.chunk_index()
        if isinstance(chunk_index, int):
            return chunk_index
        return None

    def select_for_global_scope(
        self,
        bundle: FaissIndexBundle,
        results: list[SearchMetadata],
    ) -> CoverageSelection:
        """Filter retrieval results to improve global coverage across chunk regions."""
        if not results:
            return CoverageSelection(
                selected_results=[],
                raw_chunk_indices=[],
                selected_chunk_indices=[],
            )

        # Keep deterministic ordering by retrieval score before coverage filtering.
        sorted_results = sorted(results, key=lambda result: result.score)

        selected_results: list[SearchMetadata] = []
        raw_chunk_indices: list[int | None] = []
        selected_chunk_indices: list[int | None] = []

        for result in sorted_results:
            chunk_index = self._extract_chunk_index(bundle, result)
            raw_chunk_indices.append(chunk_index)

            if chunk_index is None:
                selected_results.append(result)
                selected_chunk_indices.append(chunk_index)
                continue

            is_near_existing = any(
                existing is not None
                and abs(chunk_index - existing) <= self.nearby_chunk_distance
                for existing in selected_chunk_indices
            )
            if is_near_existing:
                continue

            selected_results.append(result)
            selected_chunk_indices.append(chunk_index)

        if not selected_results:
            selected_results = [sorted_results[0]]
            selected_chunk_indices = [raw_chunk_indices[0]]

        return CoverageSelection(
            selected_results=selected_results,
            raw_chunk_indices=raw_chunk_indices,
            selected_chunk_indices=selected_chunk_indices,
        )
