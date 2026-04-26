from collections import OrderedDict
from typing import Callable

import numpy as np

from embeddings.embedder import Embedder
from section_tasks.heuristic_task_unit_split_resolver import SemanticBoundaryScorer


class EmbeddingSemanticBoundaryScorer(SemanticBoundaryScorer):
    """Embedding-backed semantic boundary scorer for heuristic split candidate ranking."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        embedder_factory: Callable[[], Embedder] | None = None,
        similarity_service: object | None = None,
        boundary_window_chars: int = 220,
        context_window_chars: int = 700,
        score_weight: float = 1.0,
        embedding_cache_size: int = 256,
    ):
        self.embedder = embedder
        self.embedder_factory = embedder_factory
        self.similarity_service = similarity_service
        self.boundary_window_chars = max(40, int(boundary_window_chars))
        self.context_window_chars = max(self.boundary_window_chars, int(context_window_chars))
        self.score_weight = float(score_weight)
        self.embedding_cache_size = max(32, int(embedding_cache_size))
        self._embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def score_boundary(
        self,
        *,
        text: str,
        boundary_index: int,
    ) -> float:
        """Score one boundary by semantic shift vs continuity signals."""
        if not text or boundary_index <= 0 or boundary_index >= len(text):
            return 0.0

        left_tail = self._slice_left(text=text, boundary_index=boundary_index, window=self.boundary_window_chars)
        right_head = self._slice_right(text=text, boundary_index=boundary_index, window=self.boundary_window_chars)
        left_context = self._slice_left(text=text, boundary_index=boundary_index, window=self.context_window_chars)
        right_context = self._slice_right(text=text, boundary_index=boundary_index, window=self.context_window_chars)
        if min(len(left_tail), len(right_head), len(left_context), len(right_context)) < 12:
            return 0.0

        try:
            continuity_similarity = self._cosine_similarity(left_tail, right_head)
            context_similarity = self._cosine_similarity(left_context, right_context)
        except Exception:
            return 0.0

        # Prefer boundaries where local continuity is lower and wider-context shift is higher.
        context_shift = 1.0 - context_similarity
        semantic_score = self.score_weight * (context_shift - continuity_similarity)
        return float(semantic_score)

    def _get_embedder(self) -> Embedder | None:
        """Resolve embedder lazily so scorer never hard-fails split path."""
        if self.embedder is not None:
            return self.embedder
        if self.embedder_factory is None:
            return None
        try:
            self.embedder = self.embedder_factory()
            return self.embedder
        except Exception:
            return None

    def _embedding_for_text(self, text: str) -> np.ndarray | None:
        """Get normalized embedding for snippet with bounded in-memory cache."""
        normalized_text = text.strip()
        if not normalized_text:
            return None
        cached = self._embedding_cache.get(normalized_text)
        if cached is not None:
            self._embedding_cache.move_to_end(normalized_text)
            return cached

        embedder = self._get_embedder()
        if embedder is None:
            return None

        vector = embedder.get_text_embedding(normalized_text)
        normalized = self._normalize_vector(vector)
        self._embedding_cache[normalized_text] = normalized
        self._embedding_cache.move_to_end(normalized_text)
        while len(self._embedding_cache) > self.embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        return normalized

    def _cosine_similarity(self, left: str, right: str) -> float:
        """Compute cosine similarity from snippet embeddings."""
        left_vector = self._embedding_for_text(left)
        right_vector = self._embedding_for_text(right)
        if left_vector is None or right_vector is None:
            return 0.0
        if left_vector.shape[0] != right_vector.shape[0]:
            return 0.0
        similarity = float(np.dot(left_vector, right_vector))
        if similarity < -1.0:
            return -1.0
        if similarity > 1.0:
            return 1.0
        return similarity

    def _normalize_vector(self, vector: list[float]) -> np.ndarray:
        """Normalize embedding vector with optional service hook."""
        if (
            self.similarity_service is not None
            and hasattr(self.similarity_service, "normalize_embedding")
        ):
            try:
                normalized = self.similarity_service.normalize_embedding(vector)
                return np.asarray(normalized, dtype=np.float32).reshape(-1)
            except Exception:
                pass

        fallback = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(fallback))
        if norm <= 0:
            return fallback
        return fallback / norm

    @staticmethod
    def _slice_left(*, text: str, boundary_index: int, window: int) -> str:
        start = max(0, boundary_index - window)
        return text[start:boundary_index].strip()

    @staticmethod
    def _slice_right(*, text: str, boundary_index: int, window: int) -> str:
        end = min(len(text), boundary_index + window)
        return text[boundary_index:end].strip()
