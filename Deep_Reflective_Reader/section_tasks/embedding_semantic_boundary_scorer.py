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
        embedding_batch_size: int = 24,
    ):
        self.embedder = embedder
        self.embedder_factory = embedder_factory
        self.similarity_service = similarity_service
        self.boundary_window_chars = max(40, int(boundary_window_chars))
        self.context_window_chars = max(self.boundary_window_chars, int(context_window_chars))
        self.score_weight = float(score_weight)
        self.embedding_cache_size = max(32, int(embedding_cache_size))
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self._embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def score_boundary(
        self,
        *,
        text: str,
        boundary_index: int,
    ) -> float:
        """Score one boundary by semantic shift vs continuity signals."""
        scored = self.score_boundaries(
            text=text,
            boundary_indices=[boundary_index],
        )
        return float(scored.get(boundary_index, 0.0))

    def score_boundaries(
        self,
        *,
        text: str,
        boundary_indices: list[int],
    ) -> dict[int, float]:
        """Score multiple boundaries with one embedding-batch fill where possible."""
        if not text:
            return {}

        valid_indices = [
            int(index)
            for index in boundary_indices
            if isinstance(index, int) and 0 < index < len(text)
        ]
        if not valid_indices:
            return {}

        boundary_to_snippets: dict[int, tuple[str, str, str, str]] = {}
        for index in valid_indices:
            left_tail = self._slice_left(
                text=text,
                boundary_index=index,
                window=self.boundary_window_chars,
            )
            right_head = self._slice_right(
                text=text,
                boundary_index=index,
                window=self.boundary_window_chars,
            )
            left_context = self._slice_left(
                text=text,
                boundary_index=index,
                window=self.context_window_chars,
            )
            right_context = self._slice_right(
                text=text,
                boundary_index=index,
                window=self.context_window_chars,
            )
            if min(
                len(left_tail),
                len(right_head),
                len(left_context),
                len(right_context),
            ) < 12:
                continue
            boundary_to_snippets[index] = (
                left_tail,
                right_head,
                left_context,
                right_context,
            )
        if not boundary_to_snippets:
            return {}

        all_snippets: list[str] = []
        for snippets in boundary_to_snippets.values():
            all_snippets.extend(snippets)
        self._warmup_embeddings(all_snippets)

        scored: dict[int, float] = {}
        for index, snippets in boundary_to_snippets.items():
            try:
                continuity_similarity = self._cosine_similarity(snippets[0], snippets[1])
                context_similarity = self._cosine_similarity(snippets[2], snippets[3])
            except Exception:
                continue

            # Prefer boundaries where local continuity is lower and wider-context shift is higher.
            context_shift = 1.0 - context_similarity
            semantic_score = self.score_weight * (context_shift - continuity_similarity)
            scored[index] = float(semantic_score)
        return scored

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
        normalized_text = self._normalize_snippet(text)
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

    def _warmup_embeddings(self, snippets: list[str]) -> None:
        """Batch-fill snippet embedding cache before cosine scoring."""
        normalized_snippets = [
            self._normalize_snippet(snippet)
            for snippet in snippets
        ]
        normalized_snippets = [
            snippet
            for snippet in normalized_snippets
            if snippet
        ]
        if not normalized_snippets:
            return

        missing_texts = [
            snippet
            for snippet in dict.fromkeys(normalized_snippets)
            if snippet not in self._embedding_cache
        ]
        if not missing_texts:
            return

        embedder = self._get_embedder()
        if embedder is None:
            return

        supports_batch = hasattr(embedder, "get_text_embedding_batch")
        if not supports_batch or len(missing_texts) == 1:
            for snippet in missing_texts:
                self._embedding_for_text(snippet)
            return

        for start in range(0, len(missing_texts), self.embedding_batch_size):
            chunk = missing_texts[start:start + self.embedding_batch_size]
            try:
                vectors = embedder.get_text_embedding_batch(chunk)
            except Exception:
                for snippet in chunk:
                    self._embedding_for_text(snippet)
                continue

            for snippet, vector in zip(chunk, vectors):
                normalized = self._normalize_vector(vector)
                self._embedding_cache[snippet] = normalized
                self._embedding_cache.move_to_end(snippet)
            while len(self._embedding_cache) > self.embedding_cache_size:
                self._embedding_cache.popitem(last=False)

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
    def _normalize_snippet(text: str) -> str:
        """Canonicalize snippet key to improve cache hit rate."""
        return " ".join((text or "").strip().split())

    @staticmethod
    def _slice_left(*, text: str, boundary_index: int, window: int) -> str:
        start = max(0, boundary_index - window)
        return text[start:boundary_index].strip()

    @staticmethod
    def _slice_right(*, text: str, boundary_index: int, window: int) -> str:
        end = min(len(text), boundary_index + window)
        return text[boundary_index:end].strip()
