from typing import Sequence

import faiss
import numpy as np


class EmbeddingSimilarityService:
    """Utility service for embedding normalization and similarity search."""

    @staticmethod
    def normalize_embedding(
        vector: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Normalize one embedding vector to unit length."""
        normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized))
        if norm <= 0:
            return normalized
        return normalized / norm

    @classmethod
    def normalize_embedding_matrix(
        cls,
        vectors: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Normalize embedding matrix row-wise to unit length."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.size == 0:
            return matrix
        normalized_rows = [cls.normalize_embedding(row) for row in matrix]
        return np.vstack(normalized_rows)

    @classmethod
    def best_similarity_index(
        cls,
        query_vector: Sequence[float] | np.ndarray,
        candidate_vectors: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[int, float] | None:
        """Return best candidate index and similarity (cosine via IP)."""
        query = cls.normalize_embedding(query_vector)
        candidates = cls.normalize_embedding_matrix(candidate_vectors)
        if candidates.size == 0:
            return None

        dimension = int(query.shape[0])
        if candidates.shape[1] != dimension:
            raise ValueError(
                "candidate_vectors dimension does not match query_vector dimension"
            )

        index = faiss.IndexFlatIP(dimension)
        index.add(candidates)
        similarities, indices = index.search(query.reshape(1, -1), 1)
        if indices.shape[1] == 0 or int(indices[0][0]) < 0:
            return None
        return int(indices[0][0]), float(similarities[0][0])
