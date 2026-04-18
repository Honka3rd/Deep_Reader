from dataclasses import dataclass

import faiss
import numpy as np

from embeddings.embedder import Embedder
from language.language_code import LanguageCode, LanguageCodeResolver
from question.qa_enums import QuestionScope
from question.question_scope_keywords_provider import QuestionScopeKeywordsProvider
from question.standardized.standardized_question import StandardizedQuestion


@dataclass(frozen=True)
class QuestionScopeResolution:
    """Detailed scope resolution payload with diagnostics."""
    scope: QuestionScope
    query_language: LanguageCode
    matched_keyword: str | None
    similarity: float | None
    method: str


class QuestionScopeResolver:
    """Resolve question scope with keyword heuristics + embedding similarity."""
    keywords_provider: QuestionScopeKeywordsProvider
    embedder: Embedder
    global_similarity_threshold: float
    keyword_embedding_cache: dict[tuple[str, str], np.ndarray]

    def __init__(
        self,
        keywords_provider: QuestionScopeKeywordsProvider,
        embedder: Embedder,
        global_similarity_threshold: float = 0.78,
    ):
        """Initialize resolver with injected dependencies."""
        self.keywords_provider = keywords_provider
        self.embedder = embedder
        self.global_similarity_threshold = global_similarity_threshold
        self.keyword_embedding_cache = {}

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine/IP similarity."""
        norm = np.linalg.norm(vector)
        if norm <= 0:
            return vector
        return vector / norm

    @staticmethod
    def _contains_keyword(query: str, keywords: tuple[str, ...]) -> str | None:
        """Return first lexical keyword hit if any."""
        compact = " ".join(query.lower().split())
        for keyword in keywords:
            keyword_compact = " ".join(keyword.lower().split())
            if keyword_compact and keyword_compact in compact:
                return keyword
        return None

    @staticmethod
    def _resolve_query_language(question: StandardizedQuestion) -> LanguageCode:
        """Resolve query language using structured field with safe fallbacks."""
        if question.user_language != LanguageCode.UNKNOWN:
            return question.user_language

        inferred = LanguageCodeResolver.infer_from_text(question.original_query)
        if inferred != LanguageCode.UNKNOWN:
            return inferred

        if question.document_language != LanguageCode.UNKNOWN:
            print(
                "Warn:QuestionScopeResolver#resolve: unknown user language, "
                f"fallback_to_document_language={question.document_language.value}"
            )
            return question.document_language

        print(
            "Warn:QuestionScopeResolver#resolve: unknown user/document language, "
            "fallback_to_all_keywords"
        )
        return LanguageCode.UNKNOWN

    def _keyword_vectors(
        self,
        language: LanguageCode,
        keywords: tuple[str, ...],
        dimension: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Build normalized keyword vectors with cache."""
        vectors: list[np.ndarray] = []
        used_keywords: list[str] = []
        for keyword in keywords:
            cache_key = (language.value, keyword)
            cached = self.keyword_embedding_cache.get(cache_key)
            if cached is None:
                embedded = self.embedder.get_text_embedding(keyword)
                cached = self._normalize_vector(
                    np.asarray(embedded, dtype=np.float32)
                )
                self.keyword_embedding_cache[cache_key] = cached
            vectors.append(cached)
            used_keywords.append(keyword)

        if not vectors:
            return np.zeros((0, dimension), dtype=np.float32), []
        return np.vstack(vectors), used_keywords

    def _semantic_match(
        self,
        query: str,
        language: LanguageCode,
        keywords: tuple[str, ...],
    ) -> tuple[str | None, float | None]:
        """Return best semantic keyword match and cosine similarity."""
        if not keywords:
            return None, None

        query_vector_raw = self.embedder.get_text_embedding(query)
        query_vector = self._normalize_vector(
            np.asarray(query_vector_raw, dtype=np.float32)
        ).reshape(1, -1)
        dimension = int(query_vector.shape[1])

        keyword_matrix, used_keywords = self._keyword_vectors(
            language=language,
            keywords=keywords,
            dimension=dimension,
        )
        if keyword_matrix.shape[0] == 0:
            return None, None

        index = faiss.IndexFlatIP(dimension)
        index.add(keyword_matrix)
        similarities, indices = index.search(query_vector, 1)
        if indices.shape[1] == 0 or indices[0][0] < 0:
            return None, None

        best_idx = int(indices[0][0])
        best_similarity = float(similarities[0][0])
        return used_keywords[best_idx], best_similarity

    def resolve(self, question: StandardizedQuestion) -> QuestionScopeResolution:
        """Resolve question scope for one standardized question."""
        query = question.original_query.strip()
        if not query:
            return QuestionScopeResolution(
                scope=QuestionScope.LOCAL,
                query_language=LanguageCode.UNKNOWN,
                matched_keyword=None,
                similarity=None,
                method="empty_query",
            )

        language = self._resolve_query_language(question)
        keywords = (
            self.keywords_provider.get_all_keywords()
            if language == LanguageCode.UNKNOWN
            else self.keywords_provider.get_keywords(language)
        )

        lexical_hit = self._contains_keyword(query, keywords)
        if lexical_hit is not None:
            return QuestionScopeResolution(
                scope=QuestionScope.GLOBAL,
                query_language=language,
                matched_keyword=lexical_hit,
                similarity=1.0,
                method="lexical_keyword",
            )

        matched_keyword, similarity = self._semantic_match(
            query=query,
            language=language,
            keywords=keywords,
        )
        if similarity is not None and similarity >= self.global_similarity_threshold:
            return QuestionScopeResolution(
                scope=QuestionScope.GLOBAL,
                query_language=language,
                matched_keyword=matched_keyword,
                similarity=similarity,
                method="semantic_keyword",
            )

        return QuestionScopeResolution(
            scope=QuestionScope.LOCAL,
            query_language=language,
            matched_keyword=matched_keyword,
            similarity=similarity,
            method="fallback_local",
        )
