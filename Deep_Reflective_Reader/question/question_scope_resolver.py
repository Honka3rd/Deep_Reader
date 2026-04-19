import json
import re
from dataclasses import dataclass

import numpy as np

from embeddings.embedder import Embedder
from embeddings.embedding_similarity_service import EmbeddingSimilarityService
from language.language_code import LanguageCode, LanguageCodeResolver
from language.language_profile_registry import LanguageProfileRegistry
from llm.llm_provider import LLMProvider
from question.qa_enums import LocalSignalQuality, QuestionScope
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


@dataclass(frozen=True)
class LocalReferenceSignalResolution:
    """Local-reference detection result with quality grading."""
    matched: bool
    quality: LocalSignalQuality | None
    method: str | None
    matched_text: str | None
    similarity: float | None


class QuestionScopeResolver:
    """Resolve question scope with keyword heuristics + embedding similarity."""
    keywords_provider: QuestionScopeKeywordsProvider
    embedder: Embedder
    similarity_service: EmbeddingSimilarityService
    llm_provider: LLMProvider | None
    global_similarity_threshold: float
    llm_gray_zone_min_similarity: float
    llm_gray_zone_max_similarity: float
    llm_fallback_enabled: bool
    llm_summary_char_limit: int
    local_anchor_similarity_threshold: float
    text_embedding_cache: dict[tuple[str, str], np.ndarray]

    def __init__(
        self,
        keywords_provider: QuestionScopeKeywordsProvider,
        embedder: Embedder,
        similarity_service: EmbeddingSimilarityService,
        llm_provider: LLMProvider | None = None,
        global_similarity_threshold: float = 0.78,
        llm_gray_zone_min_similarity: float = 0.45,
        llm_gray_zone_max_similarity: float = 0.78,
        llm_fallback_enabled: bool = True,
        llm_summary_char_limit: int = 800,
        local_anchor_similarity_threshold: float = 0.75,
    ):
        """Initialize resolver with injected dependencies."""
        self.keywords_provider = keywords_provider
        self.embedder = embedder
        self.similarity_service = similarity_service
        self.llm_provider = llm_provider
        self.global_similarity_threshold = global_similarity_threshold
        self.llm_gray_zone_min_similarity = llm_gray_zone_min_similarity
        self.llm_gray_zone_max_similarity = llm_gray_zone_max_similarity
        self.llm_fallback_enabled = llm_fallback_enabled
        self.llm_summary_char_limit = llm_summary_char_limit
        self.local_anchor_similarity_threshold = local_anchor_similarity_threshold
        self.text_embedding_cache = {}
        if not (0 < self.global_similarity_threshold <= 1):
            raise ValueError("global_similarity_threshold must be in (0, 1]")
        if not (0 <= self.llm_gray_zone_min_similarity <= 1):
            raise ValueError("llm_gray_zone_min_similarity must be in [0, 1]")
        if not (0 <= self.llm_gray_zone_max_similarity <= 1):
            raise ValueError("llm_gray_zone_max_similarity must be in [0, 1]")
        if self.llm_gray_zone_min_similarity > self.llm_gray_zone_max_similarity:
            raise ValueError(
                "llm_gray_zone_min_similarity must be <= llm_gray_zone_max_similarity"
            )
        if self.llm_summary_char_limit <= 0:
            raise ValueError("llm_summary_char_limit must be > 0")
        if not (0 <= self.local_anchor_similarity_threshold <= 1):
            raise ValueError("local_anchor_similarity_threshold must be in [0, 1]")

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

    def _text_vectors(
        self,
        language: LanguageCode,
        texts: tuple[str, ...],
        dimension: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Build normalized text vectors with cache."""
        vectors: list[np.ndarray] = []
        used_texts: list[str] = []
        for text in texts:
            cache_key = (language.value, text)
            cached = self.text_embedding_cache.get(cache_key)
            if cached is None:
                embedded = self.embedder.get_text_embedding(text)
                cached = self.similarity_service.normalize_embedding(embedded)
                self.text_embedding_cache[cache_key] = cached
            vectors.append(cached)
            used_texts.append(text)

        if not vectors:
            return np.zeros((0, dimension), dtype=np.float32), []
        return np.vstack(vectors), used_texts

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
        query_vector = self.similarity_service.normalize_embedding(query_vector_raw)
        dimension = int(query_vector.shape[0])

        keyword_matrix, used_keywords = self._text_vectors(
            language=language,
            texts=keywords,
            dimension=dimension,
        )
        if keyword_matrix.shape[0] == 0:
            return None, None

        best = self.similarity_service.best_similarity_index(
            query_vector=query_vector,
            candidate_vectors=keyword_matrix,
        )
        if best is None:
            return None, None

        best_idx, best_similarity = best
        return used_keywords[best_idx], best_similarity

    def _semantic_local_anchor_match(
        self,
        query: str,
        language: LanguageCode,
    ) -> tuple[str | None, float | None]:
        """Return best semantic local-anchor match and cosine similarity."""
        anchors = (
            LanguageProfileRegistry.get_all_weak_session_local_anchor_signals()
            if language == LanguageCode.UNKNOWN
            else LanguageProfileRegistry.get_weak_session_local_anchor_signals(language)
        )
        if not anchors:
            return None, None

        query_vector_raw = self.embedder.get_text_embedding(query)
        query_vector = self.similarity_service.normalize_embedding(query_vector_raw)
        dimension = int(query_vector.shape[0])

        anchor_matrix, used_anchors = self._text_vectors(
            language=language,
            texts=anchors,
            dimension=dimension,
        )
        if anchor_matrix.shape[0] == 0:
            return None, None

        best = self.similarity_service.best_similarity_index(
            query_vector=query_vector,
            candidate_vectors=anchor_matrix,
        )
        if best is None:
            return None, None

        best_idx, best_similarity = best
        return used_anchors[best_idx], best_similarity

    def _has_local_reference_signal(
        self,
        query: str,
        query_language: LanguageCode,
        session_active_chunk_index: int | None,
    ) -> LocalReferenceSignalResolution:
        """Detect local-reading hints with strong/weak signal quality grading."""
        lowered = query.lower()
        if query_language == LanguageCode.UNKNOWN:
            strong_local_markers = (
                LanguageProfileRegistry.get_all_strong_local_reference_signals()
            )
        else:
            strong_local_markers = (
                LanguageProfileRegistry.get_strong_local_reference_signals(
                    query_language
                )
            )
        for marker in strong_local_markers:
            if marker.lower() in lowered:
                return LocalReferenceSignalResolution(
                    matched=True,
                    quality=LocalSignalQuality.STRONG,
                    method="strong_lexical_local_marker",
                    matched_text=marker,
                    similarity=1.0,
                )
        if session_active_chunk_index is None:
            return LocalReferenceSignalResolution(
                matched=False,
                quality=None,
                method=None,
                matched_text=None,
                similarity=None,
            )
        session_anchor_markers = LanguageProfileRegistry.get_weak_session_local_anchor_signals(
            query_language
        )
        for marker in session_anchor_markers:
            if marker.lower() in lowered:
                return LocalReferenceSignalResolution(
                    matched=True,
                    quality=LocalSignalQuality.WEAK,
                    method="weak_session_lexical_anchor",
                    matched_text=marker,
                    similarity=1.0,
                )

        matched_anchor, similarity = self._semantic_local_anchor_match(
            query=query,
            language=query_language,
        )
        if (
            similarity is not None
            and similarity >= self.local_anchor_similarity_threshold
        ):
            return LocalReferenceSignalResolution(
                matched=True,
                quality=LocalSignalQuality.WEAK,
                method="weak_session_semantic_anchor",
                matched_text=matched_anchor,
                similarity=similarity,
            )
        return LocalReferenceSignalResolution(
            matched=False,
            quality=None,
            method=None,
            matched_text=matched_anchor,
            similarity=similarity,
        )

    def _llm_decide_scope(
        self,
        question: StandardizedQuestion,
        query_language: LanguageCode,
        matched_keyword: str | None,
        similarity: float | None,
        document_summary: str | None,
        session_active_chunk_index: int | None,
    ) -> tuple[QuestionScope | None, str | None]:
        """Use LLM as a fallback classifier and parse a strict JSON output."""
        if self.llm_provider is None or not self.llm_fallback_enabled:
            return None, None

        summary = (document_summary or "").strip()
        if len(summary) > self.llm_summary_char_limit:
            summary = summary[: self.llm_summary_char_limit].rstrip()

        prompt = (
            "You are a scope classifier for a reading assistant.\n"
            "Classify whether the user question is GLOBAL or LOCAL.\n"
            "GLOBAL: asks for whole-document summary/listing/themes/main points.\n"
            "LOCAL: asks about a specific sentence/paragraph/nearby context.\n"
            "Output JSON only: {\"scope\":\"global|local\",\"reason\":\"...\"}\n\n"
            f"user_query_original: {question.original_query}\n"
            f"user_query_standardized: {question.standardized_query}\n"
            f"user_language: {query_language.value}\n"
            f"session_active_chunk_index: {session_active_chunk_index}\n"
            f"best_semantic_keyword: {matched_keyword}\n"
            f"best_semantic_similarity: {similarity}\n"
            f"document_summary: {summary if summary else '[none]'}\n"
        )
        try:
            raw = self.llm_provider.complete_text(prompt).strip()
            # Accept direct JSON or a JSON block embedded in response text.
            parsed: dict[str, str] | None = None
            try:
                candidate = json.loads(raw)
                if isinstance(candidate, dict):
                    parsed = candidate
            except Exception:
                match = re.search(r"\{[\s\S]*\}", raw)
                if match:
                    candidate = json.loads(match.group(0))
                    if isinstance(candidate, dict):
                        parsed = candidate

            if parsed is None:
                print(
                    "Warn:QuestionScopeResolver#resolve: llm_scope_parse_failed, "
                    f"raw={raw[:180]}"
                )
                return None, None

            scope_value = str(parsed.get("scope", "")).strip().lower()
            reason_value = str(parsed.get("reason", "")).strip()
            if scope_value == QuestionScope.GLOBAL.value:
                return QuestionScope.GLOBAL, reason_value
            if scope_value == QuestionScope.LOCAL.value:
                return QuestionScope.LOCAL, reason_value
            print(
                "Warn:QuestionScopeResolver#resolve: llm_scope_invalid_value, "
                f"scope={scope_value}"
            )
            return None, None
        except Exception as e:
            print(
                "Warn:QuestionScopeResolver#resolve: llm_scope_failed, "
                f"error={e}"
            )
            return None, None

    def resolve(
        self,
        question: StandardizedQuestion,
        document_summary: str | None = None,
        session_active_chunk_index: int | None = None,
    ) -> QuestionScopeResolution:
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

        is_gray_zone = (
            similarity is not None
            and self.llm_gray_zone_min_similarity <= similarity < self.llm_gray_zone_max_similarity
        )
        local_signal = self._has_local_reference_signal(
            query=query,
            query_language=language,
            session_active_chunk_index=session_active_chunk_index,
        )

        if (
            is_gray_zone
            and self.llm_fallback_enabled
            and not local_signal.matched
        ):
            llm_scope, llm_reason = self._llm_decide_scope(
                question=question,
                query_language=language,
                matched_keyword=matched_keyword,
                similarity=similarity,
                document_summary=document_summary,
                session_active_chunk_index=session_active_chunk_index,
            )
            if llm_scope is not None:
                method = "llm_fallback"
                if llm_reason:
                    method = f"{method}:{llm_reason[:80]}"
                return QuestionScopeResolution(
                    scope=llm_scope,
                    query_language=language,
                    matched_keyword=matched_keyword,
                    similarity=similarity,
                    method=method,
                )

        if is_gray_zone and local_signal.matched:
            print(
                "QuestionScopeResolver#local_signal:",
                f"quality={local_signal.quality.value if local_signal.quality else 'none'}",
                f"method={local_signal.method}",
                f"keyword={local_signal.matched_text}",
                f"similarity={local_signal.similarity}",
                f"session_active_chunk_index={session_active_chunk_index}",
            )

        return QuestionScopeResolution(
            scope=QuestionScope.LOCAL,
            query_language=language,
            matched_keyword=matched_keyword,
            similarity=similarity,
            method="fallback_local",
        )
