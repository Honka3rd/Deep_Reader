from dataclasses import dataclass

import numpy as np

from embeddings.embedder import Embedder
from embeddings.embedding_similarity_service import EmbeddingSimilarityService


@dataclass(frozen=True)
class TopicGuidanceRule:
    """Mapping rule from topic signals to task analysis guidance."""

    rule_id: str
    lexical_keywords: tuple[str, ...]
    instruction: str


class TopicGuidanceRegistry:
    """Central registry for topic -> analysis-guidance resolution."""

    _RULES: tuple[TopicGuidanceRule, ...] = (
        TopicGuidanceRule(
            rule_id="literature",
            lexical_keywords=("literary", "novel", "fiction", "literature"),
            instruction=(
                "Topic Guidance: emphasize characters, relationships, and key events."
            ),
        ),
        TopicGuidanceRule(
            rule_id="finance",
            lexical_keywords=("financial", "report", "finance", "earnings"),
            instruction=(
                "Topic Guidance: emphasize entities, periods, and concrete financial points."
            ),
        ),
        TopicGuidanceRule(
            rule_id="technical",
            lexical_keywords=("technical", "documentation", "manual", "api"),
            instruction=(
                "Topic Guidance: emphasize definitions, procedures, and constraints."
            ),
        ),
        TopicGuidanceRule(
            rule_id="history",
            lexical_keywords=("biography", "history", "historical"),
            instruction=(
                "Topic Guidance: emphasize people, timeline, and turning points."
            ),
        ),
    )

    def __init__(
        self,
        embedder: Embedder | None = None,
        similarity_service: EmbeddingSimilarityService | None = None,
        semantic_match_enabled: bool = True,
        semantic_similarity_threshold: float = 0.78,
    ):
        self.embedder = embedder
        self.similarity_service = similarity_service
        self.semantic_match_enabled = semantic_match_enabled
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self._embedding_cache: dict[str, np.ndarray] = {}

    def resolve_instruction(self, topic: str | None) -> str:
        """Resolve topic guidance instruction from one topic text."""
        normalized = (topic or "").strip().lower()
        if not normalized:
            return "Topic Guidance: None"

        lexical_rule = self._resolve_lexical_rule(normalized)
        if lexical_rule is not None:
            return lexical_rule.instruction

        semantic_rule = self._resolve_semantic_rule(normalized)
        if semantic_rule is not None:
            return semantic_rule.instruction

        return f"Topic Guidance: align style with topic '{topic}'."

    def _resolve_lexical_rule(self, normalized_topic: str) -> TopicGuidanceRule | None:
        """Resolve topic guidance by lexical keyword hit."""
        for rule in self._RULES:
            for keyword in rule.lexical_keywords:
                if keyword in normalized_topic:
                    return rule
        return None

    def _resolve_semantic_rule(self, normalized_topic: str) -> TopicGuidanceRule | None:
        """Resolve topic guidance by semantic similarity on rule keywords."""
        if not self.semantic_match_enabled:
            return None
        if self.embedder is None or self.similarity_service is None:
            return None

        query_vector = self._embed_text(normalized_topic)
        if query_vector is None:
            return None

        candidate_vectors: list[np.ndarray] = []
        candidate_rules: list[TopicGuidanceRule] = []
        for rule in self._RULES:
            for keyword in rule.lexical_keywords:
                keyword_vector = self._embed_text(keyword)
                if keyword_vector is None:
                    continue
                candidate_vectors.append(keyword_vector)
                candidate_rules.append(rule)

        if not candidate_vectors:
            return None

        best = self.similarity_service.best_similarity_index(
            query_vector=query_vector,
            candidate_vectors=np.vstack(candidate_vectors),
        )
        if best is None:
            return None

        best_index, similarity = best
        if similarity < self.semantic_similarity_threshold:
            return None
        return candidate_rules[best_index]

    def _embed_text(self, text: str) -> np.ndarray | None:
        """Embed and normalize text with in-memory cache."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        if self.embedder is None or self.similarity_service is None:
            return None
        try:
            raw = self.embedder.get_text_embedding(text)
            normalized = self.similarity_service.normalize_embedding(raw)
            self._embedding_cache[text] = normalized
            return normalized
        except Exception as error:
            print(
                "Warn:TopicGuidanceRegistry#_embed_text_failed:",
                f"text={text!r}",
                f"error={error}",
            )
            return None
