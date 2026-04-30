import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from document_structure.structured_document import StructuredSection
from language.language_code import LanguageCode
from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
from section_tasks.sentence_terminal_boundary_normalizer import (
    SentenceTerminalBoundaryNormalizer,
)
from section_tasks.task_unit_boundary_language_registry import (
    TaskUnitBoundaryLanguageConfig,
    TaskUnitBoundaryLanguageRegistry,
)
from section_tasks.task_unit import TaskUnit
from section_tasks.task_unit_split_mode import TaskUnitSplitMode


class SemanticBoundaryScorer(Protocol):
    """Optional semantic scorer contract for ranking split-boundary candidates."""

    def score_boundary(
        self,
        *,
        text: str,
        boundary_index: int,
    ) -> float:
        """Return semantic preference score for one candidate split boundary."""
        raise NotImplementedError

    def score_boundaries(
        self,
        *,
        text: str,
        boundary_indices: list[int],
    ) -> dict[int, float]:
        """Return semantic scores for multiple boundaries (optional batch path)."""
        raise NotImplementedError


class BoundaryKind(StrEnum):
    """Candidate boundary classification for semantic-safe hard-constraint policy."""

    PARAGRAPH_END = "paragraph_end"
    SENTENCE_END = "sentence_end"
    STRONG_PUNCT = "strong_punct"
    CLAUSE_END = "clause_end"
    LINE_WRAP = "line_wrap"
    HARD_WINDOW = "hard_window"


class SplitBoundaryQuality(StrEnum):
    """Observed split boundary quality label for diagnostics."""

    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE_CLUSTER = "sentence_cluster"
    CLAUSE = "clause"
    HARD_FALLBACK = "hard_fallback"
    SOURCE_QUALITY_FALLBACK = "source_quality_fallback"


@dataclass(frozen=True)
class CutBoundaryCandidate:
    """One candidate split boundary with heuristic + semantic scoring details."""

    index: int
    kind: BoundaryKind
    left_snippet: str
    right_snippet: str
    base_score: float
    semantic_score: float
    final_score: float
    quality: SplitBoundaryQuality


@dataclass
class _BoundaryQualityMetrics:
    """Split-boundary quality counters for section-level observability."""

    orphan_closing_start_count: int = 0
    boundary_inside_terminal_cluster_count: int = 0
    hard_cut_count: int = 0
    mid_sentence_cut_count: int = 0
    source_quality_issue_count: int = 0
    no_sentence_boundary_fallback_count: int = 0
    source_quality_fallback_count: int = 0
    sentence_boundary_cut_count: int = 0
    paragraph_boundary_cut_count: int = 0
    potential_mid_sentence_cut_count: int = 0


@dataclass
class _SemanticRerankContext:
    """Per-section semantic rerank runtime state."""

    remaining_section_budget: int
    section_id: str
    semantic_windows_reranked: int = 0
    semantic_candidates_scored: int = 0
    budget_fallback_hit_count: int = 0
    boundary_quality_metrics: _BoundaryQualityMetrics = field(
        default_factory=_BoundaryQualityMetrics
    )


class HeuristicTaskUnitSplitResolver(AbstractTaskUnitSplitResolver):
    """Low-cost section split resolver with semantic-safe/progressive modes."""

    _PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")

    def __init__(
        self,
        split_mode: TaskUnitSplitMode | str = TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer: SemanticBoundaryScorer | None = None,
        boundary_language_registry: TaskUnitBoundaryLanguageRegistry | None = None,
        boundary_cluster_normalizer: SentenceTerminalBoundaryNormalizer | None = None,
        boundary_cluster_max_extra_chars: int = 8,
        semantic_top_k_candidates: int = 3,
        semantic_max_scoring_per_window: int = 3,
        semantic_max_scoring_per_section: int = 12,
        semantic_scoring_debug_log: bool = True,
    ):
        self.split_mode = TaskUnitSplitMode.resolve(split_mode)
        # Extension hook: can rank candidate boundaries by semantic cohesion.
        self.semantic_boundary_scorer = semantic_boundary_scorer
        self.semantic_top_k_candidates = max(1, int(semantic_top_k_candidates))
        self.semantic_max_scoring_per_window = max(
            1, int(semantic_max_scoring_per_window)
        )
        self.semantic_max_scoring_per_section = max(
            1, int(semantic_max_scoring_per_section)
        )
        self.semantic_scoring_debug_log = bool(semantic_scoring_debug_log)
        self.boundary_language_registry = (
            boundary_language_registry or TaskUnitBoundaryLanguageRegistry()
        )
        self.boundary_cluster_max_extra_chars = max(
            1, int(boundary_cluster_max_extra_chars)
        )
        self.boundary_cluster_normalizer = boundary_cluster_normalizer or SentenceTerminalBoundaryNormalizer(
            language_registry=self.boundary_language_registry,
            boundary_cluster_max_extra_chars=self.boundary_cluster_max_extra_chars,
        )

    def split_section(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
        semantic_top_k_candidates: int | None = None,
        language_code: LanguageCode | str | None = None,
    ) -> list[TaskUnit]:
        """Split one oversized section with selected heuristic mode."""
        text = section.content.strip()
        if not text:
            return []

        min_chars = max(1, int(task_unit_min_chars))
        max_chars = max(min_chars, int(task_unit_max_chars))
        resolved_semantic_top_k_candidates = self._resolve_semantic_top_k_candidates(
            semantic_top_k_candidates
        )
        semantic_context = self._build_semantic_context(section_id=section.section_id)
        chunks = self._split_text(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
            section_id=section.section_id,
            language_code=language_code,
            semantic_context=semantic_context,
            semantic_top_k_candidates=resolved_semantic_top_k_candidates,
        )
        chunks = self._stabilize_trailing_short_chunk(
            chunks=chunks,
            min_chars=min_chars,
            max_chars=max_chars,
            section_id=section.section_id,
            language_code=language_code,
            semantic_context=semantic_context,
            semantic_top_k_candidates=resolved_semantic_top_k_candidates,
        )
        if not chunks:
            return []

        self._log_semantic_context(
            section_id=section.section_id,
            semantic_context=semantic_context,
        )
        self._validate_boundary_quality(
            chunks=chunks,
            section_id=section.section_id,
            language_code=language_code,
            metrics=(
                None
                if semantic_context is None
                else semantic_context.boundary_quality_metrics
            ),
        )

        base_title = self._normalize_optional_text(section.title)
        container_title = self._normalize_optional_text(section.container_title)
        if len(chunks) == 1:
            return [
                TaskUnit(
                    unit_id=f"task-unit-{section_index}-0",
                    title=base_title,
                    container_title=container_title,
                    content=chunks[0],
                    source_section_ids=[section.section_id],
                    is_fallback_generated=True,
                )
            ]

        units: list[TaskUnit] = []
        for chunk_index, chunk in enumerate(chunks):
            if base_title:
                resolved_title = f"{base_title} (Part {chunk_index + 1})"
            else:
                resolved_title = f"Task Unit {chunk_index + 1}"
            units.append(
                TaskUnit(
                    unit_id=f"task-unit-{section_index}-{chunk_index}",
                    title=resolved_title,
                    container_title=container_title,
                    content=chunk,
                    source_section_ids=[section.section_id],
                    is_fallback_generated=True,
                )
            )
        return units

    def _split_text(
        self,
        *,
        text: str,
        min_chars: int,
        max_chars: int,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None = None,
        semantic_top_k_candidates: int,
    ) -> list[str]:
        """Route to target heuristic split mode."""
        if self.split_mode == TaskUnitSplitMode.PROGRESSIVE:
            return self._split_text_progressive(
                text=text,
                min_chars=min_chars,
                max_chars=max_chars,
                section_id=section_id,
                language_code=language_code,
                semantic_context=semantic_context,
                semantic_top_k_candidates=semantic_top_k_candidates,
            )
        return self._split_text_semantic_safe(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
            section_id=section_id,
            language_code=language_code,
            semantic_context=semantic_context,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )

    def _split_text_semantic_safe(
        self,
        *,
        text: str,
        min_chars: int,
        max_chars: int,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None = None,
        semantic_top_k_candidates: int,
    ) -> list[str]:
        """Split with paragraph-first strategy and semantic boundary search."""
        paragraphs = [
            paragraph.strip()
            for paragraph in self._PARAGRAPH_SPLIT_PATTERN.split(text)
            if paragraph.strip()
        ]
        if not paragraphs:
            return self._split_text_progressive(
                text=text,
                min_chars=min_chars,
                max_chars=max_chars,
                section_id=section_id,
                language_code=language_code,
                semantic_context=semantic_context,
                semantic_top_k_candidates=semantic_top_k_candidates,
            )

        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
            if len(candidate) <= max_chars:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer.strip())
                buffer = ""

            if len(paragraph) <= max_chars:
                buffer = paragraph
                continue

            chunks.extend(
                self._split_text_progressive(
                    text=paragraph,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    section_id=section_id,
                    language_code=language_code,
                    semantic_context=semantic_context,
                    semantic_top_k_candidates=semantic_top_k_candidates,
                )
            )

        if buffer.strip():
            chunks.append(buffer.strip())
        return [chunk for chunk in chunks if chunk]

    def _split_text_progressive(
        self,
        *,
        text: str,
        min_chars: int,
        max_chars: int,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None = None,
        semantic_top_k_candidates: int,
    ) -> list[str]:
        """Split by target windows but search semantic-safe cut points near boundary."""
        if not text.strip():
            return []

        chunks: list[str] = []
        cursor = 0
        text_length = len(text)
        while cursor < text_length:
            remaining = text_length - cursor
            if remaining <= max_chars:
                tail = text[cursor:].strip()
                if tail:
                    chunks.append(tail)
                break

            ideal_cut = cursor + max_chars
            search_window = max(60, max_chars // 5)
            min_cut = min(text_length, cursor + min_chars)
            max_cut = min(text_length, ideal_cut + search_window)
            lower_bound = max(min_cut, ideal_cut - search_window)
            if lower_bound >= max_cut:
                lower_bound = min_cut

            cut = self._find_best_cut_index(
                text=text,
                lower_bound=lower_bound,
                upper_bound=max_cut,
                ideal_cut=ideal_cut,
                section_id=section_id,
                language_code=language_code,
                semantic_context=semantic_context,
                semantic_top_k_candidates=semantic_top_k_candidates,
            )
            if cut <= cursor:
                cut = min(text_length, cursor + max_chars)

            chunk = text[cursor:cut].strip()
            if chunk:
                chunks.append(chunk)
            cursor = cut

        return [chunk for chunk in chunks if chunk]

    def _find_best_cut_index(
        self,
        *,
        text: str,
        lower_bound: int,
        upper_bound: int,
        ideal_cut: int,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None = None,
        semantic_top_k_candidates: int,
    ) -> int:
        """Pick best cut in boundary window with hard boundary constraints."""
        safe_lower = max(1, int(lower_bound))
        safe_upper = min(len(text), int(upper_bound))
        if safe_lower >= safe_upper:
            return min(max(ideal_cut, safe_lower), safe_upper)

        candidate_tuples: list[tuple[int, BoundaryKind]] = []
        for index in range(safe_lower, safe_upper + 1):
            boundary_kind = self._classify_boundary_kind(
                text=text,
                index=index,
                language_code=language_code,
            )
            if boundary_kind == BoundaryKind.HARD_WINDOW:
                continue
            if boundary_kind == BoundaryKind.LINE_WRAP:
                # OCR/layout line wraps are weak visual artifacts, not semantic boundaries.
                continue
            normalization = self.boundary_cluster_normalizer.normalize(
                text=text,
                boundary_index=index,
                language=language_code,
            )
            normalized_index = normalization.normalized_index
            if normalized_index <= 0 or normalized_index >= len(text):
                continue
            if normalized_index < safe_lower:
                continue
            if normalized_index > (safe_upper + self.boundary_cluster_max_extra_chars):
                continue
            normalized_kind = self._classify_boundary_kind(
                text=text,
                index=normalized_index,
                language_code=language_code,
            )
            if normalized_kind == BoundaryKind.HARD_WINDOW:
                normalized_kind = boundary_kind
            candidate_tuples.append((normalized_index, normalized_kind))
            if semantic_context is not None:
                if normalization.boundary_inside_terminal_cluster:
                    semantic_context.boundary_quality_metrics.boundary_inside_terminal_cluster_count += 1
                if normalization.orphan_closing_start_shifted:
                    semantic_context.boundary_quality_metrics.orphan_closing_start_count += 1

        candidate_map: dict[int, BoundaryKind] = {}
        for normalized_index, normalized_kind in candidate_tuples:
            existing = candidate_map.get(normalized_index)
            if existing is None:
                candidate_map[normalized_index] = normalized_kind
                continue
            if self._kind_rank(normalized_kind) < self._kind_rank(existing):
                candidate_map[normalized_index] = normalized_kind

        candidate_indices = sorted(candidate_map.keys())
        if not candidate_indices:
            fallback_index = min(max(ideal_cut, safe_lower), safe_upper)
            fallback_reason = "no_reliable_sentence_or_paragraph_boundary"
            fallback_quality = SplitBoundaryQuality.SOURCE_QUALITY_FALLBACK
            if semantic_context is not None:
                semantic_context.boundary_quality_metrics.no_sentence_boundary_fallback_count += 1
            if semantic_context is not None:
                self._record_boundary_quality(
                    text=text,
                    cut_index=fallback_index,
                    metrics=semantic_context.boundary_quality_metrics,
                    language_code=language_code,
                    quality=fallback_quality,
                )
            self._log_hard_cut_fallback(
                text=text,
                ideal_cut=ideal_cut,
                chosen_index=fallback_index,
                reason=fallback_reason,
                section_id=section_id,
            )
            return fallback_index

        heuristic_scores = {
            index: self._score_cut_candidate_base(
                text=text,
                index=index,
                ideal_cut=ideal_cut,
                boundary_kind=candidate_map.get(index),
            )
            for index in candidate_indices
        }
        semantic_scores = self._score_semantic_top_candidates(
            text=text,
            candidate_indices=candidate_indices,
            heuristic_scores=heuristic_scores,
            semantic_context=semantic_context,
            semantic_top_k_candidates=semantic_top_k_candidates,
        )

        candidates = self._build_cut_candidates(
            text=text,
            candidate_indices=candidate_indices,
            boundary_kind_by_index=candidate_map,
            heuristic_scores=heuristic_scores,
            semantic_scores=semantic_scores,
            language_code=language_code,
        )

        constrained_cut = self._apply_hard_boundary_policy(
            candidates=candidates,
            ideal_cut=ideal_cut,
            text=text,
            semantic_context=semantic_context,
            section_id=section_id,
            language_code=language_code,
        )
        if semantic_context is not None:
            self._record_boundary_quality(
                text=text,
                cut_index=constrained_cut,
                metrics=semantic_context.boundary_quality_metrics,
                language_code=language_code,
            )
        return constrained_cut

    def _build_cut_candidates(
        self,
        *,
        text: str,
        candidate_indices: list[int],
        boundary_kind_by_index: dict[int, BoundaryKind],
        heuristic_scores: dict[int, float],
        semantic_scores: dict[int, float],
        language_code: LanguageCode | str | None,
    ) -> list[CutBoundaryCandidate]:
        """Build classified cut-candidate objects for hard-constraint ranking."""
        candidates: list[CutBoundaryCandidate] = []
        for index in candidate_indices:
            base_score = heuristic_scores[index]
            semantic_score = float(semantic_scores.get(index, 0.0))
            boundary_kind = boundary_kind_by_index.get(
                index,
                self._classify_boundary_kind(
                    text=text,
                    index=index,
                    language_code=language_code,
                ),
            )
            candidates.append(
                CutBoundaryCandidate(
                    index=index,
                    kind=boundary_kind,
                    left_snippet=self._snippet_left(text=text, index=index),
                    right_snippet=self._snippet_right(text=text, index=index),
                    base_score=base_score,
                    semantic_score=semantic_score,
                    final_score=base_score + semantic_score,
                    quality=self._quality_from_kind(boundary_kind),
                )
            )
        return candidates

    def _apply_hard_boundary_policy(
        self,
        *,
        candidates: list[CutBoundaryCandidate],
        ideal_cut: int,
        text: str,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None,
    ) -> int:
        """Apply mode-specific priority to avoid mid-sentence boundary selection."""
        if self.split_mode == TaskUnitSplitMode.SEMANTIC_SAFE:
            kind_priority = (
                BoundaryKind.PARAGRAPH_END,
                BoundaryKind.SENTENCE_END,
                BoundaryKind.STRONG_PUNCT,
                BoundaryKind.CLAUSE_END,
            )
        else:
            kind_priority = (
                BoundaryKind.SENTENCE_END,
                BoundaryKind.STRONG_PUNCT,
                BoundaryKind.CLAUSE_END,
                BoundaryKind.PARAGRAPH_END,
            )

        for kind in kind_priority:
            typed = [candidate for candidate in candidates if candidate.kind == kind]
            if typed:
                best = max(
                    typed,
                    key=lambda candidate: (
                        candidate.final_score,
                        -abs(candidate.index - ideal_cut),
                    ),
                )
                normalized = self.boundary_cluster_normalizer.normalize(
                    text=text,
                    boundary_index=best.index,
                    language=language_code,
                )
                if semantic_context is not None:
                    if normalized.boundary_inside_terminal_cluster:
                        semantic_context.boundary_quality_metrics.boundary_inside_terminal_cluster_count += 1
                    if normalized.orphan_closing_start_shifted:
                        semantic_context.boundary_quality_metrics.orphan_closing_start_count += 1
                return normalized.normalized_index

        # No reliable candidate remained after hard constraints.
        fallback_index = min(max(ideal_cut, 1), max(1, len(text) - 1))
        fallback_normalized = self.boundary_cluster_normalizer.normalize(
            text=text,
            boundary_index=fallback_index,
            language=language_code,
        )
        fallback_index = fallback_normalized.normalized_index
        if semantic_context is not None:
            semantic_context.boundary_quality_metrics.hard_cut_count += 1
            semantic_context.boundary_quality_metrics.no_sentence_boundary_fallback_count += 1
            semantic_context.boundary_quality_metrics.source_quality_issue_count += 1
            semantic_context.boundary_quality_metrics.source_quality_fallback_count += 1
            if fallback_normalized.boundary_inside_terminal_cluster:
                semantic_context.boundary_quality_metrics.boundary_inside_terminal_cluster_count += 1
            if fallback_normalized.orphan_closing_start_shifted:
                semantic_context.boundary_quality_metrics.orphan_closing_start_count += 1
        self._log_hard_cut_fallback(
            text=text,
            ideal_cut=ideal_cut,
            chosen_index=fallback_index,
            reason="no_reliable_sentence_or_paragraph_boundary",
            section_id=section_id,
        )
        return fallback_index

    def _score_cut_candidate_base(
        self,
        *,
        text: str,
        index: int,
        ideal_cut: int,
        boundary_kind: BoundaryKind | None = None,
    ) -> float:
        """Score one candidate boundary by structural heuristic signal only."""
        score = -abs(index - ideal_cut) / 8.0
        kind = boundary_kind or self._classify_boundary_kind(
            text=text,
            index=index,
            language_code=LanguageCode.UNKNOWN,
        )
        if kind == BoundaryKind.PARAGRAPH_END:
            score += 4.0
        elif kind == BoundaryKind.SENTENCE_END:
            score += 3.0
        elif kind == BoundaryKind.STRONG_PUNCT:
            score += 2.0
        elif kind == BoundaryKind.CLAUSE_END:
            score += 0.5
        elif kind == BoundaryKind.LINE_WRAP:
            score += 0.2
        return score

    def _score_semantic_top_candidates(
        self,
        *,
        text: str,
        candidate_indices: list[int],
        heuristic_scores: dict[int, float],
        semantic_context: _SemanticRerankContext | None,
        semantic_top_k_candidates: int,
    ) -> dict[int, float]:
        """Rerank only top heuristic candidates with bounded semantic scoring budget."""
        scorer = self.semantic_boundary_scorer
        if scorer is None or semantic_context is None:
            return {}

        if semantic_context.remaining_section_budget <= 0:
            semantic_context.budget_fallback_hit_count += 1
            return {}

        ranked_by_heuristic = sorted(
            candidate_indices,
            key=lambda index: heuristic_scores[index],
            reverse=True,
        )
        rerank_cap = min(
            len(ranked_by_heuristic),
            semantic_top_k_candidates,
            self.semantic_max_scoring_per_window,
            semantic_context.remaining_section_budget,
        )
        if rerank_cap <= 0:
            semantic_context.budget_fallback_hit_count += 1
            return {}

        rerank_indices = ranked_by_heuristic[:rerank_cap]
        semantic_context.remaining_section_budget -= rerank_cap
        semantic_context.semantic_windows_reranked += 1

        try:
            if hasattr(scorer, "score_boundaries"):
                scored = scorer.score_boundaries(
                    text=text,
                    boundary_indices=rerank_indices,
                )
            else:
                scored = {
                    index: scorer.score_boundary(
                        text=text,
                        boundary_index=index,
                    )
                    for index in rerank_indices
                }
            semantic_context.semantic_candidates_scored += len(scored)
            return {
                index: float(value)
                for index, value in scored.items()
                if index in rerank_indices
            }
        except Exception:
            # Keep split deterministic even if optional scorer is unstable.
            return {}

    def _build_semantic_context(
        self,
        *,
        section_id: str,
    ) -> _SemanticRerankContext | None:
        """Build per-section semantic rerank budget state when scorer is enabled."""
        if self.semantic_boundary_scorer is None:
            return None
        return _SemanticRerankContext(
            remaining_section_budget=self.semantic_max_scoring_per_section,
            section_id=section_id,
        )

    def _log_semantic_context(
        self,
        *,
        section_id: str,
        semantic_context: _SemanticRerankContext | None,
    ) -> None:
        """Print compact semantic rerank stats for observability."""
        if (
            not self.semantic_scoring_debug_log
            or semantic_context is None
        ):
            return

        print(
            "HeuristicTaskUnitSplitResolver#semantic_rerank:",
            f"mode={self.split_mode.value}",
            f"section_id={section_id or semantic_context.section_id}",
            f"semantic_windows={semantic_context.semantic_windows_reranked}",
            f"semantic_candidates_scored={semantic_context.semantic_candidates_scored}",
            f"remaining_section_budget={semantic_context.remaining_section_budget}",
            f"budget_fallback_hit_count={semantic_context.budget_fallback_hit_count}",
            f"paragraph_boundary_cut_count={semantic_context.boundary_quality_metrics.paragraph_boundary_cut_count}",
            f"sentence_boundary_cut_count={semantic_context.boundary_quality_metrics.sentence_boundary_cut_count}",
            f"hard_cut_count={semantic_context.boundary_quality_metrics.hard_cut_count}",
            f"orphan_closing_start_count={semantic_context.boundary_quality_metrics.orphan_closing_start_count}",
            f"boundary_inside_terminal_cluster_count={semantic_context.boundary_quality_metrics.boundary_inside_terminal_cluster_count}",
            f"mid_sentence_cut_count={semantic_context.boundary_quality_metrics.mid_sentence_cut_count}",
            f"source_quality_issue_count={semantic_context.boundary_quality_metrics.source_quality_issue_count}",
            f"no_sentence_boundary_fallback_count={semantic_context.boundary_quality_metrics.no_sentence_boundary_fallback_count}",
            f"source_quality_fallback_count={semantic_context.boundary_quality_metrics.source_quality_fallback_count}",
            f"potential_mid_sentence_cut_count={semantic_context.boundary_quality_metrics.potential_mid_sentence_cut_count}",
        )

    def _is_boundary_candidate(self, *, text: str, index: int) -> bool:
        """Return True when index is likely safe split boundary."""
        return (
            self._classify_boundary_kind(
                text=text,
                index=index,
                language_code=LanguageCode.UNKNOWN,
            )
            != BoundaryKind.HARD_WINDOW
        )

    @staticmethod
    def _quality_from_kind(boundary_kind: BoundaryKind) -> SplitBoundaryQuality:
        if boundary_kind == BoundaryKind.PARAGRAPH_END:
            return SplitBoundaryQuality.PARAGRAPH
        if boundary_kind in {BoundaryKind.SENTENCE_END, BoundaryKind.STRONG_PUNCT}:
            return SplitBoundaryQuality.SENTENCE_CLUSTER
        if boundary_kind == BoundaryKind.CLAUSE_END:
            return SplitBoundaryQuality.CLAUSE
        if boundary_kind == BoundaryKind.HARD_WINDOW:
            return SplitBoundaryQuality.HARD_FALLBACK
        return SplitBoundaryQuality.SEMANTIC

    @staticmethod
    def _kind_rank(boundary_kind: BoundaryKind) -> int:
        order = {
            BoundaryKind.PARAGRAPH_END: 0,
            BoundaryKind.SENTENCE_END: 1,
            BoundaryKind.STRONG_PUNCT: 2,
            BoundaryKind.CLAUSE_END: 3,
            BoundaryKind.LINE_WRAP: 4,
            BoundaryKind.HARD_WINDOW: 5,
        }
        return order.get(boundary_kind, 99)

    def _classify_boundary_kind(
        self,
        *,
        text: str,
        index: int,
        language_code: LanguageCode | str | None,
    ) -> BoundaryKind:
        """Classify one boundary by semantic quality priority."""
        config = self.boundary_language_registry.get_config(language_code)
        if index <= 0 or index >= len(text):
            return BoundaryKind.HARD_WINDOW
        if self._is_paragraph_boundary(text=text, index=index):
            return BoundaryKind.PARAGRAPH_END
        if self._is_sentence_boundary(text=text, index=index, config=config):
            return BoundaryKind.SENTENCE_END
        if self._is_strong_punctuation_boundary(text=text, index=index, config=config):
            return BoundaryKind.STRONG_PUNCT
        if self._is_clause_boundary(text=text, index=index, config=config):
            return BoundaryKind.CLAUSE_END
        if self._is_line_wrap_boundary(text=text, index=index):
            return BoundaryKind.LINE_WRAP
        return BoundaryKind.HARD_WINDOW

    @staticmethod
    def _snippet_left(*, text: str, index: int, width: int = 80) -> str:
        return text[max(0, index - width):index]

    @staticmethod
    def _snippet_right(*, text: str, index: int, width: int = 80) -> str:
        return text[index:min(len(text), index + width)]

    def _is_paragraph_boundary(self, *, text: str, index: int) -> bool:
        """Paragraph boundary must be blank-line style, not single layout newline."""
        left_tail = text[max(0, index - 6):index]
        right_head = text[index:min(len(text), index + 6)]
        return bool(
            re.search(r"\n\s*\n\s*$", left_tail)
            or re.match(r"^\s*\n\s*\n", right_head)
        )

    def _is_sentence_boundary(
        self,
        *,
        text: str,
        index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        return self._tail_ends_with_chars(
            text=text,
            index=index,
            boundary_chars=config.sentence_terminal_chars,
            config=config,
        )

    def _is_strong_punctuation_boundary(
        self,
        *,
        text: str,
        index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        return self._tail_ends_with_chars(
            text=text,
            index=index,
            boundary_chars=config.strong_punctuation_chars,
            config=config,
        )

    def _is_clause_boundary(
        self,
        *,
        text: str,
        index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        return self._tail_ends_with_chars(
            text=text,
            index=index,
            boundary_chars=config.weak_clause_chars,
            config=config,
        )

    def _tail_ends_with_chars(
        self,
        *,
        text: str,
        index: int,
        boundary_chars: frozenset[str],
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        """Check whether boundary tail ends with target punctuation cluster."""
        cursor = index - 1
        consumed = 0
        max_scan = self.boundary_cluster_max_extra_chars + 2
        while cursor >= 0 and consumed < max_scan:
            char = text[cursor]
            if char.isspace():
                cursor -= 1
                consumed += 1
                continue
            if self._is_orphan_closing_char(char, config=config):
                cursor -= 1
                consumed += 1
                continue
            return char in boundary_chars
        return False

    def _is_line_wrap_boundary(self, *, text: str, index: int) -> bool:
        return (
            index > 0
            and text[index - 1] == "\n"
            and not self._is_paragraph_boundary(text=text, index=index)
        )

    def _record_boundary_quality(
        self,
        *,
        text: str,
        cut_index: int,
        metrics: _BoundaryQualityMetrics,
        language_code: LanguageCode | str | None,
        quality: SplitBoundaryQuality | None = None,
    ) -> None:
        """Record split-boundary quality counters for observability."""
        kind = self._classify_boundary_kind(
            text=text,
            index=cut_index,
            language_code=language_code,
        )
        resolved_quality = quality or self._quality_from_kind(kind)
        if kind == BoundaryKind.PARAGRAPH_END:
            metrics.paragraph_boundary_cut_count += 1
        elif kind in {BoundaryKind.SENTENCE_END, BoundaryKind.STRONG_PUNCT}:
            metrics.sentence_boundary_cut_count += 1
        elif kind == BoundaryKind.HARD_WINDOW or resolved_quality in {
            SplitBoundaryQuality.HARD_FALLBACK,
            SplitBoundaryQuality.SOURCE_QUALITY_FALLBACK,
        }:
            metrics.hard_cut_count += 1

        if kind in {
            BoundaryKind.CLAUSE_END,
            BoundaryKind.LINE_WRAP,
            BoundaryKind.HARD_WINDOW,
        }:
            metrics.potential_mid_sentence_cut_count += 1
            metrics.mid_sentence_cut_count += 1
        if resolved_quality == SplitBoundaryQuality.SOURCE_QUALITY_FALLBACK:
            metrics.source_quality_fallback_count += 1
            metrics.source_quality_issue_count += 1

    def _log_hard_cut_fallback(
        self,
        *,
        text: str,
        ideal_cut: int,
        chosen_index: int,
        reason: str,
        section_id: str,
    ) -> None:
        """Warn when resolver must fall back to hard window cut."""
        left = self._snippet_left(text=text, index=chosen_index, width=60).replace(
            "\n", "\\n"
        )
        right = self._snippet_right(text=text, index=chosen_index, width=60).replace(
            "\n", "\\n"
        )
        print(
            "HeuristicTaskUnitSplitResolver#hard_cut_warning:",
            f"mode={self.split_mode.value}",
            f"section_id={section_id}",
            f"target_index={ideal_cut}",
            f"chosen_index={chosen_index}",
            f"reason={reason}",
            f"left='{left}'",
            f"right='{right}'",
        )

    def _stabilize_trailing_short_chunk(
        self,
        *,
        chunks: list[str],
        min_chars: int,
        max_chars: int,
        section_id: str,
        language_code: LanguageCode | str | None,
        semantic_context: _SemanticRerankContext | None = None,
        semantic_top_k_candidates: int,
    ) -> list[str]:
        """Avoid too-short tail chunks that can degrade downstream task quality."""
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if len(normalized_chunks) <= 1:
            return normalized_chunks

        max_rebalance_iterations = max(4, len(normalized_chunks) * 4)
        rebalance_iterations = 0
        while (
            len(normalized_chunks) > 1
            and len(normalized_chunks[-1]) < min_chars
        ):
            rebalance_iterations += 1
            if rebalance_iterations > max_rebalance_iterations:
                break
            left = normalized_chunks[-2]
            right = normalized_chunks[-1]
            merged = f"{left}\n\n{right}".strip()
            if len(merged) <= max_chars:
                normalized_chunks[-2] = merged
                normalized_chunks.pop()
                continue

            rebalance = self._find_best_cut_index(
                text=merged,
                lower_bound=min_chars,
                upper_bound=max(len(merged) - min_chars, min_chars + 1),
                ideal_cut=len(merged) // 2,
                section_id=section_id,
                language_code=language_code,
                semantic_context=semantic_context,
                semantic_top_k_candidates=semantic_top_k_candidates,
            )
            if rebalance <= 0 or rebalance >= len(merged):
                normalized_chunks[-2] = merged
                normalized_chunks.pop()
                continue

            rebalanced_left = merged[:rebalance].strip()
            rebalanced_right = merged[rebalance:].strip()
            if not rebalanced_left or not rebalanced_right:
                normalized_chunks[-2] = merged
                normalized_chunks.pop()
                continue

            if rebalanced_left == left and rebalanced_right == right:
                # Avoid infinite rebalancing loops when split index converges.
                normalized_chunks[-2] = merged
                normalized_chunks.pop()
                continue

            normalized_chunks[-2] = rebalanced_left
            normalized_chunks[-1] = rebalanced_right
            if len(normalized_chunks[-1]) >= min_chars:
                break

        return normalized_chunks

    def _validate_boundary_quality(
        self,
        *,
        chunks: list[str],
        section_id: str,
        language_code: LanguageCode | str | None,
        metrics: _BoundaryQualityMetrics | None,
    ) -> None:
        """Validate adjacent chunk boundaries and log potential quality issues."""
        if len(chunks) <= 1:
            return
        config = self.boundary_language_registry.get_config(language_code)
        for index in range(len(chunks) - 1):
            left = chunks[index].rstrip()
            right = chunks[index + 1].lstrip()
            if not left or not right:
                continue

            if right and self._is_orphan_closing_char(right[0], config=config):
                if metrics is not None:
                    metrics.orphan_closing_start_count += 1
                    metrics.source_quality_issue_count += 1
                print(
                    "HeuristicTaskUnitSplitResolver#boundary_quality_warning:",
                    f"mode={self.split_mode.value}",
                    f"section_id={section_id}",
                    "reason=orphan_closing_punctuation_at_unit_start",
                    f"boundary_index={index}",
                    f"left_tail='{left[-50:].replace(chr(10), ' ')}'",
                    f"right_head='{right[:50].replace(chr(10), ' ')}'",
                )

            if not self._looks_reasonable_left_boundary_end(left, config=config):
                if metrics is not None:
                    metrics.mid_sentence_cut_count += 1
                print(
                    "HeuristicTaskUnitSplitResolver#boundary_quality_warning:",
                    f"mode={self.split_mode.value}",
                    f"section_id={section_id}",
                    "reason=mid_sentence_cut",
                    f"boundary_index={index}",
                    f"left_tail='{left[-50:].replace(chr(10), ' ')}'",
                    f"right_head='{right[:50].replace(chr(10), ' ')}'",
                )

    def _looks_reasonable_left_boundary_end(
        self,
        text: str,
        *,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        """Return whether chunk end looks like paragraph/sentence terminal boundary."""
        stripped = text.rstrip()
        if not stripped:
            return False
        if stripped.endswith("\n\n"):
            return True
        cursor = len(stripped) - 1
        while cursor >= 0 and self._is_orphan_closing_char(stripped[cursor], config=config):
            cursor -= 1
        if cursor < 0:
            return False
        char = stripped[cursor]
        return (
            char in config.sentence_terminal_chars
            or char in config.strong_punctuation_chars
            or char in config.weak_clause_chars
        )

    @staticmethod
    def _is_orphan_closing_char(
        char: str,
        *,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        return (
            char in config.closing_quote_chars
            or char in config.closing_bracket_chars
        )

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize optional text to either None or non-empty string."""
        normalized = (value or "").strip()
        if not normalized:
            return None
        return normalized

    def _resolve_semantic_top_k_candidates(
        self,
        semantic_top_k_candidates: int | None,
    ) -> int:
        """Resolve per-request semantic top-k override with safe fallback."""
        if semantic_top_k_candidates is None:
            return self.semantic_top_k_candidates
        return max(1, int(semantic_top_k_candidates))
