import re
from dataclasses import dataclass
from typing import Protocol

from document_structure.structured_document import StructuredSection
from section_tasks.abstract_task_unit_split_resolver import AbstractTaskUnitSplitResolver
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


@dataclass
class _SemanticRerankContext:
    """Per-section semantic rerank runtime state."""

    remaining_section_budget: int
    semantic_windows_reranked: int = 0
    semantic_candidates_scored: int = 0
    budget_fallback_hit_count: int = 0


class HeuristicTaskUnitSplitResolver(AbstractTaskUnitSplitResolver):
    """Low-cost section split resolver with semantic-safe/progressive modes."""

    _PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
    _PRIMARY_BOUNDARY_CHARS = frozenset("。！？!?；;\n")
    _SECONDARY_BOUNDARY_CHARS = frozenset("，,:：、")

    def __init__(
        self,
        split_mode: TaskUnitSplitMode | str = TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer: SemanticBoundaryScorer | None = None,
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

    def split_section(
        self,
        *,
        section: StructuredSection,
        section_index: int,
        task_unit_min_chars: int,
        task_unit_max_chars: int,
    ) -> list[TaskUnit]:
        """Split one oversized section with selected heuristic mode."""
        text = section.content.strip()
        if not text:
            return []

        min_chars = max(1, int(task_unit_min_chars))
        max_chars = max(min_chars, int(task_unit_max_chars))
        semantic_context = self._build_semantic_context()
        chunks = self._split_text(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
            semantic_context=semantic_context,
        )
        chunks = self._stabilize_trailing_short_chunk(
            chunks=chunks,
            min_chars=min_chars,
            max_chars=max_chars,
            semantic_context=semantic_context,
        )
        if not chunks:
            return []

        self._log_semantic_context(
            section_id=section.section_id,
            semantic_context=semantic_context,
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
        semantic_context: _SemanticRerankContext | None = None,
    ) -> list[str]:
        """Route to target heuristic split mode."""
        if self.split_mode == TaskUnitSplitMode.PROGRESSIVE:
            return self._split_text_progressive(
                text=text,
                min_chars=min_chars,
                max_chars=max_chars,
                semantic_context=semantic_context,
            )
        return self._split_text_semantic_safe(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
            semantic_context=semantic_context,
        )

    def _split_text_semantic_safe(
        self,
        *,
        text: str,
        min_chars: int,
        max_chars: int,
        semantic_context: _SemanticRerankContext | None = None,
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
                semantic_context=semantic_context,
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
                    semantic_context=semantic_context,
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
        semantic_context: _SemanticRerankContext | None = None,
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
                semantic_context=semantic_context,
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
        semantic_context: _SemanticRerankContext | None = None,
    ) -> int:
        """Pick best cut in boundary window, preferring semantic boundaries."""
        safe_lower = max(1, int(lower_bound))
        safe_upper = min(len(text), int(upper_bound))
        if safe_lower >= safe_upper:
            return min(max(ideal_cut, safe_lower), safe_upper)

        candidate_indices: list[int] = []
        for index in range(safe_lower, safe_upper + 1):
            if self._is_boundary_candidate(text=text, index=index):
                candidate_indices.append(index)
        if not candidate_indices:
            return min(max(ideal_cut, safe_lower), safe_upper)

        heuristic_scores = {
            index: self._score_cut_candidate_base(
                text=text,
                index=index,
                ideal_cut=ideal_cut,
            )
            for index in candidate_indices
        }
        semantic_scores = self._score_semantic_top_candidates(
            text=text,
            candidate_indices=candidate_indices,
            heuristic_scores=heuristic_scores,
            semantic_context=semantic_context,
        )

        best_index = candidate_indices[0]
        best_score = float("-inf")
        for index in candidate_indices:
            score = heuristic_scores[index] + semantic_scores.get(index, 0.0)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _score_cut_candidate_base(self, *, text: str, index: int, ideal_cut: int) -> float:
        """Score one candidate boundary by structural heuristic signal only."""
        score = -abs(index - ideal_cut) / 8.0
        if index >= 2 and text[index - 2:index] == "\n\n":
            score += 4.0
        prev_char = text[index - 1] if index > 0 else ""
        if prev_char in self._PRIMARY_BOUNDARY_CHARS:
            score += 2.0
        elif prev_char in self._SECONDARY_BOUNDARY_CHARS:
            score += 0.5
        return score

    def _score_semantic_top_candidates(
        self,
        *,
        text: str,
        candidate_indices: list[int],
        heuristic_scores: dict[int, float],
        semantic_context: _SemanticRerankContext | None,
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
            self.semantic_top_k_candidates,
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

    def _build_semantic_context(self) -> _SemanticRerankContext | None:
        """Build per-section semantic rerank budget state when scorer is enabled."""
        if self.semantic_boundary_scorer is None:
            return None
        return _SemanticRerankContext(
            remaining_section_budget=self.semantic_max_scoring_per_section,
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
            f"section_id={section_id}",
            f"semantic_windows={semantic_context.semantic_windows_reranked}",
            f"semantic_candidates_scored={semantic_context.semantic_candidates_scored}",
            f"remaining_section_budget={semantic_context.remaining_section_budget}",
            f"budget_fallback_hit_count={semantic_context.budget_fallback_hit_count}",
        )

    def _is_boundary_candidate(self, *, text: str, index: int) -> bool:
        """Return True when index is likely safe split boundary."""
        if index <= 0 or index >= len(text):
            return False
        if text[index - 1:index + 1] == "\n\n":
            return True
        prev_char = text[index - 1]
        if prev_char in self._PRIMARY_BOUNDARY_CHARS:
            return True
        if prev_char in self._SECONDARY_BOUNDARY_CHARS and text[index] == " ":
            return True
        return False

    def _stabilize_trailing_short_chunk(
        self,
        *,
        chunks: list[str],
        min_chars: int,
        max_chars: int,
        semantic_context: _SemanticRerankContext | None = None,
    ) -> list[str]:
        """Avoid too-short tail chunks that can degrade downstream task quality."""
        normalized_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if len(normalized_chunks) <= 1:
            return normalized_chunks

        while (
            len(normalized_chunks) > 1
            and len(normalized_chunks[-1]) < min_chars
        ):
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
                semantic_context=semantic_context,
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
            normalized_chunks[-2] = rebalanced_left
            normalized_chunks[-1] = rebalanced_right
            if len(normalized_chunks[-1]) >= min_chars:
                break

        return normalized_chunks

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize optional text to either None or non-empty string."""
        normalized = (value or "").strip()
        if not normalized:
            return None
        return normalized
