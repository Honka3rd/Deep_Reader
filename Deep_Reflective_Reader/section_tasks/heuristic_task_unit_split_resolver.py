import re
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


class HeuristicTaskUnitSplitResolver(AbstractTaskUnitSplitResolver):
    """Low-cost section split resolver with semantic-safe/progressive modes."""

    _PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
    _PRIMARY_BOUNDARY_CHARS = frozenset("。！？!?；;\n")
    _SECONDARY_BOUNDARY_CHARS = frozenset("，,:：、")

    def __init__(
        self,
        split_mode: TaskUnitSplitMode | str = TaskUnitSplitMode.SEMANTIC_SAFE,
        semantic_boundary_scorer: SemanticBoundaryScorer | None = None,
    ):
        self.split_mode = TaskUnitSplitMode.resolve(split_mode)
        # Extension hook: can rank candidate boundaries by semantic cohesion.
        self.semantic_boundary_scorer = semantic_boundary_scorer

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
        chunks = self._split_text(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        chunks = self._stabilize_trailing_short_chunk(
            chunks=chunks,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if not chunks:
            return []

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

    def _split_text(self, *, text: str, min_chars: int, max_chars: int) -> list[str]:
        """Route to target heuristic split mode."""
        if self.split_mode == TaskUnitSplitMode.PROGRESSIVE:
            return self._split_text_progressive(
                text=text,
                min_chars=min_chars,
                max_chars=max_chars,
            )
        return self._split_text_semantic_safe(
            text=text,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    def _split_text_semantic_safe(
        self,
        *,
        text: str,
        min_chars: int,
        max_chars: int,
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

        best_index = candidate_indices[0]
        best_score = float("-inf")
        for index in candidate_indices:
            score = self._score_cut_candidate(
                text=text,
                index=index,
                ideal_cut=ideal_cut,
            )
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _score_cut_candidate(self, *, text: str, index: int, ideal_cut: int) -> float:
        """Score one candidate boundary by structure + optional semantic signal."""
        score = -abs(index - ideal_cut) / 8.0
        if index >= 2 and text[index - 2:index] == "\n\n":
            score += 4.0
        prev_char = text[index - 1] if index > 0 else ""
        if prev_char in self._PRIMARY_BOUNDARY_CHARS:
            score += 2.0
        elif prev_char in self._SECONDARY_BOUNDARY_CHARS:
            score += 0.5

        if self.semantic_boundary_scorer is not None:
            try:
                score += float(
                    self.semantic_boundary_scorer.score_boundary(
                        text=text,
                        boundary_index=index,
                    )
                )
            except Exception:
                # Keep split deterministic even if optional scorer is unstable.
                pass
        return score

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
