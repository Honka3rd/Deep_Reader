from __future__ import annotations

from dataclasses import dataclass

from language.language_code import LanguageCode
from section_tasks.task_unit_boundary_language_registry import (
    TaskUnitBoundaryLanguageConfig,
    TaskUnitBoundaryLanguageRegistry,
)


@dataclass(frozen=True)
class BoundaryNormalizationResult:
    """Normalized cut boundary and diagnostics."""

    normalized_index: int
    boundary_inside_terminal_cluster: bool
    orphan_closing_start_shifted: bool


@dataclass(frozen=True)
class _RightClusterScanResult:
    normalized_index: int
    consumed_closing_char: bool


class SentenceTerminalBoundaryNormalizer:
    """Normalize split boundaries to keep terminal punctuation cluster intact."""

    def __init__(
        self,
        language_registry: TaskUnitBoundaryLanguageRegistry | None = None,
        boundary_cluster_max_extra_chars: int = 8,
    ):
        self.language_registry = language_registry or TaskUnitBoundaryLanguageRegistry()
        self.boundary_cluster_max_extra_chars = max(1, int(boundary_cluster_max_extra_chars))

    def normalize(
        self,
        *,
        text: str,
        boundary_index: int,
        language: LanguageCode | str | None,
    ) -> BoundaryNormalizationResult:
        """Normalize one boundary index to avoid splitting terminal cluster."""
        if not text:
            return BoundaryNormalizationResult(
                normalized_index=0,
                boundary_inside_terminal_cluster=False,
                orphan_closing_start_shifted=False,
            )

        clamped = max(0, min(len(text), int(boundary_index)))
        config = self.language_registry.get_config(language)

        cluster_normalized = self._extend_terminal_cluster(
            text=text,
            boundary_index=clamped,
            config=config,
        )
        inside_cluster = cluster_normalized > clamped

        orphan_fixed = self._absorb_orphan_closing_start(
            text=text,
            boundary_index=cluster_normalized,
            config=config,
        )
        orphan_shifted = orphan_fixed > cluster_normalized
        return BoundaryNormalizationResult(
            normalized_index=orphan_fixed,
            boundary_inside_terminal_cluster=inside_cluster,
            orphan_closing_start_shifted=orphan_shifted,
        )

    def _extend_terminal_cluster(
        self,
        *,
        text: str,
        boundary_index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> int:
        """Extend boundary to include trailing closing punctuation after sentence terminal."""
        if boundary_index <= 0 or boundary_index >= len(text):
            return boundary_index
        if not self._has_terminal_before(
            text=text,
            boundary_index=boundary_index,
            config=config,
        ):
            return boundary_index
        return self._scan_right_closing_cluster(
            text=text,
            boundary_index=boundary_index,
            config=config,
        ).normalized_index

    def _absorb_orphan_closing_start(
        self,
        *,
        text: str,
        boundary_index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> int:
        """Shift right when next unit would start with orphan closing punctuation."""
        if boundary_index <= 0 or boundary_index >= len(text):
            return boundary_index
        scan = self._scan_right_closing_cluster(
            text=text,
            boundary_index=boundary_index,
            config=config,
        )
        # Only shift when the boundary really starts from orphan closing punctuation.
        # Whitespace-only advancement is not considered orphan repair.
        if not scan.consumed_closing_char:
            return boundary_index
        return scan.normalized_index

    def _has_terminal_before(
        self,
        *,
        text: str,
        boundary_index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        """Return whether a sentence/strong terminal exists immediately before boundary cluster."""
        cursor = boundary_index - 1
        consumed = 0
        while cursor >= 0 and consumed < self.boundary_cluster_max_extra_chars:
            char = text[cursor]
            if char.isspace():
                cursor -= 1
                consumed += 1
                continue
            if self._is_closing_char(char=char, config=config):
                cursor -= 1
                consumed += 1
                continue
            return (
                char in config.sentence_terminal_chars
                or char in config.strong_punctuation_chars
            )
        return False

    def _scan_right_closing_cluster(
        self,
        *,
        text: str,
        boundary_index: int,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> _RightClusterScanResult:
        """Scan right over trailing whitespace/closing punctuation cluster."""
        cursor = boundary_index
        consumed = 0
        consumed_closing_char = False
        while cursor < len(text) and consumed < self.boundary_cluster_max_extra_chars:
            char = text[cursor]
            if self._is_closing_char(char=char, config=config):
                consumed_closing_char = True
                cursor += 1
                consumed += 1
                continue
            if char.isspace():
                cursor += 1
                consumed += 1
                continue
            break
        return _RightClusterScanResult(
            normalized_index=cursor,
            consumed_closing_char=consumed_closing_char,
        )

    @staticmethod
    def _is_closing_char(
        *,
        char: str,
        config: TaskUnitBoundaryLanguageConfig,
    ) -> bool:
        return (
            char in config.closing_quote_chars
            or char in config.closing_bracket_chars
        )
