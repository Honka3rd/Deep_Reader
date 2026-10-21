from __future__ import annotations

from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass
import re
import logging

from document_structure.common_parse_policy import common_analysis_view
from document_structure.source_mapped_text import MAX_SOURCE_CHARS

from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)
from shared.task_unit_model import TaskUnitContentBlock


@dataclass(frozen=True)
class ArtifactValidityResult:
    """Existence + cache-validity decision for one persisted artifact slot."""

    exists: bool
    cache_valid: bool | None
    invalid_reason: str | None

    @classmethod
    def missing(cls) -> "ArtifactValidityResult":
        return cls(exists=False, cache_valid=None, invalid_reason=None)

    @classmethod
    def valid(cls) -> "ArtifactValidityResult":
        return cls(exists=True, cache_valid=True, invalid_reason=None)

    @classmethod
    def invalid(cls, reason: str) -> "ArtifactValidityResult":
        return cls(exists=True, cache_valid=False, invalid_reason=reason)


@dataclass(frozen=True)
class ReadingInteractionTargetValidity:
    """Target-context validity decision before an interaction spends LLM work."""

    valid: bool
    reason: str | None = None

    @classmethod
    def current(cls) -> "ReadingInteractionTargetValidity":
        return cls(valid=True, reason=None)

    @classmethod
    def stale(cls, reason: str) -> "ReadingInteractionTargetValidity":
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("stale target context requires a reason")
        return cls(valid=False, reason=normalized_reason)


def content_block_source_evidence_validity(
    content_block: TaskUnitContentBlock,
    *,
    source_hash: str | None = None,
    source_structure_version: int | None = None,
) -> ReadingInteractionTargetValidity:
    """
    Validate optional client-observed content-block source evidence.

    Missing client evidence means the target is treated as current for this
    policy slice. When the client supplies evidence, the block metadata must
    carry a matching value; missing or mismatched evidence is surfaced as a
    recoverable stale-target preflight result rather than a hard validation
    error or silent regeneration.
    """
    normalized_source_hash = _normalize_optional_text(source_hash)
    if normalized_source_hash is None and source_structure_version is None:
        return ReadingInteractionTargetValidity.current()

    metadata = content_block.metadata if isinstance(content_block.metadata, dict) else {}
    block_label = content_block.block_id.strip() or "<unknown>"

    if normalized_source_hash is not None:
        block_source_hash = _normalize_optional_text(metadata.get("source_hash"))
        if block_source_hash is None:
            return ReadingInteractionTargetValidity.stale(
                "content_block target source_hash is unavailable for stale-target check: "
                f"content_block_id='{block_label}'"
            )
        if block_source_hash != normalized_source_hash:
            return ReadingInteractionTargetValidity.stale(
                "content_block target source_hash mismatch: "
                f"content_block_id='{block_label}'"
            )

    if source_structure_version is not None:
        block_structure_version = _normalize_optional_int(
            metadata.get("source_structure_version")
        )
        if block_structure_version is None:
            return ReadingInteractionTargetValidity.stale(
                "content_block target source_structure_version is unavailable "
                f"for stale-target check: content_block_id='{block_label}'"
            )
        if block_structure_version != source_structure_version:
            return ReadingInteractionTargetValidity.stale(
                "content_block target source_structure_version mismatch: "
                f"content_block_id='{block_label}'"
            )

    return ReadingInteractionTargetValidity.current()


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ReadingInteractionPreflightResult:
    """Recoverable pre-generation status for reading interaction services."""

    status: str
    reason: str


class ReadingInteractionValidityPolicy:
    """Shared preflight policy for reading interaction generation services."""

    def __init__(
        self,
        *,
        interaction_label: str,
        min_content_chars: int,
        min_alnum_chars: int,
        require_semantic_content: bool = False,
        target_validity_checker: (
            Callable[[ReadingInteractionRequest], ReadingInteractionTargetValidity]
            | None
        ) = None,
    ) -> None:
        if min_content_chars < 0:
            raise ValueError("min_content_chars must be >= 0")
        if min_alnum_chars < 0:
            raise ValueError("min_alnum_chars must be >= 0")
        self._interaction_label = interaction_label
        self._min_content_chars = min_content_chars
        self._min_alnum_chars = min_alnum_chars
        self._require_semantic_content = require_semantic_content
        self._target_validity_checker = (
            target_validity_checker
            if target_validity_checker is not None
            else lambda request: ReadingInteractionTargetValidity.current()
        )

    def preflight(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionPreflightResult | None:
        if request.target.source_needs_review:
            return ReadingInteractionPreflightResult(status="insufficient_content",reason="source_needs_review")
        target_validity = self._target_validity_checker(request)
        if not target_validity.valid:
            reason = target_validity.reason
            if reason is None or not reason.strip():
                raise ValueError("invalid target context requires a reason")
            return ReadingInteractionPreflightResult(
                status="stale_target",
                reason=reason.strip(),
            )

        insufficient_reason = self.insufficient_content_reason(request.target.content)
        if insufficient_reason is not None:
            return ReadingInteractionPreflightResult(
                status="insufficient_content",
                reason=insufficient_reason,
            )
        return None

    def insufficient_content_reason(self, content: str) -> str | None:
        stripped_content = content.strip()
        if len(stripped_content) < self._min_content_chars:
            return f"target content is too short for {self._interaction_label}"

        alnum_count = sum(1 for character in stripped_content if character.isalnum())
        if alnum_count < self._min_alnum_chars:
            return (
                "target content lacks enough readable text for "
                f"{self._interaction_label}"
            )
        if self._require_semantic_content:
            semantic_reason = _semantic_content_insufficient_reason(
                stripped_content,
                interaction_label=self._interaction_label,
            )
            if semantic_reason is not None:
                return semantic_reason
        return None


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_SENTENCE_SIGNAL_PATTERN = re.compile(r"[。！？.!?;；:：,，、]")
_INDEX_LINE_PATTERN = re.compile(
    r"^\s*(?:[\w\u3400-\u9fff][\w\s\u3400-\u9fff:：,，、.-]{0,48})"
    r"(?:\.{2,}|…{1,}|[\s\t]{2,})?\s*\d{1,4}\s*$",
    re.UNICODE,
)
SEMANTIC_GUARD_VERSION = "fragment-evidence-v2"
_LOGGER = logging.getLogger(__name__)


def _semantic_content_insufficient_reason(
    content: str,
    *,
    interaction_label: str,
) -> str | None:
    if len(content) > MAX_SOURCE_CHARS:
        _LOGGER.info("semantic_preflight version=%s decision=analysis_budget_exceeded raw_chars=%d",
                     SEMANTIC_GUARD_VERSION, len(content))
        return f"target content exceeds semantic analysis budget for {interaction_label}"
    try:
        view = common_analysis_view(content)
    except ValueError:
        _LOGGER.info("semantic_preflight version=%s decision=evidence_budget_exceeded raw_chars=%d",
                     SEMANTIC_GUARD_VERSION, len(content))
        return f"target content exceeds semantic evidence budget for {interaction_label}"
    reason = _semantic_view_insufficient_reason(
        view.text, interaction_label=interaction_label,
        repaired=bool(view.removed_ranges), original=content,
    )
    _LOGGER.info(
        "semantic_preflight version=%s decision=%s raw_chars=%d readable_chars=%d "
        "raw_lines=%d fragment_ranges=%d removed_gaps=%d",
        SEMANTIC_GUARD_VERSION, reason or "accepted", len(content),
        sum(char.isalnum() for char in view.text), len(view.line_ranges),
        len(view.confirmed_fragment_ranges), len(view.removed_ranges),
    )
    return reason


def _semantic_view_insufficient_reason(
    content: str, *, interaction_label: str, repaired: bool, original: str,
) -> str | None:
    tokens = _TOKEN_PATTERN.findall(content.lower())
    cjk_chars = _CJK_PATTERN.findall(content)
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]

    dominant_cjk = bool(cjk_chars) and max(Counter(cjk_chars).values()) / len(cjk_chars) >= 0.55
    if _looks_repetitive(tokens) or (repaired and dominant_cjk):
        return f"target content is too repetitive for {interaction_label}"

    original_lines = [line.strip() for line in original.splitlines() if line.strip()]
    if _looks_like_index_or_toc(nonempty_lines) or _looks_like_index_or_toc(original_lines):
        return f"target content looks index-like rather than readable for {interaction_label}"

    # Repaired prose must dominate character mass, not just a few long lines.
    # Glyph-per-line extraction otherwise overwhelms the line-count heuristic.
    prose_chars = sum(len(_CJK_PATTERN.findall(line)) for line in nonempty_lines
                      if len(_CJK_PATTERN.findall(line)) > 12)
    evidenced_prose = repaired and prose_chars > len(cjk_chars) / 2
    if not evidenced_prose and _looks_like_fragmented_headings(nonempty_lines, tokens, len(cjk_chars)):
        return (
            "target content looks like fragmented headings rather than readable "
            f"source text for {interaction_label}"
        )

    if _looks_like_ocr_junk(content, tokens, len(cjk_chars)):
        return f"target content looks like OCR noise for {interaction_label}"

    return None


def _looks_repetitive(tokens: list[str]) -> bool:
    if len(tokens) < 8:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    unique_ratio = len(counts) / len(tokens)
    most_common_ratio = max(counts.values()) / len(tokens)
    return unique_ratio <= 0.25 or most_common_ratio >= 0.55


def _looks_like_index_or_toc(lines: list[str]) -> bool:
    if len(lines) < 4:
        return False
    index_like_count = sum(1 for line in lines if _INDEX_LINE_PATTERN.match(line))
    return index_like_count / len(lines) >= 0.7


def _looks_like_fragmented_headings(
    lines: list[str],
    tokens: list[str],
    cjk_char_count: int,
) -> bool:
    if len(lines) < 5:
        return False
    sentence_line_count = sum(1 for line in lines if _SENTENCE_SIGNAL_PATTERN.search(line))
    average_tokens_per_line = len(tokens) / len(lines)
    average_cjk_per_line = cjk_char_count / len(lines)
    return (
        sentence_line_count / len(lines) <= 0.25
        and average_tokens_per_line <= 3.0
        and average_cjk_per_line <= 12.0
    )


def _looks_like_ocr_junk(
    content: str,
    tokens: list[str],
    cjk_char_count: int,
) -> bool:
    visible_chars = [char for char in content if not char.isspace()]
    if len(visible_chars) < 20:
        return False
    readable_chars = sum(1 for char in visible_chars if char.isalnum())
    symbol_ratio = 1 - (readable_chars / len(visible_chars))
    short_token_ratio = (
        sum(1 for token in tokens if len(token) <= 2) / len(tokens) if tokens else 1.0
    )
    has_sentence_signal = _SENTENCE_SIGNAL_PATTERN.search(content) is not None
    return (
        symbol_ratio >= 0.45
        and short_token_ratio >= 0.7
        and cjk_char_count < 12
        and not has_sentence_signal
    )
