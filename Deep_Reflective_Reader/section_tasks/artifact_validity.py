from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from section_tasks.reading_interaction_service_contracts import (
    ReadingInteractionRequest,
)


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
        self._target_validity_checker = (
            target_validity_checker
            if target_validity_checker is not None
            else lambda request: ReadingInteractionTargetValidity.current()
        )

    def preflight(
        self,
        request: ReadingInteractionRequest,
    ) -> ReadingInteractionPreflightResult | None:
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
        return None
