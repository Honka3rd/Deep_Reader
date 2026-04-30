from __future__ import annotations

from dataclasses import dataclass


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

