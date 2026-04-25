from dataclasses import dataclass
from typing import Self

from shared.abstract_result import AbstractResult

@dataclass(frozen=True)
class SectionTaskResult(AbstractResult[str]):
    """Unified service result DTO for section task outputs."""

    @classmethod
    def ok(cls, payload: str) -> Self:
        """Build success result with LLM-generated payload."""
        return cls(success=True, payload=payload, reason="")

    @classmethod
    def fail(cls, reason: str) -> Self:
        """Build failure result with explicit reason."""
        normalized_reason = reason.strip() or "unknown section task failure"
        return cls(success=False, payload=None, reason=normalized_reason)

    @classmethod
    def from_llm_error(cls, error: Exception) -> Self:
        """Build failure result from LLM provider/runtime exception."""
        return cls.fail(cls._extract_error_reason(error))

    @staticmethod
    def _extract_error_reason(error: Exception) -> str:
        """Extract best-effort error reason including status code when available."""
        parts: list[str] = []
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code is not None:
            parts.append(f"status={status_code}")

        error_message = str(error).strip() or error.__class__.__name__
        parts.append(error_message)
        return " | ".join(parts)
