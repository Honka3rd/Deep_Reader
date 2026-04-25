from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class AbstractResult(ABC, Generic[PayloadT]):
    """Generic abstract result DTO for task/service execution."""

    success: bool
    payload: PayloadT | None
    reason: str

    @classmethod
    @abstractmethod
    def ok(cls, payload: PayloadT) -> Self:
        """Build a success result."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def fail(cls, reason: str) -> Self:
        """Build a failure result."""
        raise NotImplementedError
