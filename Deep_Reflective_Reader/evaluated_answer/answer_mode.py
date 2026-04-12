from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerMode:
    """Answer strictness decision derived from retrieval relevance."""
    level: str   # "strict" | "cautious" | "reject"
    reason: str