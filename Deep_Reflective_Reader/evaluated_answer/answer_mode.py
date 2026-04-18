from dataclasses import dataclass

from question.qa_enums import AnswerLevel


@dataclass(frozen=True)
class AnswerMode:
    """Answer strictness decision derived from retrieval relevance."""
    level: AnswerLevel
    reason: str
