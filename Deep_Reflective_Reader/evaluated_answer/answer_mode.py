from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerMode:
    level: str   # "strict" | "cautious" | "reject"
    reason: str