from dataclasses import dataclass


@dataclass(frozen=True)
class QuizQuestion:
    """One validated question; provider values must never be string-coerced."""

    question_id: str
    question_text: str
    answer_text: str

    def __post_init__(self) -> None:
        for field_name in ("question_id", "question_text", "answer_text"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"quiz_validation_failed: {field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

    @classmethod
    def from_dict(cls, payload: dict) -> "QuizQuestion":
        if not isinstance(payload, dict):
            raise ValueError("quiz_validation_failed: question must be an object")
        return cls(
            question_id=payload.get("question_id"),
            question_text=payload.get("question_text"),
            answer_text=payload.get("answer_text"),
        )

    @classmethod
    def from_items(cls, payload: list, *, minimum: int = 1) -> list["QuizQuestion"]:
        """Validate the whole result; legacy caches may contain fewer than four items."""
        if not isinstance(payload, list) or len(payload) < minimum:
            raise ValueError(f"quiz_validation_failed: expected at least {minimum} questions")
        questions = [cls.from_dict(item) for item in payload]
        if len({question.question_id for question in questions}) != len(questions):
            raise ValueError("quiz_validation_failed: duplicate question_id")
        return questions
