from dataclasses import dataclass


@dataclass(frozen=True)
class QuizQuestion:
    """Structured quiz question DTO."""

    question_id: str
    question_text: str
    answer_text: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "QuizQuestion":
        """Build one quiz question from dictionary payload."""
        question_id = str(payload.get("question_id", "")).strip()
        question_text = str(payload.get("question_text", "")).strip()
        answer_text = str(payload.get("answer_text", "")).strip()

        if not question_id:
            raise ValueError("quiz question missing non-empty 'question_id'")
        if not question_text:
            raise ValueError(
                f"quiz question '{question_id}' missing non-empty 'question_text'"
            )
        if not answer_text:
            raise ValueError(
                f"quiz question '{question_id}' missing non-empty 'answer_text'"
            )

        return cls(
            question_id=question_id,
            question_text=question_text,
            answer_text=answer_text,
        )
