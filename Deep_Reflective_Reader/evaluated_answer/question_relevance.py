from typing import List

from evaluated_answer.answer_mode import AnswerMode
from search_metadata import SearchMetadata


class QuestionRelevanceEvaluator:
    """Convert retrieval scores into strict/cautious/reject modes."""

    @staticmethod
    def evaluate(results: List[SearchMetadata]) -> AnswerMode:
        """Evaluate retrieval quality and map it to an answer mode.

Args:
    results: Ordered retrieval hits for current query.

Returns:
    Selected answer mode (`strict`, `cautious`, or `reject`) with reason."""
        if not results:
            return AnswerMode(
                level="reject",
                reason="no retrieval results",
            )

        # 你現在用的是 FAISS L2 distance，越小越相關
        best_score: float = min(result.score for result in results)
        print("QuestionRelevanceEvaluator:best_score:", best_score)
        # 這些閾值先作為起點，之後再按你的資料微調
        if best_score < 1.10:
            return AnswerMode(
                level="strict",
                reason=f"strong retrieval match: {best_score:.4f}",
            )

        if best_score < 1.28:
            return AnswerMode(
                level="cautious",
                reason=f"partial retrieval match: {best_score:.4f}",
            )

        return AnswerMode(
            level="reject",
            reason=f"weak retrieval match: {best_score:.4f}",
        )
