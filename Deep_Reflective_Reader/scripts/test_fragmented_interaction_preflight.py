"""INTERACT-01: analysis-only repair and independent service preflight coverage."""

import unittest
from unittest.mock import patch

from test_native_fragmentation_baseline import PROSE, FRAGMENTED
from section_tasks.artifact_validity import (
    _semantic_content_insufficient_reason, ReadingInteractionValidityPolicy,
)
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.reading_target_resolver import ResolvedReadingTarget


class FragmentedInteractionPreflightTests(unittest.TestCase):
    def reason(self, content):
        return _semantic_content_insufficient_reason(content, interaction_label="analysis")

    def test_partial_repair_with_residual_glyph_lines(self):
        # Unconfirmed lines must not outweigh the majority reconstructed prose.
        source = "\n".join(PROSE.replace("\u3002", "", 1)) + "\n\n" + "\n".join(PROSE[:8])
        self.assertIsNone(self.reason(source))

    def test_meaningless_controls_remain_rejected(self):
        negatives = [
            "noise " * 30,
            "\n".join(["Economics", "Markets", "Labor", "Capital", "Trade"]),
            "\n".join(f"Chapter {i} .... {i}" for i in range(1, 12)),
            "x @ # $ % ^ & * " * 8,
            "\n".join("\u7532" * 100 + "\u3002"),
            "\n".join(PROSE * 20),
        ]
        for source in negatives:
            with self.subTest(source=source[:20]):
                self.assertIsNotNone(self.reason(source))

    def test_small_prose_appendix_does_not_rescue_heading_list(self):
        headings = "\n".join(f"Chapter {i} .... {i}" for i in range(60))
        self.assertIsNotNone(self.reason(headings + "\n\n" + FRAGMENTED))

    def test_normal_prose_and_english_are_unchanged(self):
        for source in [PROSE, "The village changed its procedures after reviewing the evidence."]:
            self.assertIsNone(self.reason(source))

    def test_budget_exhaustion_is_explicit_and_bounded(self):
        with patch("section_tasks.artifact_validity.MAX_SOURCE_CHARS", 10):
            self.assertIn("budget", self.reason(PROSE))
        with patch("section_tasks.artifact_validity.common_analysis_view", side_effect=ValueError):
            self.assertIn("evidence budget", self.reason(PROSE))

    def test_diagnostics_do_not_log_source_or_identifiers(self):
        source = "PRIVATE_SOURCE_SENTINEL\n" + FRAGMENTED
        with self.assertLogs("section_tasks.artifact_validity", level="INFO") as logs:
            self.reason(source)
        self.assertNotIn("PRIVATE_SOURCE_SENTINEL", " ".join(logs.output))
        self.assertIn("version=fragment-evidence-v2", " ".join(logs.output))
        self.assertIn("removed_gaps=", " ".join(logs.output))

    def test_all_service_levels_receive_exact_source_and_identity(self):
        levels = ["document", "chapter", "section", "task_unit", "content_block"]
        for level in levels:
            for kind in ["analysis", "quiz", "critical_thinking_session"]:
                with self.subTest(level=level, kind=kind):
                    target = ResolvedReadingTarget(
                        document_id="doc", document_title="Test", target_level=level,
                        target_id="target", content=FRAGMENTED,
                        chapter_id="chapter", section_id="section",
                        task_unit_id="unit", content_block_id="unit:content:0",
                    )
                    request = ReadingInteractionRequest(target=target, interaction_type=kind)
                    received = []

                    def generator(actual, *args):
                        received.append(actual.target)
                        if kind == "quiz":
                            return {"items": [{"type": "short_answer", "question": "What changed?", "answer": "Procedures"}]}
                        if kind == "critical_thinking_session":
                            return {"question": "What evidence supports this change?"}
                        return {"summary": "The procedures reflect collective reasoning.",
                                "reasoning": "The contrast shows how deliberation changes procedure.",
                                "explanation": "The recorded changes supply evidence of revision."}

                    service = (AnalysisInteractionService(generator) if kind == "analysis"
                               else QuizInteractionService(generator) if kind == "quiz"
                               else CriticalThinkingSessionService(generator, lambda *args: {}))
                    artifact = service.generate(request)
                    self.assertEqual(len(received), 1, artifact.reason)
                    self.assertEqual(received[0], target)
                    self.assertEqual(target.content, FRAGMENTED)
                    self.assertNotEqual(artifact.status, "insufficient_content")
                    self.assertEqual(artifact.status, "question_generated" if kind == "critical_thinking_session" else "completed")
                    tiny = ReadingInteractionRequest(
                        target=ResolvedReadingTarget(document_id="doc", document_title="Test",
                            target_level=level, target_id="tiny", content="abc"),
                        interaction_type=kind)
                    self.assertEqual(service.generate(tiny).status, "insufficient_content")
                    self.assertEqual(len(received), 1)

    def test_quantity_thresholds_stay_in_force(self):
        policy = ReadingInteractionValidityPolicy(interaction_label="analysis",
            min_content_chars=20, min_alnum_chars=8, require_semantic_content=True)
        self.assertIn("too short", policy.insufficient_content_reason("abc"))
        self.assertIn("readable", policy.insufficient_content_reason("!" * 30))


if __name__ == "__main__":
    unittest.main()
