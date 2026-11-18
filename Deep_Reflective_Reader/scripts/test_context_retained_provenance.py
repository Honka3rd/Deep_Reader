"""CONTEXT-02/03: provenance describes retained content, not rejected candidates."""
from dataclasses import asdict
import re
import unittest

from context.artifact_aware_context import ArtifactAwareContextBuilder, ArtifactContextSummary
from context.reading_interaction_context import ReadingInteractionContextBuilder
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.reading_target_resolver import ResolvedReadingTarget


SOURCE = (
    "Opening evidence line one.\nOpening evidence line two.\nOpening evidence line three.\n\n"
    "Middle evidence concerns a separate source topic. It is neither present nor referenced in the opening passage.\n\n"
    "Final evidence contains another independent source discussion. It is never part of the earlier source passages."
)


def target(content=SOURCE):
    return ResolvedReadingTarget(
        document_id="synthetic.txt", document_title="Synthetic", target_level="section",
        target_id="s1", section_id="s1", chapter_id="c1", content=content,
    )


def artifact(kind="summary", **kwargs):
    return ArtifactContextSummary(
        artifact_id="artifact-" + kind, artifact_type=kind,
        target_level="task_unit", target_id="u1", **kwargs,
    )


class RetainedPrimaryEvidenceTests(unittest.TestCase):
    def check_primary(self, content, budget, expected_ids):
        source = target(content)
        snapshot = asdict(source)
        result = ReadingInteractionContextBuilder().build_context(source, max_context_tokens=budget)
        self.assertEqual(result.evidence_ids, expected_ids)
        self.assertEqual(result.to_metadata()["evidence_ids"], expected_ids)
        self.assertEqual(result.target_level, source.target_level)
        self.assertEqual(result.target_id, source.target_id)
        self.assertEqual(asdict(source), snapshot)
        self.assertNotIn("context_text", result.to_metadata())
        self.assertLessEqual(result.used_context_tokens, budget)
        return result

    def test_multiline_first_chunk_does_not_invent_other_evidence(self):
        result = self.check_primary(SOURCE, 40, ["section:s1#chunk-1"])
        self.assertEqual(result.context_text, "[section:s1#chunk-1] " + SOURCE.split("\n\n")[0])
        self.assertTrue(result.truncated)

    def test_single_and_multiline_selected_entries_keep_coverage_order(self):
        for multiline in (False, True):
            with self.subTest(multiline=multiline):
                paragraphs = [(chr(65 + i) * 40 + ("\n" if multiline else " ") + chr(65 + i) * 40) for i in range(5)]
                result = self.check_primary("\n\n".join(paragraphs), 65, ["section:s1#chunk-1", "section:s1#chunk-3"])
                self.assertEqual(result.context_text,
                                 "[section:s1#chunk-1] " + paragraphs[0] + "\n[section:s1#chunk-3] " + paragraphs[2])
                self.assertTrue(result.truncated)

    def test_clipped_first_multiline_entry_has_only_its_own_identity(self):
        first = "A" * 60 + "\n" + "B" * 60 + "\n" + "C" * 30
        result = self.check_primary(first + "\n\n" + "D" * 150, 40, ["section:s1#chunk-1"])
        self.assertTrue(result.context_text.startswith("[section:s1#chunk-1] " + "A" * 60 + "\n"))
        self.assertLess(len(result.context_text), len("[section:s1#chunk-1] " + first))
        self.assertEqual(re.findall(r"\[(section:s1#chunk-\d+)\]", result.context_text), result.evidence_ids)
        self.assertTrue(result.truncated)

    def test_zero_and_full_target_metadata_controls(self):
        empty = self.check_primary(SOURCE, 0, [])
        self.assertEqual((empty.context_text, empty.used_context_tokens), ("", 0))
        self.assertTrue(empty.truncated)
        full = self.check_primary(SOURCE, 1000, ["section:s1"])
        self.assertEqual(full.context_text, SOURCE)
        self.assertFalse(full.truncated)


class RetainedArtifactSignalTests(unittest.TestCase):
    def setUp(self):
        self.builder = ArtifactAwareContextBuilder()
        self.neutral = artifact()

    def assert_metadata(self, result, retained, dedup=False, abstraction=False):
        metadata = result.to_metadata()
        self.assertEqual(metadata["referenced_artifact_ids"], [a.artifact_id for a in retained])
        self.assertEqual(metadata["referenced_artifact_types"], [a.artifact_type for a in retained])
        self.assertEqual(metadata["referenced_artifact_target_levels"], [a.target_level for a in retained])
        self.assertEqual(metadata["coverage_counts"], {"task_unit": len(retained)} if retained else {})
        self.assertEqual(metadata["deduplication_hint_applied"], dedup)
        self.assertEqual(metadata["abstraction_hint_applied"], abstraction)
        self.assertNotIn("context_text", metadata)
        self.assertNotIn("payload", metadata)
        if not retained:
            self.assertEqual(result.context_text, "")

    def test_all_pruned_signals_and_header_only_clear_flags(self):
        signals = [artifact("quiz", concepts=("evidence",)), artifact("analysis", focus="Local claim", outcome="Supported")]
        snapshot = [asdict(a) for a in signals]
        for budget in (0, len("Secondary lower-level artifact context:")):
            with self.subTest(budget=budget):
                result = self.builder.build_secondary_context(signals, max_context_chars=budget)
                self.assert_metadata(result, [])
                self.assertEqual(result.artifact_context_mode, "pruned")
                self.assertEqual(result.artifact_context_pruned_reason, "budget_pruned=2")
        self.assertEqual([asdict(a) for a in signals], snapshot)

    def test_only_removed_signal_cannot_enable_flags(self):
        signals = [artifact("quiz"), artifact("analysis"), artifact("critical_thinking_session"),
                   artifact("other", concepts=("evidence",)), artifact("other", focus="Focus"), artifact("other", outcome="Outcome")]
        budget = len(self.builder.build_secondary_context([self.neutral]).context_text)
        for signal in signals:
            with self.subTest(signal=asdict(signal)):
                result = self.builder.build_secondary_context([self.neutral, signal], max_context_chars=budget)
                self.assert_metadata(result, [self.neutral])
                self.assertEqual(result.artifact_context_pruned_reason, "budget_pruned=1")
                self.assertNotIn(signal.artifact_type + ";", result.context_text)

    def test_retained_signals_survive_pruning_and_whitespace_is_not_a_signal(self):
        for signal, dedup, abstraction in ((artifact("quiz"), True, False),
                                           (artifact("analysis"), False, True),
                                           (artifact("other", concepts=(" term ",), focus="focus"), True, True),
                                           (artifact("other", concepts=(" ",), focus=" ", outcome=" "), False, False)):
            with self.subTest(signal=asdict(signal)):
                budget = len(self.builder.build_secondary_context([signal]).context_text)
                result = self.builder.build_secondary_context([signal, self.neutral], max_context_chars=budget)
                self.assert_metadata(result, [signal], dedup, abstraction)
                self.assertEqual(result.artifact_context_pruned_reason, "budget_pruned=1")

    def test_no_budget_and_count_pruning_controls(self):
        quiz, analysis = artifact("quiz"), artifact("analysis")
        result = self.builder.build_secondary_context([quiz, analysis])
        self.assert_metadata(result, [quiz, analysis], True, True)
        self.assertIsNone(result.artifact_context_pruned_reason)
        limited = self.builder.build_secondary_context([self.neutral, quiz, analysis], max_artifacts=1)
        self.assert_metadata(limited, [self.neutral])
        self.assertEqual(limited.artifact_context_pruned_reason, "skipped=2")
        self.assert_metadata(self.builder.build_secondary_context([], max_context_chars=0), [])

    def test_quiz_and_critical_services_use_only_retained_signal_metadata(self):
        for kind, signal in (("quiz", artifact("quiz")), ("critical_thinking_session", artifact("analysis"))):
            for mode in ("empty", "neutral", "retained"):
                with self.subTest(kind=kind, mode=mode):
                    candidates = [self.neutral, signal] if mode == "neutral" else [signal]
                    budget = (0 if mode == "empty" else len(self.builder.build_secondary_context([self.neutral]).context_text)) if mode != "retained" else None
                    context = self.builder.build_secondary_context(candidates, max_context_chars=budget)
                    seen = []
                    def generate(request, *args):
                        seen.append(request)
                        if kind == "quiz":
                            return {"items": [{"type": "short_answer", "question": "What does the source discuss?", "answer": "Distinct evidence."}]}
                        return {"question": "How can the separate claims be compared using evidence?"}
                    service = (QuizInteractionService(generate, artifact_context_provider=lambda request: context) if kind == "quiz"
                               else CriticalThinkingSessionService(generate, lambda *args: self.fail("unexpected evaluation"), artifact_context_provider=lambda request: context))
                    request = ReadingInteractionRequest(target=target(), interaction_type=kind)
                    snapshot = asdict(request.target)
                    returned = service.generate(request)
                    self.assertEqual(returned.status, "completed" if kind == "quiz" else "question_generated")
                    self.assertEqual(len(seen), 1)
                    self.assertEqual(asdict(seen[0].target), snapshot)
                    self.assertEqual(asdict(request.target), snapshot)
                    guidance_key = "quiz_deduplication" if kind == "quiz" else "critical_thinking_continuity"
                    self.assertEqual(guidance_key in returned.metadata["context"], mode == "retained")
                    self.assertEqual(returned.metadata["artifact_reference"], context.to_metadata())
                    if mode != "retained":
                        self.assertEqual(seen[0].secondary_context, context.context_text)
                    else:
                        self.assertGreater(len(seen[0].secondary_context), len(context.context_text))


if __name__ == "__main__":
    unittest.main()
