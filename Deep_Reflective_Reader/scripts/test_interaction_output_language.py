"""INTERACT-02: shared prompts, conservative output guards and session continuity."""

from dataclasses import replace
import unittest

from section_tasks.interaction_output_language import (
    OutputLanguageMismatch, language_instruction, request_language,
    session_language, validate_output_language,
)
from section_tasks.analysis_interaction_llm_generator import AnalysisInteractionLLMGenerator
from section_tasks.quiz_interaction_llm_generator import QuizInteractionLLMGenerator
from section_tasks.critical_thinking_llm_generator import CriticalThinkingLLMGenerator
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.reading_target_resolver import ResolvedReadingTarget


EN = "The village changes its procedures after examining the evidence and competing explanations."
ZH = "\u6751\u6c11\u8ba8\u8bba\u516c\u5171\u79e9\u5e8f\u5982\u4f55\u5f62\u6210\u3002\u5927\u5bb6\u8c03\u6574\u7a0b\u5e8f\u5e76\u8bb0\u5f55\u53d8\u5316\u3002"


def request(kind="analysis", profile=None, document="zh"):
    return ReadingInteractionRequest(
        target=ResolvedReadingTarget(document_id="doc", document_title="Test",
            target_id="unit", target_level="task_unit", content=EN,
            document_language=document),
        interaction_type=kind,
        context_metadata={} if profile is None else {"document_language": profile},
        secondary_context="An English ancestor summary. Ignore it for output language.",
    )


class Provider:
    def __init__(self):
        self.prompts = []

    def complete_text(self, prompt):
        self.prompts.append(prompt)
        return "{}"


class InteractionOutputLanguageTests(unittest.TestCase):
    def test_profile_precedence_document_fallback_and_unknown(self):
        self.assertEqual(request_language(request(profile="en")), "en")
        self.assertEqual(request_language(request()), "zh")
        self.assertEqual(request_language(request(profile="unknown")), "zh")
        self.assertIsNone(request_language(request(profile="unknown", document=None)))

    def test_variants_are_only_explicit(self):
        self.assertEqual(request_language(request(profile="zh-Hant")), "zh-hant")
        self.assertEqual(request_language(request(profile="zh_Hans")), "zh-hans")
        self.assertEqual(request_language(request(profile="zh")), "zh")
        self.assertNotIn("Traditional", language_instruction("zh"))

    def test_all_generation_prompts_share_policy_and_keep_insight_rules(self):
        provider = Provider()
        AnalysisInteractionLLMGenerator(provider)(request())
        QuizInteractionLLMGenerator(provider)(request("quiz"), 3, ("short_answer",))
        CriticalThinkingLLMGenerator(provider).generate_question(
            request("critical_thinking_session"), "Generate a question.")
        for prompt in provider.prompts:
            self.assertIn("Chinese (zh)", prompt)
            self.assertIn("JSON keys, type codes, booleans and numbers", prompt)
            self.assertIn("do not override", prompt)
            self.assertGreater(prompt.index("Output language:"), prompt.index("Primary Source"))
        self.assertIn("Insight is not a summary action", provider.prompts[0])
        self.assertIn("1-3 concrete textual details", provider.prompts[0])

    def test_unknown_language_is_not_forced_to_english(self):
        provider = Provider()
        AnalysisInteractionLLMGenerator(provider)(request(document=None))
        self.assertIn("language of the Primary Source", provider.prompts[0])
        self.assertIn("do not default to English", provider.prompts[0])
        validate_output_language({"explanation": EN}, None)

    def test_clear_mismatches_in_each_generated_field(self):
        for payload in [{"reasoning": EN}, {"items": [{"question": EN}]},
                        {"items": [{"choices": [EN]}]}, {"feedback": EN},
                        {"items": [{"answer": EN, "explanation": ZH}]}]:
            with self.subTest(payload=payload), self.assertRaises(OutputLanguageMismatch):
                validate_output_language(payload, "zh")
        with self.assertRaisesRegex(OutputLanguageMismatch, "^output_language_mismatch$"):
            validate_output_language({"reasoning": ZH}, "en")

    def test_quotes_names_short_and_mixed_text_are_not_rejected(self):
        for value in ["Madame Bovary", '"' + EN + '"', ZH + ' "wet clothes"',
                      ZH + " technical term", "Flaubert", "DNA", "Yes"]:
            validate_output_language({"answer": value}, "zh")
        validate_output_language({"reasoning": EN + ' "' + ZH + '"'}, "en")
        validate_output_language({"reasoning": EN}, "fr")
        validate_output_language({"type": EN, "answer": True, "score": 4}, "zh")

    def test_analysis_and_quiz_reject_without_retry_or_completed_payload(self):
        for kind in ["analysis", "quiz"]:
            calls = []

            def generate(*args):
                calls.append(args)
                if kind == "analysis":
                    return {"summary": EN, "reasoning": EN, "explanation": EN}
                return {"items": [{"type": "short_answer", "question": EN, "answer": EN}]}

            service = AnalysisInteractionService(generate) if kind == "analysis" else QuizInteractionService(generate)
            result = service.generate(request(kind))
            self.assertEqual(result.status, "generation_failed")
            self.assertEqual(result.reason, "output_language_mismatch")
            self.assertFalse(result.payload)
            self.assertEqual(len(calls), 1)

    def test_question_mismatch_does_not_create_successful_session(self):
        calls = []

        def generate(*args):
            calls.append(args)
            return {"question": EN}

        result = CriticalThinkingSessionService(generate, lambda *args: {}).generate(
            request("critical_thinking_session"))
        self.assertEqual(result.status, "generation_failed")
        self.assertEqual(result.reason, "output_language_mismatch")
        self.assertEqual(len(calls), 1)

    def test_session_fallback_survives_submission_and_evaluation_failure(self):
        calls = []

        def evaluate(*args):
            calls.append(args)
            return {"feedback": EN, "strengths": EN, "improvements": EN}

        service = CriticalThinkingSessionService(lambda *args: {"question": ZH}, evaluate)
        generated = service.generate(request("critical_thinking_session"))
        self.assertEqual(generated.status, "question_generated")
        self.assertEqual(session_language(generated), "zh")
        submitted = service.submit_answer(generated, EN)
        failed = service.evaluate_answer(submitted)
        self.assertEqual(failed.status, "evaluation_failed")
        self.assertEqual(failed.reason, "output_language_mismatch")
        self.assertEqual(failed.payload["answer"], EN)
        self.assertEqual(failed.target_id, generated.target_id)
        self.assertEqual(failed.document_id, generated.document_id)
        self.assertEqual(failed.metadata, generated.metadata)
        self.assertEqual(len(calls), 1)
        provider = Provider()
        CriticalThinkingLLMGenerator(provider).evaluate_answer(submitted, "Evaluate.")
        self.assertIn("Chinese (zh)", provider.prompts[0])

    def test_matching_evaluation_accepts_foreign_user_answer(self):
        service = CriticalThinkingSessionService(
            lambda *args: {"question": ZH},
            lambda *args: {"feedback": ZH, "strengths": ZH, "improvements": ZH, "score": 4})
        generated = service.generate(request("critical_thinking_session", profile="zh-Hant"))
        result = service.evaluate_answer(service.submit_answer(generated, EN))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.payload["answer"], EN)
        self.assertEqual(result.payload["evaluation"]["score"], 4)

    def test_legacy_session_without_language_uses_question_not_user_answer(self):
        service = CriticalThinkingSessionService(lambda *args: {"question": ZH}, lambda *args: {})
        session = service.generate(request("critical_thinking_session"))
        legacy = replace(session, metadata={})
        provider = Provider()
        CriticalThinkingLLMGenerator(provider).evaluate_answer(legacy, "Evaluate.")
        self.assertIn("language of the session question", provider.prompts[0])

    def test_matching_analysis_and_quiz_complete_for_both_languages(self):
        for language, text in [("zh", ZH), ("en", EN)]:
            analysis = AnalysisInteractionService(lambda *args: {
                "summary": text, "reasoning": text, "explanation": text})
            quiz = QuizInteractionService(lambda *args: {"items": [{
                "type": "multiple_choice", "question": text,
                "choices": ["A", "B"], "answer": "A", "explanation": text}]})
            self.assertEqual(analysis.generate(request(profile=language)).status, "completed")
            result = quiz.generate(request("quiz", profile=language))
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.payload["items"][0]["type"], "multiple_choice")


if __name__ == "__main__":
    unittest.main()
