"""Offline contract tests: abstention is not a successful invented interaction."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.analysis_interaction_llm_generator import AnalysisInteractionLLMGenerator
from section_tasks.quiz_interaction_llm_generator import QuizInteractionLLMGenerator
from section_tasks.critical_thinking_llm_generator import CriticalThinkingLLMGenerator
from section_tasks.interaction_source_sufficiency import SOURCE_SUFFICIENCY_INSTRUCTION
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.reading_target_resolver import ResolvedReadingTarget


REFUSAL = {"status": "insufficient_content", "reason": "primary_source_not_meaningful"}


class SourceSufficiencyTests(unittest.TestCase):
    def request(self, kind):
        return ReadingInteractionRequest(
            target=ResolvedReadingTarget(
                document_id="doc", document_title="A readable title",
                target_level="content_block", target_id="unit:content:7",
                content="3 AHR BY g2% eee os ree BREE us 2 eck SES a Weg : "
                "人自© eg 7 ba ee +e aie tT esage",
            ),
            interaction_type=kind,
        )

    def services(self, response):
        generator = lambda *args: response
        return [AnalysisInteractionService(generator), QuizInteractionService(generator),
                CriticalThinkingSessionService(generator, generator)]

    def test_abstention_has_no_generated_payload(self):
        for response in (REFUSAL, json.dumps(REFUSAL)):
            for service in self.services(response):
                with self.subTest(kind=service.interaction_type, response=response):
                    result = service.generate(self.request(service.interaction_type))
                    self.assertEqual(result.status, "insufficient_content")
                    self.assertEqual(result.reason, "primary_source_not_meaningful")
                    self.assertFalse(result.payload)

    def test_malformed_or_contradictory_refusal_is_failure(self):
        for response in ({"status": "insufficient_content"},
                         dict(REFUSAL, question="Invented question"),
                         dict(REFUSAL, reason="Invented reason")):
            for service in self.services(response):
                with self.subTest(kind=service.interaction_type, response=response):
                    result = service.generate(self.request(service.interaction_type))
                    self.assertEqual(result.status, "generation_failed")
                    self.assertFalse(result.payload)

    def test_all_generation_prompts_include_contract(self):
        prompts = []

        class Provider:
            def complete_text(self, prompt):
                prompts.append(prompt)
                return json.dumps(REFUSAL)

        provider = Provider()
        AnalysisInteractionLLMGenerator(provider)(self.request("analysis"))
        QuizInteractionLLMGenerator(provider)(self.request("quiz"), 3, ("true_false",))
        CriticalThinkingLLMGenerator(provider).generate_question(
            self.request("critical_thinking_session"), "Generate one question."
        )
        self.assertEqual(len(prompts), 3)
        for prompt in prompts:
            self.assertIn(SOURCE_SUFFICIENCY_INSTRUCTION, prompt)


if __name__ == "__main__":
    unittest.main()
