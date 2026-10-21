"""Regression shapes observed in private real-book provider responses."""
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.structure_plan_contract import EnhancementIntent, TocObservation
from document_structure.common_parse_policy import TocHint
from document_structure.enhancement_capacity import capacity_identity, require_capacity_identity
from document_structure.enhancement_resources import EnhancementResourceLimits
from document_structure.plan_prompt_budget import EnhancementPromptBudgetError
from language.language_code import LanguageCode
from test_structure_plan_application import Provider


class RepairTests(unittest.TestCase):
    text = "First discussion.\n\nSecond discussion."

    def setUp(self):
        self.good = {"schema_version": "semantic_structure_plan_v1",
            "boundaries": [{"start_quote": "First discussion.", "title": "Opening", "origin": "generated", "level": 1, "role": "main_body"},
                {"start_quote": "Second discussion.", "title": "Next", "origin": "generated", "level": 1, "role": "main_body"}]}
        self.provider = Provider()
        self.splitter = LLMSectionSplitter(llm_provider=self.provider)

    def run_plan(self):
        return self.splitter.split_candidate_with_provenance(raw_text=self.text, language=LanguageCode.EN)

    def test_missing_quote_duplicate_and_false_source_title_recover(self):
        for kind in ("unknown", "duplicate", "source"):
            bad = json.loads(json.dumps(self.good))
            if kind == "unknown":
                bad["boundaries"][1]["start_quote"] = "Invented passage."
            elif kind == "duplicate":
                bad["boundaries"].append({**bad["boundaries"][1], "level": 2})
            else:
                bad["boundaries"][0]["origin"] = "source"
                bad["boundaries"][0]["role"] = "front_matter"
            self.provider.complete_text = Mock(side_effect=[json.dumps(bad), json.dumps(self.good)])
            sections, _ = self.run_plan()
            self.assertEqual("".join(s.content for s in sections), self.text)
            self.assertEqual(self.provider.complete_text.call_count, 2)
            repair = self.provider.complete_text.call_args.args[0]
            self.assertIn("VALIDATION ERRORS", repair)
            if kind == "unknown":
                self.assertIn('"reason": "quote_not_found"', repair)
            if kind == "duplicate":
                self.assertIn('"reason": "duplicate_quote_position"', repair)
            self.assertNotIn("allowed_boundary_ids", repair)

    def test_permanent_invalid_plan_is_not_silently_applied(self):
        self.provider.complete_text = Mock(return_value=json.dumps({"PRIVATE_RESPONSE": "not a plan"}))
        with self.assertLogs("document_structure.llm_section_splitter", level="INFO") as logs:
            with self.assertRaisesRegex(ValueError, "enhancement_candidate_rejected"):
                self.run_plan()
        self.assertEqual(self.provider.complete_text.call_count, 3)
        self.assertNotIn("PRIVATE_RESPONSE", " ".join(logs.output))
        self.assertIn("attempts=3", " ".join(logs.output))

    def test_success_and_transport_failure_do_not_enter_repair(self):
        self.provider.complete_text = Mock(return_value=json.dumps(self.good))
        self.run_plan()
        self.provider.complete_text.assert_called_once()
        self.provider.complete_text = Mock(side_effect=TimeoutError("transport"))
        with self.assertRaisesRegex(ValueError, "enhancement_provider_transient"):
            self.run_plan()
        self.provider.complete_text.assert_called_once()

    def test_repair_resource_gate_blocks_second_call(self):
        _, prompt, _ = self.splitter.assemble_candidate_request(self.text, LanguageCode.EN,
            EnhancementIntent.RESTORE_SOURCE, TocHint.AUTO, TocObservation.NOT_CHECKED)
        self.splitter.resource_limits = EnhancementResourceLimits(max_prompt_bytes=len(prompt.encode()) + 5)
        self.provider.complete_text = Mock(return_value=json.dumps({"unexpected": "x" * 20000}))
        with self.assertRaisesRegex(EnhancementPromptBudgetError, "local_resource_limit"):
            self.run_plan()
        self.provider.complete_text.assert_called_once()

    def test_repair_preserves_confirmation_identity_guard(self):
        _, prompt, budget = self.splitter.assemble_candidate_request(self.text, LanguageCode.EN,
            EnhancementIntent.RESTORE_SOURCE, TocHint.AUTO, TocObservation.NOT_CHECKED)
        identity = capacity_identity(prompt, budget, self.splitter.resource_limits)
        self.provider.complete_text = Mock(side_effect=["invalid", json.dumps(self.good)])
        with require_capacity_identity(identity):
            self.run_plan()
        self.assertEqual(self.provider.complete_text.call_count, 2)
        caps = self.provider.get_model_capabilities()
        def change_model(_):
            self.provider.get_model_capabilities = lambda: replace(caps, model_name="changed")
            return "invalid"
        self.provider.complete_text = Mock(side_effect=change_model)
        with require_capacity_identity(identity), self.assertRaisesRegex(EnhancementPromptBudgetError, "configuration_changed"):
            self.run_plan()
        self.provider.complete_text.assert_called_once()

    def test_internal_application_failure_does_not_retry(self):
        self.provider.complete_text = Mock(return_value=json.dumps(self.good))
        with patch("document_structure.llm_section_splitter.apply_candidate", side_effect=RuntimeError("internal")):
            with self.assertRaisesRegex(ValueError, "unexpected_validation_error"):
                self.run_plan()
        self.provider.complete_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
