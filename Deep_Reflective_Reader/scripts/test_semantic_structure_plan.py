"""Model-facing semantic contract, exact backend resolution and output isolation."""
import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError
from document_structure.common_parse_policy import TocHint, common_analysis_view
from document_structure.semantic_structure_plan import (
    SemanticPlan, semantic_source_preview, semantic_plan_prompt, validate_semantic_candidate, semantic_response_schema,
)
from document_structure.structure_plan_contract import EnhancementIntent, TocObservation
from document_structure.structure_plan_application import apply_candidate
from document_structure.plan_rejection import rejection_diagnostic
from document_structure.plan_prompt_metrics import prompt_metrics
from document_structure.plan_prompt_budget import enforce_prompt_budget
from document_structure.semantic_plan_provider import SemanticPlanProvider
from llm.llm_model_capabilities import LLMModelCapabilities
from llm.openai_llm_provider import OpenAILLMProvider


def boundary(quote, **fields):
    return {"start_quote": quote, "title": "Navigation", "origin": "generated",
            "level": 1, "role": "main_body", **fields}


class SemanticTests(unittest.TestCase):
    def test_source_title_downgrade_reports_located_boundary_error(self):
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1", "boundaries": [
            boundary("interior phrase", origin="source", title="Invented title")]})
        with self.assertRaisesRegex(ValueError, "generated_boundary_inside_prose") as caught:
            validate_semantic_candidate(payload,
                preview=semantic_source_preview("Opening with an interior phrase in this sentence."),
                intent=EnhancementIntent.RESTORE_SOURCE)
        diagnostic = rejection_diagnostic(caught.exception, payload)
        self.assertEqual(diagnostic["stage"], "quote")
        self.assertEqual(diagnostic["issues"][0]["path"], ["boundaries", 0, "start_quote"])

    def resolve(self, text, boundaries):
        preview = semantic_source_preview(text)
        candidate = validate_semantic_candidate(json.dumps({"schema_version": "semantic_structure_plan_v1",
            "boundaries": boundaries}), preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        return apply_candidate(text, candidate, preview)[0]

    def test_no_model_ids_offsets_or_occurrences(self):
        good = boundary("Opening.")
        for field in ("boundary_id", "segment_id", "occurrence", "char_start", "section_id", "anchor", "preceding_quote"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SemanticPlan.model_validate({"schema_version": "semantic_structure_plan_v1",
                    "boundaries": [{**good, field: 1}]})

    def test_prompt_is_full_source_once_without_reference_map(self):
        text = "PRIVATE_UNIQUE_BODY.\nA later discussion."
        prompt = semantic_plan_prompt(preview=semantic_source_preview(text), language="en",
            intent=EnhancementIntent.RESTORE_SOURCE, toc_hint=TocHint.AUTO,
            observed_toc=TocObservation.NOT_CHECKED)
        self.assertEqual(prompt.count("PRIVATE_UNIQUE_BODY"), 1)
        for field in ("boundary_id", "segment_id", "boundary_references", "char_start", "occurrence"):
            self.assertNotIn('"' + field + '"', prompt)
        metrics = prompt_metrics(prompt)
        self.assertEqual(metrics["reference_count"], 0)
        self.assertEqual(metrics["visible_source_chars"], len(text))
        self.assertEqual(sum(metrics["component_chars"].values()), len(prompt))

    def test_backend_generates_ids_and_preserves_exact_body(self):
        text = "Opening discussion.\n\nLater discussion."
        sections = self.resolve(text, [boundary("Opening discussion."), boundary("Later discussion.", level=2)])
        self.assertEqual([s.char_start for s in sections], [0, text.index("Later")])
        self.assertEqual(len({s.section_id for s in sections}), 2)
        self.assertEqual("".join(s.content for s in sections), text)

    def test_quote_across_old_window_boundary_has_no_segment_dependency(self):
        text = "A" * 1995 + ".\nA distinctive transition across the old window."
        sections = self.resolve(text, [boundary("A distinctive transition")])
        self.assertEqual(sections[-1].char_start, text.index("A distinctive"))
        self.assertEqual("".join(s.content for s in sections), text)

    def test_repeated_quote_requires_distinct_exact_continuation(self):
        text = "First. Repeated opening. One.\nSecond. Repeated opening. Two."
        with self.assertRaisesRegex(ValueError, "quote_ambiguous"):
            self.resolve(text, [boundary("Repeated opening.")])
        sections = self.resolve(text, [boundary("Repeated opening. Two.")])
        self.assertEqual(sections[-1].char_start, text.rindex("Repeated"))
        with self.assertRaisesRegex(ValueError, "quote_not_found"):
            self.resolve(text, [boundary("Repeated opening. Three.")])

    def test_no_fuzzy_quote_or_whitespace_repair(self):
        for quote in ("A different topic.", "A  topic.", "A topic!", "A topic.\\nNext."):
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, "quote_not_found"):
                self.resolve("A topic.\nNext.", [boundary(quote)])

    def test_fragmented_readable_view_maps_to_canonical(self):
        text = "中\n文\n測\n試\n。另一個話題。"
        view = common_analysis_view(text)
        sections = self.resolve(text, [boundary(view.text[:view.text.index("。") + 1]), boundary("另一個話題。")])
        self.assertEqual("".join(s.content for s in sections), text)

    def test_duplicate_and_multiple_bad_quotes_report_all_safe_indices(self):
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1", "boundaries": [
            boundary("Opening."), boundary("Opening."), boundary("PRIVATE_MISSING"), boundary("PRIVATE_MISSING_TWO")]})
        try:
            validate_semantic_candidate(payload, preview=semantic_source_preview("Opening."),
                intent=EnhancementIntent.RESTORE_SOURCE)
        except ValueError as error:
            diagnostic = rejection_diagnostic(error, payload)
        else:
            self.fail("invalid proposal accepted")
        self.assertEqual([issue["path"][1] for issue in diagnostic["issues"]], [1, 2, 3])
        self.assertNotIn("PRIVATE", json.dumps(diagnostic))

    def test_whole_body_start_is_structurally_zero_not_arbitrary_first_match(self):
        text = "Repeated. Repeated."
        sections = self.resolve(text, [boundary("Repeated.", title=None, origin="implicit")])
        self.assertEqual(sections[0].content, text)
        with self.assertRaisesRegex(ValueError, "quote_ambiguous"):
            self.resolve(text, [boundary("Repeated.")])

    def test_role_and_quote_failures_are_reported_together(self):
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1", "boundaries": [
            boundary("Missing opening.", role="back_matter")]})
        with self.assertRaises(ValueError) as caught:
            validate_semantic_candidate(payload, preview=semantic_source_preview("Opening."),
                                        intent=EnhancementIntent.RESTORE_SOURCE)
        codes = [issue["code"] for issue in caught.exception.diagnostic["issues"]]
        self.assertEqual(codes, ["generated_navigation_is_not_source_region", "quote_not_found"])

    def test_inside_word_and_legacy_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, "generated_boundary_inside_prose") as caught:
            self.resolve("Opening discussion.", [boundary("discussion.")])
        self.assertEqual(caught.exception.diagnostic["issues"][0]["path"],
                         ["boundaries", 0, "start_quote"])
        with self.assertRaises(ValidationError):
            validate_semantic_candidate('{"schema_version":"structure_plan_v2","boundaries":[]}',
                preview=semantic_source_preview("Opening."), intent=EnhancementIntent.RESTORE_SOURCE)

    def test_unicode_paragraph_indentation_is_not_inside_prose(self):
        for indentation in ("\u3000", "\u00a0", "\u2002\u2003"):
            text = "Opening.\n" + indentation + "Next paragraph."
            sections = self.resolve(text, [boundary("Opening."), boundary("Next paragraph.")])
            self.assertEqual("".join(section.content for section in sections), text)
            self.assertEqual(len(sections), 2)
            with self.assertRaisesRegex(ValueError, "generated_boundary_inside_prose"):
                self.resolve("Opening" + indentation + "continuation.", [boundary("continuation.")])

    def test_provider_output_override_is_request_local_and_budgeted(self):
        provider = object.__new__(OpenAILLMProvider)
        provider.model_capabilities = LLMModelCapabilities("test", "responses", 1000000, 32768)
        provider.effective_output_budget = 3276
        provider.llm = Mock()
        provider.llm.complete.return_value.text = "accepted"
        wrapper = SemanticPlanProvider(provider)
        self.assertEqual(enforce_prompt_budget("text", wrapper)["output_reserve"], 32768)
        self.assertEqual(wrapper.complete_text("text"), "accepted")
        provider.llm.complete.assert_called_with("text", max_output_tokens=32768, timeout=300.0,
            text={"format": {"type": "json_schema", "name": "semantic_reading_plan",
                "strict": True, "schema": semantic_response_schema()}})
        self.assertEqual(provider.effective_output_budget, 3276)
        provider.complete_text("interaction")
        provider.llm.complete.assert_called_with("interaction", max_output_tokens=3276)
        for limit in (0, True, 32769):
            with self.assertRaisesRegex(ValueError, "invalid_output_budget"):
                provider.complete_text_with_output_budget("invalid", limit)

    def test_structured_output_schema_and_extra_budget_are_explicit(self):
        schema = semantic_response_schema()
        for obj in (schema, schema["$defs"]["SemanticBoundary"]):
            self.assertEqual(set(obj["required"]), set(obj["properties"]))
            self.assertFalse(obj["additionalProperties"])
        provider = object.__new__(OpenAILLMProvider)
        provider.model_capabilities = LLMModelCapabilities("test", "responses", 1000000, 32768)
        provider.effective_output_budget = 3276
        provider.count_input_tokens = Mock(return_value=123)
        wrapped = SemanticPlanProvider(provider)
        enforce_prompt_budget("prompt", wrapped)
        self.assertEqual(provider.count_input_tokens.call_args.args[0], "prompt" + wrapped.request_schema_text)

    def test_repair_excerpts_never_authorize_or_rewrite_a_quote(self):
        from document_structure.semantic_quote_evidence import quote_repair_evidence
        text = "Opening discussion. More evidence."
        preview = semantic_source_preview(text)
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1",
            "boundaries": [boundary("Opening discussin")]})
        with self.assertRaises(ValueError) as caught:
            validate_semantic_candidate(payload, preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        evidence = quote_repair_evidence(payload, preview, rejection_diagnostic(caught.exception, payload))
        self.assertEqual(evidence[0]["source_excerpts"], [text])
        self.assertEqual(preview["segments"][0]["boundary_references"], [])
        with self.assertRaisesRegex(ValueError, "quote_not_found"):
            validate_semantic_candidate(payload, preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        for field in ("boundary_id", "segment_id", "char_start", "offset"):
            self.assertNotIn(field, json.dumps(evidence))

    def test_ambiguous_evidence_is_not_arbitrarily_truncated(self):
        from document_structure.semantic_quote_evidence import quote_repair_evidence
        text = "Opening discussion. " * 10
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1",
            "boundaries": [boundary("Opening discussin")]})
        diagnostic = {"issues": [{"code": "quote_not_found", "path": ["boundaries", 0, "start_quote"]}]}
        self.assertEqual(quote_repair_evidence(payload, semantic_source_preview(text), diagnostic), [])

    def test_repeated_bad_quotes_do_not_multiply_repair_evidence(self):
        from document_structure.semantic_quote_evidence import quote_repair_evidence, semantic_repair_prompt
        payload = json.dumps({"schema_version": "semantic_structure_plan_v1",
            "boundaries": [boundary("Opening discussin")] * 128})
        diagnostic = {"issues": [{"code": "quote_not_found", "path": ["boundaries", i, "start_quote"]}
                                 for i in range(128)]}
        evidence = quote_repair_evidence(payload, semantic_source_preview("Opening discussion."), diagnostic)
        self.assertEqual(len(evidence), 1)
        prompt = semantic_repair_prompt({"rejected_proposal": payload,
            "validation": diagnostic, "source_evidence_for_correction": evidence})
        self.assertLess(len(prompt), 100000)

    def test_local_repair_identifies_bad_quotes_without_technical_locators(self):
        from document_structure.semantic_quote_evidence import semantic_repair_prompt
        proposal = {"schema_version": "semantic_structure_plan_v1", "boundaries": [
            boundary("Opening."), boundary("Invented opening.")]}
        correction = {"rejected_proposal": json.dumps(proposal), "validation": {
            "issues": [{"code": "quote_not_found", "path": ["boundaries", 1, "start_quote"]}]}}
        prompt = semantic_repair_prompt(correction)
        self.assertIn('"rejected_quote": "Invented opening."', prompt)
        self.assertIn("omit this boundary", prompt)
        correction["source_evidence_for_correction"] = [{"proposed_title": "Heading",
            "rejected_quote": "Invented opening.", "source_excerpts": ["Actual opening. Rest."]}]
        prompt = semantic_repair_prompt(correction)
        self.assertIn("Replace start_quote with ONE exact_opening_options value", prompt)
        self.assertIn("Actual opening. Rest.", prompt)
        self.assertEqual(len(proposal["boundaries"]), 2)
        for field in ("boundary_id", "segment_id", "char_start"):
            self.assertNotIn(field, prompt)


if __name__ == "__main__":
    unittest.main()
