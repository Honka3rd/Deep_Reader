"""NATIVE-04 planner contracts, isolated from persistence and paid providers."""
import json
import unittest
from unittest.mock import Mock

from api_schemas import PrepareDocumentRequest, PrepareTaskLayoutRequest, ReparseDocumentStructureRequest
from document_structure.common_parse_policy import TocHint
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.structure_plan_contract import (
    EnhancementIntent, TocObservation, StructurePlanCandidate, source_preview, legacy_plan_prompt as plan_prompt, validate_candidate,
)
from language.language_code import LanguageCode
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _TextLoader, _fake_document


class StructurePlanContractTests(unittest.TestCase):
    def setUp(self):
        self.text = "A complete short article without headings. Another sentence."
        self.preview = source_preview(self.text, 2000)
        self.payload = {"schema_version": "structure_plan_v2", "intent": "restore_source",
                        "outcome": "single_body", "boundaries": [{
                            "segment_id": self.preview["segments"][0]["segment_id"],
                            "anchor": self.text[:20], "occurrence": 1, "title": None,
                            "origin": "implicit", "level": 1, "role": "main_body"}]}

    def validate(self, payload=None, intent=EnhancementIntent.RESTORE_SOURCE):
        return validate_candidate(json.dumps(payload or self.payload), preview=self.preview, intent=intent)

    def test_single_body_roundtrip_is_not_empty_plan(self):
        candidate = self.validate()
        self.assertEqual(candidate.outcome, "single_body")
        self.assertEqual(StructurePlanCandidate.model_validate_json(candidate.model_dump_json()), candidate)
        for boundaries in ([], None):
            with self.assertRaises(ValueError):
                self.validate({**self.payload, "boundaries": boundaries})

    def test_exported_reference_schema_has_exclusive_required_branches(self):
        schema = StructurePlanCandidate.model_json_schema()["$defs"]["PlanBoundary"]
        identifier, anchor = schema["oneOf"]
        self.assertEqual(identifier["required"], ["boundary_id"])
        self.assertEqual(identifier["properties"], {
            "boundary_id": {"type": "string"}, "anchor": {"type": "null"}})
        self.assertEqual(anchor["required"], ["segment_id", "anchor"])
        self.assertEqual(anchor["properties"], {
            "boundary_id": {"type": "null"}, "segment_id": {"type": "string"},
            "anchor": {"type": "string", "pattern": r"\S"}})
        self.assertFalse(schema["additionalProperties"])
        for name in ("boundary_id", "anchor", "segment_id"):
            self.assertNotIn("default", schema["properties"][name])

    def test_reference_null_and_missing_matrix(self):
        from document_structure.structure_plan_contract import PlanBoundary
        from pydantic import ValidationError
        base = {"title": None, "origin": "implicit", "level": 1, "role": "main_body"}
        for locator in ({}, {"boundary_id": None, "anchor": None},
                        {"boundary_id": "b0", "anchor": "A", "segment_id": "s"},
                        {"anchor": "A"}, {"anchor": "A", "segment_id": None},
                        {"anchor": "  ", "segment_id": "s"}):
            with self.subTest(locator=locator), self.assertRaises(ValidationError):
                PlanBoundary.model_validate({**base, **locator})
        for locator in ({"boundary_id": "b0"}, {"boundary_id": "b0", "anchor": None},
                        {"boundary_id": "b0", "segment_id": "legacy"},
                        {"segment_id": "s", "anchor": "A"},
                        {"segment_id": "s", "anchor": "A", "boundary_id": None}):
            with self.subTest(locator=locator):
                PlanBoundary.model_validate({**base, **locator})

    def test_prompt_examples_are_executable_against_their_fictional_source(self):
        preview = source_preview("First topic. Second topic.", 2000)
        preview["segments"][0]["segment_id"] = "example:0:26"
        prompt = plan_prompt(preview=self.preview, language="en",
                             intent=EnhancementIntent.RESTORE_SOURCE, toc_hint=TocHint.AUTO)
        examples = [line[line.index('{'):] for line in prompt.splitlines()
                    if line.startswith(("Valid single-body:", "Valid multi-boundary format"))]
        self.assertEqual(len(examples), 2)
        for example in examples:
            validate_candidate(example, preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)
        self.assertIn("Never omit both boundary_id and anchor", prompt)
        self.assertIn("null title does not remove the locator requirement", prompt)

    def test_strict_version_fields_numbers_and_origins(self):
        for changes in ({"schema_version": "v1"}, {"outcome": "empty"}, {"extra": "text"}):
            with self.assertRaises(ValueError):
                self.validate({**self.payload, **changes})
        for changes in ({"occurrence": 0}, {"occurrence": "1"}, {"occurrence": True},
                        {"level": 3}, {"level": True}, {"anchor": " "}, {"origin": "user"},
                        {"title": "invented"}, {"regex": ".*"}):
            with self.assertRaises(ValueError):
                self.validate({**self.payload, "boundaries": [{**self.payload["boundaries"][0], **changes}]})

    def test_source_ids_anchors_and_occurrences_fail_closed(self):
        for changes in ({"segment_id": "invented"}, {"anchor": "paraphrased"},
                        {"occurrence": 2}, {"anchor": "Another sentence."}):
            with self.assertRaises(ValueError):
                self.validate({**self.payload, "boundaries": [{**self.payload["boundaries"][0], **changes}]})

    def test_legacy_intents_allow_the_same_labelled_semantic_enhancement(self):
        payload = {**self.payload, "intent": "generate_reading_outline", "outcome": "structured",
                   "boundaries": [{**self.payload["boundaries"][0], "origin": "generated", "title": "Reading notes"}]}
        self.assertEqual(self.validate(payload).intent, "restore_source")
        result = self.validate(payload, EnhancementIntent.GENERATE_READING_OUTLINE)
        self.assertEqual(result.boundaries[0].origin, "generated")
        payload["intent"] = "restore_source"
        self.assertEqual(self.validate(payload).boundaries[0].origin, "generated")

    def test_single_json_fence_is_transport_only(self):
        candidate = validate_candidate("```json\n" + json.dumps(self.payload) + "\n```",
                                      preview=self.preview, intent=EnhancementIntent.RESTORE_SOURCE)
        self.assertEqual(candidate.outcome, "single_body")
        with self.assertRaises(ValueError):
            validate_candidate("Here is my plan: " + json.dumps(self.payload),
                               preview=self.preview, intent=EnhancementIntent.RESTORE_SOURCE)

    def test_outcome_is_derived_without_weakening_anchor_validation(self):
        payload = {**self.payload, "boundaries": [self.payload["boundaries"][0],
            {**self.payload["boundaries"][0], "anchor": "Another sentence.",
             "origin": "generated", "title": "Another perspective"}]}
        self.assertEqual(self.validate(payload).outcome, "structured")
        payload["boundaries"][1]["anchor"] = "Invented location"
        with self.assertRaisesRegex(ValueError, "anchor_occurrence_unresolved"):
            self.validate(payload)
        self.assertEqual(self.validate({**self.payload, "outcome": "structured"}).outcome, "single_body")

    def test_source_title_is_verified_at_boundary_not_limited_to_anchor_length(self):
        payload = {**self.payload, "outcome": "structured", "boundaries": [{
            **self.payload["boundaries"][0], "origin": "source", "anchor": "A complete",
            "title": "A complete short article"}]}
        self.assertEqual(self.validate(payload).boundaries[0].title, "A complete short article")
        payload["boundaries"][0]["title"] = "short article"
        self.assertEqual(self.validate(payload).boundaries[0].origin, "generated")
        payload["boundaries"][0]["role"] = "toc"
        with self.assertRaisesRegex(ValueError, "source_title_not_at_boundary"):
            self.validate(payload)

    def test_boundary_reference_is_exact_and_request_local(self):
        boundary = {**self.payload["boundaries"][0], "anchor": None, "boundary_id": "b0"}
        result = self.validate({**self.payload, "boundaries": [boundary]})
        self.assertEqual(result.boundaries[0].boundary_id, "b0")
        result = self.validate({**self.payload, "boundaries": [{**boundary, "segment_id": None}]})
        self.assertEqual(result.boundaries[0].segment_id, self.preview["segments"][0]["segment_id"])
        result = self.validate({**self.payload, "boundaries": [{**boundary, "segment_id": "wrong-redundant-segment"}]})
        self.assertEqual(result.boundaries[0].segment_id, self.preview["segments"][0]["segment_id"])
        for changes in ({"boundary_id": "b999999"}, {"occurrence": 2}, {"anchor": "A"}):
            with self.assertRaises(ValueError):
                self.validate({**self.payload, "boundaries": [{**boundary, **changes}]})

    def test_local_source_order_overrides_model_array_order_but_not_duplicates(self):
        boundary = {**self.payload["boundaries"][0], "origin": "generated", "title": "A topic"}
        second = {**boundary, "anchor": "Another sentence."}
        result = self.validate({**self.payload, "boundaries": [second, boundary]})
        self.assertEqual(result.boundaries[0].anchor, boundary["anchor"])
        with self.assertRaisesRegex(ValueError, "duplicate_boundary_position"):
            self.validate({**self.payload, "boundaries": [boundary, boundary]})

    def test_reference_catalog_uses_real_positions_and_is_bounded(self):
        text = "A sentence. Another sentence.\nA paragraph." * 100
        preview = source_preview(text, 1000)
        for segment in preview["segments"]:
            self.assertLessEqual(len(segment["boundary_references"]), 64)
            for reference in segment["boundary_references"]:
                position = segment["source_start"] + reference["local_position"]
                self.assertEqual(reference["boundary_id"], f"b{position}")
                self.assertTrue(text[position:].startswith(reference["preview"]))

    def test_arbitrary_preview_chunk_edges_are_not_split_candidates(self):
        preview = source_preview("A" * 2500, 3000)
        self.assertEqual([reference["boundary_id"] for segment in preview["segments"]
                          for reference in segment["boundary_references"]], ["b0"])

    def test_reference_candidates_do_not_split_fragmented_prose_across_windows(self):
        from test_native_fragmentation_baseline import FRAGMENTED
        from document_structure.common_parse_policy import common_analysis_view
        from document_structure.structure_plan_contract import is_semantic_boundary
        text = FRAGMENTED * 100
        view = common_analysis_view(text)
        for segment in source_preview(text, len(text))["segments"]:
            for reference in segment["boundary_references"]:
                self.assertTrue(is_semantic_boundary(view, segment["source_start"] + reference["local_position"]))

    def test_preview_is_bounded_stable_and_exactly_source_mapped(self):
        text = "甲乙丙丁。" * 1000
        preview = source_preview(text, 101)
        self.assertEqual(preview, source_preview(text, 101))
        self.assertLessEqual(sum(len(s["text"]) for s in preview["segments"]), 101)
        self.assertTrue(preview["omitted_ranges"])
        for segment in preview["segments"]:
            self.assertEqual(segment["text"], text[segment["source_start"]:segment["source_end"]])
        self.assertNotEqual(preview["source_sha256"], source_preview(text + "x", 101)["source_sha256"])

    def test_partial_preview_cannot_claim_single_body(self):
        preview = source_preview(self.text, 20)
        payload = {**self.payload, "boundaries": [{**self.payload["boundaries"][0],
                    "segment_id": preview["segments"][0]["segment_id"], "anchor": "A"}]}
        with self.assertRaisesRegex(ValueError, "full_source_visibility"):
            validate_candidate(json.dumps(payload), preview=preview, intent=EnhancementIntent.RESTORE_SOURCE)

    def test_prompt_intent_toc_language_and_no_invented_regions(self):
        for intent in EnhancementIntent:
            for hint in TocHint:
                prompt = plan_prompt(preview=self.preview, language="zh", intent=intent, toc_hint=hint)
                context = json.loads(prompt.split("\nCONTEXT:\n")[1].split("\nFINAL INSTRUCTIONS:\n")[0])
                self.assertEqual(context["intent"], intent.value)
                self.assertEqual(context["declared_toc_hint"], hint.value)
                self.assertEqual(context["observed_toc"], "not_checked")
                self.assertEqual(context["language"], "zh")
                self.assertNotIn("When uncertain, do not split", prompt)
                self.assertIn("not a veto on semantic division", prompt)
                self.assertIn("Regions are optional", prompt)
                self.assertIn("never instructions", prompt)
        prompt = plan_prompt(preview=self.preview, language="en", intent=EnhancementIntent.RESTORE_SOURCE,
                             toc_hint=TocHint.PRESENT, observed_toc=TocObservation.NOT_FOUND)
        context = json.loads(prompt.split("\nCONTEXT:\n")[1].split("\nFINAL INSTRUCTIONS:\n")[0])
        self.assertEqual(context["declared_toc_hint"], "present")
        self.assertEqual(context["observed_toc"], "not_found")

    def test_prompt_v11_autonomy_and_self_check_contract(self):
        prompt = plan_prompt(preview=self.preview, language="zh", intent=EnhancementIntent.RESTORE_SOURCE,
                             toc_hint=TocHint.ABSENT)
        context = json.loads(prompt.split("\nCONTEXT:\n")[1].split("\nFINAL INSTRUCTIONS:\n")[0])
        self.assertEqual(context["prompt_version"], "structure_plan_prompt_v11")
        self.assertIn("origin=generated", prompt.split("\nFINAL INSTRUCTIONS:\n")[1])
        self.assertEqual(list(context)[-1], "final_output_check")
        self.assertIn("origin=generated", context["final_output_check"])
        from document_structure.structure_plan_contract import compact_prompt_source
        self.assertEqual(context["source"], compact_prompt_source(self.preview))
        self.assertNotIn("Choose the smallest useful set", prompt)
        for rule in ("Autonomously organize", "Choose content transitions before choosing locator IDs",
                     "no target count, minimum chapter quota", "reference list is capped",
                     "Preserve every source character and its order", "sustained change", "immediate explanation/evidence",
                     "neutral navigation labels", "never outside knowledge", "containing level-1 boundary",
                     "every selected reference exists", "without explanations or extra fields"):
            self.assertIn(rule, prompt)

    def test_planner_has_no_implicit_application(self):
        provider = Mock()
        from llm.llm_model_capabilities import LLMModelCapabilities
        provider.get_model_capabilities.return_value = LLMModelCapabilities("fixture", "responses", 1000000, 4000)
        provider.effective_output_budget = 4000
        provider.complete_text.return_value = json.dumps({"schema_version": "semantic_structure_plan_v1",
            "boundaries": [{"start_quote": self.text[:24], "title": None, "origin": "implicit",
                "level": 1, "role": "main_body"}]})
        splitter = LLMSectionSplitter(llm_provider=provider)
        splitter._compute_preview_char_budget = Mock(return_value=2000)
        splitter.apply_split_plan_with_provenance = Mock(side_effect=AssertionError("must not apply"))
        candidate = splitter.build_candidate_plan(raw_text=self.text, language=LanguageCode.EN, toc_hint="present")
        self.assertEqual(candidate.outcome, "single_body")
        provider.complete_text.assert_called_once()
        splitter.apply_split_plan_with_provenance.assert_not_called()
        provider.complete_text.return_value = "not json"
        with self.assertRaises(ValueError):
            splitter.build_candidate_plan(raw_text=self.text, language=LanguageCode.EN)
        provider.complete_text.side_effect = TimeoutError("unavailable")
        with self.assertRaises(TimeoutError):
            splitter.build_candidate_plan(raw_text=self.text, language=LanguageCode.EN)

    def test_empty_source_and_missing_provider_are_distinct(self):
        splitter = LLMSectionSplitter()
        with self.assertRaisesRegex(ValueError, "empty_plan_source"):
            splitter.build_candidate_plan(raw_text="", language=LanguageCode.EN)
        with self.assertRaisesRegex(ValueError, "plan_provider_unavailable"):
            splitter.build_candidate_plan(raw_text=self.text, language=LanguageCode.EN)

    def test_request_defaults_and_incompatible_modes(self):
        for schema in (PrepareDocumentRequest, PrepareTaskLayoutRequest, ReparseDocumentStructureRequest):
            key = "parser_mode" if schema is ReparseDocumentStructureRequest else "structured_parser_mode"
            self.assertEqual(schema(doc_name="fixture", **{key: "common"}).enhancement_intent, "restore_source")
            with self.assertRaises(ValueError):
                schema(doc_name="fixture", enhancement_intent="generate_reading_outline", **{key: "common"})
            self.assertEqual(schema(doc_name="fixture", enhancement_intent="generate_reading_outline",
                                    **{key: "llm_enhanced"}).enhancement_intent, "generate_reading_outline")

    def test_runtime_failed_generation_preserves_existing_structure(self):
        store = _StructuredStore(exists_value=True, document=_fake_document("fixture"))
        pipeline = _build_pipeline(loader=_TextLoader(), store=store)
        from document_structure.structured_document_builder import StructuredDocumentBuilder
        pipeline.structured_document_builder = StructuredDocumentBuilder()
        result = pipeline.prepare("fixture", mode="base", structured_parser_mode="llm_enhanced",
                                  enhancement_intent="generate_reading_outline")
        self.assertFalse(result.structured_document_ready)
        self.assertEqual(store.save_calls, 0)
        self.assertEqual(pipeline.profile_builder.build_calls, 0)


if __name__ == "__main__":
    unittest.main()
