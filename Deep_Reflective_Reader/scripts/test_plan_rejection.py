"""Safe rejection diagnostics and compatible reparse error responses."""
import json
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from document_structure.structure_plan_contract import StructurePlanCandidate
from document_structure.plan_rejection import rejection_diagnostic


class RejectionTests(unittest.TestCase):
    def reference_case(self, locator, occurrence=1):
        from document_structure.structure_plan_contract import source_preview, resolve_candidate_positions
        preview = source_preview("Private source sentence. Another sentence.", None)
        candidate = StructurePlanCandidate.model_validate({
            "schema_version": "structure_plan_v2", "intent": "restore_source", "outcome": "single_body",
            "boundaries": [{"boundary_id": locator, "occurrence": occurrence,
                "title": None, "origin": "implicit", "role": "main_body", "level": 1}]})
        try:
            return resolve_candidate_positions(candidate, preview)
        except ValueError as error:
            self.assertEqual(str(error), "unknown_boundary_reference")
            return rejection_diagnostic(error, "PRIVATE provider response")

    def test_reference_diagnostics_distinguish_membership_and_occurrence(self):
        missing = self.reference_case("b999999")
        occurrence = self.reference_case("b0", 2)
        self.assertEqual(missing["issues"][0]["code"], "reference_not_in_map")
        self.assertEqual(occurrence["issues"][0]["code"], "reference_occurrence_not_one")
        self.assertFalse(missing["issues"][0]["reference_found"])
        self.assertTrue(occurrence["issues"][0]["reference_found"])
        self.assertEqual(missing["issues"][0]["path"], ["boundaries", 0, "boundary_id"])
        self.assertEqual(missing["boundary_count"], 1)
        self.assertGreater(missing["reference_count"], 0)
        self.assertNotIn("999999", json.dumps(missing))

    def test_reference_diagnostics_never_log_untrusted_values(self):
        diagnostic = self.reference_case("PRIVATE_ID_OR_SOURCE")
        self.assertEqual(diagnostic["issues"][0]["locator_form"], "other")
        self.assertNotIn("PRIVATE", json.dumps(diagnostic))
        self.assertNotIn("Private source", json.dumps(diagnostic))

    def test_reference_diagnostics_do_not_relax_validators(self):
        self.assertEqual(self.reference_case("b0"), [0])
        for locator in ("b1", "b-1", "0", "b999999"):
            self.assertEqual(self.reference_case(locator)["reason"], "unknown_boundary_reference")

    def diagnostic(self, locators):
        payload = {"schema_version": "structure_plan_v2", "intent": "restore_source",
                   "outcome": "single_body", "boundaries": [dict(title=None, origin="implicit",
                   role="main_body", level=1, **locators)]}
        response = json.dumps(payload)
        try:
            StructurePlanCandidate.model_validate_json(response)
        except ValidationError as error:
            return rejection_diagnostic(error, response)
        self.fail("invalid fixture unexpectedly accepted")

    def test_missing_and_conflicting_locators_are_distinct_without_values(self):
        missing = self.diagnostic({})
        both = self.diagnostic({"boundary_id": "PRIVATE_ID", "anchor": "PRIVATE_TEXT"})
        self.assertEqual(missing["issues"][0]["path"], ["boundaries", 0])
        self.assertFalse(missing["issues"][0]["locator_present"]["boundary_id"])
        self.assertTrue(both["issues"][0]["locator_present"]["boundary_id"])
        self.assertTrue(both["issues"][0]["locator_present"]["anchor"])
        self.assertNotIn("PRIVATE", json.dumps(both))

    def test_unknown_field_names_and_json_content_are_redacted(self):
        diagnostic = self.diagnostic({"boundary_id": "b0", "PRIVATE_KEY": "PRIVATE_VALUE"})
        self.assertNotIn("PRIVATE", json.dumps(diagnostic))
        self.assertIn("unknown_field", json.dumps(diagnostic))
        try:
            StructurePlanCandidate.model_validate_json("PRIVATE malformed")
        except ValidationError as error:
            diagnostic = rejection_diagnostic(error, "PRIVATE malformed")
        self.assertEqual(diagnostic["stage"], "json")
        self.assertNotIn("PRIVATE", json.dumps(diagnostic))

    def test_reference_and_internal_failures_have_safe_categories(self):
        self.assertEqual(rejection_diagnostic(ValueError("unknown_boundary_reference"), "")["stage"], "reference")
        diagnostic = rejection_diagnostic(RuntimeError("PRIVATE"), "PRIVATE")
        self.assertEqual(diagnostic["reason"], "unexpected_validation_error")
        self.assertNotIn("PRIVATE", json.dumps(diagnostic))

    def test_reparse_route_preserves_envelope_and_maps_categories(self):
        import main
        from fastapi import Response
        from api_schemas import ReparseDocumentStructureRequest
        from app.enhancement_operations import EnhancementOperations
        from section_tasks.reparse_document_structure_result import ReparseDocumentStructureResult
        prefix = "prepare_structured_document_llm_enhanced_rejected:"
        for code, status in {
            "enhancement_candidate_rejected:schema:exactly_one_boundary_reference_required": 422,
            "enhancement_candidate_rejected:schema:json_invalid": 422,
            "enhancement_candidate_rejected:unknown_boundary_reference": 422,
            "enhancement_candidate_rejected:unexpected_validation_error": 500,
            "enhancement_prompt_budget_exceeded": 413,
            "local_resource_limit": 413,
            "enhancement_provider_failed": 502,
            "enhancement_provider_transient": 502,
            "enhancement_provider_unavailable": 503,
            "enhancement_budget_unavailable": 503,
            "enhancement_output_budget_invalid": 503,
            "enhancement_builder_failed": 500,
        }.items():
            with self.subTest(code=code):
                result = ReparseDocumentStructureResult.fail(doc_name="fixture", parser_mode="llm_enhanced", error=prefix+code)
                request = ReparseDocumentStructureRequest(doc_name="fixture", parser_mode="llm_enhanced",
                    discard_interactions=True, confirmation_token="synthetic-route-test")
                operations = EnhancementOperations()
                operations.register(request.model_dump())
                with patch.object(main.section_task_coordinator, "reparse_document_structure", return_value=result), \
                     patch.object(main.enhancement_preflight_service, "operations", operations):
                    response = Response()
                    body = main.reparse_document_structure(request, response)
                self.assertEqual(response.status_code, status)
                self.assertFalse(body.success)
                self.assertEqual(body.error, prefix+code)

    def test_unexpected_execution_exception_does_not_leak(self):
        import main
        from fastapi import Response, HTTPException
        from api_schemas import ReparseDocumentStructureRequest
        from app.enhancement_operations import EnhancementOperations
        for error in (RuntimeError("PRIVATE database detail"), ValueError("PRIVATE source")):
            request = ReparseDocumentStructureRequest(doc_name="fixture", parser_mode="llm_enhanced",
                discard_interactions=True, confirmation_token="synthetic-route-test")
            operations = EnhancementOperations()
            operations.register(request.model_dump())
            with patch.object(main.section_task_coordinator, "reparse_document_structure", side_effect=error), \
                 patch.object(main.enhancement_preflight_service, "operations", operations):
                with self.assertRaises(HTTPException) as caught:
                    main.reparse_document_structure(request, Response())
                self.assertEqual(caught.exception.status_code, 500)
                self.assertEqual(caught.exception.detail, "enhancement_execution_failed")


if __name__ == "__main__":
    unittest.main()
