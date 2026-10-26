"""Pure cleanup contracts; no provider, extraction or repository dependencies."""

from dataclasses import replace
import json
import unittest

from pydantic import ValidationError

from doc_loaders.extraction_identity import ExtractionIdentity, ExtractionResult, text_digest
from document_structure.ocr_cleanup_contract import (
    ActualOcrEvidence, BoundCleanupProposal, CanonicalOcrSpan, CleanupBinding,
    CleanupEligibility, CleanupOutcome, MAX_CONTEXT_CHARS,
    MAX_RESPONSE_BYTES, cleanup_eligibility,
    cleanup_response_schema, parse_cleanup_proposal,
)


def source(mode="ocr", source_format="pdf"):
    return ExtractionResult(ExtractionIdentity.create(
        source_sha256="a" * 64, source_format=source_format, extractor_id="fixture",
        extraction_version="v1", mode=mode, settings={}), "Opening.\nnoise?!\nBody.")


def item(**values):
    return {"action": "delete", "quote": "noise?!", "reason": "ocr_garbage", **values}


def proposal(items=()):
    return parse_cleanup_proposal(json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": items}))


def evidence(value, spans=None, **values):
    return ActualOcrEvidence(schema_version="cleanup_ocr_evidence_v1",
        extraction_identity=value.identity, text_sha256=value.text_sha256,
        spans=(CanonicalOcrSpan(char_start=9, char_end=16, page_index=1),) if spans is None else spans,
        **values)


def binding(value, **values):
    return CleanupBinding(**{
        "operation_id": "op-1", "namespace": "test", "document_id": "book",
        "extraction_identity": value.identity, "text_sha256": value.text_sha256,
        "text_length": value.text_length, "structure_version": 1,
        "structure_fingerprint": "b" * 64, "expected_reading_revision": 0,
        "source_representation": "readable_full_source_v1", "prompt_version": "test-v1",
        "policy_version": "ocr_cleanup_policy_v1", "consent_scope": "ocr_cleanup_reading_overlay_v1",
        **values})


class CleanupProposalTests(unittest.TestCase):
    def test_empty_is_valid_and_immutable(self):
        plan = proposal()
        self.assertEqual(plan.items, ())
        with self.assertRaises(ValidationError):
            plan.items = ()

    def test_exact_unicode_whitespace_not_normalized(self):
        quote = " \u4e2d\u6587\r\n\U0001f600e\u0301 "
        plan = proposal([item(quote=quote, prefix_quote="\n", suffix_quote=" ")])
        self.assertEqual(plan.items[0].quote, quote)
        self.assertEqual(plan.items[0].prefix_quote, "\n")

    def test_unknown_model_fields_rejected(self):
        for field in ("block_id", "operation_id", "char_start", "char_end", "occurrence",
                      "source_hash", "replacement", "confidence", "page_index", "binding"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "^cleanup_plan_invalid$"):
                proposal([item(**{field: "private-value"})])

    def test_action_reason_and_strict_types(self):
        for change in ({"action": "replace"}, {"action": "insert"}, {"action": "reorder"},
                       {"reason": "unfamiliar_language"}, {"quote": 23}, {"quote": True},
                       {"quote": " "}, {"quote": ""}, {"prefix_quote": ""}, {"quote": "\ud800"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                proposal([item(**change)])

    def test_large_quotes_and_batches_keep_context_boundaries(self):
        quote = "\U0001f600" * 513
        self.assertEqual(proposal([item(quote=quote)]).items[0].quote, quote)
        self.assertEqual(len(proposal([item()] * 129).items), 129)
        proposal([item(prefix_quote="x" * MAX_CONTEXT_CHARS, suffix_quote="y" * MAX_CONTEXT_CHARS)])
        for items in ([item(prefix_quote="x" * (MAX_CONTEXT_CHARS + 1))],
                      [item(suffix_quote="x" * (MAX_CONTEXT_CHARS + 1))]):
            with self.assertRaises(ValueError):
                proposal(items)

    def test_bad_envelope_and_duplicate_json_keys(self):
        for payload in ('{"schema_version":"ocr_cleanup_plan_v1","items":[],"items":[]}',
                        '{"schema_version":"ocr_cleanup_plan_v1","items":[],"binding":{}}',
                        '{"schema_version":"old","items":[]}', '{}', 'null', '[]',
                        '```json\n{}\n```', '{"PRIVATE_SOURCE":',
                        '{"schema_version":"ocr_cleanup_plan_v1","items":[{"action":"delete",'
                        '"quote":"a","quote":"b","reason":"ocr_garbage"}]}'):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "^cleanup_plan_invalid$"):
                parse_cleanup_proposal(payload)

    def test_response_byte_limit_and_safe_errors(self):
        valid = '{"schema_version":"ocr_cleanup_plan_v1","items":[]}'
        parse_cleanup_proposal(valid + " " * (MAX_RESPONSE_BYTES - len(valid)))
        with self.assertRaisesRegex(ValueError, "^cleanup_response_too_large$"):
            parse_cleanup_proposal(valid + " " * MAX_RESPONSE_BYTES)
        with self.assertRaisesRegex(ValueError, "^cleanup_plan_invalid$"):
            parse_cleanup_proposal(None)

    def test_provider_schema_matches_model_ownership(self):
        schema = cleanup_response_schema()
        self.assertEqual(set(schema["properties"]), {"schema_version", "items"})
        self.assertNotIn("maxItems", schema["properties"]["items"])
        self.assertNotIn("maxLength", schema["$defs"]["CleanupItem"]["properties"]["quote"])
        fields = schema["$defs"]["CleanupItem"]
        self.assertFalse(fields["additionalProperties"])
        self.assertEqual(set(fields["required"]), set(fields["properties"]))
        self.assertEqual(fields["properties"]["action"]["const"], "delete")
        self.assertNotIn("default", json.dumps(schema))


class CleanupEvidenceTests(unittest.TestCase):
    def test_explicit_ocr_eligible_independent_of_format(self):
        for fmt in ("pdf", "txt", "future_scan"):
            self.assertEqual(cleanup_eligibility(source(source_format=fmt)).status, "eligible")

    def test_native_does_not_guess_from_garbage_or_evidence(self):
        for mode in ("fast", "native"):
            value = source(mode)
            self.assertEqual(cleanup_eligibility(value, evidence(value)).reason, "native_source")

    def test_auto_requires_actual_evidence(self):
        value = source("auto")
        self.assertEqual(cleanup_eligibility(value).reason, "ocr_evidence_missing")
        self.assertEqual(cleanup_eligibility(value, evidence(value)).status, "eligible")
        self.assertEqual(cleanup_eligibility(value, evidence(value, ())).status, "not_applicable")
        self.assertEqual(cleanup_eligibility(value, {"mode": "ocr"}).reason, "ocr_evidence_mismatch")

    def test_unknown_mode_is_skipped(self):
        self.assertEqual(cleanup_eligibility(source("unknown")).reason, "extraction_mode_unsupported")

    def test_auto_identity_text_and_span_mismatch(self):
        value = source("auto")
        good = evidence(value)
        for update in ({"extraction_identity": replace(value.identity, settings_sha256="c" * 64)},
                       {"text_sha256": "d" * 64},
                       {"spans": (CanonicalOcrSpan(char_start=0, char_end=value.text_length + 1),)},
                       {"spans": (CanonicalOcrSpan(char_start=5, char_end=10),
                                  CanonicalOcrSpan(char_start=4, char_end=6))}):
            changed = ActualOcrEvidence(**{**dict(good), **update})
            self.assertEqual(cleanup_eligibility(value, changed).reason, "ocr_evidence_mismatch")

    def test_span_coordinates_and_eligibility_are_strict(self):
        for fields in ({"char_start": True, "char_end": 4}, {"char_start": -1, "char_end": 4},
                       {"char_start": 4, "char_end": 4}, {"char_start": 0, "char_end": 4, "page_index": -1}):
            with self.assertRaises(ValidationError):
                CanonicalOcrSpan(**fields)
        with self.assertRaises(ValidationError):
            CleanupEligibility(status="eligible", reason="native_source")


class CleanupBindingTests(unittest.TestCase):
    def test_bound_plan_preserves_source_and_serializes(self):
        value = source()
        plan = proposal([item()])
        bound = BoundCleanupProposal.bind(source=value, binding=binding(value), proposal=plan)
        self.assertEqual(BoundCleanupProposal.model_validate_json(bound.model_dump_json()), bound)
        self.assertEqual(value.canonical_text, "Opening.\nnoise?!\nBody.")
        self.assertEqual(bound.binding.text_sha256, text_digest(value.canonical_text))

    def test_source_binding_mismatch_rejected(self):
        value = source()
        for fields in ({"text_sha256": "e" * 64}, {"text_length": 1},
                       {"extraction_identity": replace(value.identity, source_sha256="e" * 64)}):
            with self.assertRaisesRegex(ValueError, "cleanup_source_mismatch"):
                BoundCleanupProposal.bind(source=value, binding=binding(value, **fields), proposal=proposal())

    def test_versions_consent_and_representation_are_not_model_authority(self):
        for fields in ({"expected_reading_revision": True}, {"expected_reading_revision": -1},
                       {"structure_version": 0}, {"structure_version": None}, {"operation_id": " "},
                       {"consent_scope": "structure_only"}, {"source_representation": "guessed"},
                       {"policy_version": "old"}, {"structure_fingerprint": "wrong"}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                binding(source(), **fields)


class CleanupOutcomeTests(unittest.TestCase):
    def outcome(self, **fields):
        return CleanupOutcome(**{"status": "no_changes", "proposed_count": 0, "applied_count": 0,
            "skipped_count": 0, "reason": None, "reading_content_changed": False, "reading_revision": 0, **fields})

    def test_all_terminal_states(self):
        self.outcome()
        self.outcome(status="not_applicable", proposed_count=None, reason="native_source")
        self.outcome(status="skipped", proposed_count=None, reason="capacity_exceeded")
        self.outcome(status="failed", proposed_count=None, reason="provider_failed")
        self.outcome(status="failed", proposed_count=2, skipped_count=2, reason="storage_failed")
        self.outcome(status="completed", proposed_count=2, applied_count=2,
                     reading_content_changed=True, reading_revision=1)
        self.outcome(status="partial", proposed_count=2, applied_count=1, skipped_count=1,
                     reason="some_items_rejected", reading_content_changed=True, reading_revision=1)

    def test_inconsistent_outcomes_rejected(self):
        cases = ({"reading_content_changed": True}, {"proposed_count": 1}, {"proposed_count": True},
                 {"status": "completed"}, {"status": "partial", "reason": "some_items_rejected"},
                 {"status": "failed", "reason": "PRIVATE_EXCEPTION"},
                 {"status": "failed", "reason": "capacity_exceeded"},
                 {"status": "skipped", "reason": "provider_failed"},
                 {"status": "not_applicable", "reason": "native_source"},
                 {"status": "completed", "proposed_count": 1, "applied_count": 1, "reading_content_changed": True},
                 {"status": "completed", "proposed_count": 2, "applied_count": 1, "skipped_count": 1,
                  "reading_content_changed": True, "reading_revision": 1},
                 {"status": "failed", "reason": "storage_failed", "proposed_count": None, "skipped_count": 1})
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                self.outcome(**fields)

    def test_outcomes_accept_counts_above_old_item_limit(self):
        self.outcome(status="completed", proposed_count=129, applied_count=129,
                     reading_content_changed=True, reading_revision=1)
        self.outcome(status="partial", proposed_count=300, applied_count=150, skipped_count=150,
                     reason="some_items_rejected", reading_content_changed=True, reading_revision=1)


if __name__ == "__main__":
    unittest.main()
