#!/usr/bin/env python3
"""Pure OCR decision contract regression; no PDF, OCR, LLM or storage needed."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.pdf_ocr_decision import (
    PdfOcrDecision,
    PdfOcrPurpose,
    PdfOcrReason,
    PdfPageOcrDecision,
)


class PdfOcrDecisionTests(unittest.TestCase):
    def make_decision(self, **overrides):
        values = dict(
            source_sha256="ab" * 32,
            page_index=0,
            purpose=PdfOcrPurpose.TEXT_RECOVERY,
            decision=PdfOcrDecision.UNCERTAIN,
            reason_codes=(PdfOcrReason.EVIDENCE_INCOMPLETE,),
            policy_version="test-policy-v1",
        )
        values.update(overrides)
        return PdfPageOcrDecision(**values)

    def test_all_decisions_and_purposes_serialize_as_stable_strings(self):
        for purpose in PdfOcrPurpose:
            for decision in PdfOcrDecision:
                with self.subTest(purpose=purpose, decision=decision):
                    result = self.make_decision(purpose=purpose, decision=decision)
                    payload = json.loads(json.dumps(result.to_dict()))
                    self.assertEqual(payload, {
                        "source_sha256": "ab" * 32,
                        "page_index": 0,
                        "purpose": purpose.value,
                        "decision": decision.value,
                        "reason_codes": ["evidence_incomplete"],
                        "policy_version": "test-policy-v1",
                        "evidence_schema_version": 1,
                    })

    def test_enum_strings_and_hash_are_normalized(self):
        result = self.make_decision(
            source_sha256="AB" * 32, purpose="text_recovery", decision="skip",
            reason_codes=["native_text_usable"],
        )
        self.assertIs(result.purpose, PdfOcrPurpose.TEXT_RECOVERY)
        self.assertIs(result.decision, PdfOcrDecision.SKIP)
        self.assertEqual(result.reason_codes, (PdfOcrReason.NATIVE_TEXT_USABLE,))
        self.assertEqual(result.source_sha256, "ab" * 32)

    def test_source_page_identity_is_validated(self):
        for field, values in (
            ("source_sha256", (None, "", "x" * 64, "a" * 63, "a" * 65, " a" * 32)),
            ("page_index", (-1, True, 1.5, "0", None)),
            ("policy_version", (None, "", " ", "v1\n", "x" * 65)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.make_decision(**{field: value})

    def test_invalid_states_and_reasons_are_rejected(self):
        for fields in (
            {"purpose": "all"}, {"decision": "auto"}, {"decision": None},
            {"reason_codes": ()}, {"reason_codes": None},
            {"reason_codes": "native_text_usable"},
            {"reason_codes": ("unrecognized_reason",)},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.make_decision(**fields)

    def test_reason_copy_and_serialization_do_not_mutate_result(self):
        reasons = [PdfOcrReason.MIXED_CONTENT, PdfOcrReason.EVIDENCE_INCOMPLETE,
                   PdfOcrReason.MIXED_CONTENT]
        result = self.make_decision(reason_codes=reasons)
        reasons.clear()
        result.to_dict()["reason_codes"].clear()
        self.assertEqual(result.reason_codes, (
            PdfOcrReason.MIXED_CONTENT, PdfOcrReason.EVIDENCE_INCOMPLETE,
        ))
        with self.assertRaises(FrozenInstanceError):
            result.decision = PdfOcrDecision.SKIP

    def test_text_skip_does_not_imply_layout_skip(self):
        text = self.make_decision(
            decision=PdfOcrDecision.SKIP,
            reason_codes=(PdfOcrReason.NATIVE_TEXT_USABLE,),
        )
        layout = replace(
            text, purpose=PdfOcrPurpose.LAYOUT_ANALYSIS,
            decision=PdfOcrDecision.UNCERTAIN,
            reason_codes=(PdfOcrReason.EVIDENCE_INCOMPLETE,),
        )
        self.assertEqual(text.source_sha256, layout.source_sha256)
        self.assertEqual(text.page_index, layout.page_index)
        self.assertEqual(text.decision, PdfOcrDecision.SKIP)
        self.assertEqual(layout.decision, PdfOcrDecision.UNCERTAIN)

    def test_contract_does_not_infer_decisions_or_accept_source_content(self):
        for fields in ({"native_text": "PRIVATE_TEXT"}, {"evidence_schema_version": 2}):
            with self.subTest(fields=fields), self.assertRaises(TypeError):
                self.make_decision(**fields)
        with self.assertRaises(TypeError):
            PdfPageOcrDecision(
                source_sha256="ab" * 32, page_index=0,
                purpose=PdfOcrPurpose.TEXT_RECOVERY,
                reason_codes=(PdfOcrReason.EVIDENCE_INCOMPLETE,),
                policy_version="test-policy-v1",
            )


if __name__ == "__main__":
    unittest.main()
