#!/usr/bin/env python3
"""Deterministic need decisions and bounded escalation without OCR/LLM calls."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_loaders.pdf_ocr_decision import PdfOcrDecision as D, PdfOcrPurpose as P, PdfOcrReason as R
from doc_loaders.pdf_ocr_policy import PdfOcrPolicy, PdfOcrPolicyConfig, PdfPageVisualCheck
from doc_loaders.pdf_page_inspection import PdfNativeTextQuality, PdfPageInspection, inspect_pdf_page
from test_pdf_page_inspection import add_image, page_fixture


BODY = "The native text contains enough varied words to represent an ordinary body paragraph."


def page(text=BODY, **changes):
    values = dict(source_sha256="ab" * 32, page_index=0, native_text=text,
                  text_quality=PdfNativeTextQuality.from_text(text), crop_box=(0, 0, 100, 100),
                  rotation_degrees=0, has_font_resources=True, image_draw_count=0,
                  image_rects=(), image_coverage_ratio=0.0, text_origins=((10, 70), (10, 40)),
                  text_positions_complete=True, has_vector_paint=False, flags=())
    values.update(changes)
    return PdfPageInspection(**values)


def check(**changes):
    values = dict(source_sha256="ab" * 32, page_index=0, detector_version="fixture-v1",
                  contains_text=True, covers_full_page=True, native_text_complete=None)
    values.update(changes)
    return PdfPageVisualCheck(**values)


class PdfOcrPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = PdfOcrPolicy()

    def assert_state(self, evidence, state, reason, *, purpose=P.TEXT_RECOVERY, visual=None):
        result = self.policy.assess(evidence, purpose, visual_check=visual)
        self.assertEqual(result.decision, state)
        self.assertIn(reason, result.reason_codes)
        return result

    def test_native_multilingual_numeric_and_mixed_text_skip(self):
        for text in (BODY, "\u4e2d\u6587\u6bb5\u843d\u6b63\u6587\u95b1\u8b80\u8cc7\u6599" * 6,
                     "0123456789" * 8, "English \u4e2d\u6587 123 caf\u00e9 " * 5):
            with self.subTest(text=text):
                self.assert_state(page(text), D.SKIP, R.NATIVE_TEXT_USABLE)
                self.assert_state(page(text), D.SKIP, R.NATIVE_LAYOUT_SUFFICIENT, purpose=P.LAYOUT_ANALYSIS)

    def test_blank_native_page_skips_without_font_requirement(self):
        self.assert_state(page("", text_origins=(), has_font_resources=False), D.SKIP, R.BLANK_PAGE)

    def test_full_page_image_is_not_automatically_a_scan(self):
        scan = page("", image_draw_count=1, image_coverage_ratio=1.0,
                    image_rects=((0, 0, 100, 100),), text_origins=(), has_font_resources=False)
        self.assert_state(scan, D.UNCERTAIN, R.TEXT_LAYER_MISSING)
        self.assert_state(scan, D.REQUIRED, R.VISUAL_TEXT_CONFIRMED, visual=check())
        self.assert_state(scan, D.REQUIRED, R.LAYOUT_EVIDENCE_REQUIRED,
                          purpose=P.LAYOUT_ANALYSIS, visual=check())
        self.assert_state(scan, D.SKIP, R.NON_TEXT_PAGE, visual=check(contains_text=False))

    def test_vector_art_needs_positive_non_text_evidence(self):
        art = page("", has_vector_paint=True, text_origins=())
        self.assert_state(art, D.UNCERTAIN, R.EVIDENCE_INCOMPLETE)
        self.assert_state(art, D.SKIP, R.NON_TEXT_PAGE, visual=check(contains_text=False))

    def test_sparse_and_header_only_do_not_certify_body_coverage(self):
        self.assert_state(page("Chapter 1"), D.UNCERTAIN, R.SPARSE_NATIVE_TEXT)
        self.assert_state(page(text_origins=((5, 98), (5, 2))), D.UNCERTAIN, R.HEADER_FOOTER_ONLY)

    def test_mixed_content_cannot_hide_behind_large_native_text_count(self):
        mixed = page(BODY * 20, image_draw_count=1, image_coverage_ratio=0.02)
        self.assert_state(mixed, D.UNCERTAIN, R.MIXED_CONTENT)
        self.assert_state(mixed, D.REQUIRED, R.UNCOVERED_IMAGE_TEXT,
                          visual=check(native_text_complete=False))
        self.assert_state(mixed, D.SKIP, R.NATIVE_TEXT_USABLE,
                          visual=check(native_text_complete=True))
        self.assert_state(mixed, D.UNCERTAIN, R.EVIDENCE_INCOMPLETE,
                          purpose=P.LAYOUT_ANALYSIS, visual=check(native_text_complete=True))

    def test_corrupt_and_duplicate_text_stays_uncertain_without_visual_evidence(self):
        for text in ("\ufffd" * 60, BODY + "\x00", "a" * 80, (BODY + "\n") * 4):
            with self.subTest(text=text):
                self.assert_state(page(text), D.UNCERTAIN, R.TEXT_LAYER_UNRELIABLE)
                self.assert_state(page(text), D.REQUIRED, R.TEXT_LAYER_UNRELIABLE, visual=check())

    def test_hidden_ocr_layer_is_not_trusted_by_presence_alone(self):
        hidden = page(flags=("invisible_text_mode",), image_draw_count=1, image_coverage_ratio=1.0)
        self.assert_state(hidden, D.UNCERTAIN, R.EVIDENCE_INCOMPLETE)
        self.assert_state(hidden, D.SKIP, R.NATIVE_TEXT_USABLE, visual=check(native_text_complete=True))

    def test_missing_and_new_unknown_evidence_fails_closed(self):
        for changes in (
            dict(crop_box=None), dict(image_draw_count=None), dict(image_coverage_ratio=None),
            dict(rotation_degrees=None), dict(has_font_resources=None), dict(has_vector_paint=None),
            dict(text_positions_complete=False), dict(flags=("future_unknown_flag",)),
            dict(flags=("form_or_unknown_xobject_uninspected",)),
            dict(flags=("custom_clipping_unresolved",)),
        ):
            with self.subTest(changes=changes):
                self.assert_state(page(**changes), D.UNCERTAIN, R.EVIDENCE_INCOMPLETE)

    def test_failed_or_invalid_inspection_cannot_be_rescued_by_probe(self):
        for changes in (dict(native_text=None), dict(text_quality=None), dict(evidence_schema_version=2),
                        dict(flags=("native_text_extraction_failed",)),
                        dict(text_quality=replace(page().text_quality, printable_ratio=float("nan")))):
            with self.subTest(changes=changes):
                self.assert_state(page(**changes), D.UNCERTAIN, R.INSPECTION_FAILED, visual=check())

    def test_partial_unknown_and_conflicting_visual_checks(self):
        self.assert_state(page(), D.UNCERTAIN, R.EVIDENCE_INCOMPLETE,
                          visual=check(covers_full_page=False, contains_text=False))
        self.assert_state(page(), D.UNCERTAIN, R.EVIDENCE_INCOMPLETE, visual=check(contains_text=None))
        self.assert_state(page(), D.UNCERTAIN, R.CONFLICTING_EVIDENCE, visual=check(contains_text=False))
        self.assert_state(page(""), D.UNCERTAIN, R.CONFLICTING_EVIDENCE,
                          visual=check(native_text_complete=True))

    def test_probe_source_and_page_identity_must_match(self):
        for visual in (check(source_sha256="cd" * 32), check(page_index=1)):
            with self.assertRaises(ValueError):
                self.policy.assess(page(), P.TEXT_RECOVERY, visual_check=visual)

    def test_text_and_layout_are_independent_and_rotation_conservative(self):
        for evidence in (page(text_origins=((20, 50),)), page(rotation_degrees=90)):
            self.assert_state(evidence, D.SKIP, R.NATIVE_TEXT_USABLE)
            self.assert_state(evidence, D.UNCERTAIN, R.EVIDENCE_INCOMPLETE, purpose=P.LAYOUT_ANALYSIS)

    def test_short_confirmed_native_label_can_skip(self):
        self.assert_state(page("1"), D.SKIP, R.NATIVE_TEXT_USABLE, visual=check(native_text_complete=True))

    def test_threshold_boundary_and_configuration_fingerprint(self):
        text = "0123456789" * 4
        self.assert_state(page(text), D.SKIP, R.NATIVE_TEXT_USABLE)
        self.assert_state(page(text[:-1]), D.UNCERTAIN, R.SPARSE_NATIVE_TEXT)
        other = PdfOcrPolicy(PdfOcrPolicyConfig(min_native_chars=39))
        self.assertEqual(other.assess(page(text[:-1]), P.TEXT_RECOVERY).decision, D.SKIP)
        self.assertNotEqual(other.config.policy_version, self.policy.config.policy_version)
        self.assertEqual(PdfOcrPolicyConfig().policy_version, self.policy.config.policy_version)

    def test_invalid_configuration_is_rejected(self):
        for fields in (dict(min_native_chars=0), dict(min_native_chars=True),
                       dict(min_printable_ratio=float("nan")), dict(max_control_ratio=1.1),
                       dict(body_margin_ratio=0.5), dict(probe_attempt_limit=3),
                       dict(probe_dpi=300), dict(probe_max_pixels=0), dict(probe_timeout_seconds=60)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                PdfOcrPolicyConfig(**fields)

    def test_quality_threshold_boundaries(self):
        for field, boundary, outside in (
            ("printable_ratio", 0.98, 0.979), ("replacement_ratio", 0.01, 0.0101),
            ("control_ratio", 0.0, 0.0001), ("longest_repeat_ratio", 0.35, 0.3501),
            ("duplicate_line_ratio", 0.5, 0.5001),
        ):
            with self.subTest(field=field):
                quality = replace(page().text_quality, **{field: boundary})
                self.assert_state(page(text_quality=quality), D.SKIP, R.NATIVE_TEXT_USABLE)
                quality = replace(quality, **{field: outside})
                self.assert_state(page(text_quality=quality), D.UNCERTAIN, R.TEXT_LAYER_UNRELIABLE)

    def test_position_and_probe_zero_budget_boundaries(self):
        self.assert_state(page(text_origins=((10, 12),)), D.UNCERTAIN, R.HEADER_FOOTER_ONLY)
        self.assert_state(page(text_origins=((10, 12.01),)), D.SKIP, R.NATIVE_TEXT_USABLE)
        self.assert_state(page(text_origins=((10, 50), (10, 65))), D.SKIP,
                          R.NATIVE_LAYOUT_SUFFICIENT, purpose=P.LAYOUT_ANALYSIS)
        self.assert_state(page(text_origins=((10, 50), (10, 64.9))), D.UNCERTAIN,
                          R.EVIDENCE_INCOMPLETE, purpose=P.LAYOUT_ANALYSIS)
        policy = PdfOcrPolicy(PdfOcrPolicyConfig(probe_attempt_limit=0))
        decision = policy.assess(page("Header"), P.TEXT_RECOVERY)
        self.assertEqual(policy.plan_probe(decision, attempts_used=0, ocr_enabled=True,
                                            detector_available=True).status, "budget_exhausted")

    def test_probe_metadata_rejects_ambiguous_types(self):
        for fields in (dict(contains_text=1), dict(covers_full_page=None), dict(page_index=-1),
                       dict(source_sha256="bad"), dict(detector_version=""),
                       dict(contains_text=False, native_text_complete=True)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                check(**fields)

    def test_probe_plan_gates_limits_and_exhaustion_preserve_uncertainty(self):
        decision = self.policy.assess(page("Header"), P.TEXT_RECOVERY)
        args = dict(attempts_used=0, ocr_enabled=True, detector_available=True)
        plan = self.policy.plan_probe(decision, **args)
        self.assertEqual((plan.status, plan.dpi, plan.max_pixels, plan.timeout_seconds),
                         ("probe", 96, 1_000_000, 5))
        for override, status in ((dict(ocr_enabled=False), "disabled"),
                                 (dict(detector_available=False), "unavailable"),
                                 (dict(attempts_used=1), "budget_exhausted")):
            limited = self.policy.plan_probe(decision, **(args | override))
            self.assertEqual(limited.status, status)
            self.assertIsNone(limited.dpi)
        self.assertEqual(decision.decision, D.UNCERTAIN)
        with self.assertRaises(ValueError):
            self.policy.plan_probe(decision, **(args | dict(attempts_used=-1)))
        with self.assertRaises(ValueError):
            PdfOcrPolicy(PdfOcrPolicyConfig(min_native_chars=39)).plan_probe(decision, **args)

    def test_skip_and_required_never_trigger_probe_plan(self):
        for decision in (self.policy.assess(page(), P.TEXT_RECOVERY),
                         self.policy.assess(page(""), P.TEXT_RECOVERY, visual_check=check())):
            self.assertEqual(self.policy.plan_probe(decision, attempts_used=0, ocr_enabled=True,
                                                   detector_available=True).status, "not_needed")

    def test_real_native_and_image_stream_handoff_without_external_calls(self):
        _, native = page_fixture(b"BT /F1 12 Tf 10 70 Td (" + BODY.encode() + b") Tj ET", with_font=True)
        _, scan = page_fixture(b"q 100 0 0 100 0 0 cm /Im Do Q")
        add_image(scan)
        with patch("subprocess.run", side_effect=AssertionError("external process")):
            for index, raw in enumerate((native, scan)):
                evidence = inspect_pdf_page(raw, source_sha256="ab" * 32, page_index=index)
                result = self.policy.assess(evidence, P.TEXT_RECOVERY)
                self.assertEqual(result.decision, D.SKIP if index == 0 else D.UNCERTAIN)
                self.assertNotIn(BODY, json.dumps(result.to_dict()))
                self.assertEqual(result, self.policy.assess(evidence, P.TEXT_RECOVERY))


if __name__ == "__main__":
    unittest.main()
