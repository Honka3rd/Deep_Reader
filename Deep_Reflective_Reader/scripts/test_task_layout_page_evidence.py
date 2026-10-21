#!/usr/bin/env python3
"""Persisted-only page evidence; real PDF source hashing, no read-time engines."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import test_pdf_mode_preparation as mode_tests
from document_preparation.pdf_extraction_identity import PdfExtractionIdentity


class PageEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixture = mode_tests.ModePreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertTrue(fixture.prepare().structured_document_ready)
        self.pipeline, self.loader = fixture.pipeline, fixture.loader
        self.store, self.path = fixture.store, fixture.path
        self.document = self.store.document
        guards = ExitStack()
        self.addCleanup(guards.close)
        for target, methods in (
            (self.loader, ("load", "extract", "load_page_boundary_evidence", "load_page_layout_evidence", "_render_page_images")),
            (self.pipeline.language_detector, ("detect",)),
            (self.pipeline.profile_builder, ("build",)),
            (self.store, ("load", "save")),
            (self.pipeline, ("load_manual_structure_source_evidence",)),
        ):
            for method in methods:
                guards.enter_context(patch.object(target, method, side_effect=AssertionError("forbidden evidence work: " + method)))

    def read(self, document=None):
        return self.pipeline.load_task_layout_page_boundaries("sample", document=document or self.document)

    def test_valid_native_cached_spans_are_read_only(self):
        before = deepcopy(self.document.parse_provenance)
        for _ in range(2):
            evidence = self.read()
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].text, self.document.raw_text)
        self.assertEqual(self.document.parse_provenance, before)
        self.assertEqual(self.store.save_calls, 1)

    def test_ocr_auto_provenance_reuses_without_engine(self):
        # Contract fixtures model already-persisted OCR/auto output, not OCR accuracy.
        record = self.document.parse_provenance["pdf_extraction"]
        for mode in ("ocr", "auto"):
            document = deepcopy(self.document)
            identity = replace(PdfExtractionIdentity(**record["identity"]), mode=mode)
            document.parse_provenance["pdf_extraction"] = identity.record(document.raw_text)
            self.assertEqual(len(self.read(document)), 1)

    def test_changed_source_is_rejected(self):
        self.path.write_bytes(b"replacement")
        self.assertEqual(self.read(), [])

    def test_missing_source_is_rejected(self):
        self.path.unlink()
        self.assertEqual(self.read(), [])

    def test_changed_projection_text_is_rejected(self):
        self.assertEqual(self.read(replace(self.document, raw_text="different snapshot")), [])

    def test_unknown_provenance_does_not_touch_loader(self):
        with patch.object(self.pipeline.loader_factory, "get", side_effect=AssertionError("legacy source load")):
            self.assertEqual(self.read(replace(self.document, parse_provenance={})), [])

    def test_unsupported_identity_schema_is_rejected(self):
        self.document.parse_provenance["pdf_extraction"]["identity"]["schema_version"] = 99
        self.assertEqual(self.read(), [])

    def test_missing_malformed_out_of_range_and_duplicate_spans(self):
        span = self.document.parse_provenance["pdf_page_spans"][0]
        for spans in (None, [], [None], [{**span, "char_start": True}],
                      [{**span, "char_end": len(self.document.raw_text) + 1}],
                      [{**span, "char_start": -1}], [span, span],
                      [{**span, "page_index": 2}], [{**span, "page_label": 2}]):
            with self.subTest(spans=spans):
                document = deepcopy(self.document)
                document.parse_provenance["pdf_page_spans"] = spans
                self.assertEqual(self.read(document), [])


if __name__ == "__main__":
    unittest.main()
