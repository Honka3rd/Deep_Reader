"""Document review visibility uses source-bound OCR evidence, never UI guesses."""

from copy import deepcopy
from dataclasses import replace
import unittest

from api_schemas import ReadingContentResponse
from doc_loaders.extraction_identity import ExtractionIdentity
from doc_loaders.ocr_review import review_record, region_record
from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.ocr_review_availability import ocr_review_available
from document_structure.reading_content import ReadingContentService
from scripts.test_reading_interaction_artifact_store import _document


class OcrReviewAvailabilityTests(unittest.TestCase):
    def document(self, mode="auto", source_format="pdf", spans=True):
        document = _document()
        text = canonical_text(document)
        identity = ExtractionIdentity.create(source_sha256="a" * 64, source_format=source_format,
            extractor_id="test-loader", extraction_version="v1", mode=mode, settings={})
        provenance = {"extraction": identity.record(text), "actual_ocr_evidence": {
            "schema_version": "cleanup_ocr_evidence_v1", "extraction_identity": identity.to_dict(),
            "text_sha256": identity.record(text)["text_sha256"],
            "spans": [{"char_start": 0, "char_end": len(text), "page_index": 0}] if spans else []}}
        return replace(document, parse_provenance=provenance)

    def test_snapshot_contract_and_no_read_side_mutation(self):
        for mode, format_, spans, expected in [
            ("ocr", "pdf", True, True), ("auto", "pdf", True, True),
            ("auto", "pdf", False, False), ("fast", "pdf", True, False),
            ("native", "txt", False, False), ("ocr", "txt", True, False),
        ]:
            with self.subTest(mode=mode, format=format_, spans=spans):
                document = self.document(mode, format_, spans)
                before = deepcopy(document.to_dict())
                state = ReadingContentService(None).snapshot(document)
                self.assertEqual(state["ocr_review_available"], expected)
                self.assertEqual(ReadingContentResponse.model_validate(state).ocr_review_available, expected)
                self.assertEqual(document.to_dict(), before)

    def test_unknown_invalid_or_stale_evidence_is_hidden(self):
        for fault in ["missing_identity", "invalid_identity", "stale_text", "stale_evidence",
                      "missing_evidence", "bad_evidence", "out_of_bounds"]:
            with self.subTest(fault=fault):
                document = self.document()
                provenance = document.parse_provenance
                if fault == "missing_identity": del provenance["extraction"]
                elif fault == "invalid_identity": provenance["extraction"]["identity"] = {}
                elif fault == "stale_text": document = replace(document, raw_text="Changed canonical source")
                elif fault == "stale_evidence": provenance["actual_ocr_evidence"]["text_sha256"] = "b" * 64
                elif fault == "missing_evidence": del provenance["actual_ocr_evidence"]
                elif fault == "bad_evidence": provenance["actual_ocr_evidence"] = "invalid"
                else: provenance["actual_ocr_evidence"]["spans"][0]["char_end"] += 1
                self.assertFalse(ocr_review_available(document))

    def test_empty_ocr_region_and_resolved_history_remain_available(self):
        document = self.document(spans=False)
        region = region_record(1, 0, [0, 0, 10, 10], 150, "recognition_empty")
        document.parse_provenance["source_review"] = review_record("a" * 64, [region])
        self.assertTrue(ocr_review_available(document))
        document.parse_provenance["reading_content"] = {"regions": {region["id"]: "discarded"}}
        self.assertTrue(ocr_review_available(document))
        document.parse_provenance["source_review"]["source_sha256"] = "b" * 64
        self.assertFalse(ocr_review_available(document))

    def test_full_ocr_legacy_record_without_evidence_is_supported(self):
        document = self.document(mode="ocr")
        record = document.parse_provenance.pop("extraction")
        del record["identity"]["source_format"]
        del record["identity"]["extractor_id"]
        document = replace(document, parse_provenance={"pdf_extraction": record})
        self.assertTrue(ocr_review_available(document))


if __name__ == "__main__":
    unittest.main()
