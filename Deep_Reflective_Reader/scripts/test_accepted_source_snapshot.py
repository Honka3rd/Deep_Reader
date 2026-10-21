"""Accepted source readiness without parsers, providers or persistence writes."""
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_preparation.accepted_source_snapshot import (
    SourceNotReady, resolve_accepted_source_snapshot,
)
from document_preparation.pdf_extraction_identity import PdfExtractionIdentity
from doc_loaders.extraction_identity import normalize_pdf_identity
from document_structure.structured_document import (
    StructuredChapter, StructuredDocument, StructuredSection,
)
from document_structure.structured_document_store import StructuredDocumentStore


class AcceptedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source.pdf"
        # Deliberately not a parseable PDF: this path must only hash bytes.
        self.source.write_bytes(b"accepted source fixture")
        self.text = "First paragraph.\n\nSecond paragraph.\n"
        self.identity = PdfExtractionIdentity.create(
            source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            mode="fast", extraction_version="older-accepted-version", settings={},
        )
        self.section = StructuredSection("s", 0, None, 2, self.text, 0, len(self.text))
        self.document = StructuredDocument(
            "doc", "title", "untrusted/path", "en", self.text,
            chapters=[StructuredChapter("c", None, 1, None, [self.section])],
            parse_provenance={"pdf_extraction": self.identity.record(self.text)},
        )

    def resolve(self, document=None, **kwargs):
        args = dict(document=document or self.document, namespace="tenant/doc",
                    expected_document_id="doc", structure_version=3,
                    source_path=self.source, accepted=True)
        args.update(kwargs)
        return resolve_accepted_source_snapshot(**args)

    def reject(self, reason, document=None, **kwargs):
        with self.assertRaises(SourceNotReady) as caught:
            self.resolve(document, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        self.assertEqual(caught.exception.recovery, "explicit_prepare_required")

    def test_file_store_readback_and_immutable_snapshot(self):
        path = Path(self.temp.name) / "stored.json"
        StructuredDocumentStore.save(self.document, str(path))
        before = path.read_bytes()
        loaded = StructuredDocumentStore.load(str(path))
        with patch.object(StructuredDocumentStore, "save", side_effect=AssertionError("write")):
            snapshot = self.resolve(loaded)
        self.assertEqual(snapshot.canonical_text, self.text)
        self.assertEqual(snapshot.text_length, len(self.text))
        self.assertEqual(snapshot.extraction_identity, normalize_pdf_identity(self.identity))
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(FrozenInstanceError):
            snapshot.language = "fr"

    def test_postgres_shaped_readback_reconstructs_exact_slices(self):
        split = 18
        sections = [replace(self.section, content=self.text[:split], char_end=split),
                    replace(self.section, section_id="s2", content=self.text[split:],
                            char_start=split)]
        document = replace(self.document, raw_text="", chapters=[
            replace(self.document.chapters[0], sections=sections)])
        before = document.to_dict()
        self.assertEqual(self.resolve(document).canonical_text, self.text)
        self.assertEqual(document.to_dict(), before)

    def test_gaps_overlap_out_of_order_and_corrupt_lengths_rejected(self):
        for section in [replace(self.section, char_start=1),
                        replace(self.section, char_start=-1),
                        replace(self.section, char_end=2),
                        replace(self.section, char_start=False)]:
            with self.subTest(section=section):
                self.reject("canonical_coverage_invalid", replace(self.document, raw_text="",
                    chapters=[replace(self.document.chapters[0], sections=[section])]))

    def test_missing_suffix_rejected_even_with_contiguous_spans(self):
        section = replace(self.section, content=self.text[:-1], char_end=len(self.text)-1)
        self.reject("canonical_hash_mismatch", replace(self.document, raw_text="",
            chapters=[replace(self.document.chapters[0], sections=[section])]))

    def test_legacy_roots_are_never_used(self):
        self.reject("canonical_text_empty", replace(self.document, raw_text="",
                    chapters=[], sections=[self.section]))

    def test_unknown_and_malformed_identity(self):
        for record in [None, {}, {"identity": None},
                       {"identity": self.identity.to_dict(), "text_sha256": "bad"}]:
            self.reject("extraction_identity_unknown", replace(self.document,
                parse_provenance={"pdf_extraction": record}))

    def test_changed_and_missing_source(self):
        self.source.write_bytes(b"changed")
        self.reject("source_changed")
        self.source.unlink()
        self.reject("source_missing")

    def test_partial_or_failed_acceptance(self):
        self.reject("source_not_accepted", accepted=False)
        self.reject("source_not_accepted", replace(self.document, structure_error_code="partial"))

    def test_identity_and_version(self):
        self.reject("document_identity_mismatch", expected_document_id="other")
        self.reject("document_identity_mismatch", namespace="")
        for version in [None, 0, True, "3"]:
            self.reject("structure_version_unknown", structure_version=version)

    def test_language_and_text_failures(self):
        for language in [None, "unknown", "not-a-language"]:
            self.reject("language_unavailable", replace(self.document, language=language))
        self.reject("canonical_hash_mismatch", replace(self.document, raw_text="tampered"))
        self.reject("canonical_text_empty", replace(self.document, raw_text="   "))

    def test_all_accepted_modes_reuse_old_identity_without_engine_calls(self):
        for mode in ["fast", "ocr", "auto"]:
            identity = replace(self.identity, mode=mode)
            document = replace(self.document,
                parse_provenance={"pdf_extraction": identity.record(self.text)})
            with patch("subprocess.run", side_effect=AssertionError("engine called")):
                self.assertEqual(self.resolve(document).extraction_identity, normalize_pdf_identity(identity))

    def test_unicode_whitespace_preserved_exactly(self):
        text = "\u6b78\u4f86\r\n\n\u6751\u838a \u5100\u5f0f\n"
        document = replace(self.document, raw_text="", chapters=[
            replace(self.document.chapters[0], sections=[
                replace(self.section, content=text, char_end=len(text))])],
            parse_provenance={"pdf_extraction": self.identity.record(text)})
        self.assertEqual(self.resolve(document).canonical_text, text)

    def test_path_replacement_during_hash_is_rejected(self):
        other = Path(self.temp.name) / "replacement.pdf"
        other.write_bytes(self.source.read_bytes())
        with patch.object(Path, "stat", return_value=other.stat()):
            self.reject("source_changed")


if __name__ == "__main__":
    unittest.main()
