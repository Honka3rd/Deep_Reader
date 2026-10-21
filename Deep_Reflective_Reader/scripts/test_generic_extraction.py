"""Format-neutral contracts and explicit legacy migration proof."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

from doc_loaders.extraction_identity import (
    ExtractionIdentity, ExtractionResult, read_extraction_record, normalize_pdf_identity,
    extraction_matches,
)
from doc_loaders.text_document_loader import TextDocumentLoader
from document_preparation.extraction_backfill import prove_txt_backfill
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
from document_structure.structured_document import StructuredDocument, StructuredChapter, StructuredSection
from test_accepted_source_snapshot import AcceptedSourceTests


class GenericExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "doc.txt"
        self.path.write_bytes(b"\xef\xbb\xbfOne\r\nTwo\rThree\n")
        self.loader = TextDocumentLoader(self.temp.name)
        self.result = self.loader.extract("doc")

    def test_txt_compatibility_and_identity(self):
        self.assertEqual(self.loader.load("doc"), self.result.canonical_text)
        with self.path.open(encoding="utf-8") as stream:
            self.assertEqual(stream.read(), self.result.canonical_text)
        self.assertEqual(self.result.identity.mode, "native")
        self.path.rename(self.path.with_name("renamed.txt"))
        self.assertEqual(self.result, self.loader.extract("renamed"))

    def test_changed_bytes_same_text_different_identity(self):
        self.path.write_text(self.result.canonical_text, encoding="utf-8")
        new = self.loader.extract("doc")
        self.assertEqual(new.canonical_text, self.result.canonical_text)
        self.assertNotEqual(new.identity, self.result.identity)

    def test_invalid_empty_utf8_rejected(self):
        for raw in (b"", b" \n", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.loader.extract("doc", source_bytes=raw)

    def test_strict_identity_and_completion(self):
        for fields in ({"schema_version": True}, {"schema_version": 2},
                       {"source_format": "../epub"}, {"mode": ""}, {"source_sha256": "bad"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.result.identity, **fields)
        for fields in ({"completion": "partial"}, {"text_length": True}, {"text_sha256": "bad"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                read_extraction_record({"extraction": {**self.result.record(), **fields}})

    def test_settings_digest_is_canonical(self):
        fields = self.result.identity.to_dict()
        fields.pop("settings_sha256")
        self.assertEqual(ExtractionIdentity.create(**fields, settings={"a": 1, "b": 2}),
                         ExtractionIdentity.create(**fields, settings={"b": 2, "a": 1}))

    def test_pdf_legacy_normalization_conflict(self):
        fixture = AcceptedSourceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        identity = normalize_pdf_identity(fixture.identity)
        provenance = {"pdf_extraction": fixture.identity.record(fixture.text),
                      "extraction": identity.record(fixture.text)}
        self.assertTrue(extraction_matches(provenance, identity, fixture.text))
        provenance["extraction"] = identity.record("changed")
        self.assertFalse(extraction_matches(provenance, identity, fixture.text))

    def stored(self):
        text = self.result.canonical_text
        section = StructuredSection("s", 0, None, 2, text, 0, len(text))
        document = StructuredDocument("doc", "title", str(self.path), "en", "",
            chapters=[StructuredChapter("c", None, 1, None, [section])])
        return SimpleNamespace(document=document, namespace="doc", structure_version=1, accepted=True)

    def test_explicit_backfill_proof_and_read_only_snapshot(self):
        stored = self.stored()
        before = stored.document.to_dict()
        record = prove_txt_backfill(stored, self.result, self.path)
        self.assertEqual(stored.document.to_dict(), before)
        stored.document.parse_provenance["extraction"] = record
        self.assertEqual(record, prove_txt_backfill(stored, self.result, self.path))
        snapshot = resolve_accepted_source_snapshot(document=stored.document, namespace="doc",
            expected_document_id="doc", structure_version=1, source_path=self.path, accepted=True)
        self.assertEqual(snapshot.canonical_text, self.result.canonical_text)

    def test_backfill_rejects_truncated_stale_and_conflicting_source(self):
        stored = self.stored()
        chapter = stored.document.chapters[0]
        section = chapter.sections[0]
        stored.document = replace(stored.document, chapters=[replace(chapter,
            sections=[replace(section, char_end=section.char_end - 1)])])
        with self.assertRaises(ValueError):
            prove_txt_backfill(stored, self.result, self.path)
        self.path.write_bytes(b"different source")
        with self.assertRaises(ValueError):
            prove_txt_backfill(self.stored(), self.result, self.path)

    def test_prepare_persists_identity_and_reuses_without_profile_work(self):
        from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore
        store = _StructuredStore(exists_value=False, document=self.stored().document)
        pipeline = _build_pipeline(loader=self.loader, store=store)
        pipeline.structured_document_builder = Mock()
        pipeline.structured_document_builder.build.side_effect = lambda **kwargs: replace(
            self.stored().document, raw_text=kwargs["raw_text"])
        with patch.object(self.loader, "extract", wraps=self.loader.extract) as extract:
            prepared = pipeline.prepare("doc", mode="base", pdf_extraction_mode="fast")
            self.assertTrue(prepared.structured_document_ready, prepared.errors)
            extract.assert_called_once()
        self.assertEqual(store.document.parse_provenance["extraction"], self.result.record())
        with patch.object(pipeline, "_prepare_profile", side_effect=AssertionError("profile generation")):
            reused = pipeline.prepare("doc", mode="base", pdf_extraction_mode="fast")
        self.assertTrue(reused.structured_document_ready, reused.errors)
        self.assertEqual(store.save_calls, 1)


if __name__ == "__main__":
    unittest.main()
