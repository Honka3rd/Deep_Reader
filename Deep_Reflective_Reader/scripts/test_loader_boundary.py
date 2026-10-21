"""Loader contracts, including a synthetic third format (not an EPUB decoder)."""
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from doc_loaders.abstract_document_loader import AbstractDocumentLoader
from doc_loaders.extraction_identity import ExtractionIdentity, ExtractionResult
from doc_loaders.source_session import SourceArtifact, SourceSession
from doc_loaders.text_document_loader import TextDocumentLoader
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _fake_document


class SyntheticLoader(AbstractDocumentLoader):
    def __init__(self, path):
        self.path = path
        self.calls = 0
        self.version = "v1"

    def source_path(self, doc_name):
        return self.path

    def load(self, doc_name):
        raise AssertionError("Pipeline must use the session")

    def open_source(self, doc_name, *, mode=None):
        source = self.path.read_bytes()
        identity = ExtractionIdentity.create(source_sha256=hashlib.sha256(source).hexdigest(),
            source_format="epub", extractor_id="synthetic-test-only", extraction_version=self.version,
            mode="native", settings={})
        def extract():
            self.calls += 1
            return SourceArtifact(ExtractionResult(identity, source.decode()))
        return SourceSession(identity=identity, source_path=self.path, extract=extract)


class LoaderBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "doc.epub"
        self.path.write_text("First chapter\nA full paragraph from a synthetic digital source.")
        self.loader = SyntheticLoader(self.path)
        self.store = _StructuredStore(exists_value=False, document=_fake_document("doc"))
        self.pipeline = _build_pipeline(loader=self.loader, store=self.store)
        self.pipeline.structured_document_builder = Mock()
        self.pipeline.structured_document_builder.build.side_effect = lambda **kw: replace(
            _fake_document("doc"), raw_text=kw["raw_text"])

    def test_third_format_prepare_cache_and_manual_read_without_format_dispatch(self):
        first = self.pipeline.prepare("doc", mode="base")
        self.assertTrue(first.structured_document_ready, first.errors)
        self.assertEqual(self.loader.calls, 1)
        with patch.object(self.pipeline, "_prepare_profile", side_effect=AssertionError("profile")):
            self.assertTrue(self.pipeline.prepare("doc", mode="base").structured_document_ready)
        self.assertEqual(self.loader.calls, 1)
        self.assertEqual(self.store.save_calls, 1)
        with patch.object(self.pipeline.language_detector, "detect", side_effect=AssertionError("language")):
            evidence = self.pipeline.load_manual_structure_source_evidence("doc")
        self.assertEqual(evidence.raw_text, self.path.read_text())
        self.assertEqual(evidence.page_boundaries, [])

    def test_request_session_extracts_only_once_and_defers_optional_evidence(self):
        session = self.loader.open_source("doc")
        self.assertEqual(self.loader.calls, 0)
        self.assertIs(session.extract(), session.extract())
        self.assertEqual(self.loader.calls, 1)

    def test_source_change_during_build_does_not_save(self):
        def mutate(**kw):
            self.path.write_text("changed source")
            return replace(_fake_document("doc"), raw_text=kw["raw_text"])
        self.pipeline.structured_document_builder.build.side_effect = mutate
        result = self.pipeline.prepare("doc", mode="base")
        self.assertFalse(result.structured_document_ready)
        self.assertEqual(self.store.save_calls, 0)
        self.assertTrue(any("source changed" in e for e in result.errors))

    def test_extractor_version_change_misses_cache(self):
        self.assertTrue(self.pipeline.prepare("doc", mode="base").structured_document_ready)
        self.loader.version = "v2"
        self.assertTrue(self.pipeline.prepare("doc", mode="base").structured_document_ready)
        self.assertEqual(self.loader.calls, 2)

    def test_txt_capture_is_lazy_and_mode_validation_is_loader_owned(self):
        path = self.path.with_suffix(".txt")
        path.write_bytes(b"first\r\nsecond")
        loader = TextDocumentLoader(path.parent)
        with patch.object(loader, "extract", wraps=loader.extract) as extract:
            session = loader.open_source("doc", mode="fast")
            extract.assert_not_called()
            self.assertEqual(session.extract().result.canonical_text, "first\nsecond")
            session.extract()
            extract.assert_called_once()
        for mode in ("ocr", "auto"):
            with self.assertRaises(ValueError):
                loader.open_source("doc", mode=mode)

    def test_mismatched_result_identity_is_rejected(self):
        session = self.loader.open_source("doc")
        result = ExtractionResult(replace(session.identity, extraction_version="wrong"), "source")
        session._extract = lambda: SourceArtifact(result)
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            session.extract()

    def test_pipeline_has_no_concrete_loader_or_private_source_dependency(self):
        source = inspect.getsource(DocumentPreparationPipeline)
        for forbidden in ("PdfDocumentLoader", "TextDocumentLoader", "_resolve_file_path",
                          "._file_sha256", 'source_format ==', 'source_format !='):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
