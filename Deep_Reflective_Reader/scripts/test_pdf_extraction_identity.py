#!/usr/bin/env python3
"""Extraction-aware reuse without external storage or LLM calls."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from document_preparation.pdf_extraction_identity import PdfExtractionIdentity, identity_for_loader
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.extraction_identity import normalize_pdf_identity, ExtractionResult
from doc_loaders.source_session import SourceArtifact
from document_preparation.prepared_document_assets import PreparedDocumentAssets
from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.section_splitter_selector import SectionSplitterMode
from test_prepare_structured_reuse_before_raw_load import _fake_document, _StructuredStore, _build_pipeline, _TextLoader


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = PdfExtractionIdentity.create(source_sha256="a" * 64, mode="fast",
            extraction_version="fast-native-v1", settings={"parser": "pypdf", "version": "test"})
        self.document = _fake_document()
        self.document.parse_provenance["pdf_extraction"] = self.identity.record(self.document.raw_text)
        self.store = _StructuredStore(exists_value=True, document=self.document)
        self.pipeline = object.__new__(DocumentPreparationPipeline)
        self.pipeline.structured_document_store = self.store

    def assets(self, identity=None):
        assets = PreparedDocumentAssets("fixture", None, None, False, False, False, False,
            None, None, requested_pdf_extraction_mode="fast", pdf_extraction_identity=identity)
        assets.source_session = Mock(identity=normalize_pdf_identity(identity) if identity else None)
        assets.source_session.compatibility_status.return_value = {"pdf_extraction_provenance": "reused"}
        return assets

    def reuse(self, assets):
        return self.pipeline._try_reuse_existing_structured_document(doc_name="fixture", assets=assets,
            force_rebuild=False, parser_mode=SectionSplitterMode.COMMON)

    def test_same_identity_reuses_without_writes(self):
        assets = self.assets(self.identity)
        self.assertTrue(self.reuse(assets))
        self.assertEqual(assets.pdf_extraction_provenance, "reused")
        self.assertEqual(self.store.save_calls, 0)

    def test_changed_source_mode_version_or_settings_misses_without_deletion(self):
        variants = [replace(self.identity, source_sha256="b" * 64),
                    replace(self.identity, mode="ocr"), replace(self.identity, mode="auto"),
                    replace(self.identity, extraction_version="v2"),
                    replace(self.identity, settings_sha256="c" * 64)]
        for identity in variants:
            assets = self.assets(identity)
            assets.requested_pdf_extraction_mode = identity.mode
            self.assertFalse(self.reuse(assets))
        self.assertIs(self.store.document, self.document)
        self.assertEqual(self.store.save_calls, 0)

    def test_legacy_unknown_and_text_corruption_cannot_reuse(self):
        self.assertFalse(self.reuse(self.assets()))
        self.document.parse_provenance.clear()
        self.assertFalse(self.reuse(self.assets(self.identity)))
        self.document.parse_provenance["pdf_extraction"] = self.identity.record("other text")
        self.assertFalse(self.reuse(self.assets(self.identity)))

    def test_json_storage_and_dto_roundtrip_preserve_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "document.json")
            StructuredDocumentStore.save(self.document, path)
            document = StructuredDocumentStore.load(path)
            self.assertTrue(self.identity.matches(document.parse_provenance["pdf_extraction"], document.raw_text))
        assets = self.assets(self.identity)
        self.assertEqual(PreparedDocumentAssets.from_dict(assets.to_dict()).pdf_extraction_identity, self.identity)

    def test_settings_order_does_not_change_identity(self):
        other = PdfExtractionIdentity.create(source_sha256="a" * 64, mode="fast",
            extraction_version="fast-native-v1", settings={"version": "test", "parser": "pypdf"})
        self.assertEqual(self.identity, other)
        with self.assertRaises(ValueError):
            replace(self.identity, source_sha256="invalid")

    def test_loader_settings_are_mode_specific(self):
        loader = PdfDocumentLoader(tesseract_cmd="nonexistent-test-engine", pdf_renderer_cmd="nonexistent-test-renderer")
        fast = identity_for_loader(loader, source_sha256="a" * 64, mode="fast")
        ocr = identity_for_loader(loader, source_sha256="a" * 64, mode="ocr")
        loader.pdf_render_dpi += 1
        self.assertEqual(fast, identity_for_loader(loader, source_sha256="a" * 64, mode="fast"))
        self.assertNotEqual(ocr, identity_for_loader(loader, source_sha256="a" * 64, mode="ocr"))
        self.assertNotEqual(ocr, identity_for_loader(loader, source_sha256="a" * 64, mode="auto"))

    def test_unproven_extraction_cannot_overwrite_existing_document(self):
        pipeline = _build_pipeline(loader=_TextLoader(), store=self.store)
        pipeline.structured_document_builder = Mock()
        pipeline.structured_document_builder.build.return_value = _fake_document()
        pipeline._load_outline = Mock(return_value=None)
        assets = self.assets(self.identity)
        ready, _ = pipeline._prepare_structured_document(
            "fixture", self.document.raw_text, "zh", None, assets, True,
            SectionSplitterMode.COMMON, page_boundaries=[],
        )
        self.assertFalse(ready)
        self.assertEqual(self.store.save_calls, 0)
        self.assertIs(self.store.document, self.document)

    def test_proven_extraction_is_recorded_at_existing_save_boundary(self):
        pipeline = _build_pipeline(loader=_TextLoader(), store=self.store)
        pipeline.structured_document_builder = Mock()
        pipeline.structured_document_builder.build.return_value = _fake_document()
        pipeline._load_outline = Mock(return_value=None)
        pipeline._persist_ocr_run = Mock()
        assets = self.assets(self.identity)
        assets.pdf_extraction_provenance = "applied"
        assets.effective_pdf_extraction_mode = "fast"
        assets.extraction_result = ExtractionResult(normalize_pdf_identity(self.identity), self.document.raw_text)
        artifact = SourceArtifact(assets.extraction_result,
            storage_metadata={"pdf_extraction": self.identity.record(self.document.raw_text)})
        ready, _ = pipeline._prepare_structured_document(
            "fixture", self.document.raw_text, "zh", None, assets, True,
            SectionSplitterMode.COMMON, page_boundaries=[], extraction=artifact,
        )
        self.assertTrue(ready, assets.errors)
        self.assertEqual(self.store.save_calls, 1)
        self.assertTrue(self.identity.matches(self.store.document.parse_provenance["pdf_extraction"], self.store.document.raw_text))

    def test_missing_raw_text_preserves_derived_resources(self):
        self.pipeline.profile_store = Mock()
        self.pipeline.profile_store.exists.return_value = False
        self.pipeline.faiss_index_store = Mock()
        self.pipeline.bundle_provider = Mock()
        assets = self.assets(self.identity)
        self.assertFalse(self.pipeline._prepare_profile("fixture", None, "zh", assets, False)[0])
        self.assertFalse(self.pipeline._prepare_faiss("fixture", None, assets, False)[0])
        self.assertFalse(self.pipeline._prepare_bundle("fixture", None, assets, False))
        self.pipeline.profile_store.save.assert_not_called()
        self.pipeline.profile_store.clear.assert_not_called()
        self.pipeline.profile_store.load.assert_not_called()
        for dependency in (self.pipeline.faiss_index_store, self.pipeline.bundle_provider):
            self.assertEqual(dependency.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
