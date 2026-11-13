"""DOCLOAD-01/02: synthetic sources, real loaders/factory/pipeline, isolated stores."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from config.storage_namespace_helper import StorageNamespaceHelper
from shared.task_artifacts import DocumentTaskArtifacts
from doc_loaders.document_loader_factory import DocumentLoaderFactory
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from doc_loaders.text_document_loader import TextDocumentLoader
from test_pdf_page_inspection import page_fixture
from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _fake_document


def write_pdf(path):
    writer, _ = page_fixture(b'BT /F1 12 Tf 10 70 Td (Synthetic boundary paragraph.) Tj ET', with_font=True)
    writer.write(path)


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / 'custom-raw'
        self.raw.mkdir()
        self.pdf = PdfDocumentLoader(self.raw, ocr_enabled=False)
        self.txt = TextDocumentLoader(self.raw)
        for target in ('socket.socket.connect', 'subprocess.run'):
            guard = patch(target, side_effect=AssertionError('network/OCR/render'))
            guard.start()
            self.addCleanup(guard.stop)

    def pipeline(self, loader, name):
        store = _StructuredStore(exists_value=False, document=_fake_document(name))
        pipeline = _build_pipeline(loader=loader, store=store)
        pipeline.loader_factory = DocumentLoaderFactory(self.raw)
        pipeline.loader_factory.pdf_loader = self.pdf
        pipeline.loader_factory.txt_loader = self.txt
        pipeline.identity_resolver = lambda n: StorageNamespaceHelper.file_identity(
            n, self.root / 'structured', suffix='.structured.json', raw_root=self.raw)
        pipeline.structured_document_builder = Mock()
        pipeline.structured_document_builder.build.side_effect = lambda **kw: replace(
            _fake_document(name), raw_text=kw['raw_text'])
        return pipeline, store

    def test_all_pdf_reads_reject_unsafe_sources_before_parsing(self):
        outside = self.root / 'outside.pdf'
        write_pdf(outside)
        (self.raw / 'linked.pdf').symlink_to(outside)
        entries = [self.pdf.load, self.pdf.load_pages, self.pdf.open_source, self.pdf.extract,
                   self.pdf.inspect, self.pdf.inspect_pages, self.pdf.load_outline,
                   self.pdf.load_page_text_boundaries, self.pdf.load_page_boundary_evidence,
                   self.pdf.load_with_page_boundary_evidence, self.pdf.load_page_layout_evidence,
                   lambda name: self.pdf.render_source_region(name, '0' * 64, {})]
        with patch('doc_loaders.pdf_document_loader.PdfReader', side_effect=AssertionError('parsed outside source')) as parser:
            for name in ('../outside.pdf', str(outside), 'linked.pdf', 'x/../linked.pdf', r'C:\outside.pdf'):
                for entry in entries:
                    with self.subTest(name=name, entry=entry.__name__), self.assertRaises(ValueError):
                        entry(name)
            parser.assert_not_called()

    def test_pdf_preparation_rejection_never_generates_or_saves(self):
        outside = self.root / 'outside.pdf'
        write_pdf(outside)
        (self.raw / 'linked.pdf').symlink_to(outside)
        for name in ('../outside.pdf', str(outside), 'linked.pdf'):
            with self.subTest(name=name):
                pipeline, store = self.pipeline(self.pdf, name)
                with patch.object(pipeline.profile_builder, 'build', side_effect=AssertionError('profile')) as profile:
                    with self.assertRaises(ValueError):
                        pipeline.prepare(name, mode='base', pdf_extraction_mode='fast')
                    profile.assert_not_called()
                pipeline.structured_document_builder.build.assert_not_called()
                self.assertEqual(store.save_calls, 0)

    def test_safe_names_roots_and_suffixes_remain_usable(self):
        for name in ('中文書', 'English: A Reader', 'Who am I?', 'Book <Draft>', 'Upper.PDF'):
            with self.subTest(name=name):
                path = self.raw / (name if name.endswith('.PDF') else name + '.pdf')
                write_pdf(path)
                session = self.pdf.open_source(name)
                self.assertEqual(session.extract().result.canonical_text, 'Synthetic boundary paragraph.')
                session.verify_source()
        (self.raw / 'inside.pdf').symlink_to(self.raw / '中文書.pdf')
        session = self.pdf.open_source('inside.pdf')
        self.assertTrue(session.extract().result.canonical_text)
        session.verify_source()
        (self.raw / '中文書.txt').write_text('Different TXT content', encoding='utf-8')
        factory = DocumentLoaderFactory(self.raw)
        self.assertIs(factory.get('中文書.pdf'), factory.pdf_loader)
        self.assertIs(factory.get('中文書.txt'), factory.txt_loader)
        self.assertEqual(factory.get('中文書.txt').load('中文書.txt'), 'Different TXT content')
        with self.assertRaisesRegex(ValueError, 'source_ambiguous'):
            factory.get('中文書')

    def source(self):
        a, b, link = (self.raw / name for name in ('a.txt', 'b.txt', 'linked.txt'))
        a.write_text('Original source A.', encoding='utf-8')
        b.write_text('Replacement source B.', encoding='utf-8')
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(a)
        return a, b, link

    def mutate(self, kind, a, b, link):
        if kind == 'replace_target':
            b.replace(a)
            return
        link.unlink()
        if kind == 'retarget':
            link.symlink_to(b)
        elif kind == 'external':
            outside = self.root / 'outside.txt'
            outside.write_bytes(a.read_bytes())
            link.symlink_to(outside)
        elif kind == 'replace_link':
            link.write_text('A new direct source.', encoding='utf-8')

    def test_txt_logical_source_rechecked_after_capture(self):
        for kind in ('retarget', 'deleted', 'replace_link', 'replace_target', 'external'):
            with self.subTest(kind=kind):
                a, b, link = self.source()
                session = self.txt.open_source('linked.txt')
                self.mutate(kind, a, b, link)
                with self.assertRaises((ValueError, OSError)):
                    session.verify_source()
                if link.exists() or link.is_symlink():
                    link.unlink()

    def test_txt_changes_during_build_preserve_saved_content_and_notes(self):
        for kind in ('retarget', 'deleted', 'replace_link', 'replace_target', 'external'):
            with self.subTest(kind=kind):
                a, b, link = self.source()
                pipeline, store = self.pipeline(self.txt, 'linked.txt')
                self.assertTrue(pipeline.prepare('linked.txt', mode='base').structured_document_ready)
                store.document = replace(store.document, document_task_artifacts=DocumentTaskArtifacts(
                    metadata={'note': {'existing-target': {'note': {'text': 'Keep this existing note'}}}}))
                before = deepcopy(store.document.to_dict())
                def build(**kw):
                    self.mutate(kind, a, b, link)
                    return replace(_fake_document('linked.txt'), raw_text=kw['raw_text'])
                pipeline.structured_document_builder.build.side_effect = build
                result = pipeline.prepare('linked.txt', mode='base', force_rebuild=True)
                self.assertFalse(result.structured_document_ready, result.errors)
                self.assertEqual(store.save_calls, 1)
                self.assertEqual(store.document.to_dict(), before)
                self.assertTrue(result.errors)
                if link.exists() or link.is_symlink():
                    link.unlink()

    def test_cache_checks_logical_source_and_unchanged_hit_never_extracts(self):
        for kind in ('retarget', 'deleted', 'replace_link', 'replace_target', 'external'):
            with self.subTest(kind=kind):
                a, b, link = self.source()
                pipeline, store = self.pipeline(self.txt, 'linked.txt')
                self.assertTrue(pipeline.prepare('linked.txt', mode='base').structured_document_ready)
                before = deepcopy(store.document.to_dict())
                with patch.object(self.txt, 'extract', side_effect=AssertionError('warm extraction')) as extract, \
                        patch.object(pipeline.profile_builder, 'build', side_effect=AssertionError('warm profile')) as profile:
                    self.assertTrue(pipeline.prepare('linked.txt', mode='base').structured_document_ready)
                    extract.assert_not_called()
                    profile.assert_not_called()
                original = store.load
                def change_source(target):
                    document = original(target)
                    self.mutate(kind, a, b, link)
                    return document
                with patch.object(store, 'load', side_effect=change_source):
                    result = pipeline.prepare('linked.txt', mode='base')
                self.assertFalse(result.structured_document_ready, result.errors)
                self.assertEqual(store.save_calls, 1)
                self.assertEqual(store.document.to_dict(), before)

    def test_pdf_session_rechecks_external_retarget_even_with_identical_bytes(self):
        target = self.raw / 'book.pdf'
        write_pdf(target)
        link = self.raw / 'linked.pdf'
        link.symlink_to(target)
        session = self.pdf.open_source('linked.pdf')
        outside = self.root / 'outside.pdf'
        outside.write_bytes(target.read_bytes())
        link.unlink()
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'source_identity_invalid'):
            session.verify_source()

    def test_txt_rejects_unsafe_initial_source(self):
        outside = self.root / 'outside.txt'
        outside.write_text('Synthetic outside text', encoding='utf-8')
        (self.raw / 'linked.txt').symlink_to(outside)
        for name in ('../outside.txt', str(outside), 'linked.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.txt.open_source(name)


if __name__ == '__main__':
    unittest.main()
