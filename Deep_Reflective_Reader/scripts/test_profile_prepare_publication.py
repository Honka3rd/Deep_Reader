"""PROFILE-05: real Prepare/store concurrency, using file and disposable PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from threading import Event
from unittest.mock import patch
from uuid import uuid4
import unittest

import test_profile_persistence_contract as contract
from profile.document_profile_store import DocumentProfileStore
from profile.profile_publication import ProfilePublicationConflict, document_binding


class PublicationCases:
    def setUp(self):
        self.c = self.contract_case('test_new_publication_wins_against_older_completion')
        self.c.setUp()
        self.addCleanup(self.c.doCleanups)
        self.f = self.c.f

    def replacement(self, title):
        document = self.f.store.load(self.f.config())
        return replace(document, title=title, parse_provenance={
            **document.parse_provenance, 'interaction_epoch': uuid4().hex})

    def save(self, document):
        return self.f.pipeline._save_structured_document_atomically(
            document=document, storage_config=self.f.config())

    def test_publication_fences_direct_and_prepare_store_writers(self):
        for direct in (True, False):
            with self.subTest(direct=direct):
                guard = self.c.guard()
                document = self.replacement('new structure')
                entered = Event()
                def write():
                    entered.set()
                    if direct:
                        return self.f.store.save(document, self.f.config())
                    return self.save(document)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with guard.publication() as validate:
                        future = pool.submit(write)
                        self.assertTrue(entered.wait(5))
                        with self.assertRaises(TimeoutError):
                            future.result(.2)
                        DocumentProfileStore.save(contract.profile('ordered first'), self.c.config,
                                                  validate=validate)
                    future.result(10)
                before = self.c.path.read_bytes()
                self.assertNotEqual(document_binding(self.f.store.load(self.f.config())), guard.binding)
                with self.assertRaises(ProfilePublicationConflict):
                    self.c.publish(guard, contract.profile('late'))
                self.assertEqual(self.c.path.read_bytes(), before)

    def test_full_prepare_waits_for_profile_publication(self):
        entered, release, saved = Event(), Event(), Event()
        original = self.f.pipeline._save_structured_document_atomically
        def paused(**kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError('release timeout')
            result = original(**kwargs)
            saved.set()
            return result
        with patch.object(self.f.pipeline, '_save_structured_document_atomically', side_effect=paused):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.f.pipeline.prepare, 'book', force_rebuild=True)
                try:
                    self.assertTrue(entered.wait(5))
                    guard = self.c.guard()
                    with guard.publication() as validate:
                        release.set()
                        self.assertFalse(saved.wait(.2))
                        DocumentProfileStore.save(contract.profile('ordered first'), self.c.config,
                                                  validate=validate)
                finally:
                    release.set()
                result = future.result(10)
        self.assertTrue(saved.is_set())
        self.assertFalse(result.profile_ready, result.errors)
        before = self.c.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.c.publish(guard, contract.profile('late'))
        self.assertEqual(self.c.path.read_bytes(), before)

    def test_completed_prepare_rejects_old_generation(self):
        old = self.c.guard()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(self.f.pipeline.prepare, 'book', force_rebuild=True).result(10)
        self.assertTrue(result.profile_ready, result.errors)
        self.assertTrue(result.structured_document_ready, result.errors)
        self.assertNotEqual(old.binding, document_binding(self.f.store.load(self.f.config())))
        before = self.c.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.c.publish(old, contract.profile('late'))
        self.assertEqual(self.c.path.read_bytes(), before)
        self.c.publish(self.c.guard(), contract.profile('new'))
        self.assertEqual(DocumentProfileStore.load(self.c.config).topic, 'new')

    def test_enrichment_publication_fences_prepare_writer(self):
        entered, release, attempting = Event(), Event(), Event()
        original = self.f.pipeline.profile_store.save
        def paused(profile, config, **kwargs):
            if profile.post_structure_metadata is not None:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('release timeout')
            return original(profile, config, **kwargs)
        with patch.object(self.f.pipeline.profile_store, 'save', side_effect=paused):
            with ThreadPoolExecutor(max_workers=2) as pool:
                prepare = pool.submit(self.f.pipeline.prepare, 'book', force_rebuild=True)
                try:
                    self.assertTrue(entered.wait(5))
                    document = self.replacement('after enrichment')
                    def write():
                        attempting.set()
                        return self.save(document)
                    writer = pool.submit(write)
                    self.assertTrue(attempting.wait(5))
                    with self.assertRaises(TimeoutError):
                        writer.result(.2)
                finally:
                    release.set()
                prepare.result(10)
                writer.result(10)
        self.assertEqual(self.f.store.load(self.f.config()).title, 'after enrichment')
        self.assertIsNotNone(DocumentProfileStore.load(self.c.config).post_structure_metadata)

    def test_late_enrichment_after_new_prepare_preserves_new_profile(self):
        entered, release = Event(), Event()
        original = self.f.pipeline.post_structure_enricher.enrich
        calls = []
        def paused(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('release timeout')
                return replace(original(**kwargs), topic='late enrichment')
            return replace(original(**kwargs), topic='new enrichment')
        with patch.object(self.f.pipeline.post_structure_enricher, 'enrich', side_effect=paused):
            with ThreadPoolExecutor(max_workers=2) as pool:
                old = pool.submit(self.f.pipeline.prepare, 'book', force_rebuild=True)
                try:
                    self.assertTrue(entered.wait(5))
                    new = pool.submit(self.f.pipeline.prepare, 'book', force_rebuild=True).result(10)
                    self.assertTrue(new.profile_ready, new.errors)
                    before = self.c.path.read_bytes()
                finally:
                    release.set()
                result = old.result(10)
        self.assertFalse(result.profile_ready, result.errors)
        self.assertFalse(result.bundle_ready)
        self.assertEqual(self.c.path.read_bytes(), before)
        self.assertEqual(DocumentProfileStore.load(self.c.config).topic, 'new enrichment')

    def test_enrichment_rejects_structure_only_prepare_write(self):
        before = []
        def enrich(**kwargs):
            # Change only hierarchy incarnation, not source or profile digest.
            before.append(self.c.path.read_bytes())
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(self.save, self.replacement('new incarnation')).result(10)
            return replace(kwargs['profile'], topic='late enrichment')
        with patch.object(self.f.pipeline.post_structure_enricher, 'enrich', side_effect=enrich):
            result = self.f.pipeline.prepare('book', force_rebuild=True)
        self.assertFalse(result.profile_ready, result.errors)
        self.assertFalse(result.bundle_ready)
        self.assertEqual(self.c.path.read_bytes(), before[0])
        self.assertEqual(self.f.store.load(self.f.config()).title, 'new incarnation')

    def readback_hook(self):
        if isinstance(self.f.store, contract.safety.StructuredDocumentStore):
            return type(self.f.store), 'load', lambda value: value
        return self.f.store, 'load_enhancement_snapshot', lambda value: value.document

    def test_save_readback_is_own_snapshot_and_holds_writer_boundary(self):
        first, second = self.replacement('first'), self.replacement('second')
        entered, release, attempting = Event(), Event(), Event()
        target, name, document_of = self.readback_hook()
        original = getattr(target, name)
        def paused(*args, **kwargs):
            value = original(*args, **kwargs)
            if document_of(value).title == 'first' and not entered.is_set():
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('release timeout')
            return value
        with patch.object(target, name, side_effect=paused):
            with ThreadPoolExecutor(max_workers=2) as pool:
                own = pool.submit(self.save, first)
                try:
                    self.assertTrue(entered.wait(5))
                    def write():
                        attempting.set()
                        return self.save(second)
                    other = pool.submit(write)
                    self.assertTrue(attempting.wait(5))
                    with self.assertRaises(TimeoutError):
                        other.result(.2)
                finally:
                    release.set()
                own_value, other_value = own.result(10), other.result(10)
        self.assertEqual(own_value.title, 'first')
        self.assertEqual(other_value.title, 'second')
        self.assertNotEqual(document_binding(own_value), document_binding(other_value))
        self.assertEqual(document_binding(other_value), document_binding(self.f.store.load(self.f.config())))

    def test_competing_write_after_own_readback_cannot_be_adopted_for_enrichment(self):
        original = self.f.store.save_and_load
        before = []
        def competing(document, target):
            own = original(document, target)
            other = replace(own, title='competitor', parse_provenance={
                **own.parse_provenance, 'interaction_epoch': uuid4().hex})
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(original, other, target).result(10)
            before.append(self.c.path.read_bytes())
            return own
        with patch.object(self.f.store, 'save_and_load', side_effect=competing):
            result = self.f.pipeline.prepare('book', force_rebuild=True)
        self.assertEqual(self.f.store.load(self.f.config()).title, 'competitor')
        self.assertFalse(result.profile_ready, result.errors)
        self.assertFalse(result.bundle_ready)
        self.assertEqual(self.c.path.read_bytes(), before[0])

    def test_save_failure_retains_old_structure_and_profile(self):
        before = self.f.store.load(self.f.config()).to_dict()
        profile_before = self.c.path.read_bytes()
        if isinstance(self.f.store, contract.safety.StructuredDocumentStore):
            with patch('document_structure.structured_document_store.os.replace', side_effect=OSError('save failure')):
                with self.assertRaises(OSError):
                    self.save(self.replacement('failed'))
        else:
            original = self.f.store.load_enhancement_snapshot
            def fail(*args, **kwargs):
                value = original(*args, **kwargs)
                if value.document.title == 'failed':
                    raise OSError('readback failure after SQL save')
                return value
            with patch.object(self.f.store, 'load_enhancement_snapshot', side_effect=fail):
                with self.assertRaises(OSError):
                    self.save(self.replacement('failed'))
        self.assertEqual(self.f.store.load(self.f.config()).to_dict(), before)
        self.assertEqual(self.c.path.read_bytes(), profile_before)


class FilePublicationTests(PublicationCases, unittest.TestCase):
    contract_case = contract.FileProfileTests


class PostgresPublicationTests(PublicationCases, unittest.TestCase):
    contract_case = contract.PostgresProfileTests

    @classmethod
    def setUpClass(cls):
        cls.contract_case.setUpClass()
        cls.addClassCleanup(cls.contract_case.doClassCleanups)


if __name__ == '__main__':
    unittest.main()
