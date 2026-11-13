"""Profile publication and read contracts with real file / disposable PostgreSQL."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import unittest

from config.faiss_storage_config import FaissStorageConfig
from config.profile_publication_factory import profile_publication_guard
from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from profile.versioned_document_profile_store import VersionedDocumentProfileStore
from profile.profile_publication import ProfilePublicationConflict
import test_preparation_runtime_safety as safety


def profile(topic="fresh"):
    return DocumentProfile(topic=topic, summary="A valid source summary.", document_language=LanguageCode.EN)


class ProfileCases:
    def setUp(self):
        self.f = self.fixture()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.seed()
        from profile.post_structure_metadata_enricher import PostStructureMetadataEnricher
        self.f.pipeline.post_structure_enricher = PostStructureMetadataEnricher()
        self.config = FaissStorageConfig("book")
        self.path = Path(self.config.get_raw_profile_path())
        self.repo = self.f.repository()
        self.f.pipeline.profile_store = VersionedDocumentProfileStore(self.repo)

    def guard(self):
        return profile_publication_guard(name="book", source=self.f.loader.open_source("book"),
            store=self.f.store, target=self.f.config(), profile_config=self.config).capture()

    def publish(self, guard, value):
        with guard.publication() as validate:
            DocumentProfileStore.save(value, self.config, validate=validate)

    def test_generation_source_change_preserves_bytes_and_stops_following_work(self):
        before = self.path.read_bytes()
        def generate(**kw):
            self.f.path.write_text("Changed during model work.")
            return profile("late")
        with patch.object(self.f.pipeline.profile_builder, "build", side_effect=generate), \
             patch.object(self.f.pipeline.structured_document_builder, "build", side_effect=AssertionError("late structure")), \
             patch.object(self.f.pipeline.node_provider, "parse", side_effect=AssertionError("late index")):
            result = self.f.pipeline.prepare("book", force_rebuild=True)
        self.assertFalse(result.profile_ready)
        self.assertFalse(result.bundle_ready)
        self.assertTrue(any("profile_publication_rejected" in e for e in result.errors), result.errors)
        self.assertEqual(self.path.read_bytes(), before)

    def test_source_change_during_serialization_preserves_bytes_and_cleans_temp(self):
        before = self.path.read_bytes()
        guard = self.guard()
        import profile.document_profile_store as module
        dump = module.json.dump
        def late(*args, **kw):
            result = dump(*args, **kw)
            self.f.path.write_text("Changed while writing candidate.")
            return result
        with patch.object(module.json, "dump", side_effect=late):
            with self.assertRaises(ProfilePublicationConflict):
                self.publish(guard, profile("late"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".profile-*")), [])

    def test_new_publication_wins_against_older_completion(self):
        old = self.guard()
        new = self.guard()
        self.publish(new, profile("new"))
        before = self.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(old, profile("late"))
        self.assertEqual(before, self.path.read_bytes())

    def test_document_reparse_rejects_old_writer_same_source(self):
        old = self.guard()
        document = self.repo.load_document("book")
        self.repo.save_reparsed_document(document, "book")
        before = self.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(old, profile("late"))
        self.assertEqual(before, self.path.read_bytes())
        self.publish(self.guard(), profile("fresh"))
        self.assertEqual(DocumentProfileStore.load(self.config).topic, "fresh")

    def http(self):
        from fastapi.testclient import TestClient
        import main
        for key, value in (("_RAW_DOCUMENT_BASE_DIR", self.f.path.parent),
                           ("_DERIVED_DOCUMENT_BASE_DIR", self.config.base_dir.parent),
                           ("document_artifact_repository", self.repo)):
            p = patch.object(main, key, value)
            p.start(); self.addCleanup(p.stop)
        client = TestClient(main.app)
        self.addCleanup(client.close)
        return client

    def test_http_delete_reimport_same_bytes_rejects_delayed_result(self):
        old = self.guard()
        original = self.f.path.read_bytes()
        client = self.http()
        deleted = client.post("/documents/lifecycle/delete", json={"doc_name":"book", "action":"delete", "confirm":True})
        self.assertEqual(deleted.status_code, 200, deleted.text)
        uploaded = client.post("/documents/upload", files={"file": ("book.txt", original, "text/plain")})
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        DocumentProfileStore.save(profile("replacement"), self.config)
        before = self.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(old, profile("late"))
        self.assertEqual(self.f.path.read_bytes(), original)
        self.assertEqual(self.path.read_bytes(), before)

    def test_http_rename_roundtrip_rejects_delayed_result(self):
        from uuid import uuid4
        old = self.guard()
        client = self.http()
        def rename(source, target):
            context = client.get("/documents/lifecycle/rename-context", params={"doc_name": source})
            self.assertEqual(context.status_code, 200, context.text)
            response = client.post("/documents/lifecycle/rename", json={"doc_name": source,
                "action":"rename", "new_doc_name":target, "command_id":str(uuid4()),
                "expected_source_token":context.json()["source_token"]})
            self.assertEqual(response.status_code, 200, response.text)
        rename("book", "renamed")
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(old, profile("late"))
        self.assertFalse(self.path.exists())
        rename("renamed", "book")
        before = self.path.read_bytes()
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(old, profile("late"))
        self.assertEqual(self.path.read_bytes(), before)
        self.publish(self.guard(), profile("new"))

    def test_enrichment_source_change_rejects_publication(self):
        saved = []
        def enrich(**kw):
            saved.append(self.path.read_bytes())
            self.f.path.write_text("Replacement during enrichment")
            return replace(kw["profile"], topic="late-enrichment")
        with patch.object(self.f.pipeline.post_structure_enricher, "enrich", side_effect=enrich):
            result = self.f.pipeline.prepare("book", force_rebuild=True)
        self.assertFalse(result.profile_ready)
        self.assertFalse(result.bundle_ready)
        self.assertEqual(self.path.read_bytes(), saved[0])

    def test_enrichment_reparse_rejects_old_snapshot(self):
        saved = []
        def enrich(**kw):
            saved.append(self.path.read_bytes())
            self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
            return replace(kw["profile"], topic="late-enrichment")
        with patch.object(self.f.pipeline.post_structure_enricher, "enrich", side_effect=enrich):
            result = self.f.pipeline.prepare("book", force_rebuild=True)
        self.assertFalse(result.profile_ready)
        self.assertFalse(result.bundle_ready)
        self.assertEqual(self.path.read_bytes(), saved[0])

    def test_guard_construction_is_pure_and_capture_cannot_rebind(self):
        from app.document_file_identity import DocumentFileIdentity
        identity = DocumentFileIdentity(self.f.path.parent)
        before = list(identity.directory.glob("**/*")) if identity.directory.exists() else []
        guard = profile_publication_guard(name="book", source=self.f.loader.open_source("book"),
            store=self.f.store, target=self.f.config(), profile_config=self.config)
        after = list(identity.directory.glob("**/*")) if identity.directory.exists() else []
        self.assertEqual(before, after)
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(guard, profile())
        guard.capture()
        with self.assertRaises(ProfilePublicationConflict):
            guard.capture()

    def test_delayed_generation_after_new_prepare_preserves_new_result(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        started, release = Event(), Event()
        calls = []
        def generate(**kw):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                if not release.wait(10):
                    raise RuntimeError("test release timeout")
                return profile("late")
            return profile("new")
        with patch.object(self.f.pipeline.profile_builder, "build", side_effect=generate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                old = pool.submit(self.f.pipeline.prepare, "book", force_rebuild=True)
                try:
                    self.assertTrue(started.wait(5))
                    new = pool.submit(self.f.pipeline.prepare, "book", force_rebuild=True).result(8)
                    self.assertTrue(new.profile_ready, new.errors)
                    before = self.path.read_bytes()
                finally:
                    release.set()
                late = old.result(5)
        self.assertFalse(late.profile_ready)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(DocumentProfileStore.load(self.config).topic, "new")

    def test_publication_snapshot_blocks_lifecycle_writer(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        guard = self.guard()
        document = self.repo.load_document("book")
        entered = Event()
        def reparse():
            entered.set()
            self.repo.save_reparsed_document(document, "book")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with guard.publication() as validate:
                writer = pool.submit(reparse)
                self.assertTrue(entered.wait(2))
                from concurrent.futures import TimeoutError
                with self.assertRaises(TimeoutError):
                    writer.result(0.1)
                DocumentProfileStore.save(profile("linearized"), self.config, validate=validate)
            writer.result(5)
        with self.assertRaises(ProfilePublicationConflict):
            self.publish(guard, profile("late"))

    def test_enrichment_failure_remains_nonblocking_for_structure(self):
        with patch.object(self.f.pipeline.post_structure_enricher, "enrich", side_effect=RuntimeError("optional enrichment")):
            result = self.f.pipeline.prepare("book", force_rebuild=True)
        self.assertTrue(result.structured_document_ready, result.errors)
        self.assertTrue(result.profile_ready, result.errors)
        self.assertTrue(any("post_structure_enrichment_failed" in e for e in result.errors))

    def test_atomic_profile_replace_failure_preserves_prior_bytes(self):
        before = self.path.read_bytes()
        with patch("profile.document_profile_store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.publish(self.guard(), profile("candidate"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".profile-*")), [])

    def test_explicit_generation_enrichment_then_zero_work_runtime(self):
        result = self.f.pipeline.prepare("book", force_rebuild=True)
        self.assertTrue(result.profile_ready, result.errors)
        self.assertTrue(result.bundle_ready, result.errors)
        self.assertIsNotNone(DocumentProfileStore.load(self.config).post_structure_metadata)
        before = self.path.read_bytes()
        with patch.object(self.f.pipeline.profile_builder, "build", side_effect=AssertionError("runtime model")), \
             patch.object(self.f.pipeline.profile_store, "save", side_effect=AssertionError("runtime save")), \
             patch.object(self.f.loader, "extract", side_effect=AssertionError("runtime extraction")):
            runtime = self.f.pipeline.prepare_and_load("book")
        self.assertTrue(runtime.assets.bundle_ready, runtime.assets.errors)
        self.assertEqual(before, self.path.read_bytes())

    def test_invalid_persisted_payload_matrix_is_readonly_not_ready(self):
        valid = profile().to_dict()
        invalid = [{}, [], None]
        for key in valid:
            invalid.append({k:v for k,v in valid.items() if k != key})
            for value in (None, "", "  ", 17, False, [], {}):
                invalid.append(dict(valid, **{key:value}))
        invalid.append(dict(valid, document_language="not-a-language"))
        for payload in invalid + ["malformed-json"]:
            with self.subTest(payload=payload):
                raw = "{bad" if payload == "malformed-json" else json.dumps(payload)
                self.path.write_text(raw)
                before = self.path.read_bytes()
                with patch.object(self.f.pipeline.profile_builder, "build", side_effect=AssertionError("implicit model")), \
                     patch.object(self.f.pipeline.profile_store, "save", side_effect=AssertionError("implicit repair")):
                    result = self.f.pipeline.prepare_and_load("book")
                self.assertFalse(result.assets.profile_ready)
                self.assertIsNone(result.bundle)
                self.assertTrue(result.assets.errors)
                with self.assertRaises((ValueError, TypeError)):
                    VersionedDocumentProfileStore(self.repo).load_source(self.config)
                self.assertEqual(self.path.read_bytes(), before)

    def test_valid_legacy_and_source_only_reads_do_not_write_or_load_repository(self):
        self.path.write_text(json.dumps(profile("legacy").to_dict()))
        before = self.path.read_bytes()
        with patch.object(self.repo, "profile_structure_invalidated", create=True,
                          side_effect=AssertionError("source-only repository read")):
            value = VersionedDocumentProfileStore(self.repo).load_source(self.config)
        self.assertEqual(value.topic, "legacy")
        self.assertIsNone(value.parser_metadata)
        self.assertIsNone(value.post_structure_metadata)
        self.assertIsNone(value.structure_profile)
        result = self.f.pipeline.prepare_and_load("book")
        self.assertTrue(result.assets.bundle_ready, result.assets.errors)
        self.assertEqual(self.path.read_bytes(), before)


class FileProfileTests(ProfileCases, unittest.TestCase):
    fixture = safety.RuntimeSafetyTests


class PostgresProfileTests(ProfileCases, unittest.TestCase):
    fixture = safety.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)


if __name__ == "__main__":
    unittest.main()
