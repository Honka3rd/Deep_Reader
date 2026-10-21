"""Preflight against a disposable PostgreSQL schema, never production documents.

Set DEEP_READER_TEST_POSTGRES_DSN explicitly to run.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from fastapi.testclient import TestClient
import main
from app.enhancement_preflight import EnhancementConflict, EnhancementPreflightService
from app.section_task_coordinator import SectionTaskCoordinator
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from db.postgres_structured_document_store import EnhancementCommitConflict
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
import test_accepted_source_snapshot as fixtures


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "isolated PostgreSQL DSN required")
class PostgresPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
        cls.schema = "enhancement_test_" + uuid4().hex
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.scoped = make_conninfo(cls.dsn, options=f"-c search_path={cls.schema}")
        with psycopg.connect(cls.scoped) as connection:
            for migration in sorted((Path(__file__).resolve().parents[1] / "db/migrations").glob("*.sql")):
                connection.execute(migration.read_text())

    @classmethod
    def drop_schema(cls):
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        self.fixture = fixtures.AcceptedSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.source.rename(Path(self.fixture.temp.name) / "doc.pdf")
        self.store = PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex)
        self.repository = PostgresStructuredDocumentArtifactRepository(self.store)
        self.repository.save_reparsed_document(self.fixture.document, "doc")
        self.coordinator = object.__new__(SectionTaskCoordinator)
        self.coordinator.document_artifact_repository = self.repository
        self.coordinator.document_preparation_pipeline = Mock()
        self.coordinator.document_preparation_pipeline.loader_factory.base_dir = self.fixture.temp.name
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        capacity_pipeline = candidate_pipeline(Provider())
        self.service = EnhancementPreflightService(self.coordinator.read_enhancement_state,
            lease=self.coordinator.enhancement_lease, availability=lambda _: None,
            capacity_reader=lambda state, hint, intent: (
                capacity_pipeline if isinstance(self.coordinator.document_preparation_pipeline, Mock)
                else self.coordinator.document_preparation_pipeline
            ).inspect_enhancement_capacity(snapshot=state.source, toc_hint=hint, enhancement_intent=intent))
        with psycopg.connect(self.scoped) as connection:
            self.doc_id = connection.execute("SELECT id FROM documents WHERE namespace=%s",
                                             (self.store._namespace,)).fetchone()[0]

    def snapshot(self):
        return self.repository.load_enhancement_snapshot("doc")

    def token(self):
        result = self.service.preflight(doc_name="doc")
        self.assertTrue(result["ready"], result)
        return result["confirmation_token"]

    def add_artifact(self, kind="analysis", source_version=1):
        with psycopg.connect(self.scoped) as connection:
            connection.execute("""INSERT INTO artifacts
                (document_id, artifact_type, target_type, target_id, source_structure_version, payload)
                VALUES (%s, %s, 'document', %s, %s, '{}')""",
                (self.doc_id, kind, self.doc_id, source_version))

    def test_real_route_reads_without_mutation_or_providers(self):
        before = self.snapshot()
        with patch.object(main, "enhancement_preflight_service", self.service):
            response = TestClient(main.app).post("/documents/reparse-structure/preflight",
                                                json={"doc_name": "doc"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ready"])
        self.assertTrue(response.json()["execution_available"])
        self.assertEqual(response.json()["structure_version"], 1)
        self.assertEqual(response.json()["interaction_count"], 0)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.coordinator.document_preparation_pipeline.mock_calls, [])

    def test_new_edited_deleted_artifacts_invalidate_consent(self):
        token = self.token()
        self.add_artifact()
        self.assert_stale(token)
        self.assertEqual(self.snapshot().interaction_count, 1)
        token = self.token()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE artifacts SET payload=%s WHERE document_id=%s",
                               (Jsonb({"answer": "changed"}), self.doc_id))
        self.assert_stale(token)
        token = self.token()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("DELETE FROM artifacts WHERE document_id=%s", (self.doc_id,))
        self.assert_stale(token)

    def assert_stale(self, token):
        with self.assertRaisesRegex(EnhancementConflict, "confirmation_stale"):
            self.service.validate(doc_name="doc", discard_interactions=True, confirmation_token=token)

    def test_embedded_unknown_counts_and_explicit_notes(self):
        token = self.token()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE sections SET metadata_payload=%s WHERE document_id=%s",
                               (Jsonb({"notes": [{"text": "user note"}]}), self.doc_id))
        self.assert_stale(token)
        state = self.snapshot()
        self.assertIsNone(state.interaction_count)
        self.assertTrue(state.includes_user_notes)

    def test_document_lease_blocks_second_connection_and_releases_on_error(self):
        with self.repository.enhancement_lease("doc"):
            with self.assertRaises(BlockingIOError):
                with self.repository.enhancement_lease("doc.pdf"):
                    self.fail("lease acquired twice")
            other = PostgresStructuredDocumentStore(self.scoped, namespace="another")
            with other.enhancement_lease(other.target_for_doc_name("doc")):
                pass
        with self.assertRaisesRegex(ValueError, "candidate failure"):
            with self.service.authorize_execution(doc_name="doc", discard_interactions=True,
                                                  confirmation_token=self.token()):
                raise ValueError("candidate failure")
        with self.repository.enhancement_lease("doc"):
            pass

    def test_namespace_isolation_and_source_change(self):
        other = PostgresStructuredDocumentStore(self.scoped, namespace="missing")
        with self.assertRaises(FileNotFoundError):
            other.load_enhancement_snapshot(other.target_for_doc_name("doc"))
        (Path(self.fixture.temp.name) / "doc.pdf").write_bytes(b"changed")
        result = self.service.preflight(doc_name="doc")
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "source_changed")
        self.assertIsNone(result["confirmation_token"])

    def test_unsupported_backend_is_not_falsely_ready(self):
        self.coordinator.document_artifact_repository = object()
        result = self.service.preflight(doc_name="doc")
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "enhancement_storage_backend_unsupported")

    def test_concurrent_insert_cannot_mix_snapshot_versions(self):
        driver, row_factory, json_type = self.store._load_psycopg()
        owner = self

        class ConnectionProbe:
            def __init__(self, connection):
                self.connection = connection
                self.inserted = False

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, *args):
                return self.connection.__exit__(*args)

            def execute(self, statement, *args):
                if "FROM chapters" in statement and not self.inserted:
                    self.inserted = True
                    settings = self.connection.execute(
                        "SELECT current_setting('transaction_read_only') AS ro, "
                        "current_setting('transaction_isolation') AS isolation").fetchone()
                    owner.assertEqual(settings, {"ro": "on", "isolation": "repeatable read"})
                    owner.add_artifact()
                return self.connection.execute(statement, *args)

        proxy = SimpleNamespace(connect=lambda *a, **kw: ConnectionProbe(driver.connect(*a, **kw)))
        with patch.object(self.store, "_load_psycopg", return_value=(proxy, row_factory, json_type)):
            old = self.snapshot()
        current = self.snapshot()
        self.assertEqual(old.interaction_count, 0)
        self.assertEqual(current.interaction_count, 1)
        self.assertNotEqual(old.interaction_revision, current.interaction_revision)

    def test_confirmed_route_remains_gated_and_releases_lease(self):
        token = self.token()
        self.service.availability = None
        before = self.snapshot()
        with patch.object(main, "section_task_coordinator", self.coordinator), \
             patch.object(main, "enhancement_preflight_service", self.service), \
             patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            response = TestClient(main.app).post("/documents/reparse-structure", json={
                "doc_name": "doc", "parser_mode": "llm_enhanced",
                "discard_interactions": True, "confirmation_token": token})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"], "enhancement_execution_unavailable")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.coordinator.document_preparation_pipeline.mock_calls, [])
        with self.repository.enhancement_lease("doc"):
            pass

    def test_internal_candidate_from_db_never_reextracts_or_persists(self):
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        provider = Provider()
        pipeline = candidate_pipeline(provider)
        pipeline.loader_factory = SimpleNamespace(base_dir=self.fixture.temp.name)
        self.coordinator.document_preparation_pipeline = pipeline
        self.add_artifact()
        before = self.snapshot()
        token = self.token()
        with patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            with self.assertRaisesRegex(EnhancementConflict, "confirmation_required"):
                with self.coordinator.enhancement_candidate(doc_name="doc"):
                    self.fail("candidate built without consent")
            self.assertEqual(provider.calls, 0)
            with self.coordinator.enhancement_candidate(doc_name="doc", discard_interactions=True,
                                                       confirmation_token=token) as (state, candidate):
                self.assertEqual(candidate.document.raw_text, self.fixture.text)
                self.assertEqual(state.source, candidate.source)
                with self.assertRaises(BlockingIOError):
                    with self.repository.enhancement_lease("doc"):
                        self.fail("candidate did not hold lease")
        self.assertEqual(before, self.snapshot())
        self.assertEqual(provider.calls, 1)
        with self.repository.enhancement_lease("doc"):
            pass

    def test_failed_internal_candidate_preserves_interactions_and_releases_lease(self):
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        from document_preparation.structure_enhancement_candidate import EnhancementCandidateRejected
        pipeline = candidate_pipeline(Provider(lambda payload, context: {"invalid": True}))
        pipeline.loader_factory = SimpleNamespace(base_dir=self.fixture.temp.name)
        self.coordinator.document_preparation_pipeline = pipeline
        self.add_artifact()
        before = self.snapshot()
        token = self.token()
        with patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            with self.assertRaises(EnhancementCandidateRejected):
                with self.coordinator.enhancement_candidate(doc_name="doc", discard_interactions=True,
                                                           confirmation_token=token):
                    self.fail("invalid plan returned a candidate")
        self.assertEqual(before, self.snapshot())
        with self.repository.enhancement_lease("doc"):
            pass

    def make_candidate(self):
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        state = self.coordinator.read_enhancement_state("doc")
        candidate = candidate_pipeline(Provider()).build_enhancement_candidate(snapshot=state.source)
        return state, candidate

    def enable_execution(self, transform=None):
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        from profile.versioned_document_profile_store import VersionedDocumentProfileStore
        self.provider = Provider(transform)
        self.coordinator.document_preparation_pipeline = candidate_pipeline(self.provider)
        self.coordinator.document_preparation_pipeline.loader_factory = SimpleNamespace(base_dir=self.fixture.temp.name)
        self.coordinator.document_profile_store = VersionedDocumentProfileStore(self.repository)
        self.service.availability = self.coordinator.enhancement_unavailable_reason

    def execute_route(self, token=None):
        with patch.object(main, "section_task_coordinator", self.coordinator), \
             patch.object(main, "enhancement_preflight_service", self.service), \
             patch("app.enhancement_preflight.enhancement_preflight_service", self.service):
            return TestClient(main.app).post("/documents/reparse-structure", json={
                "doc_name": "doc", "parser_mode": "llm_enhanced",
                "discard_interactions": True, "confirmation_token": token or self.token()})

    def test_enabled_route_commits_without_extraction_and_invalidates_file_profile(self):
        from profile.document_profile import DocumentProfile, PostStructureMetadata
        from profile.document_profile_store import DocumentProfileStore
        from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
        self.enable_execution()
        profile_path = Path(self.fixture.temp.name) / "profile.json"
        config = SimpleNamespace(doc_name="doc", get_raw_profile_path=lambda: str(profile_path))
        profile = DocumentProfile.from_dict({"document_language": "en", "summary": "keep", "topic": "topic"})
        from dataclasses import replace
        profile = replace(profile, post_structure_metadata=PostStructureMetadata())
        DocumentProfileStore.save(profile, config)
        original = profile_path.read_bytes()
        self.assertIsNotNone(self.coordinator.document_profile_store.load(config).post_structure_metadata)
        self.add_artifact("note")
        source = (Path(self.fixture.temp.name) / "doc.pdf").read_bytes()
        self.assertTrue(self.service.preflight(doc_name="doc")["execution_available"])
        with patch("subprocess.run", side_effect=AssertionError("OCR/render")), \
             patch.object(DocumentPreparationPipeline, "prepare_and_load", side_effect=AssertionError("full prepare")):
            result = self.execute_route()
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["structure_version"], 2)
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(self.snapshot().interaction_count, 0)
        self.assertEqual(self.coordinator.read_enhancement_state("doc").source.canonical_text, self.fixture.text)
        self.assertIsNone(self.coordinator.document_profile_store.load(config).post_structure_metadata)
        self.assertEqual(profile_path.read_bytes(), original)
        self.assertEqual((Path(self.fixture.temp.name) / "doc.pdf").read_bytes(), source)

    def test_enabled_route_candidate_failure_preserves_storage(self):
        self.enable_execution(lambda payload, context: {"invalid": True})
        self.add_artifact()
        before = self.snapshot()
        result = self.execute_route()
        self.assertEqual(result.status_code, 422, result.text)
        self.assertFalse(result.json()["success"], result.text)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.repository.profile_structure_invalidated("doc"))
        self.assertEqual(self.provider.calls, 3)

    def test_invalid_proposal_repair_keeps_storage_until_valid_commit(self):
        self.add_artifact("note")
        before = self.snapshot()
        def repair(payload, context):
            self.assertEqual(self.snapshot(), before)
            if self.provider.calls == 1:
                return {**payload, "boundaries": [{"start_quote": "Missing invented passage.", "title": None,
                    "origin": "implicit", "level": 1, "role": "main_body"}]}
            return payload
        self.enable_execution(repair)
        result = self.execute_route()
        self.assertEqual(result.status_code, 200, result.text)
        self.assertTrue(result.json()["success"])
        self.assertEqual(self.provider.calls, 2)

    def test_enabled_route_provider_failure_preserves_storage(self):
        self.enable_execution()
        self.provider.complete_text = Mock(side_effect=RuntimeError("provider unavailable"))
        before = self.snapshot()
        result = self.execute_route()
        self.assertEqual(result.status_code, 502, result.text)
        self.assertFalse(result.json()["success"], result.text)
        self.assertEqual(self.snapshot(), before)

    def test_plan_rejection_categories_preserve_database_and_source(self):
        cases = {
            "missing": ({"title": None, "origin": "implicit", "level": 1, "role": "main_body"},
                        "schema:missing"),
            "conflicting": ({"boundary_id": "PRIVATE_ID", "segment_id": "PRIVATE_SEGMENT",
                             "anchor": "PRIVATE_TEXT", "title": None, "origin": "implicit",
                             "level": 1, "role": "main_body"}, "extra_forbidden"),
            "unknown": ({"start_quote": "PRIVATE_UNKNOWN_QUOTE", "title": None, "origin": "implicit",
                         "level": 1, "role": "main_body"}, "quote_not_found"),
        }
        self.add_artifact("note")
        before = self.snapshot()
        raw = (Path(self.fixture.temp.name) / "doc.pdf").read_bytes()
        for name, (boundary, reason) in cases.items():
            with self.subTest(name=name):
                self.enable_execution(lambda payload, context: {**payload, "boundaries": [boundary]})
                result = self.execute_route()
                self.assertEqual(result.status_code, 422, result.text)
                self.assertIn(reason, result.json()["error"])
                self.assertNotIn("PRIVATE", result.text)
                self.assertEqual(self.provider.calls, 3)
                self.assertEqual(self.snapshot(), before)
        self.enable_execution()
        self.provider.complete_text = Mock(return_value="PRIVATE malformed JSON")
        result = self.execute_route()
        self.assertEqual(result.status_code, 422, result.text)
        self.assertIn("json_invalid", result.json()["error"])
        self.assertNotIn("PRIVATE", result.text)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual((Path(self.fixture.temp.name) / "doc.pdf").read_bytes(), raw)

    def test_budget_rejection_over_http_spends_zero_calls_and_preserves_database(self):
        self.enable_execution()
        self.add_artifact()
        before = self.snapshot()
        token = self.token()
        unavailable_token = self.token()
        self.provider.get_model_capabilities = lambda: SimpleNamespace(context_window_tokens=2000, max_output_tokens=500)
        result = self.execute_route(token)
        self.assertEqual(result.status_code, 413, result.text)
        self.assertEqual(result.json()["detail"], "model_capacity_exceeded")
        self.assertEqual(self.provider.calls, 0)
        self.assertEqual(self.snapshot(), before)
        self.provider.get_model_capabilities = Mock(side_effect=ValueError("PRIVATE capability"))
        result = self.execute_route(unavailable_token)
        self.assertEqual(result.status_code, 503, result.text)
        self.assertNotIn("PRIVATE", result.text)
        self.assertEqual(self.provider.calls, 0)
        self.assertEqual(self.snapshot(), before)

    def test_enabled_route_concurrent_interaction_prevents_commit(self):
        def concurrent(payload, context):
            self.add_artifact()
            return payload
        self.enable_execution(concurrent)
        result = self.execute_route()
        self.assertEqual(result.status_code, 409, result.text)
        self.assertEqual(result.json()["detail"], "confirmation_stale")
        self.assertEqual(self.snapshot().structure_version, 1)
        self.assertEqual(self.snapshot().interaction_count, 1)

    def test_enabled_route_commit_failure_is_unknown_until_explicit_new_operation(self):
        self.enable_execution()
        self.add_artifact()
        before = self.snapshot()
        token = self.token()
        with patch.object(self.store, "_replace_current_hierarchy", side_effect=RuntimeError("commit failure")):
            result = self.execute_route(token)
        self.assertEqual(result.status_code, 500, result.text)
        self.assertEqual(self.snapshot(), before)
        calls = self.provider.calls
        self.assertEqual(self.execute_route(token).status_code, 409)
        self.assertEqual(self.provider.calls, calls)
        # This test has authoritative rollback evidence; a new explicit operation
        # is distinct from blindly retrying an HTTP call with an unknown outcome.
        fresh = self.token()
        first = self.execute_route(fresh)
        self.assertEqual(first.status_code, 200)
        calls = self.provider.calls
        self.assertEqual(self.execute_route(fresh).json(), first.json())
        self.assertEqual(self.provider.calls, calls)

    def test_enabled_route_guards_are_checked_before_provider(self):
        self.enable_execution()
        token = self.token()
        self.coordinator.document_profile_store = object()
        result = self.execute_route(token)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(self.provider.calls, 0)
        self.assertFalse(self.service.preflight(doc_name="doc")["execution_available"])
        self.enable_execution()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("ALTER TABLE artifacts DISABLE TRIGGER artifacts_current_version")
        try:
            self.assertFalse(self.service.preflight(doc_name="doc")["execution_available"])
            self.assertEqual(self.execute_route(token).status_code, 503)
            self.assertEqual(self.provider.calls, 0)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("ALTER TABLE artifacts ENABLE TRIGGER artifacts_current_version")

    def test_committed_structure_requires_explicit_layout_only_preparation(self):
        from test_task_layout_persistence_cache import _FakeTaskUnitResolver, _FakeSectionTaskService, _FakeEnhancedParseEvaluator
        self.enable_execution()
        SectionTaskCoordinator.__init__(self.coordinator,
            document_preparation_pipeline=self.coordinator.document_preparation_pipeline,
            document_artifact_repository=self.repository,
            document_profile_store=self.coordinator.document_profile_store,
            chapter_summary_service=_FakeSectionTaskService(), chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=_FakeTaskUnitResolver(), enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator())
        self.assertEqual(self.execute_route().status_code, 200)
        config = SimpleNamespace(doc_name="doc", get_raw_profile_path=lambda: str(Path(self.fixture.temp.name) / "missing.json"))
        with patch.object(main, "section_task_coordinator", self.coordinator), \
             patch("app.section_task_coordinator.FaissStorageConfig", return_value=config):
            client = TestClient(main.app)
            body = {"doc_name": "doc", "task_unit_split_mode": "progressive"}
            before = self.snapshot()
            response = client.post("/documents/task-layout", json=body)
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(before, self.snapshot())
            response = client.post("/documents/prepare-task-layout", json={**body, "layout_only": True})
            self.assertEqual(response.status_code, 200, response.text)
            response = client.post("/documents/task-layout", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["chapters"][0]["sections"][0]["task_units"])
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(self.snapshot().structure_version, 2)

    @unittest.skipUnless(os.environ.get("ENHANCEMENT_SOURCE_DOCUMENT"), "opt-in source readback not requested")
    def test_opt_in_source_read_only_copy_to_disposable_namespace(self):
        from dataclasses import replace
        import shutil
        name = os.environ["ENHANCEMENT_SOURCE_DOCUMENT"]
        root = Path(os.environ.get("ENHANCEMENT_RAW_ROOT", "data/raw")).resolve()
        original_store = PostgresStructuredDocumentStore(self.dsn,
            namespace=os.environ.get("ENHANCEMENT_SOURCE_NAMESPACE", "default"))
        target = original_store.target_for_doc_name(name)
        before = original_store.load_enhancement_snapshot(target)
        source = (root / f"{name}.pdf").resolve()
        self.assertEqual(source.parent, root)
        from document_preparation.accepted_source_snapshot import source_file_sha256
        original_hash = source_file_sha256(source)
        shutil.copyfile(source, Path(self.fixture.temp.name) / "doc.pdf")
        copied = replace(before.document, document_id="doc", title="isolated source", document_task_artifacts=None)
        self.repository.save_reparsed_document(copied, "doc")
        self.enable_execution()
        if "pdf_extraction" not in copied.parse_provenance:
            self.assertEqual(self.service.preflight(doc_name="doc")["reason"], "extraction_identity_unknown")
            # Legacy evidence is not fabricated from stored slices. Explicitly prepare
            # the isolated copy once, then forbid extraction during enhancement.
            from doc_loaders.pdf_document_loader import PdfDocumentLoader
            from document_preparation.pdf_extraction_identity import identity_for_loader
            from document_structure.structured_document_builder import StructuredDocumentBuilder
            from language.language_code import LanguageCode
            loader = PdfDocumentLoader(Path(self.fixture.temp.name))
            extracted = loader.extract("doc", mode="fast", source_bytes=source.read_bytes())
            text = extracted.text
            prepared = StructuredDocumentBuilder().build(document_id="doc", title="isolated source",
                raw_text=text, language=LanguageCode(before.document.language), parser_mode="common")
            identity = identity_for_loader(loader, source_sha256=original_hash, mode="fast")
            prepared = replace(prepared, parse_provenance={**prepared.parse_provenance,
                               "pdf_extraction": identity.record(text)})
            self.repository.save_reparsed_document(prepared, "doc")
        accepted = self.coordinator.read_enhancement_state("doc").source
        from doc_loaders.pdf_document_loader import PdfDocumentLoader
        with patch.object(PdfDocumentLoader, "extract", side_effect=AssertionError("enhance re-extracted")), \
             patch.object(PdfDocumentLoader, "load", side_effect=AssertionError("enhance loaded raw")), \
             patch("subprocess.run", side_effect=AssertionError("enhance OCR/render")):
            result = self.execute_route()
        self.assertEqual(result.status_code, 200, result.text)
        after = self.coordinator.read_enhancement_state("doc").source
        self.assertEqual(after.text_sha256, accepted.text_sha256)
        self.assertEqual(after.text_length, accepted.text_length)
        self.assertEqual(after.extraction_identity, accepted.extraction_identity)
        self.assertEqual(self.provider.calls, 1)
        self.assertEqual(before, original_store.load_enhancement_snapshot(target))
        self.assertEqual(source_file_sha256(source), original_hash)
        print(f"Isolated source acceptance: canonical_chars={after.text_length}, planner_calls=1, extraction_calls=0")

    def commit_candidate(self, state, candidate):
        with self.repository.enhancement_lease("doc"):
            return self.repository.commit_enhancement("doc", candidate=candidate,
                interaction_revision=state.interaction_revision,
                source_path=Path(self.fixture.temp.name) / "doc.pdf")

    def test_atomic_commit_cleans_all_interactions_and_preserves_source(self):
        self.add_artifact("note")
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE sections SET metadata_payload=%s WHERE document_id=%s",
                (Jsonb({"task_artifacts": {"summary": {"content": "old"}}, "notes": ["old"]}), self.doc_id))
            connection.execute("INSERT INTO document_profile (document_id, profile_payload) VALUES (%s,%s)",
                (self.doc_id, Jsonb({"summary": "keep", "structure_profile": {"old": True},
                                    "post_structure_metadata": {"old": True}})))
            raw_before = connection.execute("SELECT row_to_json(r) FROM raw_source_metadata r WHERE document_id=%s",
                                             (self.doc_id,)).fetchone()[0]
        old = self.repository.load_document("doc")
        state, candidate = self.make_candidate()
        committed = self.commit_candidate(state, candidate)
        self.assertEqual(committed.structure_version, 2)
        self.assertEqual(committed.interaction_count, 0)
        self.assertIsNone(committed.document.document_task_artifacts)
        self.assertNotEqual(old.chapters[0].chapter_id, committed.document.chapters[0].chapter_id)
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM artifacts WHERE document_id=%s",
                                                (self.doc_id,)).fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT row_to_json(r) FROM raw_source_metadata r WHERE document_id=%s",
                                                (self.doc_id,)).fetchone()[0], raw_before)
            self.assertEqual(connection.execute("SELECT profile_payload FROM document_profile WHERE document_id=%s",
                                                (self.doc_id,)).fetchone()[0], {"summary": "keep"})
            self.assertEqual(connection.execute("SELECT source_structure_version FROM structured_document_snapshots WHERE document_id=%s",
                                                (self.doc_id,)).fetchall(), [(2,)])
        with self.assertRaisesRegex(EnhancementCommitConflict, "stale_structure_write"):
            self.repository.save_document(old, "doc")

    def test_commit_rejects_new_interaction_and_duplicate_submission(self):
        state, candidate = self.make_candidate()
        self.add_artifact()
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "confirmation_stale"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())
        state, candidate = self.make_candidate()
        self.commit_candidate(state, candidate)
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "confirmation_stale"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_commit_failure_after_cleanup_rolls_back_everything(self):
        self.add_artifact()
        state, candidate = self.make_candidate()
        before = self.snapshot()
        with patch.object(self.store, "_replace_current_hierarchy", side_effect=RuntimeError("insert failed")):
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_source_changed_during_commit_rolls_back(self):
        state, candidate = self.make_candidate()
        before = self.snapshot()
        save = self.store.save
        def change_source(*args, **kwargs):
            save(*args, **kwargs)
            (Path(self.fixture.temp.name) / "doc.pdf").write_bytes(b"changed")
        with patch.object(self.store, "save", side_effect=change_source):
            with self.assertRaisesRegex(EnhancementCommitConflict, "source_changed"):
                self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_late_relational_and_embedded_writers_are_rejected(self):
        from dataclasses import replace
        from shared.task_artifacts import DocumentTaskArtifacts
        state, candidate = self.make_candidate()
        self.commit_candidate(state, candidate)
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.add_artifact()
        current = self.repository.load_document("doc")
        stale = replace(current, document_task_artifacts=DocumentTaskArtifacts(metadata={
            "reading_interaction_artifacts": {"analysis": {"old": {
                "source_structure_version": 1, "payload": {"analysis": "late"}}}}}))
        with self.assertRaisesRegex(EnhancementCommitConflict, "stale_interaction_write"):
            self.repository.save_document(stale, "doc")
        self.assertEqual(self.snapshot().interaction_count, 0)

    def test_commit_rejects_concurrent_writer_lock_without_waiting(self):
        state, candidate = self.make_candidate()
        before = self.snapshot()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("LOCK TABLE artifacts IN ROW EXCLUSIVE MODE")
            with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_commit_busy"):
                self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_commit_requires_database_version_guards(self):
        state, candidate = self.make_candidate()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("ALTER TABLE artifacts DISABLE TRIGGER artifacts_current_version")
        try:
            with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_storage_guards_unavailable"):
                self.commit_candidate(state, candidate)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("ALTER TABLE artifacts ENABLE TRIGGER artifacts_current_version")

    def test_commit_rejects_existing_document_row_lock_without_deadlock(self):
        state, candidate = self.make_candidate()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("SELECT id FROM documents WHERE id=%s FOR UPDATE", (self.doc_id,))
            with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_commit_busy"):
                self.commit_candidate(state, candidate)

    def test_current_version_metadata_writes_still_work(self):
        from dataclasses import replace
        from shared.task_artifacts import DocumentTaskArtifacts
        state, candidate = self.make_candidate()
        self.commit_candidate(state, candidate)
        current = self.repository.load_document("doc")
        valid = replace(current, document_task_artifacts=DocumentTaskArtifacts(metadata={
            "reading_interaction_artifacts": {"analysis": {"new": {
                "source_structure_version": 2, "payload": {"analysis": "current"}}}}}))
        self.repository.save_document(valid, "doc")
        self.assertEqual(self.snapshot().structure_version, 2)
        self.assertIsNotNone(self.repository.load_document("doc").document_task_artifacts)

    def test_source_version_is_server_bound_before_interaction_generation(self):
        from app.section_task_coordinator import ReadingTargetResolver
        self.coordinator.reading_target_resolver = ReadingTargetResolver()
        document = self.repository.load_document("doc")
        self.coordinator.document_preparation_pipeline.prepare_and_load.return_value = SimpleNamespace(
            structured_document=document)
        target = self.coordinator.resolve_reading_target(doc_name="doc", target_level="document")
        self.assertEqual(target.source_structure_version, 1)
        with self.assertRaisesRegex(ValueError, "stale_structure_version"):
            self.coordinator.resolve_reading_target(doc_name="doc", target_level="document", source_structure_version=2)

    def test_commit_rejects_candidate_with_changed_accepted_text(self):
        from dataclasses import replace
        from hashlib import sha256
        from test_structure_enhancement_candidate import candidate_pipeline, Provider
        state, _ = self.make_candidate()
        text = "This is a different complete document."
        forged = replace(state.source, canonical_text=text, text_length=len(text), text_sha256=sha256(text.encode()).hexdigest())
        candidate = candidate_pipeline(Provider()).build_enhancement_candidate(snapshot=forged)
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_source_identity_invalid"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_real_artifact_store_cannot_resurrect_delayed_book_insight(self):
        from app.section_task_coordinator import ReadingTargetResolver
        from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact, ReadingInteractionRequest
        from section_tasks.interaction_policy_identity import policy_metadata
        self.coordinator.reading_target_resolver = ReadingTargetResolver()
        self.coordinator.document_preparation_pipeline.prepare_and_load.side_effect = lambda **kw: SimpleNamespace(
            structured_document=self.repository.load_document("doc"))
        def artifact():
            target = self.coordinator.resolve_reading_target(doc_name="doc", target_level="document")
            request = ReadingInteractionRequest(target=target, interaction_type="analysis")
            return ReadingInteractionArtifact.from_target(target=target, interaction_type="analysis",
                status="completed", payload={"analysis": "insight"},
                metadata={**policy_metadata(request), "context": {"doc_name": "doc"}})
        store = DocumentReadingInteractionArtifactStore(self.repository)
        delayed = artifact()
        store.save_analysis_artifact(delayed)
        state, candidate = self.make_candidate()
        self.commit_candidate(state, candidate)
        before = self.snapshot()
        # The artifact owner may reject before reaching the DB guard.
        with self.assertRaisesRegex((EnhancementCommitConflict, ValueError), "stale_interaction_write"):
            store.save_analysis_artifact(delayed)
        self.assertEqual(before, self.snapshot())
        store.save_analysis_artifact(artifact())
        self.assertIsNotNone(self.repository.load_document("doc").document_task_artifacts)


if __name__ == "__main__":
    unittest.main()
