"""Shared cleanup production ports with real layout, files and isolated SQL."""

from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import main
from app.cleanup_runtime import build_cleanup_service
from app.standalone_cleanup import StandaloneCleanupService
from document_preparation.cleanup_handoff import prepare_enhancement_cleanup
from document_structure.ocr_cleanup_planner import REMAINING_CLEANUP_PROMPT_VERSION
from document_structure.reading_content import ReadingContentConflict, state_for
from section_tasks.task_unit_resolver import TaskUnitResolver
from smart_cleanup_routes import get_smart_cleanup_service
from scripts import test_standalone_cleanup_service as fixtures
from scripts import test_ocr_cleanup_overlay as overlays
from scripts import test_task_layout_readiness as readiness
from scripts import test_enhancement_cleanup_api as enhance


class CleanupRuntimeTests(unittest.TestCase):
    fixture_type = fixtures.StandaloneCleanupServiceTests

    def setUp(self):
        self.f = self.fixture_type()
        if hasattr(self, "scoped"):
            self.f.scoped = self.scoped
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        r = readiness.ReadinessTests()
        r.setUp()
        self.addCleanup(r.doCleanups)
        self.coordinator = r.coordinator
        self.coordinator.document_artifact_repository = self.f.repo
        self.coordinator.task_unit_resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=100)
        self.coordinator.document_preparation_pipeline = SimpleNamespace(
            loader_factory=SimpleNamespace(base_dir=self.f.temp.name),
            prepare_and_load=Mock(side_effect=AssertionError("prepare/profile/extraction")))
        if not hasattr(self.f.repo, "load_enhancement_snapshot"):
            # File storage has no production enhancement lease/snapshot port.
            # Exercise actual layout using its existing test-owned coherent reader.
            self.coordinator.read_cleanup_input = self.f.read_input
        self.f.flow = self.flow = self.make_flow()

    def make_flow(self):
        if hasattr(self.f.repo, "enhancement_lease"):
            return build_cleanup_service(namespace=self.f.namespace,
                coordinator=self.coordinator, provider=self.f.provider)
        return StandaloneCleanupService(namespace=self.f.namespace, repository=self.f.repo,
            read_input=self.coordinator.read_cleanup_input, provider=self.f.provider,
            lease=self.f.lease, layout_ready=self.coordinator.cleanup_layout_ready,
            prepare_layout=self.coordinator.prepare_cleanup_layout)

    def strip_layout(self, *, units=False):
        document = self.f.repo.load_document("book")
        document.document_task_artifacts.metadata.pop("task_layout", None)
        if units:
            document = replace(document, chapters=[replace(c, sections=[replace(s, task_units=[])
                for s in c.sections]) for c in document.chapters])
        self.f.repo.save_document(document, "book")

    def execute(self):
        consent = self.f.preflight(self.flow)
        return consent, self.f.execute(consent, self.flow)

    def test_ready_content_only_keeps_ids_and_does_not_resolve(self):
        self.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
        before = self.f.repo.load_document("book")
        with patch.object(self.coordinator, "prepare_cleanup_layout", side_effect=AssertionError("ready")), \
                patch.object(self.coordinator.task_unit_resolver, "resolve_for_task_layout", side_effect=AssertionError("split")):
            _, result = self.execute()
        self.assertEqual(result.phase, "finished")
        self.assertEqual(result.state.outcome.status, "completed")
        self.assertEqual(self.f.repo.load_document("book").chapters, before.chapters)
        self.assertEqual(len(self.f.provider.calls), 1)

    def test_common_without_units_gets_layout_in_same_post(self):
        self.strip_layout(units=True)
        with patch.object(self.f.repo, "commit_cleanup", wraps=self.f.repo.commit_cleanup) as commit:
            consent, result = self.execute()
            self.assertEqual(result.phase, "finished")
            self.assertEqual(self.f.execute(consent, self.flow).phase, "finished")
            self.assertEqual(commit.call_count, 1)
        snapshot = self.coordinator.read_cleanup_input("book")
        self.assertTrue(self.coordinator.cleanup_layout_ready(snapshot))
        self.assertTrue(snapshot.document().chapters[0].sections[0].task_units)
        self.assertNotIn("@@??", self.f.content())
        self.assertEqual(len(self.f.provider.calls), 1)
        self.coordinator.document_preparation_pipeline.prepare_and_load.assert_not_called()

    def test_noop_stale_layout_preserves_units_and_generated_metadata(self):
        self.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
        self.strip_layout()
        self.f.plan([])
        before = self.f.repo.load_document("book")
        with patch.object(self.coordinator.task_unit_resolver, "resolve_for_task_layout", side_effect=AssertionError("reuse units")):
            _, result = self.execute()
        self.assertEqual(result.phase, "finished")
        self.assertEqual(result.state.outcome.status, "no_changes")
        current = self.f.repo.load_document("book")
        self.assertEqual(current.chapters, before.chapters)
        self.assertEqual(state_for(current).get("revision", 0), 0)
        self.assertEqual(current.document_task_artifacts.metadata["public_summary"],
                         before.document_task_artifacts.metadata["public_summary"])

    def test_committed_layout_failure_restart_and_get_are_generation_free(self):
        self.strip_layout(units=True)
        with patch.object(self.coordinator, "ensure_document_task_layout", side_effect=RuntimeError("layout")):
            consent, result = self.execute()
        self.assertEqual(result.phase, "layout")
        self.assertEqual(result.retry, "resume_layout")
        restarted = self.make_flow()
        with patch.object(self.f.repo, "commit_cleanup", side_effect=AssertionError("second commit")), \
                patch.object(self.f.provider, "complete_text_with_output_budget", side_effect=AssertionError("second model")):
            before = self.f.snapshot()
            self.assertEqual(restarted.lookup("book", consent.operation_id).phase, "layout")
            self.assertEqual(self.f.snapshot(), before)
            result = self.f.execute(consent, restarted)
        self.assertEqual(result.phase, "finished")

    def test_changed_reading_revision_blocks_late_layout_write(self):
        self.strip_layout(units=True)
        resolver = self.coordinator.task_unit_resolver.resolve_for_task_layout
        def concurrent(**kwargs):
            units = resolver(**kwargs)
            self.f.change("undo")
            return units
        with patch.object(self.coordinator.task_unit_resolver, "resolve_for_task_layout", side_effect=concurrent), \
                patch.object(self.f.repo, "update_task_layout", side_effect=AssertionError("stale write")):
            _, result = self.execute()
        self.assertEqual(result.phase, "superseded")
        self.assertFalse(self.f.repo.load_document("book").chapters[0].sections[0].task_units)

    def test_source_change_during_layout_blocks_write(self):
        self.strip_layout(units=True)
        resolver = self.coordinator.task_unit_resolver.resolve_for_task_layout
        def concurrent(**kwargs):
            units = resolver(**kwargs)
            self.f.path.write_bytes(b"replaced source")
            return units
        with patch.object(self.coordinator.task_unit_resolver, "resolve_for_task_layout", side_effect=concurrent), \
                patch.object(self.f.repo, "update_task_layout", side_effect=AssertionError("stale source")):
            _, result = self.execute()
        self.assertEqual(result.phase, "superseded")

    def test_input_changed_before_layout_never_prepares(self):
        snapshot = self.coordinator.read_cleanup_input("book")
        block = self.f.service.read("book")["blocks"][1]
        self.f.change("edit", block_id=block["block_id"], text="Keep this correction")
        with patch.object(self.f.repo, "update_task_layout", side_effect=AssertionError("stale write")):
            with self.assertRaises(ReadingContentConflict):
                self.coordinator.prepare_cleanup_layout(snapshot)

    def test_default_route_dependency_uses_composed_service(self):
        overrides = dict(main.app.dependency_overrides)
        self.addCleanup(setattr, main.app, "dependency_overrides", overrides)
        main.app.dependency_overrides.pop(get_smart_cleanup_service, None)
        with patch.object(main.app.state, "smart_cleanup_service", self.flow), TestClient(main.app) as client:
            preflight = client.post("/documents/book/smart-cleanup/preflight", json={})
            self.assertEqual(preflight.status_code, 200, preflight.text)
            consent = {key: preflight.json()[key] for key in ("operation_id", "confirmation_token")}
            result = client.post("/documents/book/smart-cleanup", json=consent)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertTrue(result.json()["reading_ready"], result.text)

    def test_factory_requires_cross_process_ports(self):
        unsupported = SimpleNamespace(document_artifact_repository=SimpleNamespace())
        self.assertIsNone(build_cleanup_service(namespace="test", coordinator=unsupported, provider=self.f.provider))

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"), "opt-in immutable backups")
    def test_real_backups_complete_actual_layout_without_touching_sources(self):
        from document_structure.structured_document import StructuredDocument
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                folder = root / name
                paths = [folder / p for p in ("book.pdf", "document.json", "manifest.json")]
                before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                self.assertEqual(before[paths[0]], json.loads(paths[2].read_text())["source_sha256"])
                self.f.path.write_bytes(paths[0].read_bytes())
                document = StructuredDocument.from_dict(json.loads(paths[1].read_text()))
                self.f.repo.save_reparsed_document(replace(document, document_id="book", source_path=str(self.f.path)), "book")
                items = []
                if name == "sample2":
                    record, = json.loads((folder / "20261023T030347436965Z/deletions.json").read_text())
                    items = [{"action": "delete", "quote": record["quote"], "reason": "ocr_garbage"}]
                self.f.plan(items)
                consent, result = self.execute()
                self.assertEqual(result.phase, "finished", result.reason)
                self.assertEqual(result.state.outcome.applied_count, len(items))
                self.assertEqual(self.make_flow().lookup("book", consent.operation_id), result)
                self.assertTrue(self.coordinator.cleanup_layout_ready(self.coordinator.read_cleanup_input("book")))
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class PostgresCleanupRuntimeTests(CleanupRuntimeTests):
    fixture_type = fixtures.PostgresStandaloneCleanupServiceTests
    setUpClass = classmethod(fixture_type.setUpClass.__func__)
    drop_schema = classmethod(fixture_type.drop_schema.__func__)

    def remove_persisted_path(self):
        import psycopg
        with psycopg.connect(self.scoped) as connection:
            row = connection.execute("SELECT id FROM documents WHERE namespace=%s AND document_name='book'",
                                     (self.f.namespace,)).fetchone()
            connection.execute("DELETE FROM raw_source_metadata WHERE document_id=%s", (row[0],))
            connection.execute("UPDATE documents SET metadata_payload=metadata_payload - 'source_path' WHERE id=%s",
                               (row[0],))
        self.assertIsNone(self.f.repo.load_enhancement_snapshot("book").document.source_path)

    def test_missing_persisted_path_uses_accepted_source_and_commits(self):
        self.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
        self.remove_persisted_path()
        snapshot = self.coordinator.read_cleanup_input("book")
        self.assertEqual(snapshot.accepted_source_path, str(self.f.path.resolve()))
        _, result = self.execute()
        self.assertEqual(result.state.commit, "committed")
        self.assertEqual(result.state.outcome.status, "completed")
        self.assertNotIn("@@??", self.f.content())
        self.assertIsNone(self.f.repo.load_enhancement_snapshot("book").document.source_path)

    def test_accepted_path_still_rejects_source_change_during_planning(self):
        self.remove_persisted_path()
        consent = self.f.preflight(self.flow)
        before = self.f.snapshot()
        commit = self.f.repo.commit_cleanup
        def changed(command):
            self.f.path.write_bytes(b"Changed source during model request")
            return commit(command)
        with patch.object(self.f.repo, "commit_cleanup", side_effect=changed):
            result = self.f.execute(consent, self.flow)
        self.assertEqual(result.reason, "source_changed")
        self.assertEqual(result.state.commit, "not_committed")
        self.assertEqual(before, self.f.snapshot())

    def test_production_reader_distinguishes_missing_document(self):
        with self.assertRaises(FileNotFoundError):
            self.coordinator.read_cleanup_input("not-present")

    def test_same_lease_blocks_other_entry(self):
        with self.coordinator.enhancement_lease("book"):
            self.assertEqual(self.flow.preflight("book").reason, "cleanup_in_progress")
        self.assertFalse(self.f.provider.calls)

    def test_container_injects_one_provider_and_service(self):
        from config.container import ApplicationLookupContainer
        from config.app_DI_config import AppDIConfig
        container = ApplicationLookupContainer.build(AppDIConfig())
        container.config.postgres_namespace.from_value(self.f.namespace)
        with container.llm_provider.override(self.f.provider), \
                container.section_task_coordinator.override(self.coordinator):
            service = container.smart_cleanup_service()
            self.assertIs(service, container.smart_cleanup_service())
            self.assertIs(service.provider, self.f.provider)
            self.assertIs(container.document_preparation_pipeline.kwargs["cleanup_provider"], container.llm_provider)
            self.assertTrue(service.preflight("book").ready)


class SharedEnhanceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.f = overlays.CleanupOverlayTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        helper = fixtures.StandaloneCleanupServiceTests()
        helper.__dict__.update(self.f.__dict__)
        self.baseline = helper.read_input("book")
        from scripts.test_ocr_cleanup_planner import Provider
        self.provider = Provider()
        self.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
            {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"}]})

    def prepare(self):
        candidate, legacy = self.f.prepare("structure_changed")
        result = prepare_enhancement_cleanup(candidate=candidate, baseline=self.baseline,
            binding=legacy.plan.binding.model_copy(update={"prompt_version": REMAINING_CLEANUP_PROMPT_VERSION}),
            provider=self.provider)
        return candidate, result

    def test_shared_input_file_combined_commit_is_single_and_atomic(self):
        candidate, result = self.prepare()
        self.assertIsNotNone(result.application.cleanup_input)
        with patch.object(self.f.repo, "_atomic_save", wraps=self.f.repo._atomic_save) as save:
            committed = self.f.commit(candidate, result.application)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(committed.cleanup["status"], "completed")
        self.assertNotIn("@@??", self.f.content())

    def test_remaining_view_not_legacy_canonical_planner(self):
        with patch("document_preparation.cleanup_handoff.plan_ocr_cleanup", side_effect=AssertionError("old planner")):
            self.prepare()
        self.assertEqual(len(self.provider.calls), 1)

    def test_changed_candidate_or_baseline_rejects_before_write(self):
        candidate, result = self.prepare()
        changed = replace(candidate, document=replace(candidate.document, title="changed"))
        before = self.f.snapshot()
        with self.assertRaises(ReadingContentConflict):
            self.f.commit(changed, result.application)
        self.assertEqual(self.f.snapshot(), before)

    def test_final_file_failure_keeps_old_document(self):
        candidate, result = self.prepare()
        before = self.f.snapshot()
        with patch.object(self.f.repo, "_atomic_save", side_effect=OSError("write")), self.assertRaises(OSError):
            self.f.commit(candidate, result.application)
        self.assertEqual(self.f.snapshot(), before)


class EnhanceToStandaloneTests(unittest.TestCase):
    setUpClass = classmethod(enhance.CleanupApiTests.setUpClass.__func__)
    drop_schema = classmethod(enhance.CleanupApiTests.drop_schema.__func__)

    def setup_api(self):
        api = enhance.CleanupApiTests()
        api.scoped = self.scoped
        api.setUp()
        self.addCleanup(api.doCleanups)
        api.kind = "structure_changed"
        return api

    def test_failed_automatic_cleanup_then_independent_retry_without_resplit(self):
        api = self.setup_api()
        good = api.provider.payload
        api.provider.payload = "not json"
        result = api.assert_completed(api.execute(api.request()))
        self.assertEqual(result["cleanup"]["status"], "failed")
        api.provider.payload = good
        flow = build_cleanup_service(namespace=api.f.namespace, coordinator=api.coordinator, provider=api.provider)
        with patch.object(api.pipeline, "build_enhancement_candidate", side_effect=AssertionError("resplit")), \
                patch.object(api.f.repo, "commit_enhancement", side_effect=AssertionError("combined commit")):
            consent = flow.preflight("book")
            self.assertTrue(consent.ready, consent.reason)
            result = flow.execute("book", consent.operation_id, confirmed=True,
                confirmation_token=consent.confirmation_token)
        self.assertEqual(result.phase, "finished")
        self.assertEqual(result.state.outcome.status, "completed")
        self.assertEqual(len(api.provider.calls), 2)

    def test_partial_auto_retry_receives_remaining_view_and_keeps_revision(self):
        api = self.setup_api()
        api.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
            {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"},
            {"action": "delete", "quote": "not present", "reason": "ocr_garbage"}]})
        result = api.assert_completed(api.execute(api.request()))
        self.assertEqual(result["cleanup"]["status"], "partial")
        flow = build_cleanup_service(namespace=api.f.namespace, coordinator=api.coordinator, provider=api.provider)
        seen = []
        complete = api.provider.complete_text_with_output_budget
        def capture(prompt, *args, **kwargs):
            seen.append(prompt)
            return complete(prompt, *args, **kwargs)
        with patch.object(api.provider, "complete_text_with_output_budget", side_effect=capture), \
                patch.object(api.pipeline, "build_enhancement_candidate", side_effect=AssertionError("resplit")):
            consent = flow.preflight("book")
            result = flow.execute("book", consent.operation_id, confirmed=True,
                confirmation_token=consent.confirmation_token)
        self.assertEqual(result.phase, "finished")
        self.assertEqual(result.state.outcome.status, "skipped")
        self.assertEqual(result.state.outcome.reading_revision, 1)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("@@??", seen[0])

    def test_unavailable_cleanup_adapter_keeps_valid_split_candidate(self):
        from document_preparation.cleanup_input_adapter import CleanupInputUnavailable
        api = self.setup_api()
        request = api.request()
        with patch("document_preparation.cleanup_input_adapter.adapt_candidate_cleanup_input",
                   side_effect=CleanupInputUnavailable("cleanup_reading_mapping_invalid")):
            result = api.assert_completed(api.execute(request))
        self.assertEqual(result["cleanup"]["status"], "failed")
        self.assertEqual(result["change_kind"], "structure_changed")
        self.assertIn("@@??", api.f.content())
        self.assertFalse(api.provider.calls)


if __name__ == "__main__":
    unittest.main()
