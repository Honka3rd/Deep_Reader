"""Standalone consent/recovery tests with actual adapters, engine and repositories."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch
import unittest

from app.standalone_cleanup import CleanupUnavailable, StandaloneCleanupService
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
from document_preparation.cleanup_input_adapter import adapt_document_cleanup_input
from document_structure.reading_content import state_for
from document_structure.structured_document import StructuredDocument
from scripts import test_standalone_cleanup_commit as files
from scripts import test_standalone_cleanup_postgres as sql
from scripts.test_ocr_cleanup_contract import item
from scripts.test_ocr_cleanup_planner import Provider


class StandaloneCleanupServiceTests(unittest.TestCase):
    repository = files.StandaloneCleanupCommitTests.repository
    snapshot = files.StandaloneCleanupCommitTests.snapshot
    change = files.StandaloneCleanupCommitTests.change
    content = files.StandaloneCleanupCommitTests.content
    fail_write = files.StandaloneCleanupCommitTests.fail_write

    def setUp(self):
        files.StandaloneCleanupCommitTests.setUp(self)
        self.provider = Provider()
        self.plan([item(quote="@@??")])
        self.mutex = Lock()
        self.ready = True
        self.layout_calls = 0
        self.now = 0
        self.flow = self.make_flow()

    def plan(self, items):
        self.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": items})

    @contextmanager
    def lease(self, document_id):
        if not self.mutex.acquire(blocking=False):
            raise BlockingIOError("occupied")
        try:
            yield
        finally:
            self.mutex.release()

    def read_input(self, document_id):
        self.assertEqual(document_id, "book")
        document, version, _ = self.snapshot()
        source = resolve_accepted_source_snapshot(document=document, namespace=self.namespace,
            expected_document_id=document_id, structure_version=version, source_path=self.path, accepted=True)
        return adapt_document_cleanup_input(document=document, snapshot=source,
            expected_reading_revision=state_for(document).get("revision", 0))

    def layout(self, snapshot):
        self.assertEqual(snapshot.fingerprint(), self.read_input("book").fingerprint())
        self.layout_calls += 1
        self.ready = True

    def make_flow(self, **kwargs):
        return StandaloneCleanupService(namespace=self.namespace, repository=self.repo,
            read_input=self.read_input, provider=self.provider, lease=self.lease,
            layout_ready=lambda snapshot: self.ready, prepare_layout=self.layout,
            clock=lambda: self.now, **kwargs)

    def preflight(self, flow=None):
        result = (flow or self.flow).preflight("book")
        self.assertTrue(result.ready, result.reason)
        return result

    def execute(self, consent, flow=None, **kwargs):
        return (flow or self.flow).execute("book", consent.operation_id,
            **{"confirmed": True, "confirmation_token": consent.confirmation_token, **kwargs})

    def lookup(self, consent, flow=None):
        return (flow or self.flow).lookup("book", consent.operation_id)

    def test_preflight_no_generation_and_requires_separate_consent(self):
        before = self.snapshot()
        consent = self.preflight()
        self.assertEqual(before, self.snapshot())
        self.assertFalse(self.provider.calls)
        for kwargs, reason in (({"confirmed": False}, "required"),
                               ({"confirmed": 1}, "required"),
                               ({"confirmation_token": "enhancement-discard-token"}, "invalid")):
            with self.assertRaisesRegex(CleanupUnavailable, reason):
                self.execute(consent, **kwargs)
        self.assertFalse(self.provider.calls)
        self.assertEqual(self.execute(consent).phase, "finished")

    def test_expired_consent_and_bounded_registry(self):
        consent = self.preflight()
        self.now = 300
        with self.assertRaisesRegex(CleanupUnavailable, "expired"):
            self.execute(consent)
        flow = self.make_flow(limit=1)
        self.preflight(flow)
        self.assertEqual(flow.preflight("book").reason, "cleanup_operation_capacity")
        self.now += 3601
        self.preflight(flow)
        self.assertFalse(self.provider.calls)

    def test_success_replay_and_restart_make_one_call_one_commit(self):
        consent = self.preflight()
        with patch.object(self.repo, "commit_cleanup", wraps=self.repo.commit_cleanup) as commit:
            result = self.execute(consent)
            self.assertEqual(result.state.outcome.status, "completed")
            self.assertEqual(self.execute(consent), self.lookup(consent))
            restarted = self.make_flow()
            self.assertEqual(self.lookup(consent, restarted), result)
            self.assertEqual(self.execute(consent, restarted), result)
            self.assertEqual(commit.call_count, 1)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(result.durability, "persisted_commit")
        self.assertEqual(result.state.outcome.reading_revision, 1)

    def test_restart_before_dispatch_is_unknown_not_permission_to_regenerate(self):
        consent = self.preflight()
        restarted = self.make_flow()
        self.assertEqual(self.execute(consent, restarted).retry, "reconcile")
        self.assertEqual(self.lookup(consent, restarted).phase, "unknown")
        self.assertFalse(self.provider.calls)
        self.assertEqual(self.execute(self.preflight(restarted), restarted).phase, "finished")

    def test_read_failure_never_looks_like_missing_receipt(self):
        consent = self.preflight()
        with patch.object(self.repo, "lookup_cleanup", side_effect=OSError("PRIVATE_STORAGE")):
            result = self.execute(consent)
        self.assertEqual(result.reason, "cleanup_reconciliation_unavailable")
        self.assertEqual(result.retry, "reconcile")
        self.assertFalse(self.provider.calls)
        self.assertEqual(self.execute(consent).phase, "finished")

    def test_current_snapshot_read_failure_does_not_claim_commit_is_absent(self):
        consent = self.preflight()
        self.execute(consent)
        with patch.object(self.flow, "read_input", side_effect=OSError("PRIVATE_STORAGE")):
            self.assertEqual(self.lookup(consent).phase, "unknown")
        self.assertEqual(self.lookup(consent).state.commit, "committed")
        fresh = self.preflight()
        with patch.object(self.flow, "read_input", side_effect=OSError("PRIVATE_STORAGE")):
            self.assertEqual(self.execute(fresh).reason, "cleanup_execution_unavailable")
        self.assertEqual(self.execute(fresh).phase, "unknown")
        self.assertEqual(len(self.provider.calls), 1)

    def test_stale_manual_edit_before_dispatch_makes_zero_calls(self):
        consent = self.preflight()
        block = self.service.read("book")["blocks"][1]
        self.change("edit", block_id=block["block_id"], text="Retained correction")
        result = self.execute(consent)
        self.assertEqual(result.state.commit, "not_committed")
        self.assertEqual(result.reason, "reading_revision_changed")
        self.assertFalse(self.provider.calls)

    def test_changed_source_before_dispatch_makes_zero_calls(self):
        consent = self.preflight()
        self.path.write_bytes(b"changed")
        self.assertEqual(self.execute(consent).reason, "source_changed")
        self.assertFalse(self.provider.calls)

    def test_changed_policy_or_model_refuses_old_consent(self):
        consent = self.preflight()
        self.provider.capabilities = replace(self.provider.capabilities, model_name="other")
        self.assertEqual(self.execute(consent).reason, "cleanup_policy_changed")
        self.assertFalse(self.provider.calls)
        consent = self.preflight()
        with patch("document_structure.ocr_cleanup_planner.REMAINING_CLEANUP_POLICY_VERSION", "new-policy"):
            self.assertEqual(self.execute(consent).reason, "cleanup_policy_changed")
        self.assertFalse(self.provider.calls)

    def test_capacity_refusal_no_token_no_call(self):
        self.provider.count = 10000000
        consent = self.flow.preflight("book")
        self.assertFalse(consent.ready)
        self.assertEqual(consent.reason, "capacity_exceeded")
        self.assertIsNone(consent.confirmation_token)
        self.provider.capabilities = None
        self.assertEqual(self.flow.preflight("book").reason, "capacity_unavailable")
        self.assertFalse(self.provider.calls)

    def test_capacity_change_at_dispatch_and_after_model_refuses_commit(self):
        consent = self.preflight()
        self.provider.count = 10000000
        self.assertEqual(self.execute(consent).reason, "capacity_exceeded")
        self.assertFalse(self.provider.calls)
        self.provider.count = None
        consent = self.preflight()
        complete = self.provider.complete_text_with_output_budget
        def change(*args, **kwargs):
            payload = complete(*args, **kwargs)
            self.provider.capabilities = replace(self.provider.capabilities, model_name="other")
            return payload
        with patch.object(self.provider, "complete_text_with_output_budget", side_effect=change):
            self.assertEqual(self.execute(consent).reason, "cleanup_policy_changed")
        self.assertEqual(self.service.read("book")["revision"], 0)

    def test_engine_rechecks_capacity_identity_immediately_before_dispatch(self):
        consent = self.preflight()
        capacity = self.flow._capacity
        def change_after_check(*args):
            value = capacity(*args)
            self.provider.capabilities = replace(self.provider.capabilities, model_name="changed-at-dispatch")
            return value
        with patch.object(self.flow, "_capacity", side_effect=change_after_check):
            result = self.execute(consent)
        self.assertEqual(result.reason, "cleanup_policy_changed")
        self.assertFalse(self.provider.calls)

    def test_native_and_unknown_provenance_never_issue_consent(self):
        original = self.repo.load_document("book")
        for mode in ("fast", "unknown", "auto"):
            with self.subTest(mode=mode):
                provenance = json.loads(json.dumps(original.parse_provenance))
                provenance["extraction"]["identity"]["mode"] = mode
                self.repo.save_reparsed_document(replace(original, parse_provenance=provenance), "book")
                result = self.flow.preflight("book")
                self.assertFalse(result.ready)
                self.assertIsNone(result.confirmation_token)
                self.assertFalse(self.provider.calls)

    def test_transient_provider_failure_retries_but_same_operation_does_not(self):
        consent = self.preflight()
        self.provider.error = TimeoutError("PRIVATE_PROVIDER")
        result = self.execute(consent)
        self.assertEqual(result.retry, "regenerate_cleanup")
        self.assertEqual(result.reason, "provider_failed")
        self.assertEqual(len(self.provider.calls), 2)
        self.provider.error = None
        self.assertEqual(self.execute(consent), result)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(self.execute(self.preflight()).phase, "finished")
        self.assertEqual(len(self.provider.calls), 3)

    def test_invalid_plan_and_safe_semantic_skip_are_not_storage_failure(self):
        self.provider.payload = "not JSON"
        result = self.execute(self.preflight())
        self.assertEqual(result.reason, "invalid_plan")
        self.assertEqual(result.state.commit, "not_committed")
        self.plan([item(quote="not present")])
        result = self.execute(self.preflight())
        self.assertEqual(result.reason, "no_safe_items")
        self.assertEqual(result.state.commit, "committed")
        self.assertEqual(result.retry, "regenerate_cleanup")
        self.assertEqual(self.service.read("book")["revision"], 0)

    def test_noop_keeps_generated_content_and_is_replayable(self):
        self.plan([])
        before = self.snapshot()[0]
        consent = self.preflight()
        result = self.execute(consent)
        self.assertEqual(result.state.outcome.status, "no_changes")
        self.assertEqual(result.state.outcome.reading_revision, 0)
        self.assertEqual(result.retry, "none")
        self.assertEqual(before.document_task_artifacts, self.snapshot()[0].document_task_artifacts)
        self.assertEqual(self.lookup(consent, self.make_flow()), result)

    def test_partial_retains_rejected_items_and_fresh_attempt_uses_remaining_view(self):
        self.plan([item(quote="@@??"), item(quote="not present")])
        result = self.execute(self.preflight())
        self.assertEqual(result.state.outcome.status, "partial")
        self.assertEqual(result.retry, "regenerate_cleanup")
        self.plan([])
        self.execute(self.preflight())
        prompt = self.provider.calls[-1][0].split("\nSOURCE:\n")[1]
        self.assertNotIn("@@??", prompt)

    def test_commit_acknowledgement_loss_reconciles_without_second_write(self):
        consent = self.preflight()
        commit = self.repo.commit_cleanup
        def lost(command):
            commit(command)
            raise OSError("lost response")
        with patch.object(self.repo, "commit_cleanup", side_effect=lost) as write:
            result = self.execute(consent)
            self.assertEqual(result.phase, "finished")
            self.execute(consent)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(len(self.provider.calls), 1)

    def test_unconfirmed_storage_failure_requires_reconciliation(self):
        consent = self.preflight()
        before = self.snapshot()
        with self.fail_write():
            result = self.execute(consent)
        self.assertEqual(result.phase, "unknown")
        self.assertEqual(result.retry, "reconcile")
        self.assertEqual(before, self.snapshot())
        self.execute(consent)
        self.assertEqual(len(self.provider.calls), 1)
        with patch.object(self.repo, "lookup_cleanup", side_effect=OSError("still unavailable")):
            self.assertEqual(self.flow.preflight("book").reason, "cleanup_reconciliation_unavailable")
        fresh = self.preflight()
        self.assertNotEqual(fresh.operation_id, consent.operation_id)
        self.assertEqual(self.execute(fresh).phase, "finished")
        self.assertEqual(len(self.provider.calls), 2)

    def test_commit_conflict_preserves_concurrent_manual_change(self):
        consent = self.preflight()
        commit = self.repo.commit_cleanup
        def concurrent(command):
            block = self.service.read("book")["blocks"][1]
            self.change("edit", block_id=block["block_id"], text="Concurrent correction")
            return commit(command)
        with patch.object(self.repo, "commit_cleanup", side_effect=concurrent):
            result = self.execute(consent)
        self.assertEqual(result.state.commit, "not_committed")
        self.assertEqual(result.reason, "reading_revision_changed")
        self.assertIn("Concurrent correction", self.content())

    def test_source_unavailable_is_not_reported_as_reading_edit(self):
        from document_structure.reading_content import ReadingContentConflict
        consent = self.preflight()
        before = self.snapshot()
        with patch.object(self.repo, "commit_cleanup", side_effect=ReadingContentConflict("cleanup_source_unavailable")):
            result = self.execute(consent)
        self.assertEqual(result.reason, "storage_failed")
        self.assertEqual(result.state.outcome.status, "failed")
        self.assertEqual(result.state.commit, "not_committed")
        self.assertEqual(before, self.snapshot())

    def test_layout_failure_can_resume_after_restart_without_generation_or_commit(self):
        self.ready = False
        consent = self.preflight()
        with patch.object(self.flow, "prepare_layout", side_effect=RuntimeError("PRIVATE_LAYOUT")):
            result = self.execute(consent)
        self.assertEqual(result.phase, "layout")
        self.assertEqual(result.reason, "cleanup_layout_failed")
        self.assertEqual(result.retry, "resume_layout")
        restarted = self.make_flow()
        self.assertEqual(self.lookup(consent, restarted).phase, "layout")
        self.assertEqual(self.layout_calls, 0)
        with patch.object(self.repo, "commit_cleanup", side_effect=AssertionError("no second commit")):
            result = restarted.resume_layout("book", consent.operation_id)
        self.assertEqual(result.phase, "finished")
        self.assertEqual(self.layout_calls, 1)
        self.assertEqual(len(self.provider.calls), 1)

    def test_readiness_port_not_receipt_flag_is_authority(self):
        consent = self.preflight()
        self.execute(consent)
        self.ready = False
        self.assertEqual(self.lookup(consent).phase, "layout")
        self.assertEqual(self.layout_calls, 0)
        self.flow.resume_layout("book", consent.operation_id)
        self.assertEqual(self.layout_calls, 1)

    def test_undo_supersedes_receipt_and_never_prepares_layout(self):
        consent = self.preflight()
        self.execute(consent)
        self.change("undo")
        self.ready = False
        restarted = self.make_flow()
        result = restarted.resume_layout("book", consent.operation_id)
        self.assertEqual(result.phase, "superseded")
        self.assertEqual(result.retry, "none")
        self.assertEqual(self.layout_calls, 0)

    def test_raw_source_change_after_commit_cannot_resume_layout(self):
        consent = self.preflight()
        self.execute(consent)
        self.path.write_bytes(b"replacement raw source")
        self.ready = False
        self.assertEqual(self.flow.resume_layout("book", consent.operation_id).phase, "superseded")
        self.assertEqual(self.layout_calls, 0)

    def test_legacy_receipt_without_lineage_cannot_authorize_layout(self):
        consent = self.preflight()
        self.execute(consent)
        lookup = self.repo.lookup_cleanup
        def legacy(**kwargs):
            return lookup(**kwargs).model_copy(update={"source_structure_fingerprint": None})
        self.ready = False
        with patch.object(self.repo, "lookup_cleanup", side_effect=legacy):
            self.assertEqual(self.flow.resume_layout("book", consent.operation_id).phase, "superseded")
        self.assertEqual(self.layout_calls, 0)

    def test_provider_retry_rechecks_changed_model(self):
        consent = self.preflight()
        calls = []
        def fail(*args, **kwargs):
            calls.append(1)
            self.provider.capabilities = replace(self.provider.capabilities, model_name="changed-on-retry")
            raise TimeoutError("transient")
        with patch.object(self.provider, "complete_text_with_output_budget", side_effect=fail):
            result = self.execute(consent)
        self.assertEqual(result.reason, "cleanup_policy_changed")
        self.assertEqual(len(calls), 1)

    def test_disappearing_receipt_after_commit_is_unknown_not_success(self):
        consent = self.preflight()
        with patch.object(self.repo, "lookup_cleanup", return_value=None):
            result = self.execute(consent)
        self.assertEqual(result.phase, "unknown")
        self.assertEqual(self.lookup(consent).phase, "finished")

    def test_hard_reparse_removes_receipt_does_not_replay_cached_success(self):
        consent = self.preflight()
        self.execute(consent)
        self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
        self.assertEqual(self.lookup(consent).phase, "superseded")
        self.assertEqual(self.lookup(consent, self.make_flow()).phase, "unknown")
        self.assertEqual(len(self.provider.calls), 1)

    def test_unknown_operation_and_cross_document_lookup_do_not_generate(self):
        self.assertEqual(self.flow.lookup("book", "unknown").phase, "unknown")
        consent = self.preflight()
        self.assertEqual(self.flow.lookup("another", consent.operation_id).phase, "unknown")
        self.assertFalse(self.provider.calls)

    def test_concurrent_duplicate_and_new_attempt_share_document_lease(self):
        consent = self.preflight()
        entered, release = Event(), Event()
        complete = self.provider.complete_text_with_output_budget
        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise AssertionError("test did not release provider")
            return complete(*args, **kwargs)
        with patch.object(self.provider, "complete_text_with_output_budget", side_effect=delayed):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(self.execute, consent)
                try:
                    self.assertTrue(entered.wait(10))
                    self.assertEqual(self.lookup(consent).phase, "planning")
                    with self.assertRaisesRegex(CleanupUnavailable, "in_progress"):
                        self.execute(consent)
                    self.assertEqual(self.make_flow().preflight("book").reason, "cleanup_in_progress")
                    self.now += 7200
                    self.assertEqual(self.lookup(consent).phase, "planning")
                finally:
                    release.set()
                self.assertEqual(future.result().phase, "finished")
        self.assertEqual(len(self.provider.calls), 1)

    def test_privacy_safe_service_logs(self):
        consent = self.preflight()
        with self.assertLogs("app.standalone_cleanup", level="INFO") as logs:
            self.execute(consent)
        output = "\n".join(logs.output)
        for private in (consent.confirmation_token, consent.operation_id, "@@??", self.text):
            self.assertNotIn(private, output)
        self.assertIn("origin=standalone", output)
        self.assertIn("stage=execute", output)
        self.assertIn("elapsed_ms=", output)

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"), "opt-in immutable backups")
    def test_real_book_inputs_through_service_without_backup_changes(self):
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                folder = root / name
                paths = [folder / p for p in ("book.pdf", "document.json", "manifest.json")]
                before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                self.assertEqual(before[paths[0]], json.loads(paths[2].read_text())["source_sha256"])
                self.path.write_bytes(paths[0].read_bytes())
                document = StructuredDocument.from_dict(json.loads(paths[1].read_text()))
                self.repo.save_reparsed_document(replace(document, document_id="book", source_path=str(self.path)), "book")
                items = []
                if name == "sample2":
                    record, = json.loads((folder / "20261023T030347436965Z/deletions.json").read_text())
                    items = [item(quote=record["quote"])]
                self.plan(items)
                consent = self.preflight()
                result = self.execute(consent)
                self.assertEqual(result.phase, "finished")
                self.assertEqual(result.state.outcome.applied_count, len(items))
                self.assertEqual(self.lookup(consent, self.make_flow()), result)
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class PostgresStandaloneCleanupServiceTests(StandaloneCleanupServiceTests):
    setUpClass = classmethod(sql.PostgresStandaloneCleanupTests.setUpClass.__func__)
    drop_schema = classmethod(sql.PostgresStandaloneCleanupTests.drop_schema.__func__)
    repository = sql.PostgresStandaloneCleanupTests.repository
    snapshot = sql.PostgresStandaloneCleanupTests.snapshot
    fail_write = sql.PostgresStandaloneCleanupTests.fail_write

    def lease(self, document_id):
        return self.repo.enhancement_lease(document_id)


if __name__ == "__main__":
    unittest.main()
