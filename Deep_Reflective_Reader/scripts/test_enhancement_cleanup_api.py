"""Public cleanup workflow against disposable PostgreSQL and controlled models."""

from contextlib import ExitStack
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import main
from app.enhancement_operations import EnhancementOperations
from app.enhancement_preflight import EnhancementPreflightService
from document_structure.reading_content import state_for
from scripts.test_ocr_cleanup_overlay_postgres import PostgresCleanupOverlayTests
from scripts.test_ocr_cleanup_planner import Provider
from scripts.test_structure_enhancement_candidate import candidate_pipeline
from scripts.test_task_layout_readiness import ReadinessTests
from section_tasks.task_unit_resolver import TaskUnitResolver


class CleanupApiTests(unittest.TestCase):
    setUpClass = classmethod(PostgresCleanupOverlayTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresCleanupOverlayTests.drop_schema.__func__)

    def setUp(self):
        self.f = PostgresCleanupOverlayTests()
        self.f.scoped = self.scoped
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        readiness = ReadinessTests()
        readiness.setUp()
        self.addCleanup(readiness.doCleanups)
        self.coordinator = readiness.coordinator
        self.coordinator.document_artifact_repository = self.f.repo
        self.coordinator.task_unit_resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=100)
        self.provider = Provider()
        self.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
            {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"}]})
        self.pipeline = candidate_pipeline(self.provider)
        self.pipeline.loader_factory = SimpleNamespace(base_dir=self.f.temp.name)
        self.coordinator.document_preparation_pipeline = self.pipeline
        self.kind = "unchanged"
        def build(**kwargs):
            candidate, _ = self.f.prepare(self.kind)
            return replace(candidate, source=kwargs["snapshot"])
        self.pipeline.build_enhancement_candidate = Mock(side_effect=build)
        self.service = EnhancementPreflightService(self.coordinator.read_enhancement_state,
            lease=self.coordinator.enhancement_lease, availability=lambda _: None,
            capacity_reader=self.coordinator.inspect_enhancement_capacity)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(main, "section_task_coordinator", self.coordinator))
        stack.enter_context(patch.object(main, "enhancement_preflight_service", self.service))
        stack.enter_context(patch("app.enhancement_preflight.enhancement_preflight_service", self.service))
        stack.enter_context(patch.object(self.pipeline, "prepare_and_load", side_effect=AssertionError("prepare/profile")))
        self.client = TestClient(main.app)

    def preflight(self, **kwargs):
        return self.client.post("/documents/reparse-structure/preflight",
            json={"doc_name": "book", "allow_ocr_cleanup": True, **kwargs})

    def request(self):
        response = self.preflight()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["execution_available"], response.text)
        return dict(doc_name="book", parser_mode="llm_enhanced", allow_ocr_cleanup=True,
                    discard_interactions=True, confirmation_token=response.json()["confirmation_token"])

    def execute(self, request):
        return self.client.post("/documents/reparse-structure", json=request)

    def lookup(self, request):
        return self.client.post("/documents/reparse-structure/result",
            json={k: v for k, v in request.items() if k != "discard_interactions"})

    def assert_completed(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["phase"], "complete", response.text)
        self.assertFalse(result["reading_layout_required"])
        return result

    def test_cleanup_only_one_revision_notes_retained_and_layout_automatic(self):
        before = self.f.snapshot()[0]
        result = self.assert_completed(self.execute(self.request()))
        self.assertEqual(result["change_kind"], "unchanged")
        self.assertFalse(result["committed"])
        self.assertFalse(result["interactions_cleared"])
        self.assertEqual(result["cleanup"]["status"], "completed")
        self.assertEqual(result["cleanup"]["reading_revision"], 1)
        self.assertFalse(result["review_required"])
        document = self.f.repo.load_document("book")
        self.assertEqual(document.chapters, before.chapters)
        self.assertIn("note", document.document_task_artifacts.metadata)
        self.assertNotIn("public_summary", document.document_task_artifacts.metadata)
        self.assertNotIn("@@??", self.f.content())
        self.assertEqual(len(self.provider.calls), 1)

    def test_split_then_cleanup_single_final_commit_before_layout(self):
        self.kind = "structure_changed"
        request = self.request()
        with patch.object(self.f.store, "save", wraps=self.f.store.save) as save, \
                patch.object(self.coordinator, "ensure_document_task_layout",
                    wraps=self.coordinator.ensure_document_task_layout) as layout:
            result = self.assert_completed(self.execute(request))
        self.assertTrue(result["interactions_cleared"])
        self.assertEqual(result["structure_version"], 2)
        hierarchy_commits = [call for call in save.call_args_list if call.kwargs.get("replace_existing_hierarchy")]
        self.assertEqual(len(hierarchy_commits), 1)
        self.assertIsNotNone(hierarchy_commits[0].kwargs["_connection"])
        self.assertEqual(layout.call_count, 1)
        self.assertEqual(self.pipeline.build_enhancement_candidate.call_count, 1)

    def test_restart_reconciles_and_replay_does_not_repeat_mutation(self):
        request = self.request()
        result = self.assert_completed(self.execute(request))
        before = self.f.snapshot()
        self.service.operations = EnhancementOperations()
        self.service.secret = b"different-process-secret"
        with patch.object(self.coordinator, "ensure_document_task_layout", side_effect=AssertionError("rewrite")):
            lookup = self.lookup(request).json()
            self.assertEqual(lookup["durability"], "persisted_commit")
            self.assertEqual(lookup["result"], result)
            self.assertEqual(self.execute(request).json(), result)
        self.assertEqual(before, self.f.snapshot())
        self.assertEqual(len(self.provider.calls), 1)

    def test_layout_failure_retries_only_layout_even_after_restart(self):
        request = self.request()
        with patch.object(self.coordinator, "ensure_document_task_layout", side_effect=RuntimeError("PRIVATE")):
            first = self.execute(request)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["retry_stage"], "layout")
        self.assertEqual(first.json()["layout_error"], "enhancement_layout_failed")
        self.assertNotIn("PRIVATE", first.text)
        self.service.operations = EnhancementOperations()
        self.assertEqual(self.lookup(request).json()["result"]["phase"], "layout")
        self.assert_completed(self.execute(request))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(state_for(self.f.snapshot()[0])["revision"], 1)

    def test_result_read_never_prepares_and_unknown_never_regenerates(self):
        request = self.request()
        self.service.operations = EnhancementOperations()
        with patch.object(self.coordinator, "ensure_document_task_layout", side_effect=AssertionError("read write")):
            self.assertEqual(self.lookup(request).json()["status"], "unknown")
            self.assertEqual(self.execute(request).status_code, 409)
        self.assertEqual(self.provider.calls, [])

    def test_provider_timeout_is_retried_then_optional_failure_keeps_split(self):
        self.kind = "structure_changed"
        request = self.request()
        self.provider.error = TimeoutError("PRIVATE")
        result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["reason"], "provider_failed")
        self.assertTrue(result["review_required"])
        self.assertIn("@@??", self.f.content())
        self.assertEqual(len(self.provider.calls), 2)
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_cleanup_capacity_skip_does_not_dispatch_or_block_structure(self):
        request = self.request()
        original = self.provider.count_input_tokens
        def count(prompt):
            return 2000000 if 'ocr_cleanup_plan_v1' in prompt else original(prompt)
        with patch.object(self.provider, "count_input_tokens", side_effect=count):
            result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["reason"], "capacity_exceeded")
        self.assertEqual(self.provider.calls, [])

    def test_structural_capacity_failure_has_no_token_or_calls(self):
        self.provider.count = 2000000
        response = self.preflight()
        self.assertFalse(response.json()["execution_available"])
        self.assertIsNone(response.json()["confirmation_token"])
        self.assertEqual(self.provider.calls, [])

    def test_invalid_cleanup_keeps_source_and_reviews(self):
        request = self.request()
        self.provider.payload = '{"not_a_plan":true}'
        result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["reason"], "invalid_plan")
        self.assertEqual(result["pending_review_count"], 1)
        self.assertIn("@@??", self.f.content())

    def test_empty_cleanup_plan_is_valid_noop_with_durable_receipt(self):
        self.provider.payload = '{"schema_version":"ocr_cleanup_plan_v1","items":[]}'
        request = self.request()
        result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["status"], "no_changes")
        self.assertEqual(result["cleanup"]["reading_revision"], 0)
        self.assertEqual(self.lookup(request).json()["durability"], "persisted_commit")

    def test_receipt_and_all_writes_rollback_on_final_storage_failure(self):
        self.kind = "structure_changed"
        request = self.request()
        before = self.f.snapshot()
        save = self.f.store._commit_enhancement_receipt
        def fail(*args):
            save(*args)
            raise RuntimeError("PRIVATE injected after receipt")
        with patch.object(self.f.store, "_commit_enhancement_receipt", side_effect=fail):
            response = self.execute(request)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("PRIVATE", response.text)
        self.assertEqual(before, self.f.snapshot())
        self.assertEqual(self.lookup(request).json()["status"], "unknown")

    def test_lost_commit_ack_recovers_without_second_generation(self):
        request = self.request()
        commit = self.f.repo.commit_enhancement
        def lost(*args, **kwargs):
            commit(*args, **kwargs)
            raise ConnectionError("lost acknowledgement")
        with patch.object(self.f.repo, "commit_enhancement", side_effect=lost):
            self.assertEqual(self.execute(request).status_code, 500)
        self.service.operations = EnhancementOperations()
        self.assertEqual(self.lookup(request).json()["durability"], "persisted_commit")
        self.assert_completed(self.execute(request))
        self.assertEqual(len(self.provider.calls), 1)

    def test_stale_reading_revision_rejects_before_models(self):
        request = self.request()
        block = self.f.service.read("book")["blocks"][1]
        self.f.change("edit", block_id=block["block_id"], text="Authoritative edit.")
        response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.provider.calls, [])
        self.pipeline.build_enhancement_candidate.assert_not_called()

    def test_undo_restores_review_without_replaying_old_cleanup(self):
        request = self.request()
        self.assert_completed(self.execute(request))
        self.f.change("undo")
        result = self.lookup(request).json()["result"]
        self.assertEqual(result["phase"], "superseded")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["pending_review_count"], 1)
        self.assertFalse(result["retryable"])
        self.assertEqual(self.execute(request).json(), result)
        self.assertIn("@@??", self.f.content())
        self.assertEqual(len(self.provider.calls), 1)

    def test_consent_toggle_cannot_enable_cleanup(self):
        response = self.preflight(allow_ocr_cleanup=False)
        request = dict(doc_name="book", parser_mode="llm_enhanced", allow_ocr_cleanup=True,
            discard_interactions=True, confirmation_token=response.json()["confirmation_token"])
        self.assertEqual(self.execute(request).status_code, 409)
        self.assertEqual(self.provider.calls, [])

    def test_strict_dtos_reject_coercion_unknown_fields_and_wrong_mode(self):
        for value in ("true", 1, None):
            self.assertEqual(self.preflight(allow_ocr_cleanup=value).status_code, 422)
        self.assertEqual(self.preflight(unrecognized=True).status_code, 422)
        self.assertEqual(self.execute(dict(doc_name="book", parser_mode="common", allow_ocr_cleanup=True)).status_code, 422)

    def test_failed_cleanup_preserves_manual_edit_across_structural_commit(self):
        block = self.f.service.read("book")["blocks"][1]
        self.f.change("edit", block_id=block["block_id"], text="Authoritative user replacement.")
        self.kind = "structure_changed"
        request = self.request()
        self.provider.error = RuntimeError("permanent provider failure")
        result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["reason"], "provider_failed")
        self.assertEqual(result["cleanup"]["reading_revision"], 1)
        self.assertIn("Authoritative user replacement.", self.f.content())
        self.assertEqual(len(self.provider.calls), 1)

    def test_native_and_unknown_auto_skip_without_model_dispatch(self):
        for mode, reason in (("native", "native_source"), ("auto", "ocr_evidence_missing")):
            with self.subTest(mode=mode):
                document, _, _ = self.f.snapshot()
                from doc_loaders.extraction_identity import read_extraction_record
                from document_structure.ocr_cleanup_overlay import canonical_text
                identity, _ = read_extraction_record(document.parse_provenance)
                document = replace(document, parse_provenance={**document.parse_provenance,
                    "extraction": replace(identity, mode=mode).record(canonical_text(document))})
                self.f.repo.save_reparsed_document(document, "book")
                request = self.request()
                result = self.assert_completed(self.execute(request))
                self.assertEqual(result["cleanup"]["reason"], reason)
                self.assertEqual(self.provider.calls, [])

    def test_cleanup_policy_change_invalidates_confirmation(self):
        request = self.request()
        with patch("document_structure.ocr_cleanup_contract.CLEANUP_POLICY_VERSION", "future_policy"):
            response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.provider.calls, [])

    def test_source_change_blocks_layout_recovery_without_calls(self):
        request = self.request()
        with patch.object(self.coordinator, "ensure_document_task_layout", side_effect=RuntimeError("layout")):
            self.assertEqual(self.execute(request).status_code, 200)
        self.f.path.write_bytes(b"changed source")
        response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], "source_changed")
        self.assertEqual(len(self.provider.calls), 1)

    def test_partial_cleanup_does_not_claim_remaining_regions_resolved(self):
        document, _, _ = self.f.snapshot()
        regions = document.parse_provenance["source_review"]["regions"]
        regions.append({"id": "textless", "page_index": 1, "raw_candidate": ""})
        self.f.repo.save_document(document, "book")
        self.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
            {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"},
            {"action": "delete", "quote": "not in source", "reason": "ocr_garbage"}]})
        result = self.assert_completed(self.execute(self.request()))
        self.assertEqual(result["cleanup"]["status"], "partial")
        self.assertEqual(result["pending_review_count"], 1)
        self.assertTrue(result["review_required"])

    def test_transient_cleanup_can_recover_before_single_commit(self):
        request = self.request()
        complete = self.provider.complete_text_with_output_budget
        def transient(*args, **kwargs):
            if not self.provider.calls:
                self.provider.calls.append(("failed", 0, {}))
                raise TimeoutError("PRIVATE")
            return complete(*args, **kwargs)
        with patch.object(self.provider, "complete_text_with_output_budget", side_effect=transient):
            result = self.assert_completed(self.execute(request))
        self.assertEqual(result["cleanup"]["status"], "completed")
        self.assertEqual(result["cleanup"]["reading_revision"], 1)
        self.assertEqual(len(self.provider.calls), 2)

    def test_cleanup_only_receipt_failure_rolls_back_overlay_and_invalidation(self):
        request = self.request()
        before = self.f.snapshot()
        with patch.object(self.f.store, "_commit_enhancement_receipt", side_effect=RuntimeError("storage failed")):
            self.assertEqual(self.execute(request).status_code, 500)
        self.assertEqual(before, self.f.snapshot())

    def test_private_model_text_and_token_not_in_correlated_logs_or_receipt(self):
        request = self.request()
        self.provider.error = RuntimeError("PRIVATE_RESPONSE")
        with self.assertLogs(level="INFO") as captured:
            self.assert_completed(self.execute(request))
        logs = "\n".join(captured.output)
        self.assertIn(EnhancementOperations._key(request), logs)
        self.assertNotIn(request["confirmation_token"], logs)
        self.assertNotIn("PRIVATE_RESPONSE", logs)
        self.assertNotIn(self.f.text, logs)
        receipts = self.f.snapshot()[0].parse_provenance["enhancement_receipts"]
        self.assertNotIn(self.f.text, json.dumps(receipts))
        self.assertNotIn(request["confirmation_token"], json.dumps(receipts))

    def test_concurrent_edit_during_model_call_rejects_stale_commit(self):
        request = self.request()
        block = self.f.service.read("book")["blocks"][1]
        complete = self.provider.complete_text_with_output_budget
        def concurrent(*args, **kwargs):
            # A second connection can commit here: no generation transaction/locks.
            self.f.change("edit", block_id=block["block_id"], text="Concurrent user text.")
            return complete(*args, **kwargs)
        with patch.object(self.provider, "complete_text_with_output_budget", side_effect=concurrent):
            response = self.execute(request)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["detail"], "confirmation_stale")
        self.assertIn("Concurrent user text.", self.f.content())
        self.assertNotIn("enhancement_receipts", self.f.snapshot()[0].parse_provenance)

    def test_generation_failure_has_no_cleanup_or_commit_and_can_retry(self):
        from document_preparation.structure_enhancement_candidate import EnhancementCandidateRejected
        request = self.request()
        with patch.object(self.pipeline, "build_enhancement_candidate",
                          side_effect=EnhancementCandidateRejected("enhancement_provider_transient")):
            response = self.execute(request)
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["retry_stage"], "generation")
        self.assertEqual(self.provider.calls, [])
        self.assertNotIn("enhancement_receipts", self.f.snapshot()[0].parse_provenance)
        self.assert_completed(self.execute(request))

    def test_outcome_query_storage_failure_is_safe_not_permission_to_rerun(self):
        request = self.request()
        self.assert_completed(self.execute(request))
        with patch.object(self.f.repo, "load_enhancement_snapshot", side_effect=RuntimeError("PRIVATE STORAGE")):
            response = self.lookup(request)
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json()["detail"], "enhancement_result_unavailable")
            self.assertEqual(self.execute(request).status_code, 500)
        self.assertEqual(len(self.provider.calls), 1)


if __name__ == "__main__":
    unittest.main()
