"""Preflight observability must not add reads, generation or authority."""

import asyncio
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from app import standalone_cleanup as service
from document_preparation.accepted_source_snapshot import SourceNotReady
from document_preparation.cleanup_input_adapter import CleanupInputUnavailable
from llm.offline_token_counter import OfflineTokenCounter
from logging_config import RequestContextFilter, request_id_context, stage_id_context
from scripts import test_standalone_cleanup_service as fixtures


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.addFilter(RequestContextFilter())

    def emit(self, record):
        self.records.append(record)


class CleanupPreflightLoggingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.StandaloneCleanupServiceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.flow = self.f.flow
        self.capture = _Capture()
        for logger in (service.LOGGER, logging.getLogger("llm.offline_token_counter")):
            previous = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(self.capture)
            self.addCleanup(logger.setLevel, previous)
            self.addCleanup(logger.removeHandler, self.capture)
        self.addCleanup(self.capture.close)

    def stages(self, suffix=None, event=None):
        result = []
        for record in self.capture.records:
            message = record.getMessage()
            if "stage=cleanup_preflight_" not in message:
                continue
            head, metadata = message.split(" metadata=", 1)
            stage = re.search(r"stage=cleanup_preflight_(\w+)", head).group(1)
            if suffix is not None and stage != suffix:
                continue
            if event is not None and not head.startswith(event + " "):
                continue
            result.append((record, head, json.loads(metadata)))
        return result

    def terminal(self, reason=None, ready=False):
        records = self.stages("overall", "stage_finished")
        self.assertEqual(len(records), 1)
        metadata = records[0][2]
        self.assertEqual(metadata["execution_available"], ready)
        self.assertEqual(metadata["reason"], reason)
        self.assertEqual(metadata["status"], "ready" if ready else "unavailable")
        return records[0]

    def test_success_pairs_parentage_single_read_count_and_no_writes(self):
        before = self.f.snapshot()
        with patch.object(self.flow, "read_input", wraps=self.flow.read_input) as read, \
                patch.object(self.f.provider, "count_input_tokens", wraps=self.f.provider.count_input_tokens) as count, \
                patch.object(self.f.repo, "commit_cleanup", side_effect=AssertionError("no commit")), \
                patch.object(self.flow, "prepare_layout", side_effect=AssertionError("no layout")):
            result = self.flow.preflight("book")
        self.assertTrue(result.ready)
        self.assertEqual((read.call_count, count.call_count), (1, 1))
        self.assertFalse(self.f.provider.calls)
        self.assertEqual(before, self.f.snapshot())
        overall = self.terminal(ready=True)
        parent = re.search(r"stage_id=(\w+)", overall[1]).group(1)
        for name in ("overall", "lease", "reconciliation", "input", "capacity", "consent"):
            self.assertEqual(len(self.stages(name, "stage_started")), 1)
            completed = self.stages(name, "stage_finished")
            self.assertEqual(len(completed), 1)
            self.assertRegex(completed[0][1], r"duration_ms=\d+\.\d+")
            self.assertEqual(completed[0][2]["parent_stage_id"], "-" if name == "overall" else parent)
        messages = "\n".join(record.getMessage() for record in self.capture.records)
        for private in (result.confirmation_token, result.operation_id, str(self.f.path),
                        self.flow._attempts[result.operation_id].binding.input_fingerprint,
                        self.f.provider.counted[0]):
            self.assertNotIn(private, messages)
        self.assertEqual(stage_id_context.get(), "-")

    def test_independent_stage_durations_not_lease_hold_time(self):
        clock = [0.0]
        read, capacity, issue = self.flow.read_input, self.flow._capacity, service.secrets.token_urlsafe

        @contextmanager
        def lease(document_id):
            clock[0] += 1
            yield

        def delayed_read(document_id):
            clock[0] += 2
            return read(document_id)

        def delayed_capacity(*args):
            clock[0] += 3
            return capacity(*args)

        def delayed_issue(*args):
            clock[0] += 4
            return issue(*args)

        with patch("logging_config.time.perf_counter", side_effect=lambda: clock[0]), \
                patch.object(self.flow, "lease", lease), patch.object(self.flow, "read_input", delayed_read), \
                patch.object(self.flow, "_capacity", delayed_capacity), \
                patch.object(service.secrets, "token_urlsafe", delayed_issue):
            self.assertTrue(self.flow.preflight("book").ready)
        for name, duration in (("lease", 1000), ("input", 2000), ("capacity", 3000),
                               ("consent", 4000), ("overall", 10000)):
            self.assertIn(f"duration_ms={duration:.1f}", self.stages(name, "stage_finished")[0][1])

    def test_capacity_refusal_and_unavailable_have_failed_stage_and_safe_outcome(self):
        for count, capabilities, reason in ((10000000, self.f.provider.capabilities, "capacity_exceeded"),
                                             (None, None, "capacity_unavailable")):
            self.capture.records.clear()
            self.f.provider.count, self.f.provider.capabilities = count, capabilities
            result = self.flow.preflight("book")
            self.assertEqual(result.reason, reason)
            self.assertIsNone(result.confirmation_token)
            self.assertEqual(len(self.stages("capacity", "stage_failed")), 1)
            self.assertFalse(self.stages("consent"))
            self.terminal(reason)
        self.assertFalse(self.f.provider.calls)

    def test_unavailable_local_asset_fallback_has_no_network_or_duplicate_count(self):
        counter = OfflineTokenCounter(Path(self.f.temp.name) / "missing-assets")
        with patch("socket.socket.connect", side_effect=AssertionError("no network")), \
                patch.object(self.f.provider, "count_input_tokens",
                             side_effect=lambda text: counter.count("gpt-4.1-mini", text)) as count:
            self.assertTrue(self.flow.preflight("book").ready)
        self.assertEqual(count.call_count, 1)
        self.assertTrue(any("status=asset_unavailable" in r.getMessage() for r in self.capture.records))
        self.terminal(ready=True)
        self.assertFalse(self.f.provider.calls)

    def test_input_exception_matrix_preserves_reasons_and_excludes_details(self):
        for error, reason in ((FileNotFoundError("PRIVATE_PATH"), "cleanup_document_missing"),
                              (SourceNotReady("PRIVATE_SOURCE"), "cleanup_source_not_ready"),
                              (CleanupInputUnavailable("PRIVATE_SOURCE"), "cleanup_source_not_ready"),
                              (service.CleanupUnavailable("cleanup_identity_changed"), "cleanup_identity_changed"),
                              (RuntimeError("PRIVATE_STORAGE"), "cleanup_preflight_unavailable")):
            self.capture.records.clear()
            with patch.object(self.flow, "read_input", side_effect=error):
                self.assertEqual(self.flow.preflight("book").reason, reason)
            self.assertEqual(len(self.stages("input", "stage_failed")), 1)
            self.assertFalse(self.stages("capacity"))
            self.terminal(reason)
            self.assertNotIn("PRIVATE_", "\n".join(r.getMessage() for r in self.capture.records))
        self.assertFalse(self.f.mutex.locked())

    def test_arbitrary_use_case_error_is_not_logged_as_a_safe_reason(self):
        with patch.object(self.flow, "read_input", side_effect=service.CleanupUnavailable("PRIVATE_KEY\nTEXT")):
            result = self.flow.preflight("book")
        self.assertEqual(result.reason, "PRIVATE_KEY\nTEXT")  # Transport sanitization stays with the route.
        self.terminal("cleanup_preflight_unavailable")
        self.assertNotIn("PRIVATE_KEY", "\n".join(r.getMessage() for r in self.capture.records))

    def test_busy_lease_skips_input_capacity_and_consent(self):
        self.f.mutex.acquire()
        try:
            result = self.flow.preflight("book")
        finally:
            self.f.mutex.release()
        self.assertEqual(result.reason, "cleanup_in_progress")
        self.assertEqual(len(self.stages("lease", "stage_failed")), 1)
        self.assertFalse(self.stages("input"))
        self.terminal("cleanup_in_progress")

    def test_ineligible_source_stops_before_capacity_without_issuing_consent(self):
        snapshot = self.flow.read_input("book")
        for reason in ("native_source", "extraction_mode_unsupported", "ocr_evidence_missing", "ocr_evidence_mismatch"):
            self.capture.records.clear()
            with patch.object(type(snapshot), "eligibility", return_value=SimpleNamespace(status="skipped", reason=reason)):
                result = self.flow.preflight("book")
            self.assertFalse(result.ready)
            self.assertIsNone(result.confirmation_token)
            self.assertFalse(self.stages("capacity"))
            self.assertEqual(self.stages("input", "stage_finished")[0][2]["reason"], reason)
            self.terminal(reason)
        self.assertFalse(self.f.provider.counted)
        self.assertFalse(self.f.provider.calls)

    def test_registry_full_and_consent_exception_attribute_stage(self):
        self.flow.limit = 0
        self.assertEqual(self.flow.preflight("book").reason, "cleanup_operation_capacity")
        self.assertEqual(self.stages("consent", "stage_finished")[0][2]["reason"], "cleanup_operation_capacity")
        self.terminal("cleanup_operation_capacity")
        self.capture.records.clear()
        with patch.object(service.secrets, "token_urlsafe", side_effect=RuntimeError("PRIVATE_TOKEN")):
            self.assertEqual(self.flow.preflight("book").reason, "cleanup_preflight_unavailable")
        self.assertEqual(len(self.stages("consent", "stage_failed")), 1)
        self.terminal("cleanup_preflight_unavailable")
        self.assertNotIn("PRIVATE_TOKEN", "\n".join(r.getMessage() for r in self.capture.records))

    def test_reconciliation_refusal_has_terminal_outcome(self):
        consent = self.flow.preflight("book")
        entry = self.flow._attempts[consent.operation_id]
        entry.result = self.flow._unknown(consent.operation_id, entry)
        self.capture.records.clear()
        with patch.object(self.f.repo, "lookup_cleanup", side_effect=OSError("PRIVATE_STORAGE")):
            result = self.flow.preflight("book")
        self.assertEqual(result.reason, "cleanup_reconciliation_unavailable")
        self.terminal(result.reason)
        self.assertFalse(self.stages("input"))
        self.assertEqual(self.stages("reconciliation", "stage_finished")[0][2]["reason"], result.reason)

    def test_logging_sink_failure_preserves_success_and_failure(self):
        with patch.object(service.LOGGER, "info", side_effect=OSError("broken log")), \
                patch.object(service.LOGGER, "warning", side_effect=OSError("broken log")):
            self.assertTrue(self.flow.preflight("book").ready)
            with patch.object(self.flow, "read_input", side_effect=FileNotFoundError()):
                self.assertEqual(self.flow.preflight("book").reason, "cleanup_document_missing")
        self.assertFalse(self.f.mutex.locked())
        self.assertEqual(stage_id_context.get(), "-")

    def test_cancellation_propagates_and_resets_context_and_lease(self):
        error = asyncio.CancelledError("PRIVATE_CANCEL")
        with patch.object(self.flow, "read_input", side_effect=error):
            with self.assertRaises(asyncio.CancelledError) as caught:
                self.flow.preflight("book")
        self.assertIs(caught.exception, error)
        self.assertFalse(self.f.mutex.locked())
        self.assertEqual(stage_id_context.get(), "-")
        self.assertEqual(len(self.stages("overall", "stage_failed")), 1)
        self.assertFalse(self.stages("overall", "stage_failed")[0][2]["execution_available"])
        self.assertNotIn("PRIVATE_CANCEL", "\n".join(r.getMessage() for r in self.capture.records))

    def test_concurrent_worker_request_and_stage_isolation(self):
        barrier = Barrier(2)
        read = self.flow.read_input

        @contextmanager
        def lease(document_id):
            barrier.wait(timeout=5)
            yield

        def reader(document_id):
            if request_id_context.get() == "failure-request":
                raise FileNotFoundError("PRIVATE_PATH")
            return read(document_id)

        async def invoke(request_id):
            token = request_id_context.set(request_id)
            try:
                return await asyncio.to_thread(self.flow.preflight, "book")
            finally:
                request_id_context.reset(token)

        async def run():
            return await asyncio.gather(invoke("success-request"), invoke("failure-request"))

        with patch.object(self.flow, "lease", lease), patch.object(self.flow, "read_input", reader):
            success, failure = asyncio.run(run())
        self.assertTrue(success.ready)
        self.assertEqual(failure.reason, "cleanup_document_missing")
        parents = {r.request_id: re.search(r"stage_id=(\w+)", head).group(1)
                   for r, head, _ in self.stages("overall", "stage_finished")}
        self.assertEqual(len(set(parents.values())), 2)
        for r, head, metadata in self.stages():
            self.assertIn(r.request_id, parents)
            if "stage=cleanup_preflight_overall " not in head:
                self.assertEqual(metadata["parent_stage_id"], parents[r.request_id])
        self.assertEqual(request_id_context.get(), "-")
        self.assertEqual(stage_id_context.get(), "-")

    def test_actual_http_request_id_reaches_worker_and_dto_is_unchanged(self):
        import main
        from fastapi.testclient import TestClient
        from smart_cleanup_routes import get_smart_cleanup_service

        overrides = dict(main.app.dependency_overrides)
        self.addCleanup(setattr, main.app, "dependency_overrides", overrides)
        main.app.dependency_overrides[get_smart_cleanup_service] = lambda: self.flow
        with TestClient(main.app) as client:
            response = client.post("/documents/book/smart-cleanup/preflight", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"execution_available", "reason", "operation_id",
                                               "confirmation_token", "expires_in_seconds", "impact"})
        for record, _, _ in self.stages():
            self.assertEqual(record.request_id, response.headers["X-Request-ID"])
        self.terminal(ready=True)
        self.assertEqual(request_id_context.get(), "-")


if __name__ == "__main__":
    unittest.main()
