#!/usr/bin/env python3
"""Disposable matched orchestration baselines; no paid APIs or production data."""

import io
import asyncio
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logging_config import (
    ExternalResponseFilter, RequestContextFilter, log_stage,
    request_id_context, stage_id_context,
)
from test_task_layout_persistence_cache import (
    SectionTaskCoordinator, StructuredDocumentArtifactRepository, StructuredDocumentStore,
    _FakeTaskUnitResolver, _FakeProfileStore, _FakeSectionTaskService,
    _FakeEnhancedParseEvaluator,
    _build_base_document,
)
from test_prepare_structured_reuse_before_raw_load import (
    _build_pipeline,
)
from embeddings.openai_embedder import OpenAIEmbedder
from app.task_layout_readiness import TaskLayoutNotReadyError


class TemporaryStore(StructuredDocumentStore):
    def __init__(self, path):
        self.path = path

    def location(self, target):
        return str(self.path)

    def exists(self, target):
        return self.path.exists()

    def load(self, target):
        return super().load(str(self.path))

    def save(self, document, target):
        return super().save(document, str(self.path))


class FixtureLoader:
    def load(self, doc_name):
        return _build_base_document().raw_text


class FixtureBuilder:
    def build(self, **kwargs):
        return _build_base_document()


class LayoutLoggingTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.handler = logging.StreamHandler(self.output)
        self.handler.addFilter(RequestContextFilter())
        self.handler.setFormatter(logging.Formatter("request_id=%(request_id)s %(message)s"))
        self.root = logging.getLogger()
        self.old_level = self.root.level
        self.root.setLevel(logging.INFO)
        self.root.addHandler(self.handler)
        self.filter = ExternalResponseFilter()
        logging.getLogger("httpx").addFilter(self.filter)
        self.token = request_id_context.set("layout-test")

    def tearDown(self):
        request_id_context.reset(self.token)
        logging.getLogger("httpx").removeFilter(self.filter)
        self.root.removeHandler(self.handler)
        self.root.setLevel(self.old_level)
        self.handler.close()

    def test_nested_failure_restores_context_and_excludes_error_content(self):
        with self.assertRaises(ValueError):
            with log_stage(logging.getLogger(__name__), "outer"):
                outer = stage_id_context.get()
                with log_stage(logging.getLogger(__name__), "inner"):
                    raise ValueError("PRIVATE_PROMPT")
        self.assertEqual(stage_id_context.get(), "-")
        output = self.output.getvalue()
        self.assertIn('"parent_stage_id": "' + outer + '"', output)
        self.assertNotIn("PRIVATE_PROMPT", output)
        self.assertEqual(output.count("stage_failed"), 2)

    def test_sdk_batch_is_not_http_exchange_count(self):
        statuses = iter([429, 200, 200])
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(next(statuses), json={})
        )) as client:
            class Backend:
                def get_text_embedding_batch(self, texts):
                    for _ in range(3):
                        client.post("https://test.invalid/v1/embeddings", json={"input": texts},
                                    headers={"Authorization": "Bearer PRIVATE_KEY"})
                    return [[1.0]]
            embedder = object.__new__(OpenAIEmbedder)
            embedder.embed_model = Backend()
            self.assertEqual(embedder.get_text_embedding_batch(["PRIVATE_PROMPT"]), [[1.0]])
        output = self.output.getvalue()
        events = [line for line in output.splitlines() if "external_response" in line]
        self.assertEqual(len(events), 3)
        self.assertEqual(output.count("stage_finished stage=embedding.batch"), 1)
        self.assertTrue(all("request_id=layout-test" in line and "stage_id=-" not in line for line in events))
        self.assertNotIn("PRIVATE_PROMPT", output)
        self.assertNotIn("PRIVATE_KEY", output)
        self.assertTrue(all("test.invalid" not in line for line in events))

    def test_transport_failure_is_not_a_completed_exchange(self):
        def fail(request):
            raise httpx.ConnectError("PRIVATE_ERROR")
        with httpx.Client(transport=httpx.MockTransport(fail)) as client:
            with self.assertRaises(httpx.ConnectError):
                client.post("https://test.invalid/v1/responses")
        self.assertNotIn("external_response", self.output.getvalue())

    def test_concurrent_stage_context_isolated_and_worker_inherited(self):
        async def run_one(name):
            token = request_id_context.set(name)
            try:
                self.assertEqual(stage_id_context.get(), "-")
                with log_stage(logging.getLogger(__name__), name):
                    own_stage = stage_id_context.get()
                    await asyncio.sleep(0)
                    self.assertEqual(stage_id_context.get(), own_stage)
                    self.assertEqual(await asyncio.to_thread(stage_id_context.get), own_stage)
                    self.assertEqual(await asyncio.to_thread(request_id_context.get), name)
                self.assertEqual(stage_id_context.get(), "-")
            finally:
                request_id_context.reset(token)

        async def run():
            await asyncio.gather(run_one("first"), run_one("second"))

        asyncio.run(run())

    def test_matched_baselines(self):
        results = []
        for state in ("cold", "structured-ready", "layout-ready"):
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                store = TemporaryStore(Path(directory) / "cache-doc.structured.json")
                repository = StructuredDocumentArtifactRepository(store=store, base_dir=directory)
                pipeline = _build_pipeline(loader=FixtureLoader(), store=store)
                pipeline.structured_document_builder = FixtureBuilder()
                resolver = _FakeTaskUnitResolver()
                coordinator = SectionTaskCoordinator(
                    document_preparation_pipeline=pipeline,
                    document_artifact_repository=repository,
                    document_profile_store=_FakeProfileStore(),
                    chapter_summary_service=_FakeSectionTaskService(),
                    chapter_quiz_service=_FakeSectionTaskService(),
                    task_unit_resolver=resolver,
                    enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator(),
                )
                if state != "cold":
                    pipeline.prepare_and_load("cache-doc", mode="base")
                if state == "layout-ready":
                    coordinator.ensure_document_task_layout("cache-doc")
                resolver.calls = 0
                pipeline.language_detector.detect_calls = 0
                self.output.seek(0)
                self.output.truncate(0)
                token = request_id_context.set(state)
                outcome = "ready"
                try:
                    coordinator.get_document_task_layout("cache-doc")
                except TaskLayoutNotReadyError as error:
                    outcome = error.reason
                finally:
                    request_id_context.reset(token)
                output = self.output.getvalue()
                spans = []
                for line in output.splitlines():
                    match = re.search(r"stage_(?:finished|failed) stage=(\S+) stage_id=(\S+) duration_ms=([\d.]+)(?: error_type=\S+)? metadata=(.*)", line)
                    if match:
                        stage, span_id, duration, metadata = match.groups()
                        spans.append((stage, span_id, float(duration), json.loads(metadata)))
                total = next(span for span in spans if span[0] == "task_layout.total")
                direct = [span for span in spans if span[3]["parent_stage_id"] == total[1]]
                self.assertTrue(all("request_id=" + state in line for line in output.splitlines()))
                self.assertNotIn(_build_base_document().raw_text, output)
                self.assertEqual(resolver.calls, 0)
                self.assertEqual(pipeline.language_detector.detect_calls, 0)
                self.assertEqual(outcome, {"cold": "missing_structure", "structured-ready": "missing_task_units", "layout-ready": "ready"}[state])
                self.assertIn("stage=task_layout.repository_read", output)
                if state == "layout-ready":
                    anchor = next(span for span in spans if span[0] == "task_layout.anchor_evidence")
                    self.assertTrue(anchor[3]["skipped"])
                results.append({
                    "outcome": outcome,
                    "state": state, "total_ms": total[2],
                    "direct_children_ms": {span[0]: span[2] for span in direct},
                    "residual_ms": round(total[2] - sum(span[2] for span in direct), 1),
                    "unit_resolution_calls": resolver.calls,
                    "language_stub_calls": pipeline.language_detector.detect_calls,
                    "completed_http_exchanges": output.count("external_response"),
                })
        print(json.dumps({"scope": "real coordinator/preparation + temporary JSON; fake raw/parser/profile/language/resolver, no network", "baselines": results}, indent=2))


if __name__ == "__main__":
    unittest.main()
