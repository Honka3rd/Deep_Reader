"""Browser -> real FastAPI -> disposable PostgreSQL; deterministic planner only.

Requires DEEP_READER_TEST_POSTGRES_DSN, an existing UI_URL, and Playwright env.
Only enhancement/layout routes for the disposable document are served.
"""
from contextlib import ExitStack
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch
import psycopg
from psycopg.types.json import Jsonb

from test_enhancement_postgres_preflight import main, SectionTaskCoordinator
from test_enhancement_conditional_commit import ConditionalCommitTests as PostgresPreflightTests
from test_task_layout_persistence_cache import _FakeTaskUnitResolver, _FakeSectionTaskService, _FakeEnhancedParseEvaluator
from fastapi.testclient import TestClient
from document_structure.enhancement_resources import EnhancementResourceLimits
from llm.llm_model_capabilities import LLMModelCapabilities


def run():
    if not os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"):
        raise SystemExit("Explicit disposable PostgreSQL test DSN required")
    PostgresPreflightTests.setUpClass()
    case = PostgresPreflightTests()
    try:
        case.setUp()
        # Longer than the removed 120000-character clamp; all text is synthetic.
        raw = ("A readable idea. \u4e2d\u6587\u5168\u6587\u3002 Une autre idee.\n" * 5000)[:147401]
        raw = ("Unique opening discussion. " + raw[:73000] + "\nUnique later discussion. " + raw)[:147401]
        document = case.fixture.document
        section = replace(case.fixture.section, content=raw, char_end=len(raw))
        document = replace(document, raw_text=raw,
            chapters=[replace(document.chapters[0], sections=[section])],
            parse_provenance={"pdf_extraction": case.fixture.identity.record(raw)})
        case.repository.save_reparsed_document(document, "doc")

        plan = {"title": None, "divisions": 1, "fail_once": False}
        structural_commits = []
        def verify_dispatch(payload, context):
            source = context["source"]
            assert source["full_source"] is True
            assert source["omitted_ranges"] == []
            assert source["text"] == raw
            if plan["fail_once"]:
                plan["fail_once"] = False
                raise TimeoutError("synthetic provider transport failure")
            if plan["title"]:
                payload["boundaries"][0].update(title=plan["title"], origin="generated")
            if plan["divisions"] == 2:
                payload["boundaries"].append({"start_quote": "Unique later discussion.",
                    "title": "Later discussion", "origin": "generated", "level": 1, "role": "main_body"})
            return payload

        case.enable_execution(verify_dispatch)
        initial_version = case.snapshot().structure_version
        splitter = case.coordinator.document_preparation_pipeline.structured_document_builder.section_splitter_selector.llm_splitter
        default_capabilities = case.provider.get_model_capabilities
        source_path = Path(case.fixture.temp.name) / "doc.pdf"
        source_before = source_path.read_bytes()

        def configure(scenario):
            source_path.write_bytes(source_before)
            case.provider.get_model_capabilities = default_capabilities
            splitter.resource_limits = EnhancementResourceLimits()
            if scenario == "exceeded":
                case.provider.get_model_capabilities = lambda: LLMModelCapabilities("small", "responses", 2000, 500)
            elif scenario == "unavailable":
                case.provider.get_model_capabilities = lambda: None
            elif scenario == "resource_limited":
                splitter.resource_limits = EnhancementResourceLimits(max_source_chars=1)
            elif scenario == "changed_model":
                case.provider.get_model_capabilities = lambda: LLMModelCapabilities("changed", "responses", 1000000, 4000)
            elif scenario == "changed_policy":
                splitter.resource_limits = EnhancementResourceLimits(max_source_chars=1000000)
            elif scenario == "changed_source":
                source_path.write_bytes(b"changed synthetic source")
            elif scenario.startswith("titles_"):
                plan["title"] = scenario
            elif scenario == "structural":
                plan["divisions"] = 3 - plan["divisions"]
            elif scenario == "transient":
                plan["fail_once"] = True
            elif scenario == "seed":
                for kind in ("note", "quiz", "critical_thinking_session", "analysis"):
                    case.add_artifact(kind, source_version=case.snapshot().structure_version)
                with psycopg.connect(case.scoped) as connection:
                    connection.execute("UPDATE artifacts SET payload=%s WHERE document_id=%s",
                        (Jsonb({"answer": "saved answer", "note": "saved note", "question": "saved question"}), case.doc_id))
            elif scenario != "fits":
                raise ValueError("Unknown synthetic scenario")
        SectionTaskCoordinator.__init__(case.coordinator,
            document_preparation_pipeline=case.coordinator.document_preparation_pipeline,
            document_artifact_repository=case.repository, document_profile_store=case.coordinator.document_profile_store,
            chapter_summary_service=_FakeSectionTaskService(), chapter_quiz_service=_FakeSectionTaskService(),
            task_unit_resolver=_FakeTaskUnitResolver(), enhanced_parse_trigger_evaluator=_FakeEnhancedParseEvaluator())
        case.coordinator.ensure_document_task_layout(doc_name="doc", task_unit_split_mode="progressive")
        case.add_artifact("note", source_version=initial_version)
        config = SimpleNamespace(doc_name="doc", get_raw_profile_path=lambda: str(Path(case.fixture.temp.name) / "missing.json"))
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "section_task_coordinator", case.coordinator))
            stack.enter_context(patch.object(main, "enhancement_preflight_service", case.service))
            stack.enter_context(patch("app.enhancement_preflight.enhancement_preflight_service", case.service))
            stack.enter_context(patch("app.section_task_coordinator.FaissStorageConfig", return_value=config))
            stack.enter_context(patch.object(case.coordinator.document_preparation_pipeline,
                "prepare_and_load", side_effect=AssertionError("Enhancement must reuse accepted source")))
            client = TestClient(main.app)
            stack.callback(client.close)

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    allowed = {"/documents/reparse-structure/preflight", "/documents/reparse-structure",
                               "/documents/task-layout", "/documents/prepare-task-layout",
                               "/documents/reparse-structure/result"}
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    # Test-only control, never installed in the product FastAPI app.
                    if self.path == "/fixture":
                        configure(body["scenario"])
                        self.send_response(204)
                        self.end_headers()
                        return
                    if self.path not in allowed or body.get("doc_name") != "doc":
                        self.send_error(400)
                        return
                    before = case.snapshot()
                    tables = ("documents", "chapters", "sections", "task_units", "content_blocks",
                              "artifacts", "structured_document_snapshots", "parse_events")
                    rows_before = {table: case.rows(table) for table in tables}
                    calls_before = case.provider.calls
                    response = client.post(self.path, json=body)
                    if self.path.endswith(("/preflight", "/result", "/task-layout")) or (
                            self.path == "/documents/reparse-structure" and response.status_code != 200):
                        assert case.snapshot() == before, "Read/rejection mutated stored hierarchy or interactions"
                        assert rows_before == {table: case.rows(table) for table in tables}
                        assert case.provider.calls == calls_before + (response.status_code == 502), "Unexpected generation count"
                    if self.path == "/documents/reparse-structure" and response.status_code == 200:
                        assert case.provider.calls == calls_before + 1
                        if response.json()["change_kind"] == "structure_changed":
                            assert case.snapshot().interaction_count == 0
                            assert case.snapshot().structure_version == before.structure_version + 1
                            structural_commits.append(response.json()["structure_version"])
                        elif response.json()["change_kind"] == "titles_only":
                            assert case.snapshot().structure_version == before.structure_version
                            for table in ("documents", "task_units", "content_blocks", "artifacts", "parse_events"):
                                assert rows_before[table] == case.rows(table), table
                            for table in ("chapters", "sections"):
                                assert [row[0]["id"] for row in rows_before[table]] == [row[0]["id"] for row in case.rows(table)]
                            snapshot = case.rows("structured_document_snapshots")[0][0]["structured_document_payload"]
                            assert snapshot["chapters"][0]["title"] == case.snapshot().document.chapters[0].title == plan["title"]
                        else:
                            assert response.json()["change_kind"] == "unchanged"
                            assert case.snapshot() == before, "No-op changed storage or interactions"
                            assert rows_before == {table: case.rows(table) for table in tables}, "No-op wrote rows"
                        assert case.coordinator.read_enhancement_state("doc").source.canonical_text == raw
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(response.content)

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                script = Path(__file__).resolve().parents[2] / "Deep_Reader_UI/scripts/test-enhancement-live.mjs"
                subprocess.run(["node", str(script)], check=True, timeout=120,
                    env={**os.environ, "ENHANCEMENT_TEST_API": f"http://127.0.0.1:{server.server_port}"})
                script = script.with_name("test-enhancement-lifecycle-live.mjs")
                subprocess.run(["node", str(script)], check=True, timeout=120,
                    env={**os.environ, "ENHANCEMENT_TEST_API": f"http://127.0.0.1:{server.server_port}"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        assert case.provider.calls == 14
        assert case.snapshot().structure_version == initial_version + 3
        assert len(structural_commits) == 3
        assert (Path(case.fixture.temp.name) / "doc.pdf").read_bytes() == source_before
        print("PASS browser/API/DB: full-source three-way lifecycle, preserved rows/payloads, transient retry, lost-response reconciliation, three structural commits; no paid LLM")
    finally:
        case.doCleanups()
        PostgresPreflightTests.doClassCleanups()


if __name__ == "__main__":
    run()
