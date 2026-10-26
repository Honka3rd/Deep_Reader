"""UI CLEAN-07: production components/routes, isolated SQL, controlled LLM only."""

from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Deep_Reflective_Reader"
sys.path[:0] = [str(BACKEND), str(BACKEND / "scripts")]


def run():
    if not os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"):
        raise SystemExit("Explicit disposable PostgreSQL DSN required")
    from scripts.test_enhancement_cleanup_api import CleanupApiTests
    from app.enhancement_operations import EnhancementOperations
    from app.cleanup_runtime import build_cleanup_service
    from document_structure.reading_content import state_for
    import main

    CleanupApiTests.setUpClass()
    case = None
    stack = ExitStack()
    lock = threading.Lock()
    def configure(scenario):
        nonlocal case, stack
        if scenario == "restart":
            case.service.operations = EnhancementOperations()
            case.service.secret = b"replacement-worker"
            main.app.state.smart_cleanup_service = build_cleanup_service(namespace=case.f.namespace,
                coordinator=case.coordinator, provider=case.provider)
            return
        if scenario == "provider_ready":
            case.provider.error = None
            case.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
                {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"}]})
            return
        if scenario == "layout_ready":
            case.coordinator.ensure_document_task_layout = case.real_layout
            return
        if scenario == "undo":
            case.f.change("undo")
            return
        if scenario == "stale":
            block = case.f.service.read("book")["blocks"][1]
            case.f.change("edit", block_id=block["block_id"], text="Concurrent user edit.")
            return
        if scenario == "long_history":
            def add_history(document):
                document = deepcopy(document)
                state = state_for(document)
                entry = state["audit"][-1]
                state["audit"] = [{**entry, "revision": i + 1} for i in range(23)]
                state["revision"] = 23
                return document, None
            case.f.repo.mutate_reading_content("book", add_history)
            return
        stack.close()
        stack = ExitStack()
        if case:
            case.doCleanups()
        case = CleanupApiTests()
        case.setUp()
        if os.environ.get("CLEANUP_BROWSER_SCRIPT") in {"test-smart-cleanup.mjs", "test-smart-cleanup-recovery.mjs"}:
            document = case.f.repo.load_document("book")
            chapters = document.chapters
            if scenario in {"missing_layout", "standalone_layout_failed"}:
                chapters = [replace(chapter, sections=[replace(section, task_units=[])
                    for section in chapter.sections]) for chapter in chapters]
            case.f.repo.save_document(replace(document, chapters=chapters,
                parse_provenance={**document.parse_provenance,
                    "effective_parser_mode": "common", "source": "common"}), "book")
        stack.enter_context(patch.object(main, "document_artifact_repository", case.f.repo))
        cleanup_service = build_cleanup_service(namespace=case.f.namespace,
            coordinator=case.coordinator, provider=case.provider)
        stack.enter_context(patch.object(main.app.state, "smart_cleanup_service", cleanup_service))
        case.real_layout = case.coordinator.ensure_document_task_layout
        if scenario not in {"missing_layout", "standalone_layout_failed"}:
            case.real_layout("book", task_unit_split_mode="progressive")
        if scenario == "failed":
            case.provider.error = RuntimeError("fixture provider failure")
        elif scenario == "skipped":
            original = case.provider.count_input_tokens
            case.provider.count_input_tokens = lambda prompt: 2000000 if "ocr_cleanup_plan_v1" in prompt else original(prompt)
        elif scenario == "partial":
            case.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
                {"action": "delete", "quote": "@@??", "reason": "ocr_garbage"},
                {"action": "delete", "quote": "absent quote", "reason": "ocr_garbage"}]})
        elif scenario == "no_changes":
            case.provider.payload = '{"schema_version":"ocr_cleanup_plan_v1","items":[]}'
        elif scenario == "unsafe":
            case.provider.payload = json.dumps({"schema_version": "ocr_cleanup_plan_v1", "items": [
                {"action": "delete", "quote": "absent quote", "reason": "ocr_garbage"}]})
        elif scenario in {"layout_failed", "standalone_layout_failed"}:
            def fail(*args, **kwargs):
                raise RuntimeError("fixture layout failure")
            case.coordinator.ensure_document_task_layout = fail
        elif scenario == "structure_changed":
            case.kind = "structure_changed"
        elif scenario == "rollback":
            original = case.f.store._commit_enhancement_receipt
            def fail(*args):
                original(*args)
                raise RuntimeError("fixture final write failure")
            case.f.store._commit_enhancement_receipt = fail
        elif scenario == "remaining_regions":
            def add_regions(document):
                provenance = deepcopy(document.parse_provenance)
                provenance["source_review"]["regions"].extend([
                    {"id": "textless", "page_index": 1, "raw_candidate": ""},
                    {"id": "unmapped", "page_index": 2, "raw_candidate": "Uncertain source text"}])
                return replace(document, parse_provenance=provenance), None
            case.f.repo.mutate_reading_content("book", add_regions)
        elif scenario not in {"completed", "missing_layout"}:
            raise ValueError("Unsupported fixture")

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            with lock:
                if self.path == "/fixture":
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    configure(body["scenario"])
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/stats":
                    payload = dict(calls=len(case.provider.calls),
                        builds=case.pipeline.build_enhancement_candidate.call_count,
                        revision=state_for(case.f.snapshot()[0]).get("revision", 0),
                        receipts=len(case.f.snapshot()[0].parse_provenance.get("enhancement_receipts", {})))
                    status, content = 200, json.dumps(payload).encode()
                else:
                    allowed = {"/documents/task-layout", "/documents/reparse-structure",
                        "/documents/reparse-structure/preflight", "/documents/reparse-structure/result"}
                    if self.path not in allowed and not self.path.startswith("/documents/book/"):
                        self.send_error(400)
                        return
                    body = None
                    if self.command == "POST":
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        if self.path in allowed and body.get("doc_name") != "book":
                            self.send_error(400)
                            return
                    response = case.client.request(self.command, self.path, json=body)
                    status, content = response.status_code, response.content
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(content)

        do_GET = do_POST = handle_request
        def log_message(self, *args):
            pass

    try:
        configure("completed")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            script = os.environ.get("CLEANUP_BROWSER_SCRIPT", "test-cleanup-flow.mjs")
            if script not in {"test-cleanup-flow.mjs", "test-cleanup-review.mjs", "test-smart-cleanup.mjs", "test-smart-cleanup-recovery.mjs"}:
                raise ValueError("Unsupported browser suite")
            subprocess.run(["node", str(Path(__file__).with_name(script))], check=True, timeout=300,
                env={**os.environ, "CLEANUP_TEST_API": f"http://127.0.0.1:{server.server_port}"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    finally:
        stack.close()
        if case:
            case.doCleanups()
        CleanupApiTests.doClassCleanups()


if __name__ == "__main__":
    run()
