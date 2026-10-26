"""Disposable SQL browser acceptance; optional READING_TEST_REAL_CASE=sample2/noise.

Real mode constructs review evidence from accepted backup page text, never a
natural-positive claim or canonical_span injection. Preview remains a UI fixture.
"""

from contextlib import ExitStack
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Deep_Reflective_Reader"), str(ROOT / "Deep_Reflective_Reader/scripts")]


def run():
    from scripts.test_enhancement_cleanup_api import CleanupApiTests
    from document_structure.reading_content import ReadingContentService
    from section_tasks.note_interaction_service import NoteInteractionService
    from document_structure.ocr_cleanup_overlay import canonical_text
    import main

    real_case = os.environ.get("READING_TEST_REAL_CASE", "")
    source_case = None
    backup_hashes = {}
    if real_case:
        from scripts.accept_source_review_mapping import load_cases, real_page_case, correspondence
        cases, backup_hashes = load_cases()
        source_case = next((item for item in cases if item[0] == real_case), None)
        if source_case is None:
            raise ValueError("READING_TEST_REAL_CASE must name a backed-up acceptance case")
    CleanupApiTests.setUpClass()
    case = None
    stack = ExitStack()
    lock = threading.Lock()

    def configure():
        nonlocal case, stack
        stack.close()
        if case:
            case.doCleanups()
        stack = ExitStack()
        case = CleanupApiTests()
        case.setUp()
        stack.enter_context(patch.object(main, "document_artifact_repository", case.f.repo))
        case.region_id = "r1"
        case.evidence = {"kind": "synthetic_source", "preview": "ui_fixture"}
        if source_case:
            from section_tasks.task_unit_resolver import TaskUnitResolver
            from shared.task_artifacts import DocumentTaskArtifacts
            name, source, original = source_case
            document, evidence = real_page_case(original)
            copied_source = Path(case.f.temp.name) / "book.pdf"
            shutil.copyfile(source, copied_source)
            case.f.repo.save_reparsed_document(replace(document, source_path=str(copied_source),
                document_task_artifacts=DocumentTaskArtifacts()), "book")
            case.coordinator.task_unit_resolver = TaskUnitResolver()
            case.region_id = "real-subquote"
            case.evidence = {**evidence, "case": name, "preview": "ui_fixture"}
        case.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
        if source_case:
            case.evidence["correspondence"] = correspondence(case.f.repo.load_document("book"))
        service = ReadingContentService(case.f.repo)
        case.notes = NoteInteractionService(case.f.repo)
        state = service.read("book", region_id=case.region_id)
        assert any(r["id"] == case.region_id and r["body_status"] == "linked" for r in state["regions"])
        block = state["blocks"][0]
        case.target = dict(target_level="content_block", task_unit_id=block["task_unit_id"], content_block_id=block["block_id"])
        token = case.notes.read("book", **case.target)["target_token"]
        case.notes.save("book", **case.target, text="Keep this annotation", target_token=token,
                        expected_revision=0, mutation_id="initial-note")

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            with lock:
                if self.path == "/fixture":
                    configure()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/stats":
                    document = case.f.repo.load_document("book")
                    entries = document.document_task_artifacts.metadata.get("note", {})
                    payload = dict(note_count=sum(isinstance(v, dict) and bool(v.get("note")) for v in entries.values()),
                        model_calls=len(case.provider.calls), prepare_calls=case.pipeline.prepare_and_load.call_count,
                        source_sha256=hashlib.sha256(Path(document.source_path).read_bytes()).hexdigest(),
                        canonical_sha256=hashlib.sha256(canonical_text(document).encode()).hexdigest(),
                        region_id=case.region_id, evidence=case.evidence)
                    status, content = 200, json.dumps(payload).encode()
                else:
                    if not (self.path.startswith("/documents/book/") or self.path == "/documents/task-layout"):
                        self.send_error(400)
                        return
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))) if self.command == "POST" else None
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
        configure()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            print(json.dumps({"case": real_case or "synthetic", "evidence": case.evidence}), flush=True)
            subprocess.run(["node", str(Path(__file__).with_name("test-source-review.mjs"))], check=True, timeout=420,
                env={**os.environ, "READING_TEST_API": f"http://127.0.0.1:{server.server_port}"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    finally:
        stack.close()
        if case:
            case.doCleanups()
        CleanupApiTests.doClassCleanups()
        assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected
                   for path, expected in backup_hashes.items()), "Immutable backup changed"


if __name__ == "__main__":
    run()
