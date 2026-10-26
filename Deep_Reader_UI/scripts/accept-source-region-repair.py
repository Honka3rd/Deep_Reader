"""SRREPAIR backup-only actual API/SQL/browser acceptance. No production reads/writes."""

from contextlib import ExitStack
from copy import copy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch
from urllib.parse import unquote, urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Deep_Reflective_Reader"
sys.path[:0] = [str(BACKEND), str(BACKEND / "scripts")]
TABLES = ("documents", "raw_source_metadata", "document_profile", "parse_events", "chapters", "sections",
          "task_units", "content_blocks", "artifacts", "structured_document_snapshots", "ocr_runs", "ocr_pages")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def run():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from fastapi.testclient import TestClient
    from db.postgres_structured_document_store import PostgresStructuredDocumentStore
    from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
    from document_structure.reading_content import ReadingContentService, state_for
    from document_structure.ocr_cleanup_overlay import canonical_text
    from profile.document_profile import DocumentProfile
    from scripts.private_book_corpus import preserve

    backup = Path(os.environ["SRREPAIR_BACKUP"]).resolve()
    assert backup.is_relative_to((BACKEND / "data/test-books").resolve())
    manifest = json.loads((backup / "manifest.json").read_bytes())
    hashes = {**manifest["sha256"], "manifest.json": digest((backup / "manifest.json").read_bytes())}
    assert all(digest((backup / key).read_bytes()) == sha for key, sha in hashes.items())
    rows = json.loads((backup / "rows.json").read_bytes())
    profile_data = json.loads((backup / "profile.json").read_bytes())
    name = rows["documents"][0]["document_name"]
    dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    schema = "srrepair_" + uuid4().hex
    scoped = make_conninfo(dsn, options="-c search_path=" + schema)
    evidence = BACKEND / "data/test-books/srrepair07" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    evidence.mkdir(parents=True)
    report = {"status": "failed", "backup": str(backup), "document": name, "sha256": hashes, "schema": schema}
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        with psycopg.connect(scoped) as conn:
            for migration in sorted((BACKEND / "db/migrations").glob("*.sql")):
                conn.execute(migration.read_text())

        def reset():
            with psycopg.connect(scoped) as conn:
                conn.execute("TRUNCATE documents CASCADE")
                for table in TABLES:
                    conn.execute(sql.SQL("INSERT INTO {} SELECT * FROM jsonb_populate_recordset(NULL::{}, %s::jsonb)")
                                 .format(sql.Identifier(table), sql.Identifier(table)), (json.dumps(rows[table]),))
                    conn.execute(sql.SQL("SELECT setval(pg_get_serial_sequence(%s, 'id'), COALESCE(MAX(id),1), COUNT(*)>0) FROM {}")
                                 .format(sql.Identifier(table)), (table,))

        reset()
        repo = PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(
            scoped, namespace=rows["documents"][0]["namespace"]))
        service = ReadingContentService(repo)
        original_text = canonical_text(repo.load_document(name))
        initial = service.read(name)
        eligible = [(index, row) for index, row in enumerate(initial["regions"])
                    if row["page_index"] == 2 and len(row["block_ids"]) >= 20]
        assert eligible, "No natural Page 3 multi-block region in this backup"
        index, target = eligible[0]
        region_id = target["id"]
        block_id = next(b["block_id"] for b in initial["blocks"] if b["block_id"] in target["block_ids"]
                        and b["review_repair"]["can_edit"])
        report["natural_regions"] = len(initial["regions"])
        report["eligible_multi_block_regions"] = len(eligible)
        report["target"] = {"region_id": region_id, "block_id": block_id, "index": index}

        def stats():
            document = repo.load_document(name)
            assert canonical_text(document) == original_text
            state = state_for(document)
            with psycopg.connect(scoped) as conn:
                pair = conn.execute("SELECT d.metadata_payload->'parse_provenance'->'reading_content', "
                    "s.structured_document_payload->'parse_provenance'->'reading_content' FROM documents d "
                    "JOIN structured_document_snapshots s ON s.document_id=d.id "
                    "ORDER BY s.source_structure_version DESC LIMIT 1").fetchone()
                assert pair[0] == pair[1]
                for table in ("raw_source_metadata", "document_profile", "chapters", "sections", "task_units", "content_blocks", "ocr_runs", "ocr_pages"):
                    actual = [r[0] for r in conn.execute(sql.SQL("SELECT to_jsonb(t) FROM {} t ORDER BY id").format(sql.Identifier(table)))]
                    assert actual == rows[table], table + " changed outside overlays"
            snapshot = service.read(name)
            region = next((r for r in snapshot["regions"] if r["id"] == region_id), None)
            return {"revision": snapshot["revision"], "region": region,
                    "blocks": [b for b in snapshot["blocks"] if b["block_id"] in target["block_ids"]],
                    "kept": state.get("review_kept_blocks", {}),
                    "applications": state.get("region_applications", {}), "sql_copies_equal": True}

        import main
        coordinator = copy(main.section_task_coordinator)
        coordinator.document_artifact_repository = repo
        coordinator._load_existing_document_profile = lambda doc_name: DocumentProfile.from_dict(profile_data) if doc_name == name else None
        coordinator.analysis_interaction_orchestrator = None
        coordinator.quiz_interaction_orchestrator = None
        coordinator.critical_thinking_session_store = None
        def forbidden(*args, **kwargs):
            raise AssertionError("Preparation/model work forbidden in region acceptance")
        coordinator.ensure_document_task_layout = forbidden
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "section_task_coordinator", coordinator))
            stack.enter_context(patch.object(main, "document_artifact_repository", repo))
            stack.enter_context(patch.object(main, "_reading_content_service", return_value=service))
            client = stack.enter_context(TestClient(main.app))
            lock = threading.Lock()
            class Handler(BaseHTTPRequestHandler):
                def handle_request(self):
                    with lock:
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))) if self.command == "POST" else None
                        path = unquote(urlsplit(self.path).path)
                        if path == "/fixture" and self.command == "POST":
                            reset()
                            status, content = 200, encoded({"name": name, "region_id": region_id, "block_id": block_id,
                                                           "region_index": index, "initial": stats()})
                        elif path == "/stats":
                            status, content = 200, encoded(stats())
                        elif path in {"/documents/task-layout", f"/documents/{name}/reading-content", f"/documents/{name}/task-units/content"}:
                            if path == "/documents/task-layout" and body.get("doc_name") != name:
                                self.send_error(400); return
                            response = client.request(self.command, self.path, json=body)
                            status, content = response.status_code, response.content
                        else:
                            self.send_error(400); return
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(content)
                do_GET = do_POST = handle_request
                def log_message(self, *args):
                    pass
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print("SRREPAIR evidence: " + str(evidence), flush=True)
            try:
                subprocess.run(["node", str(Path(__file__).with_name("test-source-review-blocks-real.mjs"))], check=True, timeout=600,
                    env={**os.environ, "SRREPAIR_API": f"http://127.0.0.1:{server.server_port}", "SRREPAIR_EVIDENCE": str(evidence)})
                report["final"] = stats()
                report["status"] = "passed"
            finally:
                server.shutdown(); server.server_close(); thread.join()
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            assert conn.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (schema,)).fetchone() is None
        report["schema_removed"] = True
        report["backup_unchanged"] = all(digest((backup / key).read_bytes()) == sha for key, sha in hashes.items())
        preserve(evidence / "report.json", encoded(report))
        assert report["backup_unchanged"]


if __name__ == "__main__":
    run()
