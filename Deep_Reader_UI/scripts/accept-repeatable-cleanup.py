"""RCR-05: immutable incident backup, real routes and disposable SQL/browser writes.

Requires DEEP_READER_TEST_POSTGRES_DSN and either RCR_BACKUP or RCR_SOURCE_DSN.
Existing backups are hash-verified; source connections are read-only when used.
The target is a unique schema. Never reparse/re-OCR the incident.
RCR_SCENARIO=hidden-navigation selects the cleaned Madame Bovary block case;
the default repeatable scenario retains the original restored-block acceptance.
"""

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
TABLES = ("documents", "raw_source_metadata", "document_profile", "parse_events",
          "chapters", "sections", "task_units", "content_blocks", "artifacts",
          "structured_document_snapshots", "ocr_runs", "ocr_pages")
SCENARIO = os.environ.get("RCR_SCENARIO", "repeatable")
SCENARIOS = {
    "repeatable": (68, "19652:content:2", "7d588760645951da3c241c2a8f2d695e07b47d8d9a5ff432b2a55781e222d9b0", "restored"),
    "hidden-navigation": (69, "20303:content:2", "fece86284fd4af176e974473db1aeead31a498796ef59b5fdc0516bc98207bd2", "cleaned"),
}
DOC_ID, BLOCK_ID, RECORD_ID, INITIAL_STATUS = SCENARIOS[SCENARIO]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def capture(dsn):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    result = {}
    options = conninfo_to_dict(dsn).get("options", "") + " -c default_transaction_read_only=on"
    with psycopg.connect(make_conninfo(dsn, options=options)) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        for table in TABLES:
            predicate = ("id=%s" if table == "documents" else
                         "ocr_run_id IN (SELECT id FROM ocr_runs WHERE document_id=%s)"
                         if table == "ocr_pages" else "document_id=%s")
            query = sql.SQL("SELECT to_jsonb(t) FROM {} t WHERE " + predicate + " ORDER BY id")
            result[table] = [row[0] for row in conn.execute(query.format(sql.Identifier(table)), (DOC_ID,))]
    return result


def run():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from fastapi.testclient import TestClient
    from scripts.private_book_corpus import preserve
    from db.postgres_structured_document_store import PostgresStructuredDocumentStore
    from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
    from document_structure.reading_content import ReadingContentService
    from profile.document_profile import DocumentProfile

    target_dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    input_backup = Path(os.environ["RCR_BACKUP"]).resolve() if os.environ.get("RCR_BACKUP") else None
    if input_backup:
        assert input_backup.is_relative_to((BACKEND / "data/test-books").resolve())
        manifest = json.loads((input_backup / "manifest.json").read_bytes())
        assert manifest["document_id"] == DOC_ID and manifest["record_id"] == RECORD_ID
        input_hashes = {**manifest["sha256"], "manifest.json": digest((input_backup / "manifest.json").read_bytes())}
        assert all(digest((input_backup / key).read_bytes()) == sha for key, sha in input_hashes.items())
        backup = {key: (input_backup / key).read_bytes() for key in ("rows.json", "source.pdf", "profile.json")}
        before = json.loads(backup["rows.json"])
    else:
        source_dsn = os.environ["RCR_SOURCE_DSN"]
        before = capture(source_dsn)
    name = before["documents"][0]["document_name"]
    raw = BACKEND / "data/raw" / (name + ".pdf")
    profile = BACKEND / "data/faiss" / name / "profile.json"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    evidence = BACKEND / "data/test-books/rcr05" / run_id
    if not input_backup:
        backup = {"rows.json": encoded(before), "source.pdf": raw.read_bytes(), "profile.json": profile.read_bytes()}
    hashes = {key: digest(value) for key, value in backup.items()}
    for key, value in backup.items():
        preserve(evidence / key, value)
    preserve(evidence / "manifest.json", encoded({"scenario": SCENARIO, "document_id": DOC_ID, "document_name": name,
        "block_id": BLOCK_ID, "record_id": RECORD_ID, "sha256": hashes}))
    # All target loads come from this verified backup, not subsequent source reads.
    rows = json.loads((evidence / "rows.json").read_bytes())
    profile_data = json.loads((evidence / "profile.json").read_bytes())
    schema = "rcr05_" + uuid4().hex
    scoped = make_conninfo(target_dsn, options="-c search_path=" + schema)
    lock = threading.Lock()
    report = {"status": "failed", "scenario": SCENARIO, "backup": str(evidence), "sha256": hashes, "schema": schema,
              "input_backup": str(input_backup) if input_backup else None}
    created = False
    try:
        with psycopg.connect(target_dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        created = True
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
            assert capture(scoped) == rows

        reset()
        repo = PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(
            scoped, namespace=rows["documents"][0]["namespace"]))
        service = ReadingContentService(repo)
        initial = service.read(name, include_cleanup_audit=True)
        batch = next(b for b in initial["cleanup_audit"]["items"]
                     if any(r.get("record_id") == RECORD_ID for r in b["deletions"]))
        record = next(r for r in batch["deletions"] if r.get("record_id") == RECORD_ID)
        assert record["can_restore"] and record["status"] == INITIAL_STATUS
        target = next(b for b in initial["blocks"] if b["block_id"] == BLOCK_ID)
        assert not target["manually_edited"]
        report["initial"] = {"revision": initial["revision"], "record": record, "block": target}
        report["natural_non_navigable_records"] = sum(
            not any(link["can_navigate"] for link in row.get("links", []))
            for batch in initial["cleanup_audit"]["items"] for row in batch["deletions"])

        def stats():
            current = capture(scoped)
            for table in ("raw_source_metadata", "document_profile", "parse_events", "chapters", "sections", "task_units", "content_blocks", "ocr_runs", "ocr_pages"):
                assert current[table] == rows[table], table + " changed outside reading overlays"
            assert [r for r in current["artifacts"] if r["artifact_type"] == "note"] == [
                r for r in rows["artifacts"] if r["artifact_type"] == "note"]
            state = current["documents"][0]["metadata_payload"]["parse_provenance"]["reading_content"]
            snapshot = max(current["structured_document_snapshots"], key=lambda row: row["source_structure_version"])
            assert snapshot["structured_document_payload"]["parse_provenance"]["reading_content"] == state
            data = service.read(name, include_cleanup_audit=True)
            found = next((r for b in data["cleanup_audit"]["items"] for r in b["deletions"]
                          if r.get("record_id") == RECORD_ID), None)
            block = next(b for b in data["blocks"] if b["block_id"] == BLOCK_ID)
            return {"revision": data["revision"], "record": found, "block": block,
                    "applications": state.get("cleanup_record_applications", {}), "sql_copies_equal": True,
                    "sql_sha256": digest(encoded(current))}

        import main
        coordinator = copy(main.section_task_coordinator)
        coordinator.document_artifact_repository = repo
        # Reuse the exact backed-up profile; reads cannot create or write a profile.
        coordinator._load_existing_document_profile = lambda doc_name: DocumentProfile.from_dict(profile_data) if doc_name == name else None
        coordinator.analysis_interaction_orchestrator = None
        coordinator.quiz_interaction_orchestrator = None
        coordinator.critical_thinking_session_store = None
        def forbidden(*args, **kwargs):
            raise AssertionError("Preparation/model work is forbidden in RCR acceptance")
        coordinator.ensure_document_task_layout = forbidden
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "section_task_coordinator", coordinator))
            stack.enter_context(patch.object(main, "document_artifact_repository", repo))
            stack.enter_context(patch.object(main, "_reading_content_service", return_value=service))
            client = stack.enter_context(TestClient(main.app))
            class Handler(BaseHTTPRequestHandler):
                def handle_request(self):
                    with lock:
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))) if self.command == "POST" else None
                        path = unquote(urlsplit(self.path).path)
                        if path == "/fixture" and self.command == "POST":
                            reset()
                            payload = {"name": name, "block_id": BLOCK_ID, "record_id": RECORD_ID,
                                       "batch_revision": batch["revision"], "record_index": batch["deletions"].index(record),
                                       "non_navigable": [{"record_id": r["record_id"], "index": i}
                                           for i, r in enumerate(batch["deletions"])
                                           if r.get("links") and not any(link["can_navigate"] for link in r["links"])],
                                       "section_id": target["section_id"], "initial": stats()}
                            status, content = 200, encoded(payload)
                        elif path == "/stats" and self.command == "GET":
                            status, content = 200, encoded(stats())
                        elif path in {"/documents/task-layout", f"/documents/{name}/reading-content", f"/documents/{name}/task-units/content"}:
                            if path == "/documents/task-layout" and (not body or body.get("doc_name") != name):
                                self.send_error(400)
                                return
                            response = client.request(self.command, self.path, json=body)
                            status, content = response.status_code, response.content
                        else:
                            self.send_error(400)
                            return
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
            print("RCR evidence: " + str(evidence), flush=True)
            try:
                script = "test-cleanup-navigation-real.mjs" if SCENARIO == "hidden-navigation" else "test-repeatable-cleanup-real.mjs"
                subprocess.run(["node", str(Path(__file__).with_name(script))],
                    check=True, timeout=600, env={**os.environ, "RCR_API": f"http://127.0.0.1:{server.server_port}",
                                                 "RCR_EVIDENCE": str(evidence)})
                report["final"] = stats()
                report["status"] = "passed"
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
    finally:
        if created:
            with psycopg.connect(target_dsn, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                assert conn.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (schema,)).fetchone() is None
        report["schema_removed"] = True
        if input_backup:
            report["source_not_accessed"] = True
            report["input_backup_unchanged"] = all(digest((input_backup / key).read_bytes()) == sha for key, sha in input_hashes.items())
        else:
            report["source_unchanged"] = capture(source_dsn) == before
            report["source_files_unchanged"] = digest(raw.read_bytes()) == hashes["source.pdf"] and digest(profile.read_bytes()) == hashes["profile.json"]
        report["backup_unchanged"] = all(digest((evidence / key).read_bytes()) == value for key, value in hashes.items())
        preserve(evidence / "report.json", encoded(report))
        assert report["backup_unchanged"]
        if input_backup:
            assert report["input_backup_unchanged"]
        else:
            assert report["source_unchanged"] and report["source_files_unchanged"]


if __name__ == "__main__":
    run()
