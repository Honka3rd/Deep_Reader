"""Explicit destructive acceptance for the two user-authorized, backed-up books."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import psycopg
from psycopg.rows import dict_row
from scripts.private_book_corpus import preserve

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/test-books/summary-note-20261021"
BOOKS = ("Madame Bovary.txt", "红色赌盘_261014_145614.pdf")
TABLES = ("raw_source_metadata", "document_profile", "parse_events", "chapters", "sections",
          "task_units", "content_blocks", "artifacts", "structured_document_snapshots", "ocr_runs")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2).encode()


def backup(dsn):
    manifest_path = CORPUS / "manifest.json"
    if manifest_path.exists():
        return verify_backup()
    files = {}
    for filename in BOOKS:
        source = ROOT / "data/raw" / filename
        data = source.read_bytes()
        preserve(CORPUS / "raw" / filename, data)
        name = Path(filename).stem
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            rows = conn.execute("SELECT * FROM documents WHERE document_name=%s", (name,)).fetchall()
            snapshot = {"documents": rows}
            for table in TABLES:
                if conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()["name"]:
                    snapshot[table] = conn.execute(
                        f"SELECT * FROM {table} WHERE document_id=ANY(%s)",
                        ([row["id"] for row in rows],)).fetchall()
        preserve(CORPUS / "database" / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".json"), encoded(snapshot))
        derived = ROOT / "data/faiss" / name
        if derived.exists():
            for path in derived.rglob("*"):
                if path.is_file():
                    preserve(CORPUS / "derived" / name / path.relative_to(derived), path.read_bytes())
    for path in CORPUS.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
            files[str(path.relative_to(CORPUS))] = hashlib.sha256(path.read_bytes()).hexdigest()
    preserve(manifest_path, encoded({"files": files, "books": list(BOOKS)}))
    return verify_backup()


def verify_backup():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if hashlib.sha256((CORPUS / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("backup_hash_mismatch:" + relative)
    return manifest


def post(client, path, payload):
    response = client.post(path, json=payload)
    if response.status_code >= 400:
        raise AssertionError(f"{path}: HTTP {response.status_code}: {response.text[:800]}")
    return response.json()


def rebuild(client, dsn):
    verify_backup()
    results = []
    for filename in BOOKS:
        name = Path(filename).stem
        result = post(client, "/documents/lifecycle/delete", {"doc_name": name, "action": "delete", "confirm": True})
        assert result["deleted"]
        assert not (ROOT / "data/raw" / filename).exists()
        assert not (ROOT / "data/faiss" / name / "profile.json").exists()
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT count(*) FROM documents WHERE document_name=%s", (name,)).fetchone()[0] == 0
        print(json.dumps({"book": name, "deleted_verified": True}, ensure_ascii=False), flush=True)
        with (CORPUS / "raw" / filename).open("rb") as source:
            upload = client.post("/documents/upload", files={"file": (filename, source)})
        assert upload.status_code in {200, 201}, upload.text
        prepared = post(client, "/documents/prepare-task-layout", {
            "doc_name": name, "structured_parser_mode": "common", "pdf_extraction_mode": "fast",
            "task_unit_split_mode": "progressive"})
        assert prepared.get("chapters"), prepared
        units = [unit for chapter in prepared["chapters"] for section in chapter["sections"] for unit in section["task_units"]]
        assert units
        results.append({"book": name, "deleted_verified": True, "chapters": len(prepared["chapters"]), "units": len(units)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return results


def accept(client):
    verify_backup()
    results = []
    for filename in BOOKS:
        name = Path(filename).stem
        post(client, "/documents/prepare-task-layout", {"doc_name": name,
             "layout_only": True, "task_unit_split_mode": "progressive"})
        layout = post(client, "/documents/task-layout", {"doc_name": name, "task_unit_split_mode": "progressive"})
        candidates = [(c, s, u) for c in layout["chapters"] for s in c["sections"] for u in s["task_units"]]
        chosen = None
        for chapter, section, unit in candidates:
            unit_id = str(unit.get("task_unit_id") or unit.get("unit_id"))
            response = client.get(f"/documents/{quote(name, safe='')}/task-units/{unit_id}/content", params={"segmented": "true"})
            response.raise_for_status()
            blocks = response.json()["content_blocks"]
            block = next((b for b in blocks if len(b.get("content", "")) >= 250), None)
            if block:
                chosen = chapter, section, unit_id, block
                break
        assert chosen, "no readable target"
        chapter, section, unit_id, block = chosen
        targets = []
        for level in ("document", "chapter", "section", "task_unit", "content_block"):
            target = {"doc_name": name, "target_type": level}
            if level != "document": target["chapter_id"] = str(chapter["chapter_id"])
            if level in {"section", "task_unit", "content_block"}: target["section_id"] = str(section["section_id"])
            if level in {"task_unit", "content_block"}: target["task_unit_id"] = unit_id
            if level == "content_block": target["content_block_id"] = block["block_id"]
            targets.append(target)
            prefix = "/documents/reading-interactions/"
            read = post(client, prefix + "note/read", {"target": target})
            fields = {"target": target, "text": "Acceptance note: source-grounded observation.",
                      "expected_revision": read["note"]["revision"] if read["note"] else 0,
                      "mutation_id": "accept_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
                      "target_token": read["target_token"]}
            saved = post(client, prefix + "note/save", fields)
            assert post(client, prefix + "note/save", fields) == saved
            assert post(client, prefix + "note/read", {"target": target}) == saved
            summary = post(client, prefix + "summary/generate", {"target": target})
            assert summary["envelope"]["status"] == "completed", summary
            assert summary["payload"]["body"]
            assert post(client, prefix + "summary/read", {"target": target}) == summary
            assert post(client, prefix + "summary/generate", {"target": target}) == summary
            result = {"book": name, "level": level, "summary": summary, "note_saved": True}
            results.append(result)
            print(json.dumps({"book": name, "level": level, "passed": True}, ensure_ascii=False), flush=True)
        preserve(CORPUS / "targets" / (hashlib.sha256(name.encode()).hexdigest()[:16] + ".json"), encoded(targets))
    return results


def cleanup_notes(client):
    """Remove only the exact fixture notes created by this acceptance run."""
    verify_backup()
    removed = 0
    for filename in BOOKS:
        case = hashlib.sha256(Path(filename).stem.encode()).hexdigest()[:16]
        targets = json.loads((CORPUS / "targets" / (case + ".json")).read_text())
        for target in targets:
            prefix = "/documents/reading-interactions/note/"
            current = post(client, prefix + "read", {"target": target})
            if current["note"] is None:
                continue
            if current["note"]["text"] != "Acceptance note: source-grounded observation.":
                raise ValueError("test_note_changed: preserving non-fixture content")
            post(client, prefix + "delete", {"target": target,
                 "expected_revision": current["note"]["revision"], "target_token": current["target_token"]})
            assert post(client, prefix + "read", {"target": target})["note"] is None
            removed += 1
    return {"fixture_notes_removed": removed, "real_summaries_retained": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("backup", "rebuild", "accept", "cleanup-notes"), required=True)
    parser.add_argument("--allow-destructive-rebuild", action="store_true")
    parser.add_argument("--allow-live-model", action="store_true")
    args = parser.parse_args()
    dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    if args.operation == "backup":
        result = backup(dsn)
        print(json.dumps({"backup_verified": True, "files": len(result["files"])}))
        return
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=1800) as client:
        if args.operation == "rebuild":
            assert args.allow_destructive_rebuild, "explicit rebuild flag required"
            result = rebuild(client, dsn)
        elif args.operation == "cleanup-notes":
            result = cleanup_notes(client)
        else:
            assert args.allow_live_model, "explicit provider flag required"
            result = accept(client)
    report = CORPUS / (args.operation + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    preserve(report, encoded(result))
    verify_backup()
    print(f"PASS report={report}")


if __name__ == "__main__":
    main()
