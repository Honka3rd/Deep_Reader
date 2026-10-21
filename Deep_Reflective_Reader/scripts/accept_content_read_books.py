"""Opt-in real-book reset and full-body latency/profile immutability acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from scripts.accept_summary_note_books import BOOKS, CORPUS, ROOT, post, rebuild, verify_backup, encoded
from scripts.private_book_corpus import preserve


def profile_hash(name):
    path = ROOT / "data/faiss" / name / "profile.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accept(client, *, read_only=False):
    results = []
    for filename in BOOKS:
        name = Path(filename).stem
        before = profile_hash(name)
        initial_layout = post(client, "/documents/task-layout", {"doc_name": name,
             "task_unit_split_mode": "progressive"})
        # Idempotent preparation is reuse, not authority to refresh a profile.
        if not read_only:
            post(client, "/documents/prepare", {"doc_name": name, "mode": "base",
                 "structured_parser_mode": "common", "pdf_extraction_mode": "fast"})
        layout = post(client, "/documents/task-layout", {"doc_name": name,
             "task_unit_split_mode": "progressive"})
        assert layout == initial_layout, "repeated preparation changed accepted layout"
        chapter, section = max(((c, s) for c in layout["chapters"] for s in c["sections"]),
                              key=lambda pair: len(pair[1]["task_units"]))
        ids = [str(u["unit_id"]) for u in section["task_units"]]
        path = f"/documents/{quote(name, safe='')}/task-units/content"
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            response = client.post(path, json={"task_unit_ids": ids, "segmented": True})
            response.raise_for_status()
            contents = response.json()["contents"]
            seconds = time.perf_counter() - start
            assert seconds < 10, f"full-body too slow: {name}: {seconds}"
            assert [c["task_unit_id"] for c in contents] == ids
            timings.append(round(seconds, 3))
        target = {"doc_name": name, "target_type": "task_unit", "chapter_id": chapter["chapter_id"],
                  "section_id": section["section_id"], "task_unit_id": ids[0]}
        for kind in ("insight", "quiz", "critical-thinking", "summary", "note"):
            post(client, f"/documents/reading-interactions/{kind}/read", {"target": target})
        assert profile_hash(name) == before, "profile changed during reads/reuse"
        results.append({"book": name, "section_id": section["section_id"], "units": len(ids),
            "blocks": sum(len(c["content_blocks"]) for c in contents),
            "full_body_http_seconds": timings, "profile_sha256": before,
            "profile_unchanged": True, "interaction_reads_passed": True})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-destructive-rebuild", action="store_true")
    parser.add_argument("--read-only", action="store_true",
        help="Never prepare/rebuild: preserve the Library's current parser settings and hierarchy.")
    args = parser.parse_args()
    if args.read_only and args.allow_destructive_rebuild:
        parser.error("read-only cannot be combined with destructive rebuild")
    verify_backup()
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=1800) as client:
        reset = rebuild(client, os.environ["DEEP_READER_TEST_POSTGRES_DSN"]) if args.allow_destructive_rebuild else None
        if reset:
            preserve(CORPUS / ("content-reset-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"), encoded(reset))
        report = {"reset": reset, "read_only": args.read_only, "acceptance": accept(client, read_only=args.read_only)}
    output = CORPUS / ("content-recovery-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    preserve(output, encoded(report))
    verify_backup()
    print(f"PASS report={output}")


if __name__ == "__main__":
    main()
