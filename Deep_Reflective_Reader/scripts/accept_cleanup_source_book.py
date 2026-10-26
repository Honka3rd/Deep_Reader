"""Opt-in live cleanup acceptance on an immutable private book and isolated SQL.

Requires DEEP_READER_TEST_POSTGRES_DSN pointing to a disposable test database.
The caller supplies a manually reviewed noise boundary; raw text stays private.
"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import psycopg

from scripts.private_book_corpus import load_book, preserve


def run(case_id, noise_start, attempts):
    import main
    from app.cleanup_runtime import build_cleanup_service
    from document_structure.ocr_cleanup_overlay import canonical_text, reading_slice
    from document_structure.reading_content import state_for
    from scripts.accept_ocr_cleanup_books import check_targets
    from scripts.test_enhancement_cleanup_api import CleanupApiTests

    root, source, original = load_book(case_id)
    assert 0 < noise_start < source.text_length
    evidence = root / datetime.now(timezone.utc).strftime("cleanup-fix-%Y%m%dT%H%M%S%fZ")
    provider = main.app.state.smart_cleanup_service.provider
    real_resolver = main.section_task_coordinator.task_unit_resolver
    complete = provider.complete_text_with_output_budget
    results = []
    CleanupApiTests.setUpClass()
    try:
        for index in range(attempts):
            case = CleanupApiTests()
            try:
                case.setUp()
                shutil.copyfile(root / "source.pdf", case.f.path)
                # Reproduce the actual missing metadata; do not repair the fixture path.
                case.f.repo.save_reparsed_document(replace(original, source_path=None), "book")
                with psycopg.connect(case.scoped) as connection:
                    row = connection.execute("SELECT id FROM documents WHERE namespace=%s AND document_name='book'",
                                             (case.f.namespace,)).fetchone()
                    connection.execute("DELETE FROM raw_source_metadata WHERE document_id=%s", (row[0],))
                    connection.execute("UPDATE documents SET metadata_payload=metadata_payload - 'source_path' WHERE id=%s",
                                       (row[0],))
                case.coordinator.task_unit_resolver = real_resolver
                case.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
                before = case.f.snapshot()[0]
                assert before.source_path is None
                before_text = canonical_text(before)
                assert before_text == source.canonical_text
                flow = build_cleanup_service(namespace=case.f.namespace,
                    coordinator=case.coordinator, provider=provider)
                base = "/documents/book/smart-cleanup"
                attempt = evidence / str(index + 1)
                calls = []

                def capture(prompt, *args, **kwargs):
                    response = complete(prompt, *args, **kwargs)
                    calls.append(response)
                    preserve(attempt / f"model-{len(calls)}.json", json.dumps({
                        "prompt": prompt, "response": response}, ensure_ascii=False).encode())
                    return response

                with ExitStack() as stack:
                    stack.enter_context(patch.object(main.app.state, "smart_cleanup_service", flow))
                    stack.enter_context(patch.object(provider, "complete_text_with_output_budget", side_effect=capture))
                    stack.enter_context(patch.object(case.pipeline, "build_enhancement_candidate",
                                                   side_effect=AssertionError("No split allowed")))
                    commit = stack.enter_context(patch.object(case.f.repo, "commit_cleanup", wraps=case.f.repo.commit_cleanup))
                    preflight = case.client.post(base + "/preflight", json={})
                    assert preflight.status_code == 200 and preflight.json()["execution_available"], preflight.text
                    consent = {k: preflight.json()[k] for k in ("operation_id", "confirmation_token")}
                    response = case.client.post(base, json=consent)
                    result = response.json()
                    preserve(attempt / "result.json", json.dumps(result, ensure_ascii=False).encode())
                    assert response.status_code == 200, result
                    assert result["commit"] == "committed" and result["reading_ready"], result
                    assert result["outcome"]["reading_content_changed"], result
                    assert result["outcome"]["applied_count"] > 0, result
                    after = case.f.snapshot()[0]
                    state = state_for(after)
                    spans = state["cleanup_spans"]
                    preserve(attempt / "spans.json", json.dumps(spans, ensure_ascii=False).encode())
                    assert all(span["char_start"] >= noise_start for span in spans), "Readable prefix deleted"
                    effective = reading_slice(before_text, 0, len(before_text), state)
                    assert effective[:noise_start] == before_text[:noise_start]
                    assert len(effective) < len(before_text)
                    assert after.chapters == before.chapters, "Hierarchy/IDs mutated"
                    assert canonical_text(after) == before_text
                    for key in ("extraction", "source_page_spans", "actual_ocr_evidence"):
                        assert after.parse_provenance.get(key) == before.parse_provenance.get(key)
                    target_count = check_targets(after)
                    assert case.client.post(base, json=consent).json() == result
                    assert case.client.get(base + "/" + consent["operation_id"]).json() == result
                    assert len(calls) == 1 and commit.call_count == 1
                    case.f.change("undo")
                    assert reading_slice(before_text, 0, len(before_text), state_for(case.f.snapshot()[0])) == before_text
                    assert load_book(case_id)[1] == source
                    report = dict(attempt=index + 1, outcome=result["outcome"], targets=target_count,
                        deleted_characters=len(before_text) - len(effective), replay_without_generation=True,
                        undo_restored=True, source_sha256=hashlib.sha256((root / "source.pdf").read_bytes()).hexdigest())
                    results.append(report)
                    preserve(attempt / "report.json", json.dumps(report).encode())
                    print(json.dumps(report), flush=True)
            finally:
                case.doCleanups()
    finally:
        CleanupApiTests.doClassCleanups()
        print("private_evidence=" + str(evidence), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--noise-start", required=True, type=int)
    parser.add_argument("--attempts", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    run(args.case, args.noise_start, args.attempts)
