"""CLEAN/SCLEAN final gates: backup-only extraction and opt-in live API/SQL acceptance.

Private source, model responses and deletion evidence stay below data/test-books.
Extraction runs separately (e.g. in the pinned OCR image); enhancement reuses it.
"""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

from scripts.private_book_corpus import CORPUS_ROOT, preserve
from scripts.accept_enhancement_real_book import private_json


ROOT = CORPUS_ROOT / "cleanup-20261023"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_targets(document):
    from document_structure.ocr_cleanup_overlay import canonical_text, reading_slice, unit_ranges
    from document_structure.reading_content import state_for, project_document
    from section_tasks.reading_target_resolver import ReadingTargetResolver
    from document_structure.text_normalization import normalize_ocr_text

    state, resolver = state_for(document), ReadingTargetResolver()
    compact = lambda value: "".join(normalize_ocr_text(value).split())
    count = 0
    def check(scope, expected):
        nonlocal count
        actual = resolver.resolve(document=document, **scope).content
        assert compact(actual) == compact(expected), (scope, compact(actual), compact(expected))
        count += 1
    text = canonical_text(document)
    check(dict(target_level="document"), reading_slice(text, 0, len(text), state))
    for chapter in document.chapters:
        check(dict(target_level="chapter", chapter_id=chapter.chapter_id), "".join(
            reading_slice(text, s.char_start, s.char_end, state) for s in chapter.sections))
        for section in chapter.sections:
            check(dict(target_level="section", section_id=section.section_id),
                  reading_slice(text, section.char_start, section.char_end, state))
            spans = unit_ranges(section)
            for unit in section.task_units:
                span = spans.get(unit.unit_id)
                if span:
                    check(dict(target_level="task_unit", task_unit_id=unit.unit_id),
                          reading_slice(text, span.start, span.end, state))
    projected = project_document(document)
    for c in projected.chapters:
        for s in c.sections:
            for u in s.task_units:
                for block in resolver.content_blocks_for_target_validation(u, s, c):
                    check(dict(target_level="content_block", task_unit_id=u.unit_id,
                               content_block_id=block.block_id), block.content)
    return count


def extract():
    from doc_loaders.pdf_document_loader import PdfDocumentLoader
    from document_preparation.cleanup_handoff import extraction_cleanup_metadata
    from document_structure.structured_document_builder import StructuredDocumentBuilder
    from language.language_code import LanguageCode

    sources = []
    for name, folder in (("sample", "ocr-direction-20261021"), ("sample2", "ocr-review-20261021")):
        parent = CORPUS_ROOT / folder
        manifest = json.loads((parent / "manifest.json").read_text())
        source = parent / manifest["source"]
        assert sha(source) == manifest["sha256"], "Backup hash mismatch"
        sources.append((name, source, manifest["sha256"]))
    noise = ROOT / "noise" / "manifest.json"
    if noise.exists():
        manifest = json.loads(noise.read_text())
        sources.append(("noise", noise.parent / "book.pdf", manifest["source_sha256"]))
    for name, source, expected in sources:
        directory = ROOT / name
        preserve(directory / "book.pdf", source.read_bytes())
        assert sha(directory / "book.pdf") == expected
        artifact = PdfDocumentLoader(directory, ocr_enabled=True).open_source("book", mode="ocr").extract()
        result = artifact.result
        document = StructuredDocumentBuilder().build(document_id="book", title=name,
            raw_text=result.canonical_text, language=LanguageCode.ZH,
            source_path=str(directory / "book.pdf"), parser_mode="common")
        assert not document.structure_error_code, document.structure_error_code
        assert "".join(s.content for c in document.chapters for s in c.sections) == result.canonical_text
        metadata = dict(artifact.storage_metadata)
        metadata.pop("pdf_extraction", None)
        metadata["source_page_spans"] = metadata.pop("pdf_page_spans", [])
        document = replace(document, parse_provenance={**document.parse_provenance, **metadata,
            "extraction": result.identity.record(result.canonical_text), **extraction_cleanup_metadata(artifact)})
        preserve(directory / "document.json", json.dumps(document.to_dict(), ensure_ascii=False, sort_keys=True).encode())
        preserve(directory / "manifest.json", json.dumps({"source_sha256": expected}, sort_keys=True).encode())
        private_json(directory / "extraction.json", dict(document_sha256=sha(directory / "document.json"),
            source_sha256=expected, chars=result.text_length, ocr_run=artifact.ocr_run))
        assert sha(source) == expected
        print(json.dumps(dict(case=name, chars=result.text_length, pages=len(artifact.page_boundaries))), flush=True)


def browser(case, attempt, recovery=None):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import subprocess
    import threading
    mutations = []

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            if not (self.path.startswith("/documents/book/") or self.path == "/documents/task-layout"):
                self.send_error(400)
                return
            body = None
            if self.command == "POST":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/documents/task-layout" and body.get("doc_name") != "book":
                    self.send_error(400)
                    return
            response = case.client.request(self.command, self.path, json=body)
            if self.command == "POST" and self.path == "/documents/book/smart-cleanup":
                from document_structure.reading_content import state_for
                mutations.append(response.json())
                private_json(attempt / "browser-cleanup-results.json", mutations)
                private_json(attempt / "browser-deletions.json",
                    state_for(case.f.snapshot()[0]).get("cleanup_spans", []))
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers.get("content-type", "application/json"))
            self.end_headers()
            self.wfile.write(response.content)
        do_GET = do_POST = handle_request
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        subprocess.run(["node", str(CORPUS_ROOT.parents[2] / "Deep_Reader_UI/scripts/test-cleanup-books.mjs")],
            check=True, timeout=180, env={**os.environ,
                "CLEANUP_TEST_API": f"http://127.0.0.1:{server.server_port}",
                "CLEANUP_EVIDENCE_DIR": str(attempt),
                "CLEANUP_RECOVERY": json.dumps(recovery) if recovery else ""})
        return mutations
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def controls():
    from scripts.private_book_corpus import load_book
    from scripts.test_ocr_cleanup_contract import binding
    from doc_loaders.extraction_identity import ExtractionResult
    from document_structure.ocr_cleanup_planner import plan_ocr_cleanup, CLEANUP_PROMPT_VERSION

    results = []
    for case_id in ("fa794acef055f235", "085747b65094d52f"):
        _, snapshot, _ = load_book(case_id)
        source = ExtractionResult(snapshot.extraction_identity, snapshot.canonical_text)
        def forbidden(*args, **kwargs):
            raise AssertionError("Non-OCR source reached cleanup provider")
        provider = SimpleNamespace(get_model_capabilities=forbidden, complete_text=forbidden)
        outcome = plan_ocr_cleanup(source=source, binding=binding(source,
            source_representation="canonical_text_v1", prompt_version=CLEANUP_PROMPT_VERSION),
            provider=provider, structure_ready=True).outcome
        assert outcome.status == "not_applicable", outcome
        assert load_book(case_id)[1] == snapshot
        results.append(dict(case=case_id, chars=snapshot.text_length,
            format=source.identity.source_format, mode=source.identity.mode,
            status=outcome.status, reason=outcome.reason, provider_calls=0))
    private_json(ROOT / "controls.json", results)
    print(json.dumps(results), flush=True)


def standalone(case, provider, attempt, *, with_browser=False):
    from app.cleanup_runtime import build_cleanup_service
    from document_structure.reading_content import state_for
    from document_structure.ocr_cleanup_overlay import canonical_text
    import main

    flow = build_cleanup_service(namespace=case.f.namespace, coordinator=case.coordinator, provider=provider)
    base = "/documents/book/smart-cleanup"
    case.coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
    before = case.f.snapshot()[0]
    original_spans = state_for(before).get("cleanup_spans", [])
    undo_spans = original_spans
    results = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(main.app.state, "smart_cleanup_service", flow))
        stack.enter_context(patch.object(case.coordinator.document_preparation_pipeline,
            "build_enhancement_candidate", side_effect=AssertionError("Standalone must not split")))
        commit = stack.enter_context(patch.object(case.f.repo, "commit_cleanup", wraps=case.f.repo.commit_cleanup))
        for index in range(2):
            snapshot = case.coordinator.read_cleanup_input("book")
            for part in snapshot.parts:
                assert all(part.char_end <= gap.char_start or part.char_start >= gap.char_end
                    for gap in snapshot.prior_deletions), "Deleted text returned to cleanup input"
            preflight = case.client.post(base + "/preflight", json={})
            assert preflight.status_code == 200 and preflight.json()["execution_available"], preflight.text
            consent = {key: preflight.json()[key] for key in ("operation_id", "confirmation_token")}
            started = time.monotonic()
            response = case.client.post(base, json=consent)
            result = response.json()
            private_json(attempt / f"standalone-{index + 1}.json", result)
            assert response.status_code == 200 and result["phase"] == "finished", result
            assert result["outcome"]["status"] in {"completed", "partial", "no_changes", "skipped"}, result
            current = case.f.snapshot()[0]
            if result["outcome"]["reading_content_changed"]:
                undo_spans = state_for(snapshot.document()).get("cleanup_spans", [])
            assert current.chapters == before.chapters, "Standalone changed hierarchy/IDs"
            assert canonical_text(current) == canonical_text(before)
            for key in ("source_page_spans", "source_review", "extraction"):
                assert current.parse_provenance.get(key) == before.parse_provenance.get(key), key
            targets = check_targets(current)
            count = commit.call_count
            assert count == index + 1, "Expected exactly one receipt commit per new operation"
            assert case.client.post(base, json=consent).json() == result
            assert commit.call_count == count, "Replay committed twice"
            restarted = build_cleanup_service(namespace=case.f.namespace,
                coordinator=case.coordinator, provider=provider)
            main.app.state.smart_cleanup_service = restarted
            assert case.client.get(base + "/" + consent["operation_id"]).json() == result
            results.append(dict(result=result, seconds=round(time.monotonic() - started, 3),
                effective_targets_checked=targets, commit_calls=count))
        private_json(attempt / "after.json", current.to_dict())
        spans = state_for(current).get("cleanup_spans", [])
        private_json(attempt / "deletions.json", spans)
        assert len({(item["char_start"], item["char_end"]) for item in spans}) == len(spans)
        browser_results = []
        if with_browser:
            browser_results = browser(case, attempt, recovery=consent)
            assert len(browser_results) == 1 and browser_results[0]["phase"] == "finished"
            assert browser_results[0]["outcome"]["status"] in {"completed", "partial", "no_changes", "skipped"}
            assert commit.call_count == 3, "Browser duplicated cleanup commit"
            if browser_results[0]["outcome"]["reading_content_changed"]:
                undo_spans = spans
        elif spans != original_spans:
            case.f.change("undo")
        if spans != original_spans:
            undone = case.f.snapshot()[0]
            check_targets(undone)
            assert undone.chapters == current.chapters
            assert canonical_text(undone) == canonical_text(before)
            assert state_for(undone).get("cleanup_spans", []) == undo_spans, "Undo did not restore prior deletions"
        return dict(standalone=results, deletion_count=len(spans), hierarchy_ids_unchanged=True,
            source_unchanged=True, page_maps_unchanged=True, standalone_commits=commit.call_count,
            browser_results=browser_results, prior_deletions_excluded=True,
            browser_viewports=[1280, 390] if with_browser else [], undo_checked=spans != original_spans)


def live(name, *, with_browser=False, entry="combined"):
    import main
    from scripts.test_enhancement_cleanup_api import CleanupApiTests
    from scripts.test_structure_enhancement_candidate import candidate_pipeline
    from document_structure.structured_document import StructuredDocument
    from document_structure.reading_content import state_for, project_document
    from document_structure.ocr_cleanup_overlay import canonical_text
    from document_structure.structure_plan_contract import PROMPT_VERSION
    from document_structure.ocr_cleanup_planner import REMAINING_CLEANUP_PROMPT_VERSION, REMAINING_CLEANUP_POLICY_VERSION

    directory = ROOT / name
    manifest = json.loads((directory / "extraction.json").read_text())
    assert sha(directory / "book.pdf") == manifest["source_sha256"]
    assert sha(directory / "document.json") == manifest["document_sha256"]
    document = StructuredDocument.from_dict(json.loads((directory / "document.json").read_text()))
    attempt = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + entry)
    attempt.mkdir(mode=0o700)
    original = main.section_task_coordinator
    provider = original.document_preparation_pipeline.structured_document_builder.section_splitter_selector.llm_splitter.llm_provider
    resolver = original.task_unit_resolver
    report = dict(case=name, accepted=False, model=provider.get_model_capabilities().model_name,
        structure_prompt=PROMPT_VERSION, cleanup_prompt=REMAINING_CLEANUP_PROMPT_VERSION,
        cleanup_policy=REMAINING_CLEANUP_POLICY_VERSION,
        entry=entry, source_sha256=manifest["source_sha256"], stages=[],
        extraction_calls=0, profile_build_calls=0)
    backend = Path(__file__).resolve().parents[1]
    report["code_hashes"] = {str(p.relative_to(backend)): sha(p)
        for folder in ("app", "document_preparation", "document_structure", "db", "section_tasks")
        for p in sorted((backend / folder).glob("*.py"))}
    CleanupApiTests.setUpClass()
    case = CleanupApiTests()
    try:
        case.setUp()
        shutil.copyfile(directory / "book.pdf", case.f.path)
        document = replace(document, source_path=str(case.f.path))
        case.f.repo.save_reparsed_document(document, "book")
        pipeline = candidate_pipeline(provider)
        pipeline.loader_factory = SimpleNamespace(base_dir=case.f.temp.name)
        case.coordinator.document_preparation_pipeline = pipeline
        case.coordinator.task_unit_resolver = resolver
        method = "complete_text_with_output_budget"
        complete = getattr(provider, method)
        calls = []
        def capture(prompt, *args, **kwargs):
            calls.append("cleanup" if "ocr_cleanup_plan_v1" in prompt else "structure")
            index = len(calls)
            private_json(attempt / f"request-{index}.json", dict(prompt=prompt, schema=kwargs.get("response_schema")))
            started = time.monotonic()
            if entry == "retry" and calls[-1] == "cleanup" and "structure" in calls and calls.count("cleanup") == 1:
                private_json(attempt / f"response-{index}.json", {"injected_error": "automatic_cleanup_failed"})
                report["injected_auto_failure"] = True
                raise ValueError("Acceptance injected automatic cleanup failure")
            response = complete(prompt, *args, **kwargs)
            report["stages"].append(dict(kind=calls[-1], seconds=round(time.monotonic() - started, 3)))
            private_json(attempt / f"response-{index}.json", dict(response=response))
            return response
        before = case.f.snapshot()[0]
        with ExitStack() as stack:
            stack.enter_context(patch.object(provider, method, side_effect=capture))
            stack.enter_context(patch.object(pipeline, "prepare_and_load", side_effect=AssertionError("No extraction/profile")))
            stack.enter_context(patch.object(pipeline, "_prepare_profile", side_effect=AssertionError("No profile build")))
            stack.enter_context(patch.object(main, "document_artifact_repository", case.f.repo))
            if entry == "standalone":
                report.update(standalone(case, provider, attempt, with_browser=with_browser))
                report.update(accepted=True, provider_calls=calls)
                return
            commit = stack.enter_context(patch.object(case.f.repo, "commit_enhancement", wraps=case.f.repo.commit_enhancement))
            request = case.request()
            response = case.execute(request)
            private_json(attempt / "result.json", response.json())
            result = case.assert_completed(response)
            current = case.f.snapshot()[0]
            private_json(attempt / "after.json", current.to_dict())
            assert result["cleanup"]["status"] in ({"failed"} if entry == "retry" else
                {"completed", "partial", "no_changes", "skipped"}), result
            assert current.raw_text == before.raw_text
            reconstructed = "".join(s.content for c in current.chapters for s in c.sections)
            assert reconstructed == canonical_text(before)
            for key in ("source_page_spans", "source_review", "extraction"):
                assert current.parse_provenance.get(key) == before.parse_provenance.get(key), key
            state = state_for(current)
            deletions = state.get("cleanup_spans", [])
            private_json(attempt / "deletions.json", deletions)
            effective = project_document(current)
            assert effective.raw_text == before.raw_text
            report["effective_targets_checked"] = check_targets(current)
            replay = case.execute(request)
            assert replay.json() == result
            assert commit.call_count == 1
            assert case.lookup(request).json()["durability"] == "persisted_commit"
            report.update(result=result, provider_calls=calls, deletion_count=len(deletions),
                source_unchanged=True, page_maps_unchanged=True, single_commit=True)
            if entry == "retry":
                split_count = calls.count("structure")
                report.update(standalone(case, provider, attempt, with_browser=with_browser))
                assert calls.count("structure") == split_count
                report.update(accepted=True, provider_calls=calls)
                return
            if with_browser:
                browser(case, attempt)
                report["browser_viewports"] = [1280, 390]
            if deletions:
                if not with_browser:
                    case.f.change("undo")
                undone = case.f.snapshot()[0]
                assert canonical_text(undone) == canonical_text(before)
                assert not state_for(undone).get("cleanup_spans")
                check_targets(undone)
                assert undone.chapters == current.chapters
                assert case.lookup(request).json()["result"]["phase"] == "superseded"
                report["undo_restored"] = True
            report["accepted"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error) if isinstance(error, AssertionError) else "See private stage logs"
        raise
    finally:
        private_json(attempt / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"error", "result", "code_hashes"}}, ensure_ascii=False), flush=True)
        print("private_evidence=" + str(attempt), flush=True)
        case.doCleanups()
        CleanupApiTests.doClassCleanups()
        assert sha(directory / "book.pdf") == manifest["source_sha256"]
        assert sha(directory / "document.json") == manifest["document_sha256"]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--backup-noise", type=Path)
    parser.add_argument("--case", choices=("sample", "sample2", "noise"))
    parser.add_argument("--allow-live-model", action="store_true")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--entry", choices=("combined", "standalone", "retry"), default="combined")
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.backup_noise:
        raw = args.backup_noise.read_bytes()
        preserve(ROOT / "noise" / "book.pdf", raw)
        preserve(ROOT / "noise" / "manifest.json", json.dumps({
            "source_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True).encode())
    if args.extract:
        extract()
    if args.controls:
        controls()
    if args.case:
        if not args.allow_live_model or not os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"):
            parser.error("Explicit live-model consent and disposable PostgreSQL DSN required")
        live(args.case, with_browser=args.browser, entry=args.entry)


if __name__ == "__main__":
    run()
