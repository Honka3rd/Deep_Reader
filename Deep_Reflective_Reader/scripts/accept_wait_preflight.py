"""WAIT-05: immutable incident-book HTTP preflight, no generation or Library writes.

Run --image on the host to recreate five offline containers. The worker uses
disposable file storage; production PostgreSQL composition is covered separately.
"""

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch
from urllib.parse import quote
from urllib.request import Request, urlopen

from scripts.private_book_corpus import CORPUS_ROOT, fixture_path, load_book, preserve
from scripts.accept_enhancement_real_book import private_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_profile(case_id, source):
    directory = fixture_path(case_id)
    load_book(case_id)
    data = source.read_bytes()
    json.loads(data)
    preserve(directory / "profile.json", data)
    preserve(directory / "wait-profile-manifest.json", json.dumps({
        "profile.json": hashlib.sha256(data).hexdigest()}, sort_keys=True).encode())
    assert sha(source) == sha(directory / "profile.json")


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        message = record.getMessage()
        if ("stage=cleanup_preflight_" in message or message.startswith("tokenizer ")
                or message.startswith("cleanup_input ")):
            self.events.append({"request_id": getattr(record, "request_id", "-"), "message": message})


def worker(case_id, runs, browser=False, evidence_dir=None):
    from fastapi import FastAPI
    import uvicorn
    from app.standalone_cleanup import StandaloneCleanupService
    from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot
    from document_preparation.cleanup_input_adapter import adapt_document_cleanup_input
    from document_preparation.document_preparation_pipeline import DocumentPreparationPipeline
    from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
    from document_structure.reading_content import digest, state_for, ReadingContentService
    from doc_loaders.pdf_document_loader import PdfDocumentLoader
    from profile.document_profile_builder import DocumentProfileBuilder
    from profile.document_profile_store import DocumentProfileStore
    from logging_config import RequestContextFilter, request_id_context
    from scripts.test_offline_token_counter import provider
    from smart_cleanup_routes import router

    directory, baseline, original = load_book(case_id)
    assert baseline.document_id == "\u6697\u6c34\u5e7d\u7075" and baseline.text_length == 176196, "Full incident book required"
    expected_profile = json.loads((directory / "wait-profile-manifest.json").read_text())
    assert sha(directory / "profile.json") == expected_profile["profile.json"]
    paths = [directory / name for name in json.loads((directory / "manifest.json").read_text())["files"]]
    paths += [directory / "profile.json", directory / "manifest.json", directory / "wait-profile-manifest.json"]
    before = {p.name: sha(p) for p in paths}
    cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
    assert not cache.exists() or not list(cache.iterdir()), "Ephemeral cache must start empty"
    capture = Capture()
    capture.addFilter(RequestContextFilter())
    counts = {"input": 0, "token_count": 0, "external_network": 0, "loopback_http": 0}
    real_connect = socket.socket.connect

    def connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
            counts["external_network"] += 1
            raise AssertionError("External network forbidden")
        counts["loopback_http"] += 1
        return real_connect(sock, address)

    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        # Keep exact accepted hierarchy bytes; only the storage location is temporary.
        shutil.copyfile(directory / "original-document.json", root / f"{baseline.document_id}.structured.json")
        shutil.copyfile(directory / "profile.json", root / "profile.json")
        repo = StructuredDocumentArtifactRepository(base_dir=str(root), namespace=baseline.namespace)
        stored_before = {p.name: sha(p) for p in root.iterdir()}
        document_before = repo.load_document(baseline.document_id)
        mutex = threading.Lock()

        @contextmanager
        def lease(name):
            assert name == baseline.document_id
            with mutex:
                yield

        def read_input(name):
            assert name == baseline.document_id
            counts["input"] += 1
            document = repo.load_document(name)
            source = resolve_accepted_source_snapshot(document=document, namespace=baseline.namespace,
                expected_document_id=name, structure_version=baseline.structure_version,
                source_path=directory / "source.pdf", accepted=True)
            return adapt_document_cleanup_input(document=document, snapshot=source,
                expected_reading_revision=state_for(document).get("revision", 0))

        model = provider()  # Actual OpenAILLMProvider methods/capabilities, no SDK credentials.
        count_tokens = model.count_input_tokens

        def count(text):
            counts["token_count"] += 1
            return count_tokens(text)

        stack.enter_context(patch.object(model, "count_input_tokens", side_effect=count))
        forbidden = {}
        for owner, method in ((model, "complete_text"), (model, "complete_text_with_output_budget"),
                (repo, "commit_cleanup"), (repo, "save_document"),
                (DocumentPreparationPipeline, "prepare_and_load"),
                (DocumentPreparationPipeline, "build_enhancement_candidate"),
                (PdfDocumentLoader, "extract"), (PdfDocumentLoader, "_run_process"),
                (DocumentProfileBuilder, "build"), (DocumentProfileStore, "save")):
            forbidden[method] = stack.enter_context(patch.object(owner, method,
                side_effect=AssertionError("Forbidden preflight side effect")))
        for name in ("tiktoken.load.read_file", "tiktoken.get_encoding", "tiktoken.encoding_for_model"):
            forbidden[name] = stack.enter_context(patch(name, side_effect=AssertionError("No registry/download")))
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        for name in ("app.standalone_cleanup", "llm.offline_token_counter", "document_preparation.cleanup_input_adapter"):
            logger = logging.getLogger(name)
            previous = logger.level
            logger.setLevel(logging.INFO)
            logger.addHandler(capture)
            stack.callback(logger.setLevel, previous)
            stack.callback(logger.removeHandler, capture)
        app = FastAPI()
        app.include_router(router)
        @app.get("/documents/{name}/reading-content")
        def reading_content(name: str):
            assert name == baseline.document_id
            return ReadingContentService(repo).read(name)

        app.state.smart_cleanup_service = StandaloneCleanupService(namespace=baseline.namespace,
            repository=repo, read_input=read_input, provider=model, lease=lease,
            layout_ready=lambda value: True, prepare_layout=forbidden["prepare_and_load"])

        @app.middleware("http")
        async def correlation(request, call_next):
            token = request_id_context.set(request.headers.get("X-Request-ID", "wait05-browser"))
            try:
                return await call_next(request)
            finally:
                request_id_context.reset(token)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        samples = []
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert thread.is_alive() and time.monotonic() < deadline, "HTTP startup failed"
                time.sleep(.01)
            endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}/documents/{quote(baseline.document_id)}/smart-cleanup/preflight"
            for index in range(runs):
                request_id = f"wait05-{index}"
                started = time.perf_counter()
                with urlopen(Request(endpoint, data=b"{}", headers={"Content-Type": "application/json",
                        "X-Request-ID": request_id}), timeout=10) as response:
                    result = json.load(response)
                    assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
                elapsed = (time.perf_counter() - started) * 1000
                assert result["execution_available"] is True and result["reason"] is None, result["reason"]
                assert elapsed < 10000
                events = [event for event in capture.events if event["request_id"] == request_id]
                assert any("stage=cleanup_preflight_overall" in event["message"] for event in events)
                assert any("stage=count" in event["message"] for event in events)
                assert any(f"cache={'cold' if index == 0 else 'memory'}" in event["message"] for event in events)
                for secret in (result["confirmation_token"], result["operation_id"], baseline.canonical_text):
                    assert all(secret not in event["message"] for event in events)
                samples.append({"temperature": "cold" if index == 0 else "warm", "http_ms": elapsed, "events": events})
            if browser:
                script = CORPUS_ROOT.parents[2] / "Deep_Reader_UI/scripts/test-cleanup-books.mjs"
                subprocess.run(["node", str(script)], check=True, timeout=180, env={**os.environ,
                    "CLEANUP_PREFLIGHT_ONLY": "1", "CLEANUP_BOOK_DOCUMENT": str(directory / "original-document.json"),
                    "CLEANUP_TEST_API": endpoint.split("/documents/", 1)[0],
                    "CLEANUP_EVIDENCE_DIR": str(evidence_dir)})
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            listener.close()
            assert not thread.is_alive()
        counts.update({key: value.call_count for key, value in forbidden.items()})
        assert all(value.call_count == 0 for value in forbidden.values()), counts
        assert counts["input"] == counts["token_count"] >= runs and counts["external_network"] == 0, counts
        if not browser:
            assert counts["input"] == runs
        after = {p.name: sha(p) for p in paths}
        assert before == after
        assert stored_before == {p.name: sha(p) for p in root.iterdir()}
        current = repo.load_document(baseline.document_id)
        assert current.to_dict() == document_before.to_dict()
        assert current.chapters == original.chapters
        assert not cache.exists() or not list(cache.iterdir())
        return {"accepted": True, "case_id": case_id, "chars": baseline.text_length,
            "pages": len(baseline.cleanup_context.pages), "ocr_spans": len(baseline.cleanup_context.evidence.spans),
            "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "encoding_identity": model.tokenizer_identity, "hashes_before": before, "hashes_after": after,
            "hierarchy_sha256": digest([c.to_dict() for c in current.chapters]),
            "reading_state_sha256": digest(state_for(current)), "counters": counts, "samples": samples,
            "storage": "disposable file repository; test-local lease; production routes/adapter/provider/use-case"}


def containers(args):
    directory = fixture_path(args.case_id).resolve()
    assert args.report.parent.resolve().is_relative_to(CORPUS_ROOT.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    image = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
    results = []
    for index in range(5):
        name = f"wait05-{os.getpid()}-{index}"
        command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp", "-e", "TIKTOKEN_CACHE_DIR=/tmp/empty-token-cache",
            "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{directory}:/app/Deep_Reflective_Reader/data/test-books/{args.case_id}:ro",
            args.image, "python", "-m", "scripts.accept_wait_preflight", "--case-id", args.case_id,
            "--runs", "6" if index == 0 else "1"]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=90)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        if process.returncode:
            private_json(args.report, {"accepted": False, "image": image, "failed_container": index,
                "stdout": process.stdout, "stderr": process.stderr})
            raise RuntimeError("Offline acceptance failed; see private report")
        result = json.loads(process.stdout.split("WAIT_REPORT=", 1)[1])
        results.append(result)
        print(f"Container {index + 1}/5 accepted", flush=True)
    distributions = {}
    for temperature in ("cold", "warm"):
        values = [s["http_ms"] for r in results for s in r["samples"] if s["temperature"] == temperature]
        assert len(values) == 5
        distributions[temperature] = {"count": len(values), "min_ms": min(values),
            "median_ms": statistics.median(values), "max_ms": max(values)}
    stages = {}
    for result in results:
        for sample in result["samples"]:
            for event in sample["events"]:
                match = re.search(r"stage=(\w+).*?(?:duration|elapsed)_ms=([\d.]+)", event["message"])
                if match:
                    stages.setdefault(sample["temperature"] + ":" + match[1], []).append(float(match[2]))
    stages = {name: {"count": len(values), "min_ms": min(values), "median_ms": statistics.median(values),
        "max_ms": max(values)} for name, values in stages.items()}
    private_json(args.report, {"accepted": True, "image": image, "host": platform.platform(),
        "network": "none", "root": "read-only", "ephemeral_cache": "fresh tmpfs per container",
        "test_gate_ms": 10000, "not_production_sla": True, "distribution": distributions,
        "stage_distributions": stages, "containers": results})
    print(json.dumps(distributions), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--image")
    parser.add_argument("--runs", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--capture-profile", type=Path)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    if args.report:
        if not args.report.parent.resolve().is_relative_to(CORPUS_ROOT.resolve()):
            parser.error("Reports must stay under data/test-books")
        args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.browser and (args.image or not args.report):
        parser.error("--browser runs on the host and requires --report")
    if args.capture_profile:
        capture_profile(args.case_id, args.capture_profile)
    elif args.image:
        if not args.report:
            parser.error("--image requires --report under data/test-books")
        containers(args)
    else:
        result = worker(args.case_id, args.runs, args.browser, args.report.parent if args.report else None)
        if args.report:
            private_json(args.report, result)
        else:
            print("WAIT_REPORT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
