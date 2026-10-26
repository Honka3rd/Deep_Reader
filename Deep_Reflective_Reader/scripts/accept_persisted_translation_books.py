"""Opt-in real-model, real-App and disposable-SQL acceptance on immutable book excerpts."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from unittest.mock import Mock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.translation_action import TranslationAction
from config.faiss_storage_config import FaissStorageConfig
from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
from db.postgres_structured_document_store import PostgresStructuredDocumentStore
from document_structure.reading_content import ReadingContentService, original_blocks, project_document
from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from scripts.private_book_corpus import CORPUS_ROOT, load_book, preserve
from scripts.test_reading_target_resolver import _document
from scripts.test_summary_note_postgres import PostgresNotes
from scripts.test_task_layout_readiness import ReadinessTests
from section_tasks.task_unit_resolver import TaskUnitResolver
from section_tasks.translation_service import TranslationService


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    import main
    from auth.openai_api_key_provider import OpenAIAPIKeyProvider
    from config.app_DI_config import AppDIConfig
    from llm.openai_llm_provider import OpenAILLMProvider
    settings = AppDIConfig()
    provider = OpenAILLMProvider(OpenAIAPIKeyProvider(), model=settings.llm_model,
                                target_max_output_tokens=settings.target_max_output_tokens)
    evidence = CORPUS_ROOT / "persisted-translation" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    evidence.mkdir(parents=True, mode=0o700)
    report = {"status": "failed", "scope": "Real backup excerpts, model, API and disposable SQL; not literary certification", "cases": []}
    backups, books = {}, {}
    for case in ("fa794acef055f235", "085747b65094d52f"):
        root, _, document = load_book(case)
        books[case] = document
        for name in ("manifest.json", "snapshot.json", "original-document.json", "source.txt", "source.pdf"):
            path = root / name
            if path.is_file():
                backups[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    PostgresNotes.setUpClass()
    readiness = ReadinessTests()
    readiness.setUp()
    try:
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            coordinator = readiness.coordinator
            coordinator.task_unit_resolver = TaskUnitResolver(task_unit_min_chars=20, task_unit_max_chars=1600)
            from fastapi.testclient import TestClient
            client = TestClient(main.app)
            stack.callback(client.close)
            stack.enter_context(patch.object(main, "section_task_coordinator", coordinator))
            profile_config = FaissStorageConfig("book", read_only=True)
            profile_config.profile_path = Path(directory) / "profile.json"
            stack.enter_context(patch("app.translation_action.FaissStorageConfig", return_value=profile_config))
            original_action = main.app.state.translation_action
            stack.callback(setattr, main.app.state, "translation_action", original_action)
            repo, language, profile_hash = None, None, None

            def configure(case):
                nonlocal repo, language, profile_hash
                source = books[case]
                rows = [b.content for _, _, _, b in original_blocks(project_document(source)) if 180 <= len(b.content) <= 700]
                assert len(rows) >= 2, "real excerpt blocks unavailable"
                language = "en" if case == "fa794acef055f235" else "zh"
                base = _document()
                sections = [replace(s, title=f"Excerpt {i + 1}", content=rows[i], char_start=0,
                                    char_end=len(rows[i]), task_units=[]) for i, s in enumerate(base.chapters[0].sections)]
                document = replace(base, document_id="book", title=source.title, source_path=None, language=language,
                    raw_text="\n\n".join(rows[:2]), sections=[], chapters=[replace(base.chapters[0], title="Reading", sections=sections)])
                repo = PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(PostgresNotes.scoped, namespace=uuid4().hex))
                repo.save_reparsed_document(document, "book")
                coordinator.document_artifact_repository = repo
                coordinator.ensure_document_task_layout("book", task_unit_split_mode="progressive")
                DocumentProfileStore.save(DocumentProfile("Existing test profile", "Fixed profile summary", LanguageCode(language)), profile_config)
                profile_hash = hashlib.sha256(profile_config.profile_path.read_bytes()).hexdigest()
                main.app.state.translation_action = TranslationAction(repo, DocumentProfileStore(), TranslationService(provider))
                main.document_artifact_repository = repo
                state = ReadingContentService(repo).read("book")
                return {"name": "book", "language": language, "block_id": state["blocks"][0]["block_id"], "case": case}

            stack.enter_context(patch.object(main, "document_artifact_repository", main.document_artifact_repository))

            def stats():
                assert hashlib.sha256(profile_config.profile_path.read_bytes()).hexdigest() == profile_hash
                coordinator.document_preparation_pipeline.prepare_and_load.assert_not_called()
                state = ReadingContentService(repo).read("book")
                return {"revision": state["revision"], "blocks": state["blocks"],
                    "translations": (repo.load_document("book").document_task_artifacts.metadata.get("translations", {}))}

            if args.browser:
                class Handler(BaseHTTPRequestHandler):
                    def handle_request(self):
                        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))) if self.command == "POST" else None
                        if self.path == "/fixture":
                            status, content = 200, json.dumps(configure(body["case"])).encode()
                        elif self.path == "/stats":
                            status, content = 200, json.dumps(stats()).encode()
                        else:
                            response = client.request(self.command, self.path, json=body)
                            status, content = response.status_code, response.content
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
                try:
                    subprocess.run(["node", str(Path(__file__).resolve().parents[2] / "Deep_Reader_UI/scripts/test-translation-real.mjs")],
                        check=True, env={**os.environ, "TRANSLATION_API": f"http://127.0.0.1:{server.server_port}",
                                         "TRANSLATION_EVIDENCE": str(evidence)})
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()
            else:
                for case in books:
                    fixture = configure(case)
                    target = {"doc_name": "book", "content_block_id": fixture["block_id"]}
                    for code in ("fr", "de"):
                        request = {**target, "target_language": code, "mutation_id": uuid4().hex}
                        response = client.post("/documents/reading-interactions/translate/generate", json=request)
                        assert response.status_code == 200, response.text
                        assert response.json()["envelope"]["persisted"], response.text
                        assert client.post("/documents/reading-interactions/translate/generate", json=request).json() == response.json()
                        report["cases"].append({"case": case, "target": code, "response": response.json()})
                    saved = client.post("/documents/reading-interactions/translate/read", json=target).json()
                    assert len(saved["translations"]) == 2
                    stats()
            report["status"] = "checks_passed"
    finally:
        readiness.doCleanups()
        PostgresNotes.doClassCleanups()
        unchanged = all(hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in backups.items())
        report["backups_unchanged"] = unchanged
        preserve(evidence / "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode())
        print(f"Translation acceptance evidence: {evidence}", flush=True)
        assert unchanged, "Immutable backup changed"


if __name__ == "__main__":
    run()
