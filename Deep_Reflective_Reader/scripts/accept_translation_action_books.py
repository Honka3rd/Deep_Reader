"""Opt-in live translation API on existing immutable backup blocks, no production writes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.translation_action import TranslationAction
from config.faiss_storage_config import FaissStorageConfig
from document_structure.reading_content import original_blocks, project_document
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from language.language_code import LanguageCode
from profile.document_profile import DocumentProfile
from profile.document_profile_store import DocumentProfileStore
from scripts.private_book_corpus import CORPUS_ROOT, load_book, preserve
from section_tasks.translation_service import TranslationService
from translation_routes import router


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", required=True, action="store_true")
    parser.parse_args()
    from auth.openai_api_key_provider import OpenAIAPIKeyProvider
    from config.app_DI_config import AppDIConfig
    from llm.openai_llm_provider import OpenAILLMProvider

    settings = AppDIConfig()
    provider = OpenAILLMProvider(OpenAIAPIKeyProvider(), model=settings.llm_model,
                                target_max_output_tokens=settings.target_max_output_tokens)
    directory = CORPUS_ROOT / "translation-api" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    report = {"status": "failed", "cases": [], "model": str(settings.llm_model),
              "scope": "live HTTP response-contract acceptance, not literary accuracy certification"}
    try:
        with tempfile.TemporaryDirectory() as temporary:
            repo = StructuredDocumentArtifactRepository(base_dir=str(Path(temporary) / "structured"))
            app = FastAPI()
            app.include_router(router)
            app.state.translation_action = TranslationAction(repo, DocumentProfileStore(), TranslationService(provider))
            with TestClient(app) as client:
                for case, source, target in [("fa794acef055f235", "en", "zh-Hant"),
                                             ("085747b65094d52f", "zh", "en")]:
                    root, _, document = load_book(case)
                    backups = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in root.iterdir() if p.is_file() and p.name in
                               {"manifest.json", "source.txt", "source.pdf", "snapshot.json", "original-document.json"}}
                    repo.save_document(document, case)
                    config = FaissStorageConfig(case, read_only=True)
                    config.profile_path = Path(temporary) / f"{case}.profile.json"
                    # Isolated existing profile fixture, never generate or load production profiles.
                    DocumentProfileStore.save(DocumentProfile("fixture", "", LanguageCode(source)), config)
                    candidates = [b for _, _, _, b in original_blocks(project_document(document))
                                  if 150 <= len(b.content) <= 700]
                    assert candidates, f"missing bounded real block: {case}"
                    block = candidates[0]
                    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(temporary).rglob("*") if p.is_file()}
                    request = {"doc_name": case, "content_block_id": block.block_id, "target_language": target}
                    with patch("app.translation_action.FaissStorageConfig", return_value=config):
                        response = client.post("/documents/reading-interactions/translate/generate", json=request)
                    report["cases"].append({"case": case, "request": request, "source_text": block.content,
                                            "http_status": response.status_code, "response": response.json()})
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert body["envelope"]["status"] == "completed"
                    assert body["envelope"]["target"]["content_block_id"] == block.block_id
                    assert body["payload"]["source_language"] == source
                    assert body["payload"]["target_language"] == target
                    assert body["envelope"]["effective_content_hash"] == hashlib.sha256(block.content.encode()).hexdigest()
                    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items())
                    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha for name, sha in backups.items())
                    print(f"PASS {case}: live block API with profile default; source/profile/backups unchanged", flush=True)
        report["status"] = "checks_passed"
    finally:
        preserve(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2).encode())
        print(f"Translation API evidence: {directory}", flush=True)


if __name__ == "__main__":
    run()
