"""Backup-only real OCR plus review/edit contracts; optional disposable HTTP fixture."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from doc_loaders.pdf_document_loader import PdfDocumentLoader
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.reading_content import ReadingContentService
from scripts.test_reading_interaction_artifact_store import _document
from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.summary_interaction_service import SummaryInteractionService


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--serve", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    source = args.corpus / manifest["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    loader = PdfDocumentLoader(source.parent, ocr_enabled=True)
    clean = loader.extract(source.stem, mode="ocr")
    assert len(clean.pages) == 3 and not clean.provenance["source_review"]["regions"]
    # Reproduce the previously failing margin without replacing OCR output.
    with patch("doc_loaders.chinese_ocr_strategy.read_margin", return_value=None):
        partial = PdfDocumentLoader(source.parent, ocr_enabled=True).extract(source.stem, mode="ocr")
    review = partial.provenance["source_review"]
    assert review["regions"] and len(partial.pages) == 3
    assert partial.page_quality[1]["blank_render_confirmed"]
    assert "source_needs_review" in partial.warnings
    assert all(r["page_index"] == 2 for r in review["regions"])
    document = _document()
    section = document.chapters[0].sections[0]
    section = replace(section, content=partial.text, char_end=len(partial.text),
        task_units=[replace(section.task_units[0], content=partial.text)])
    document = replace(document, document_id=source.stem, title=source.stem, language="zh",
        source_path=str(source), raw_text=partial.text,
        chapters=[replace(document.chapters[0], sections=[section])], parse_provenance={"source_review":review})
    with TemporaryDirectory() as directory:
        repo = StructuredDocumentArtifactRepository(base_dir=directory)
        repo.save_document(document, "book")
        service = ReadingContentService(repo, SimpleNamespace(get=lambda _: loader))
        # API fixture name differs from the backup filename, but source rendering never does.
        real_preview = loader.render_source_region
        loader.render_source_region = lambda _, *rest: real_preview(source.stem, *rest)
        state = service.read("book")
        assert state["disposition"] == "needs_review"
        png = service.preview("book", review["regions"][0]["id"])
        assert png.startswith(b"\x89PNG")
        try:
            real_preview(source.stem, "0"*64, review["regions"][0])
        except ValueError:
            pass
        else:
            raise AssertionError("Replaced source identity was accepted")
        def forbidden(*a, **kw):
            raise AssertionError("Unreviewed source reached generation/context")
        target = ReadingTargetResolver().resolve(document=repo.load_document("book"), target_level="document")
        for kind, generation in [
            ("analysis", AnalysisInteractionService(forbidden, artifact_context_provider=forbidden)),
            ("quiz", QuizInteractionService(forbidden, artifact_context_provider=forbidden)),
            ("critical_thinking_session", CriticalThinkingSessionService(forbidden, forbidden, artifact_context_provider=forbidden)),
            ("summary", SummaryInteractionService(repo, SimpleNamespace(generate=forbidden))),
        ]:
            result = generation.generate(ReadingInteractionRequest(target=target, interaction_type=kind,
                context_metadata={"doc_name":"book", "document_language":"zh"}))
            assert result.reason == "source_needs_review", (kind, result)
        def mutate(action, **kwargs):
            state = service.read("book")
            return service.mutate("book", action=action, expected_revision=state["revision"],
                source_token=state["source_token"], mutation_id=str(state["revision"]), **kwargs)
        for region in review["regions"]:
            mutate("discard_region", region_id=region["id"])
        assert service.read("book")["disposition"] == "completed"
        mutate("undo")
        assert service.read("book")["disposition"] == "needs_review"
        first = state["blocks"][0]
        mutate("edit", block_id=first["block_id"], text="Verified correction")
        mutate("delete", block_id=first["block_id"])
        assert service.read("book")["blocks"][0]["deleted"]
        mutate("undo")
        assert service.read("book")["blocks"][0]["content"] == "Verified correction"
        assert repo.load_document("book").raw_text == partial.text
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
        report = {"passed":True, "source_sha256":manifest["sha256"], "clean_page_chars":[len(p) for p in clean.pages],
            "partial_page_chars":[len(p) for p in partial.pages], "unresolved_regions":len(review["regions"]),
            "partial_scenario":"real OCR with margin recovery disabled (fault injection)",
            "guards":"analysis/quiz/thinking/summary: no generator or context invocation",
            "mutations":"discard, undo, edit, delete, undo; immutable source"}
        (args.corpus / "reading-review-acceptance.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if args.serve:
            import main
            import uvicorn
            main._reading_content_service = lambda: service
            # Only expose the three review endpoints in this isolated fixture.
            from fastapi import FastAPI
            app = FastAPI()
            app.router.routes.extend(r for r in main.app.routes if
                "/reading-content" in getattr(r,"path","") or "/source-review/" in getattr(r,"path",""))
            uvicorn.run(app, host="0.0.0.0", port=args.serve)


if __name__ == "__main__":
    run()
