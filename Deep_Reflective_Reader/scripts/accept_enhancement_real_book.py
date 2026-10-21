"""Explicit paid-model acceptance on an accepted book, with no production writes.

Run from the backend after loading .env.local; --allow-live-model is mandatory.
Private source/prompt/response evidence stays in the gitignored corpus directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from unittest.mock import patch


def private_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id")
    selection.add_argument("--backup-doc-name")
    parser.add_argument("--allow-live-model", action="store_true")
    parser.add_argument("--require-structure-change", action="store_true")
    args = parser.parse_args()
    if args.case_id and not args.allow_live_model:
        parser.error("Explicit --allow-live-model required; this sends the book to the configured provider")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import main
    from private_book_corpus import backup_book, load_book
    from document_structure.structure_plan_contract import PROMPT_VERSION
    coordinator = main.section_task_coordinator
    if args.backup_doc_name:
        print("backup_case_id=" + backup_book(coordinator, args.backup_doc_name))
        return 0
    case_id = args.case_id
    corpus, snapshot, original_document = load_book(case_id)
    attempt = corpus / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(mode=0o700)
    pipeline = coordinator.document_preparation_pipeline
    splitter = pipeline.structured_document_builder.section_splitter_selector.llm_splitter
    provider = splitter.llm_provider
    method = "complete_text_with_output_budget" if callable(getattr(type(provider), "complete_text_with_output_budget", None)) else "complete_text"
    complete = getattr(provider, method)
    calls = 0
    def capture(prompt, *args, **kwargs):
        nonlocal calls
        calls += 1
        private_json(attempt / f"request-{calls}.json", {"prompt": prompt,
            "output_tokens": args[0] if args else None, "response_schema": kwargs.get("response_schema")})
        response = complete(prompt, *args, **kwargs)
        private_json(attempt / f"response-{calls}.json", {"response": response})
        return response
    report = {"case_id": case_id, "prompt_version": PROMPT_VERSION,
        "model": provider.get_model_capabilities().model_name,
        "source_chars": snapshot.text_length, "source_sha256": snapshot.text_sha256,
        "accepted": False, "production_write": False}
    try:
        with patch.object(coordinator, "read_enhancement_state", side_effect=AssertionError("Use backup only")), patch.object(coordinator.document_artifact_repository, "load_enhancement_snapshot", side_effect=AssertionError("Use backup only")), patch.object(provider, method, side_effect=capture), patch.object(
                pipeline, "prepare_and_load", side_effect=AssertionError("No extraction during enhancement")):
            candidate = pipeline.build_enhancement_candidate(snapshot=snapshot)
        sections = [s for c in candidate.document.chapters for s in c.sections]
        assert "".join(s.content for s in sections) == snapshot.canonical_text
        from document_structure.enhancement_change import compare_enhancement_documents
        change = compare_enhancement_documents(original_document, candidate.document,
                                               canonical_text=snapshot.canonical_text)
        report["change_kind"] = change.kind.value
        if args.require_structure_change:
            assert change.kind.value == "structure_changed", "No effective structural improvement"
        private_json(attempt / "candidate.json", candidate.document.to_dict())
        report.update(accepted=True, chapter_count=len(candidate.document.chapters), section_count=len(sections),
                      exact_body_preserved=True)
    except Exception as error:
        report["error"] = str(error) if isinstance(error, ValueError) else type(error).__name__
    finally:
        assert load_book(case_id)[1] == snapshot, "Backup changed during evaluation"
        report["provider_calls"] = calls
        private_json(attempt / "report.json", report)
        print(json.dumps(report), flush=True)
        print("private_evidence=" + str(attempt), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(run())
