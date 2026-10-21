"""Replay a private real-model plan through real API/PG/layout in a disposable schema."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch


def verify(attempt, *, live_model=False):
    from accept_enhancement_real_book import private_json
    from fastapi.testclient import TestClient
    from test_enhancement_postgres_preflight import main, SectionTaskCoordinator
    from test_enhancement_conditional_commit import ConditionalCommitTests
    from test_structure_enhancement_candidate import candidate_pipeline

    report = json.loads((attempt / "report.json").read_text())
    assert report["accepted"], "Only accepted real-model plans can be replayed"
    from private_book_corpus import load_book
    corpus, snapshot, original_document = load_book(report["case_id"])
    assert attempt.parent == corpus.resolve(), "Attempt must belong to the backed-up case"
    source = {"source": snapshot.canonical_text, "doc_name": snapshot.document_id}
    original = main.section_task_coordinator
    profile_root = Path(__file__).resolve().parents[1] / "data" / "faiss"
    def profile_hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in profile_root.glob("*/profile.json")}
    profiles_before = profile_hashes()
    live_evidence = attempt / ("api-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if live_model:
        live_evidence.mkdir(mode=0o700)
    assert hashlib.sha256(source["source"].encode()).hexdigest() == report["source_sha256"]
    response_path = attempt / f"response-{report['provider_calls']}.json"
    if not response_path.exists():
        response_path = attempt / "response.json"
    response = json.loads(response_path.read_text())["response"]
    provider = original.document_preparation_pipeline.structured_document_builder.section_splitter_selector.llm_splitter.llm_provider
    ConditionalCommitTests.setUpClass()
    case = ConditionalCommitTests()
    try:
        case.setUp()
        source_format = snapshot.extraction_identity.source_format
        (Path(case.fixture.temp.name) / "doc.pdf").unlink()
        shutil.copyfile(attempt.parent / ("source." + source_format),
                        Path(case.fixture.temp.name) / ("doc." + source_format))
        copied = replace(original_document, document_id="doc", title="isolated real book", document_task_artifacts=None)
        case.repository.save_reparsed_document(copied, "doc")
        case.enable_execution()
        pipeline = candidate_pipeline(provider)
        pipeline.loader_factory = SimpleNamespace(base_dir=case.fixture.temp.name)
        SectionTaskCoordinator.__init__(case.coordinator,
            document_preparation_pipeline=pipeline, document_artifact_repository=case.repository,
            document_profile_store=case.coordinator.document_profile_store,
            chapter_summary_service=original.chapter_summary_service,
            chapter_quiz_service=original.chapter_quiz_service,
            task_unit_resolver=original.task_unit_resolver,
            enhanced_parse_trigger_evaluator=original.enhanced_parse_trigger_evaluator)
        case.add_artifact("note", source_version=case.snapshot().structure_version)
        config = SimpleNamespace(doc_name="doc", get_raw_profile_path=lambda: str(Path(case.fixture.temp.name) / "missing.json"))
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "section_task_coordinator", case.coordinator))
            stack.enter_context(patch.object(original, "read_enhancement_state", side_effect=AssertionError("Use backup only")))
            stack.enter_context(patch.object(original.document_artifact_repository, "load_enhancement_snapshot", side_effect=AssertionError("Use backup only")))
            stack.enter_context(patch.object(main, "enhancement_preflight_service", case.service))
            stack.enter_context(patch("app.enhancement_preflight.enhancement_preflight_service", case.service))
            stack.enter_context(patch("app.section_task_coordinator.FaissStorageConfig", return_value=config))
            stack.enter_context(patch.object(pipeline, "prepare_and_load", side_effect=AssertionError("No raw/OCR re-entry")))
            stack.enter_context(patch.object(pipeline, "_prepare_profile", side_effect=AssertionError("No profile creation")))
            method = "complete_text_with_output_budget" if callable(getattr(type(provider), "complete_text_with_output_budget", None)) else "complete_text"
            complete = getattr(provider, method)
            calls = 0
            def capture(prompt, *args, **kwargs):
                nonlocal calls
                calls += 1
                private_json(live_evidence / f"request-{calls}.json", {"prompt": prompt})
                output = complete(prompt, *args, **kwargs)
                private_json(live_evidence / f"response-{calls}.json", {"response": output})
                return output
            replay = stack.enter_context(patch.object(provider, method,
                **({"side_effect": capture} if live_model else {"return_value": response})))
            client = stack.enter_context(TestClient(main.app))
            preflight = client.post("/documents/reparse-structure/preflight", json={"doc_name": "doc"})
            assert preflight.status_code == 200 and preflight.json()["ready"], preflight.text
            result = client.post("/documents/reparse-structure", json={"doc_name": "doc", "parser_mode": "llm_enhanced",
                "discard_interactions": True, "confirmation_token": preflight.json()["confirmation_token"]})
            if live_model:
                private_json(live_evidence / "result.json", result.json())
            assert result.status_code == 200 and result.json()["success"], result.text
            assert result.json()["change_kind"] == "structure_changed", result.text
            assert result.json()["interactions_cleared"] is True
            if live_model:
                assert replay.call_count > 0, "Real model was not called"
            else:
                replay.assert_called_once()
            body = {"doc_name": "doc", "task_unit_split_mode": "progressive"}
            prepared = client.post("/documents/prepare-task-layout", json={**body, "layout_only": True})
            assert prepared.status_code == 200, prepared.text
            layout = client.post("/documents/task-layout", json=body)
            assert layout.status_code == 200, layout.text
            chapters = layout.json()["chapters"]
            units = [u for c in chapters for s in c["sections"] for u in s["task_units"]]
            assert units
            if not live_model:
                assert len(chapters) == report["chapter_count"]
            accepted = case.coordinator.read_enhancement_state("doc").source
            assert accepted.canonical_text == source["source"]
            assert case.rows("artifacts") == [], "Seeded interaction must be deleted"
            evidence = {"api": "passed", "live_model": live_model, "provider_calls": replay.call_count,
                "real_task_layout": True, "chapter_count": len(chapters),
                "task_unit_count": len(units), "exact_body_preserved": True, "production_write": False,
                "profiles_unchanged": profile_hashes() == profiles_before,
                "source_format": source_format, "case_id": report["case_id"]}
            assert evidence["profiles_unchanged"], "Profiles changed during enhancement"
            private_json(attempt / ("api-live.json" if live_model else "api-replay.json"), evidence)
            print(json.dumps(evidence))
            return evidence
    finally:
        case.doCleanups()
        ConditionalCommitTests.doClassCleanups()
        assert load_book(report["case_id"])[1] == snapshot
        assert profile_hashes() == profiles_before, "Profile write during acceptance"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--allow-live-embeddings", action="store_true")
    parser.add_argument("--allow-live-model", action="store_true")
    args = parser.parse_args()
    if not args.allow_live_embeddings:
        parser.error("Explicit --allow-live-embeddings required for the real layout resolver")
    verify(args.attempt_dir.resolve(), live_model=args.allow_live_model)
