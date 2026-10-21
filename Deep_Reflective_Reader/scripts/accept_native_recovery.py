"""NATIVE-06 acceptance: isolated storage, opt-in PDFs/Postgres, fake providers.

Prints counts/hashes only. Never sends private source to a remote model.
Postgres requires DEEP_READER_TEST_POSTGRES_DSN and uses a disposable schema.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accept_selective_ocr import IsolatedStore
from diagnose_native_fragmentation import coverage_report
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.section_splitter_selector import SectionSplitterSelector
from document_structure.llm_section_splitter import LLMSectionSplitter
from document_structure.heading_acceptance import heading_decision
from language.language_code import LanguageCode
from section_tasks.reading_target_resolver import ReadingTargetResolver
from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
from section_tasks.reading_interaction_service_contracts import ReadingInteractionRequest
from section_tasks.analysis_interaction_service import AnalysisInteractionService
from section_tasks.quiz_interaction_service import QuizInteractionService
from section_tasks.critical_thinking_session_service import CriticalThinkingSessionService
from section_tasks.analysis_interaction_orchestrator import AnalysisInteractionOrchestrator
from section_tasks.quiz_interaction_orchestrator import QuizInteractionOrchestrator
from shared.task_unit_model import TaskUnit
from test_prepare_structured_reuse_before_raw_load import _build_pipeline
from test_structure_plan_application import Provider


SUITES = (
    "test_native_fragmentation_baseline", "test_source_mapped_text", "test_common_parse_policy",
    "test_structure_plan_contract", "test_structure_plan_application",
    "test_fragmented_interaction_preflight", "test_interaction_output_language",
    "test_interaction_policy_identity", "test_reader_provenance", "test_task_layout_readiness",
    "test_pdf_fast_extraction", "test_pdf_full_ocr_extraction", "test_pdf_ocr_decision",
    "test_pdf_mode_preparation", "test_pdf_extraction_mode_contract", "test_prepare_lazy_pdf_layout",
)


def regressions():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in SUITES)
    cleanup = importlib.import_module("test_hard_reparse_reading_interaction_cleanup")
    for name in ("test_file_repository_save_reparsed_document_clears_interaction_artifacts",
                 "test_cleanup_helper_removes_referenced_artifact_metadata"):
        suite.addTest(unittest.FunctionTestCase(getattr(cleanup, name)))
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and not result.skipped and not result.expectedFailures, "regression matrix incomplete"
    return {"tests": result.testsRun, "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures)}


def assert_coverage(document, text):
    assert document.structure_error_code is None, "structure rejected"
    sections = [s for c in document.chapters for s in c.sections]
    assert "".join(s.content for s in sections) == text, "canonical text changed"
    report = coverage_report(text, sections)
    assert all(report[key] == 0 for key in ("missing_nonspace_chars", "overlapping_nonspace_chars", "invalid_ranges")), "invalid coverage"
    return sections


def interactions(document, root):
    # Deterministic unit fixtures isolate interaction/persistence from embeddings.
    chapters = []
    for chapter in document.chapters:
        sections = [replace(section, task_units=[TaskUnit(
            unit_id=f"unit-{section.section_id}", title=section.title, container_title=None,
            content=section.content, source_section_ids=[section.section_id], is_fallback_generated=False,
            parent_section_id=section.section_id)]) for section in chapter.sections]
        chapters.append(replace(chapter, sections=sections))
    document = replace(document, chapters=chapters)
    repository = StructuredDocumentArtifactRepository(base_dir=str(root))
    repository.save_document(document, doc_name="fixture")
    document = repository.load_document("fixture")
    store = DocumentReadingInteractionArtifactStore(repository)
    chapter, section = max(((c, s) for c in document.chapters for s in c.sections), key=lambda pair: len(pair[1].content))
    unit = section.task_units[0]
    resolver = ReadingTargetResolver()
    block = max(resolver.content_blocks_for_target_validation(unit, section, chapter), key=lambda item: len(item.content))
    identifiers = {"chapter_id": chapter.chapter_id, "section_id": section.section_id,
                   "task_unit_id": unit.unit_id, "content_block_id": block.block_id}
    calls = []

    def generate(request, *args):
        calls.append(request.target.content)
        if request.interaction_type == "quiz":
            return {"items": [{"type": "short_answer", "question": "哪些證據支持這個判斷？", "answer": "需要對照原文的具體細節。"}]}
        if request.interaction_type == "critical_thinking_session":
            return {"question": "哪些證據支持這個判斷，還有哪些替代解釋？"}
        return {"summary": "這是一個驗收用的中文分析。", "reasoning": "分析需要對照具體證據。", "explanation": "這些固定文字僅驗證語言與儲存契約。"}

    checked = 0
    for index, level in enumerate(("document", "chapter", "section", "task_unit", "content_block")):
        target = resolver.resolve(document=document, target_level=level,
                                  **dict(list(identifiers.items())[:index]))
        for kind in ("analysis", "quiz", "critical_thinking_session"):
            request = ReadingInteractionRequest(target=target, interaction_type=kind,
                context_metadata={"doc_name": "fixture", "document_language": "zh"})
            count = len(calls)
            if kind == "critical_thinking_session":
                artifact = CriticalThinkingSessionService(generate, lambda *args: {}).generate(request)
                assert artifact.status == "question_generated", "thinking preflight/generation failed"
                saved = store.save_critical_thinking_session(artifact)
                before = (root / "fixture.structured.json").read_bytes()
                assert store.get_critical_thinking_session(request, saved.session_id).payload == artifact.payload
            else:
                orchestrator = (AnalysisInteractionOrchestrator(service=AnalysisInteractionService(generate), artifact_store=store)
                    if kind == "analysis" else QuizInteractionOrchestrator(service=QuizInteractionService(generate), artifact_store=store))
                artifact = orchestrator.generate(request)
                assert artifact.status == "completed", "interaction preflight/generation failed"
                before = (root / "fixture.structured.json").read_bytes()
                assert orchestrator.read(request).payload == artifact.payload
                assert orchestrator.generate(request).payload == artifact.payload
            assert len(calls) == count + 1 and calls[-1] == target.content, "read/reuse generated or source changed"
            assert (root / "fixture.structured.json").read_bytes() == before, "read/reuse wrote storage"
            checked += 1
    return checked


def accept_pdf(path):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    with tempfile.TemporaryDirectory(prefix="native-recovery-") as directory:
        root = Path(directory)
        with patch("subprocess.run", side_effect=AssertionError("fast path invoked OCR/renderer")), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            extracted = PdfDocumentLoader(path.parent).extract(path.stem, mode="fast", source_bytes=source)
            text = extracted.text
            builder = StructuredDocumentBuilder()
            common_shapes = {}
            for hint in ("auto", "present", "absent"):
                document = builder.build("fixture", "fixture", text, LanguageCode.ZH, toc_hint=hint)
                sections = assert_coverage(document, text)
                for section in sections:
                    if section.title == "\u5e8f":
                        assert heading_decision(text, section.char_start, section.title) == "accepted", "false short heading"
                common_shapes[hint] = {"chapters": len(document.chapters), "sections": len(sections)}
            checked = interactions(document, root / "interactions")
            store = IsolatedStore(root / "structure.json")
            loader = type("SnapshotLoader", (), {"load": lambda self, doc_name: text})()
            pipeline = _build_pipeline(loader=loader, store=store)
            pipeline.structured_document_builder = builder
            assert pipeline.prepare("fixture", mode="base").structured_document_ready
            attempts = 0
            for hint in ("auto", "present", "absent"):
                for intent in ("restore_source", "generate_reading_outline"):
                    provider = Provider()
                    pipeline.structured_document_builder = StructuredDocumentBuilder(
                        section_splitter_selector=SectionSplitterSelector(llm_splitter=LLMSectionSplitter(llm_provider=provider)))
                    result = pipeline.prepare("fixture", mode="base", force_rebuild=True,
                        structured_parser_mode="llm_enhanced", toc_hint=hint, enhancement_intent=intent)
                    assert result.structured_document_ready and provider.calls == 1, "explicit enhancement failed"
                    assert_coverage(store.load(None), text)
                    before = store.path.read_bytes()
                    provider.transform = lambda payload, context: {}
                    result = pipeline.prepare("fixture", mode="base", force_rebuild=True,
                        structured_parser_mode="llm_enhanced", toc_hint=hint, enhancement_intent=intent)
                    assert not result.structured_document_ready, "invalid candidate accepted"
                    assert store.path.read_bytes() == before, "failed enhancement changed stored structure"
                    attempts += 2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, "source PDF changed"
    return {"source_sha256": digest, "pages": len(extracted.pages), "raw_chars": len(text),
            "common_hints": common_shapes, "enhancement_attempts": attempts, "interactions": checked,
            "source_unchanged": True, "ocr_calls": 0, "live_llm_calls": 0}


def postgres():
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    tests = importlib.import_module("test_postgres_manual_reparse_duplicate_chapter_order")
    dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
    schema = "native_accept_" + uuid4().hex
    scoped = make_conninfo(dsn, options=f"-c search_path={schema}")
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(scoped) as connection:
                for path in sorted((Path(__file__).resolve().parents[1] / "db/migrations").glob("*.sql")):
                    connection.execute(path.read_text())
            with patch.dict(os.environ, {"DEEP_READER_TEST_POSTGRES_DSN": scoped}):
                tests.test_duplicate_chapter_order_reparse_replaces_current_hierarchy()
                tests.test_failed_replacement_rolls_back_current_hierarchy()
            repository = tests.PostgresStructuredDocumentArtifactRepository(tests.PostgresStructuredDocumentStore(scoped))
            name = "artifact-lifecycle"
            original = tests._document(doc_name=name, title="Initial", chapter_ids=["original"])
            replacement = tests._document(doc_name=name, title="Replacement", chapter_ids=["replacement"])
            repository.save_reparsed_document(original, doc_name=name)
            with psycopg.connect(scoped) as connection:
                doc_id = connection.execute("SELECT id FROM documents WHERE document_name=%s", (name,)).fetchone()[0]
                for kind in ("analysis", "quiz", "critical_thinking_session"):
                    connection.execute("""INSERT INTO artifacts
                        (document_id, artifact_type, target_type, target_id, source_structure_version, payload)
                        VALUES (%s, %s, 'document', %s, 1, '{"sentinel":"preserve on failure"}')""", (doc_id, kind, doc_id))
            failing = tests.PostgresStructuredDocumentArtifactRepository(tests._FailingReplacementStore(scoped))
            try:
                failing.save_reparsed_document(replacement, doc_name=name)
            except RuntimeError as error:
                assert str(error) == "forced replacement failure"
            else:
                raise AssertionError("replacement failure did not propagate")
            with psycopg.connect(scoped) as connection:
                assert connection.execute("SELECT count(*) FROM artifacts WHERE document_id=%s", (doc_id,)).fetchone()[0] == 3
                assert connection.execute("SELECT current_structure_version FROM documents WHERE id=%s", (doc_id,)).fetchone()[0] == 1
            repository.save_reparsed_document(replacement, doc_name=name)
            with psycopg.connect(scoped) as connection:
                assert connection.execute("SELECT count(*) FROM artifacts WHERE document_id=%s", (doc_id,)).fetchone()[0] == 0
                assert connection.execute("SELECT current_structure_version FROM documents WHERE id=%s", (doc_id,)).fetchone()[0] == 2
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    return {"version_replacement": "passed", "rollback": "passed", "artifact_types": 3,
            "failed_replacement_preserves_artifacts": True, "accepted_replacement_clears_artifacts": True,
            "schema_removed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", default=[])
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--regressions", action="store_true")
    args = parser.parse_args()
    report = {"provider": "deterministic fake", "live_llm": "not tested", "live_ui": "separate browser suite"}
    # Existing debug prints may contain fixture metadata; only bounded results are emitted.
    with redirect_stdout(io.StringIO()):
        report["regressions"] = regressions() if args.regressions else "not run"
        report["pdfs"] = [accept_pdf(path) for path in args.pdf]
        report["postgres"] = postgres() if args.postgres else "not run"
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
