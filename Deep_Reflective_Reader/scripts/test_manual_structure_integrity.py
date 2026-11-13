"""Manual reparse audit: real file/SQL owners, canonical coverage and commit guards.

PDF mode records are deterministic accepted fixtures, not live OCR/LLM validation.
"""
import os
import unittest
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import test_preparation_runtime_safety as safety
from app.section_task_coordinator import (
    SectionTaskCoordinator, ManualStructurePlanDTO, ManualStructureEntryDTO, ManualStructureAnchorDTO,
)
from doc_loaders.extraction_identity import read_extraction_record
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, CanonicalOcrSpan
from section_tasks.note_interaction_service import NoteInteractionService

TEXT = "  Preface.\nChapter one introduction.\nSection body.\nTail with whitespace.\n  "


def entry(start, end, *, level=1, page=False):
    anchor = (ManualStructureAnchorDTO(anchor_type="page_range", page_start_index=start,
              page_end_index=end) if page else ManualStructureAnchorDTO(
              anchor_type="char_range", char_start=start, char_end=end))
    return ManualStructureEntryDTO(title=f"Title {start}", level=level, anchor=anchor)


class FileManualIntegrityTests(unittest.TestCase):
    fixture = safety.RuntimeSafetyTests

    @contextmanager
    def prepared(self, kind="txt", mode="fast"):
        f = self.fixture()
        f.setUp()
        try:
            f.path.write_text(TEXT)
            text = f.seed(kind, mode)
            repo = f.repository()
            coordinator = SectionTaskCoordinator(document_preparation_pipeline=f.pipeline,
                document_artifact_repository=repo, document_profile_store=None,
                chapter_summary_service=None, chapter_quiz_service=None, task_unit_resolver=None,
                enhanced_parse_trigger_evaluator=None)
            yield f, repo, coordinator, text
        finally:
            f.doCleanups()

    def commit(self, coordinator, entries):
        return coordinator.commit_manual_structure_reparse(doc_name="book",
            manual_structure=ManualStructurePlanDTO(entries=entries))

    def assert_coverage(self, document, text):
        sections = [s for c in document.chapters for s in c.sections]
        self.assertEqual("".join(s.content for s in sections), text)
        cursor = 0
        for index, section in enumerate(sections):
            self.assertEqual(section.char_start, cursor)
            self.assertEqual(section.section_index, index)
            self.assertEqual(section.content, text[section.char_start:section.char_end])
            self.assertGreater(section.char_end, cursor)
            cursor = section.char_end
        self.assertEqual(cursor, len(text))
        self.assertEqual(len({s.section_id for s in sections}), len(sections))

    def test_character_gap_matrix_roundtrips_complete_sections(self):
        for shape in ("prefix", "middle", "tail", "sections", "chapter_only", "whitespace"):
            with self.subTest(shape=shape), self.prepared() as (f, repo, coordinator, text):
                n = len(text)
                plans = {
                    "prefix": [entry(8, n)],
                    "middle": [entry(0, 12), entry(22, n)],
                    "tail": [entry(0, n - 6)],
                    "sections": [entry(8, n-6), entry(16, 22, level=2), entry(30, 36, level=2)],
                    "chapter_only": [entry(0, n)],
                    "whitespace": [entry(2, n-3)],
                }
                result = self.commit(coordinator, plans[shape])
                self.assertTrue(result.success, result.error)
                document = repo.load_document("book")
                self.assert_coverage(document, text)
                self.assertTrue(f.pipeline.prepare_and_load("book").assets.bundle_ready)

    def test_page_gap_matrix_roundtrips_complete_sections(self):
        # Persisted synthetic page spans bind the same accepted canonical text.
        for shape in ("prefix", "middle", "tail", "sections", "chapter_only"):
            with self.subTest(shape=shape), self.prepared() as (f, repo, coordinator, text):
                edges = [0, 8, 16, 24, 32, 40, len(text)]
                spans = [dict(page_index=i, char_start=a, char_end=b, page_label=str(i+1))
                         for i, (a, b) in enumerate(zip(edges, edges[1:]))]
                current = repo.load_document("book")
                repo.save_document(replace(current, parse_provenance={**current.parse_provenance,
                    "source_page_spans": spans}), "book")
                plans = {
                    "prefix": [entry(1, 5, page=True)],
                    "middle": [entry(0, 1, page=True), entry(3, 5, page=True)],
                    "tail": [entry(0, 4, page=True)],
                    "sections": [entry(1, 4, page=True), entry(2, 2, page=True, level=2)],
                    "chapter_only": [entry(0, 5, page=True)],
                }
                result = self.commit(coordinator, plans[shape])
                self.assertTrue(result.success, result.error)
                current = repo.load_document("book")
                self.assert_coverage(current, text)
                self.assertEqual(len(f.pipeline.load_task_layout_page_boundaries("book", document=current)), 6)

    def test_accepted_modes_keep_source_proof_and_zero_generation_reads(self):
        for kind, mode in (("txt", "fast"), ("pdf", "fast"), ("pdf", "auto"), ("pdf", "ocr")):
            with self.subTest(kind=kind, mode=mode), self.prepared(kind, mode) as (f, repo, coordinator, text):
                current = repo.load_document("book")
                identity, record = read_extraction_record(current.parse_provenance)
                if mode in ("auto", "ocr"):
                    ocr = ActualOcrEvidence(schema_version="cleanup_ocr_evidence_v1",
                        extraction_identity=identity, text_sha256=record["text_sha256"],
                        spans=(CanonicalOcrSpan(char_start=0, char_end=len(text), page_index=0),))
                    repo.save_document(replace(current, parse_provenance={**current.parse_provenance,
                        "actual_ocr_evidence": ocr.model_dump(mode="json")}), "book")
                notes = NoteInteractionService(repo)
                state = notes.read("book", target_level="document")
                notes.save("book", target_level="document", text="Old hierarchy note",
                    expected_revision=0, mutation_id=uuid4().hex, target_token=state["target_token"])
                before = repo.load_document("book")
                files = f.files()
                profile_config = safety.FaissStorageConfig("book")
                profile = f.pipeline.profile_store.load(profile_config)
                with ExitStack() as stack:
                    for owner, name in ((f.loader, "open_source"), (f.loader, "extract"),
                            (f.pipeline.language_detector, "detect"), (f.pipeline.profile_builder, "build"),
                            (f.pipeline.faiss_index_builder, "build_from_parsed_document")):
                        stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
                    result = self.commit(coordinator, [entry(0, len(text))])
                    self.assertTrue(result.success, result.error)
                    current = repo.load_document("book")
                    self.assert_coverage(current, text)
                    self.assertTrue(f.pipeline.prepare_and_load("book").assets.bundle_ready)
                    evidence = f.pipeline.load_manual_structure_source_evidence("book")
                    self.assertEqual(evidence.raw_text, text)
                    pages = f.pipeline.load_task_layout_page_boundaries("book", document=current)
                    self.assertEqual(len(pages), 1 if kind == "pdf" else 0)
                    self.assertEqual(current.to_dict(), repo.load_document("book").to_dict())
                self.assertEqual(f.files(), files)
                self.assertEqual(profile, f.pipeline.profile_store.load(profile_config))
                for key in ("extraction", "actual_ocr_evidence", "source_page_spans"):
                    self.assertEqual(current.parse_provenance.get(key), before.parse_provenance.get(key))
                self.assertNotEqual(current.parse_provenance.get("interaction_epoch"), before.parse_provenance.get("interaction_epoch"))
                self.assertEqual(current.parse_provenance["storage_structure_version"], before.parse_provenance.get("storage_structure_version", 1) + 1)
                self.assertNotIn("reading_content", current.parse_provenance)
                self.assertIsNone(notes.read("book", target_level="document").get("note"))

    def test_stale_evidence_rejects_source_version_epoch_and_reading_changes(self):
        for change in ("bytes", "deleted", "same_bytes_replace", "reparse", "reading_revision", "page_evidence", "rename", "reimport"):
            with self.subTest(change=change), self.prepared() as (f, repo, coordinator, text):
                if change == "reading_revision":
                    from shared.task_unit_model import TaskUnit
                    current = repo.load_document("book")
                    chapter = current.chapters[0]; section = chapter.sections[0]
                    unit = TaskUnit(unit_id="manual-test-unit", title=section.title,
                        container_title=chapter.title, content=section.content,
                        source_section_ids=[section.section_id], is_fallback_generated=False,
                        parent_section_id=section.section_id)
                    repo.update_section_task_units("book", section.section_id, [unit])
                evidence = f.pipeline.load_manual_structure_source_evidence("book")
                current = repo.load_document("book")
                if change == "bytes":
                    f.path.write_text(text + " changed")
                elif change == "deleted":
                    f.path.unlink()
                elif change == "same_bytes_replace":
                    saved = f.path.read_bytes(); f.path.unlink(); f.path.write_bytes(saved)
                elif change == "reparse":
                    repo.save_reparsed_document(current, "book")
                elif change == "reading_revision":
                    from document_structure.reading_content import ReadingContentService
                    service = ReadingContentService(repo); state = service.read("book")
                    service.mutate("book", action="edit", expected_revision=state["revision"],
                        source_token=state["source_token"], mutation_id=uuid4().hex,
                        block_id=state["blocks"][0]["block_id"], text="A corrected reading paragraph.")
                elif change == "page_evidence":
                    repo.save_document(replace(current, parse_provenance={**current.parse_provenance,
                        "source_page_spans": [dict(page_index=0, char_start=0, char_end=len(text))]}), "book")
                elif change == "rename":
                    repo.rename_document("book", "renamed", lambda document, new_name: replace(document, document_id=new_name))
                elif change == "reimport":
                    repo.delete_document("book")
                    repo.save_reparsed_document(current, "book")
                name = "renamed" if change == "rename" else "book"
                before = repo.load_document(name).to_dict(); files = f.files()
                with patch.object(f.pipeline, "load_manual_structure_source_evidence", return_value=evidence):
                    result = self.commit(coordinator, [entry(0, len(text))])
                self.assertFalse(result.success)
                self.assertEqual(result.status_code, 409, result.error)
                self.assertEqual(repo.load_document(name).to_dict(), before)
                self.assertEqual(f.files(), files)

    def test_source_change_at_publication_preserves_old_document(self):
        with self.prepared() as (f, repo, coordinator, text):
            before = repo.load_document("book").to_dict(); files = f.files()
            if self.fixture is safety.RuntimeSafetyTests:
                import document_structure.structured_document_artifact_repository as owner
                original = owner.os.fsync
                def interrupt(fd):
                    original(fd); f.path.write_text(text + " changed at publication")
                target, name = owner.os, "fsync"
            else:
                original = repo.store.save
                def interrupt(*args, **kwargs):
                    result = original(*args, **kwargs)
                    f.path.write_text(text + " changed at publication")
                    return result
                target, name = repo.store, "save"
            with patch.object(target, name, side_effect=interrupt):
                result = self.commit(coordinator, [entry(0, len(text))])
            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 409, result.error)
            self.assertEqual(before, repo.load_document("book").to_dict())
            self.assertEqual(files, f.files())

    def test_persistence_failure_rolls_back_and_retry_succeeds(self):
        with self.prepared() as (f, repo, coordinator, text):
            before = repo.load_document("book").to_dict(); files = f.files()
            if self.fixture is safety.RuntimeSafetyTests:
                import document_structure.structured_document_artifact_repository as owner
                def fail(*args, **kwargs):
                    raise OSError("after temporary write")
                target, name = owner.os, "fsync"
            else:
                original = repo.store.save
                def fail(*args, **kwargs):
                    original(*args, **kwargs)
                    raise OSError("after SQL replacement")
                target, name = repo.store, "save"
            with patch.object(target, name, side_effect=fail):
                result = self.commit(coordinator, [entry(0, len(text))])
            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 500)
            self.assertEqual(before, repo.load_document("book").to_dict())
            self.assertEqual(files, f.files())
            self.assertFalse(list(Path(f.temp.name).rglob("*.tmp")))
            self.assertTrue(self.commit(coordinator, [entry(0, len(text))]).success)

    def test_commit_uses_server_proof_and_rejects_incomplete_draft(self):
        from document_structure.manual_structure_document_builder import build_manual_structure_document_draft
        from document_structure.manual_structure_projection import ManualStructureEntry, ManualStructureAnchor
        from shared.task_artifacts import DocumentTaskArtifacts
        for invalid in (False, True):
            with self.subTest(invalid=invalid), self.prepared() as (f, repo, coordinator, text):
                before = repo.load_document("book")
                evidence = f.pipeline.load_manual_structure_source_evidence("book")
                draft = build_manual_structure_document_draft(document_id="book", title="manual",
                    raw_text=text, entries=[ManualStructureEntry(title="Whole book", level=1,
                    anchor=ManualStructureAnchor(anchor_type="char_range", char_start=0,
                    char_end=len(text)))]).document
                draft = replace(draft, parse_provenance={**draft.parse_provenance,
                    "extraction": {"forged": True}, "reading_content": {"revision": 100},
                    "actual_ocr_evidence": {"forged": True}, "source_review_mapping": {"stale": True}},
                    document_task_artifacts=DocumentTaskArtifacts(metadata={"note": "stale"}))
                if invalid:
                    chapter = draft.chapters[0]; section = chapter.sections[0]
                    draft = replace(draft, chapters=[replace(chapter, sections=[replace(section,
                        content=text[1:], char_start=1)])])
                    with self.assertRaises(ValueError):
                        repo.commit_manual_structure(draft, doc_name="book", validate_current=evidence.validate_current)
                    self.assertEqual(before.to_dict(), repo.load_document("book").to_dict())
                else:
                    repo.commit_manual_structure(draft, doc_name="book", validate_current=evidence.validate_current)
                    current = repo.load_document("book")
                    self.assertEqual(current.parse_provenance["extraction"], before.parse_provenance["extraction"])
                    self.assertNotIn("reading_content", current.parse_provenance)
                    self.assertNotIn("actual_ocr_evidence", current.parse_provenance)
                    self.assertNotEqual(current.parse_provenance.get("source_review_mapping"), {"stale": True})
                    self.assertIsNone(current.document_task_artifacts)
                    self.assertTrue(f.pipeline.prepare_and_load("book").assets.bundle_ready)

    def test_missing_commit_evidence_cannot_write(self):
        with self.prepared() as (f, repo, coordinator, text):
            evidence = replace(f.pipeline.load_manual_structure_source_evidence("book"), validate_current=None)
            before = repo.load_document("book").to_dict()
            with patch.object(f.pipeline, "load_manual_structure_source_evidence", return_value=evidence):
                result = self.commit(coordinator, [entry(0, len(text))])
            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 422)
            self.assertEqual(before, repo.load_document("book").to_dict())


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "requires disposable PostgreSQL DSN")
class PostgresManualIntegrityTests(FileManualIntegrityTests):
    fixture = safety.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)


if __name__ == "__main__":
    unittest.main()
