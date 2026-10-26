"""One-commit cleanup contracts in temporary files and disposable PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from doc_loaders.extraction_identity import ExtractionResult, read_extraction_record
from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot
from document_preparation.structure_enhancement_candidate import StructureEnhancementCandidate
from document_structure.ocr_cleanup_contract import BoundCleanupProposal
from document_structure.ocr_cleanup_overlay import CleanupApplication, canonical_text
from document_structure.reading_content import (
    ReadingContentService, ReadingContentConflict, digest, project_document, state_for,
)
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from section_tasks.reading_target_resolver import ReadingTargetResolver
from shared.task_artifacts import DocumentTaskArtifacts
from shared.task_unit_model import TaskUnit
from scripts.test_ocr_cleanup_contract import binding, item, proposal
from scripts.test_ocr_cleanup_validation import fixture, BODY


class CleanupOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "book.pdf"
        self.path.write_bytes(b"immutable fixture, not parsed")
        self.namespace = uuid4().hex
        self.repo = self.repository()
        value, _, document = fixture("@@??\n\n" + BODY)
        identity = replace(value.identity, source_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.text = value.canonical_text
        region = {"id": "r1", "page_index": 0, "raw_candidate": "@@??",
                  "canonical_span": {"char_start": 0, "char_end": 4, "text_sha256": value.text_sha256}}
        section = document.chapters[0].sections[0]
        unit = TaskUnit(unit_id="u1", title=None, container_title=None, content=self.text,
                        source_section_ids=[section.section_id], is_fallback_generated=False,
                        parent_section_id=section.section_id)
        section = replace(section, task_units=[unit])
        document = replace(document, source_path=str(self.path),
            chapters=[replace(document.chapters[0], sections=[section])],
            parse_provenance={**document.parse_provenance, "extraction": identity.record(self.text),
                "effective_parser_mode": "llm_enhanced", "source": "validated_v2_plan",
                "fallback_used": False, "source_coverage": "complete",
                "source_page_spans": [{"page_index": 0, "char_start": 0, "char_end": len(self.text)}],
                "source_review": {"source_sha256": identity.source_sha256, "regions": [region]}},
            document_task_artifacts=DocumentTaskArtifacts(metadata={"note": {"keep": "annotation"},
                "public_summary": {"obsolete": True}, "reading_interaction_artifacts": {"obsolete": True},
                "ancestor_summary_artifacts": {"obsolete": True}}))
        self.repo.save_document(document, "book")
        self.service = ReadingContentService(self.repo)

    def repository(self):
        return StructuredDocumentArtifactRepository(base_dir=self.temp.name)

    def snapshot(self):
        document = self.repo.load_document("book")
        return document, document.parse_provenance.get("storage_structure_version", 1), digest(document.to_dict())

    def prepare(self, kind="unchanged", items=None, operation="op-1"):
        current, version, fingerprint = self.snapshot()
        identity, _ = read_extraction_record(current.parse_provenance)
        source = ExtractionResult(identity, canonical_text(current))
        snapshot = AcceptedSourceSnapshot(self.namespace, "book", version, identity,
            source.canonical_text, source.text_sha256, source.text_length, current.language)
        old = current.chapters[0]
        template = old.sections[0]
        cuts = (0, 6, source.text_length) if kind == "structure_changed" else (0, source.text_length)
        sections = [replace(template, section_id=f"candidate-{index}", section_index=index,
            parent_chapter_id="candidate-chapter", content=source.canonical_text[start:end],
            char_start=start, char_end=end, task_units=[], task_artifacts=None,
            title="New heading" if kind == "titles_only" else template.title)
            for index, (start, end) in enumerate(zip(cuts, cuts[1:]))]
        document = replace(current, raw_text=source.canonical_text, document_task_artifacts=None,
            chapters=[replace(old, chapter_id="candidate-chapter", sections=sections, task_artifacts=None)],
            parse_provenance={"extraction": identity.record(source.canonical_text),
                "effective_parser_mode": "llm_enhanced", "source": "validated_v2_plan",
                "fallback_used": False, "source_coverage": "complete", "boundary_origins": [
                    {"char_start": section.char_start, "char_end": section.char_end, "title_origin": "generated"}
                    for section in sections]})
        scope = binding(source, document_id="book", namespace=self.namespace,
            structure_version=version, structure_fingerprint=fingerprint,
            expected_reading_revision=state_for(current).get("revision", 0),
            source_representation="canonical_text_v1", operation_id=operation)
        application = CleanupApplication(BoundCleanupProposal.bind(source=source, binding=scope,
            proposal=proposal([item(quote="@@??")] if items is None else items)))
        return StructureEnhancementCandidate(snapshot, document), application

    def commit(self, candidate, application):
        return self.repo.commit_cleanup_enhancement("book", candidate=candidate,
            cleanup=application, source_path=self.path)

    def change(self, action, **kwargs):
        state = self.service.read("book")
        return self.service.mutate("book", action=action, expected_revision=state["revision"],
            source_token=state["source_token"], mutation_id=uuid4().hex, **kwargs)

    def content(self):
        return ReadingTargetResolver().resolve(document=self.repo.load_document("book"), target_level="document").content

    def test_cleanup_only_preserves_source_ids_notes_and_invalidates_generated(self):
        original, version, _ = self.snapshot()
        result = self.commit(*self.prepare())
        current, after_version, _ = self.snapshot()
        self.assertEqual(result.change_kind, "unchanged")
        self.assertEqual(result.cleanup["applied_count"], 1)
        self.assertEqual(after_version, version)
        self.assertEqual(current.chapters, original.chapters)
        self.assertEqual(canonical_text(current), self.text)
        self.assertNotIn("@@??", self.content())
        self.assertEqual(current.document_task_artifacts.metadata, {"note": {"keep": "annotation"}})
        self.assertEqual(current.parse_provenance["source_page_spans"], original.parse_provenance["source_page_spans"])

    def test_titles_only_preserves_unit_and_block_ids(self):
        before = self.service.read("book")["blocks"]
        result = self.commit(*self.prepare("titles_only"))
        after = self.service.read("book")["blocks"]
        self.assertEqual(result.change_kind, "titles_only")
        self.assertEqual([b["block_id"] for b in before], [b["block_id"] for b in after])
        self.assertTrue(after[0]["deleted"])
        self.assertFalse(after[1]["deleted"])

    def test_structural_commit_before_units_and_undo_does_not_restore_hierarchy(self):
        result = self.commit(*self.prepare("structure_changed"))
        current, version, _ = self.snapshot()
        self.assertEqual(result.change_kind, "structure_changed")
        self.assertEqual(version, 2)
        self.assertEqual(len(current.chapters[0].sections), 2)
        self.assertFalse(current.chapters[0].sections[0].task_units)
        self.assertNotIn("@@??", self.content())
        self.assertFalse(current.document_task_artifacts and current.document_task_artifacts.metadata.get("note"))
        self.change("undo")
        self.assertIn("@@??", self.content())
        self.assertEqual(len(self.repo.load_document("book").chapters[0].sections), 2)
        self.assertEqual(self.service.read("book")["revision"], 2)

    def test_replay_no_duplicate_revision_audit_or_write(self):
        command = self.prepare("structure_changed")
        first = self.commit(*command)
        before = self.snapshot()
        second = self.commit(*command)
        self.assertEqual(first.cleanup, second.cleanup)
        self.assertEqual(before, self.snapshot())
        altered = replace(command[1], plan=command[1].plan.model_copy(update={"proposal": proposal([])}))
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_operation_conflict"):
            self.commit(command[0], altered)

    def test_source_change_rejects_without_writing(self):
        command = self.prepare()
        before = self.snapshot()
        self.path.write_bytes(b"new source")
        with self.assertRaises(ValueError):
            self.commit(*command)
        self.assertEqual(before, self.snapshot())

    def test_manual_edit_during_generation_rejects_stale_commit(self):
        command = self.prepare()
        self.change("edit", block_id=self.service.read("book")["blocks"][1]["block_id"], text="User edit")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.commit(*command)
        self.assertEqual(before, self.snapshot())

    def test_user_edits_survive_structural_change_and_cleanup_undo(self):
        block = self.service.read("book")["blocks"][1]
        self.change("edit", block_id=block["block_id"], text="User-authoritative replacement")
        self.commit(*self.prepare("structure_changed"))
        self.assertIn("User-authoritative replacement", self.content())
        self.assertNotIn("@@??", self.content())
        self.change("undo")
        self.assertIn("User-authoritative replacement", self.content())
        self.assertIn("@@??", self.content())

    def test_intersecting_manual_edit_is_not_deleted(self):
        block = self.service.read("book")["blocks"][0]
        self.change("edit", block_id=block["block_id"], text="Meaningful correction")
        result = self.commit(*self.prepare())
        self.assertEqual(result.cleanup["status"], "skipped")
        self.assertIn("Meaningful correction", self.content())

    def test_manual_edits_survive_structural_commit_when_cleanup_skips(self):
        block = self.service.read("book")["blocks"][1]
        self.change("edit", block_id=block["block_id"], text="Retained correction")
        result = self.commit(*self.prepare("structure_changed", items=[item(quote="not found")]))
        self.assertEqual(result.cleanup["status"], "skipped")
        self.assertIn("Retained correction", self.content())

    def test_application_mapping_failure_preserves_uncleaned_candidate(self):
        current = self.repo.load_document("book")
        chapter = current.chapters[0]
        section = chapter.sections[0]
        unit = replace(section.task_units[0], content="Unmappable legacy unit", content_blocks=[])
        self.repo.save_document(replace(current, chapters=[replace(chapter, sections=[
            replace(section, task_units=[unit])])]), "book")
        result = self.commit(*self.prepare())
        self.assertEqual(result.cleanup["status"], "skipped")
        self.assertEqual(self.service.read("book")["revision"], 0)
        self.assertIn("Unmappable legacy unit", self.content())
        self.assertEqual(len(self.service.read("book")["regions"]), 1)

    def test_cumulative_cleanup_cannot_erase_remaining_document(self):
        current = self.repo.load_document("book")
        # Previously accepted deletion leaves only the first noise fragment visible.
        state = {"revision": 1, "cleanup_source_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                 "cleanup_spans": [{"char_start": 6, "char_end": len(self.text), "quote": self.text[6:]}]}
        self.repo.mutate_reading_content("book", lambda document: (
            replace(document, parse_provenance={**document.parse_provenance, "reading_content": state}), None))
        result = self.commit(*self.prepare())
        self.assertEqual(result.cleanup["status"], "skipped")
        self.assertIn("@@??", self.content())

    def test_cleanup_log_is_correlated_without_source_or_operation_text(self):
        with self.assertLogs("document_structure.ocr_cleanup_overlay", level="INFO") as logs:
            self.commit(*self.prepare())
        output = "\n".join(logs.output)
        self.assertIn("stage=staged", output)
        self.assertIn("elapsed_ms=", output)
        self.assertNotIn("@@??", output)
        self.assertNotIn("op-1", output)
        self.assertNotIn(BODY, output)

    def test_late_generated_write_is_rejected_after_cleanup(self):
        from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact, ReadingInteractionRequest
        from section_tasks.interaction_policy_identity import policy_metadata
        target = ReadingTargetResolver().resolve(document=self.repo.load_document("book"), target_level="document")
        request = ReadingInteractionRequest(target=target, interaction_type="analysis", context_metadata={"doc_name": "book"})
        artifact = ReadingInteractionArtifact.from_target(target=target, interaction_type="analysis", status="completed",
            payload={"interpretation": "Obsolete generated text"},
            metadata={**policy_metadata(request), "context": {"doc_name": "book"}})
        self.commit(*self.prepare())
        with self.assertRaisesRegex(ValueError, "stale_interaction_write"):
            DocumentReadingInteractionArtifactStore(self.repo).save_analysis_artifact(artifact)

    def test_late_structure_save_cannot_resurrect_old_units(self):
        old = self.repo.load_document("book")
        self.commit(*self.prepare("structure_changed"))
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.repo.save_document(old, "book")
        self.assertEqual(before, self.snapshot())

    def test_concurrent_distinct_plans_one_winner(self):
        commands = [self.prepare(operation=f"op-{index}") for index in range(2)]
        def run(command):
            try:
                self.commit(*command)
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sum(pool.map(run, commands)), 1)
        self.assertEqual(self.service.read("book")["revision"], 1)

    def test_stale_review_source_is_not_auto_resolved(self):
        current = self.repo.load_document("book")
        self.repo.save_document(replace(current, parse_provenance={**current.parse_provenance,
            "source_review": {**current.parse_provenance["source_review"], "source_sha256": "0" * 64}}), "book")
        self.commit(*self.prepare())
        self.assertEqual(len(self.service.read("book")["regions"]), 1)

    def test_full_review_resolution_and_undo(self):
        self.assertEqual(len(self.service.read("book")["regions"]), 1)
        self.commit(*self.prepare())
        self.assertEqual(self.service.read("book")["regions"], [])
        self.assertEqual(self.service.read("book")["disposition"], "completed")
        self.assertFalse(self.repo.list_documents()[0].review_required)
        self.assertEqual(len(self.repo.load_document("book").parse_provenance["source_review"]["regions"]), 1)
        self.change("undo")
        self.assertEqual(len(self.service.read("book")["regions"]), 1)

    def test_partial_missing_or_unmapped_region_remains(self):
        current = self.repo.load_document("book")
        review = current.parse_provenance["source_review"]
        regions = [review["regions"][0], {"id": "empty", "raw_candidate": ""},
                   {"id": "unmapped", "raw_candidate": "@@??"},
                   {"id": "partial", "raw_candidate": self.text[:10], "canonical_span": {
                       "char_start": 0, "char_end": 10,
                       "text_sha256": hashlib.sha256(self.text.encode()).hexdigest()}}]
        self.repo.save_document(replace(current, parse_provenance={**current.parse_provenance,
            "source_review": {**review, "regions": regions}}), "book")
        self.commit(*self.prepare())
        self.assertEqual({r["id"] for r in self.service.read("book")["regions"]}, {"empty", "unmapped", "partial"})

    def test_invalid_cleanup_keeps_valid_split_candidate(self):
        result = self.commit(*self.prepare("structure_changed", items=[item(quote="absent")]))
        self.assertEqual(result.cleanup["status"], "skipped")
        self.assertEqual(len(self.repo.load_document("book").chapters[0].sections), 2)
        self.assertIn("@@??", self.content())
        self.assertEqual(len(self.service.read("book")["regions"]), 1)

    def test_all_five_targets_use_overlay_and_empty_block_not_resurrected(self):
        before = self.service.read("book")["blocks"]
        self.commit(*self.prepare())
        document = self.repo.load_document("book")
        block = before[1]
        for scope in ({"target_level": "document"},
                      {"target_level": "chapter", "chapter_id": block["chapter_id"]},
                      {"target_level": "section", "section_id": block["section_id"]},
                      {"target_level": "task_unit", "task_unit_id": block["task_unit_id"]},
                      {"target_level": "content_block", "task_unit_id": block["task_unit_id"],
                       "content_block_id": block["block_id"]}):
            self.assertNotIn("@@??", ReadingTargetResolver().resolve(document=document, **scope).content)
        with self.assertRaises(ValueError):
            ReadingTargetResolver().resolve(document=document, target_level="content_block",
                task_unit_id=before[0]["task_unit_id"], content_block_id=before[0]["block_id"])

    def test_new_manual_action_supersedes_cleanup_undo(self):
        self.commit(*self.prepare())
        block = self.service.read("book")["blocks"][1]
        self.change("edit", block_id=block["block_id"], text="Newest manual text")
        self.change("undo")
        self.assertNotIn("@@??", self.content())
        self.assertNotIn("Newest manual text", self.content())
        self.assertFalse(self.service.read("book")["can_undo"])

    def test_new_units_after_commit_keep_canonical_ids_and_apply_spans(self):
        self.commit(*self.prepare("structure_changed"))
        current = self.repo.load_document("book")
        chapter = current.chapters[0]
        sections = [replace(s, task_units=[TaskUnit(unit_id=f"new-{i}", title=None, container_title=None,
            content=s.content, source_section_ids=[s.section_id], is_fallback_generated=False,
            parent_section_id=s.section_id)]) for i, s in enumerate(chapter.sections)]
        self.repo.save_document(replace(current, chapters=[replace(chapter, sections=sections)]), "book")
        self.assertNotIn("@@??", self.content())
        projected = project_document(self.repo.load_document("book"))
        self.assertEqual(projected.chapters[0].sections[0].content, "")
        self.assertEqual(projected.chapters[0].sections[0].task_units[0].content, "")

    def test_failed_final_file_write_preserves_everything(self):
        command = self.prepare("structure_changed")
        before = self.snapshot()
        with patch.object(self.repo, "_atomic_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.commit(*command)
        self.assertEqual(before, self.snapshot())

    def test_single_atomic_save_no_saved_candidate_reload(self):
        command = self.prepare("structure_changed")
        with patch.object(self.repo, "_atomic_save", wraps=self.repo._atomic_save) as save, \
                patch.object(self.repo, "load_document", wraps=self.repo.load_document) as load:
            self.commit(*command)
        self.assertEqual(save.call_count, 1)
        self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
