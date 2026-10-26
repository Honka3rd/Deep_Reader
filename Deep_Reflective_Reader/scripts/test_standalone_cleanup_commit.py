"""Content-only cleanup contract shared by file and isolated PostgreSQL tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

from doc_loaders.extraction_identity import ExtractionResult, read_extraction_record
from document_preparation.cleanup_input_adapter import adapt_document_cleanup_input
from document_structure.cleanup_operation import CleanupCommitCommand, StagedCleanup
from document_structure.cleanup_commit import RECEIPTS
from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.ocr_cleanup_planner import REMAINING_CLEANUP_POLICY_VERSION, REMAINING_CLEANUP_PROMPT_VERSION
from document_structure.ocr_cleanup_validation import validate_cleanup_input
from document_structure.reading_content import ReadingContentConflict, state_for, original_blocks
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from document_structure.structured_document import StructuredDocument
from section_tasks.reading_target_resolver import ReadingTargetResolver
from scripts import test_ocr_cleanup_overlay as legacy
from scripts.test_cleanup_input_adapter import accepted
from scripts.test_cleanup_operation_contract import binding
from scripts.test_ocr_cleanup_contract import item, proposal


class StandaloneCleanupCommitTests(unittest.TestCase):
    setUp = legacy.CleanupOverlayTests.setUp
    snapshot = legacy.CleanupOverlayTests.snapshot
    change = legacy.CleanupOverlayTests.change
    content = legacy.CleanupOverlayTests.content

    def repository(self):
        return StructuredDocumentArtifactRepository(base_dir=self.temp.name, namespace=self.namespace)

    def prepare(self, items=None, operation="standalone-1"):
        document, version, _ = self.snapshot()
        identity, _ = read_extraction_record(document.parse_provenance)
        source = ExtractionResult(identity, canonical_text(document))
        baseline = replace(accepted(document, source), namespace=self.namespace, structure_version=version)
        value = adapt_document_cleanup_input(document=document, snapshot=baseline,
                                              expected_reading_revision=state_for(document).get("revision", 0))
        plan = proposal([item(quote="@@??")] if items is None else items)
        result = validate_cleanup_input(plan, snapshot=value, expected_fingerprint=value.fingerprint())
        scope = binding(value, operation_id=operation, prompt_version=REMAINING_CLEANUP_PROMPT_VERSION,
                        policy_version=REMAINING_CLEANUP_POLICY_VERSION)
        return CleanupCommitCommand(snapshot=value, staged=StagedCleanup(binding=scope, proposal=plan,
            deletions=result.deletions, rejections=result.rejections))

    def lookup(self, operation="standalone-1"):
        return self.repo.lookup_cleanup(namespace=self.namespace, document_id="book", operation_id=operation)

    def fail_write(self):
        return patch.object(self.repo, "_atomic_save", side_effect=RuntimeError("injected write failure"))

    def test_content_only_preserves_identity_source_notes_and_pages(self):
        before, version, _ = self.snapshot()
        before_blocks = [b.block_id for _, _, _, b in original_blocks(before)]
        result = self.repo.commit_cleanup(self.prepare())
        current, after_version, _ = self.snapshot()
        self.assertEqual(result.outcome.status, "completed")
        self.assertEqual(result.outcome.reading_revision, 1)
        self.assertEqual(after_version, version)
        self.assertEqual(current.chapters, before.chapters)
        self.assertEqual(before_blocks, [b.block_id for _, _, _, b in original_blocks(current)])
        self.assertEqual(canonical_text(current), canonical_text(before))
        for key in ("extraction", "source_page_spans", "source_review"):
            self.assertEqual(current.parse_provenance[key], before.parse_provenance[key])
        self.assertEqual(current.document_task_artifacts.metadata, {"note": {"keep": "annotation"}})
        self.assertNotIn("@@??", self.content())
        self.assertEqual(self.lookup().outcome, result.outcome)

    def test_noop_receipt_preserves_revision_generated_and_undo(self):
        before = self.repo.load_document("book")
        result = self.repo.commit_cleanup(self.prepare([]))
        current = self.repo.load_document("book")
        self.assertEqual(result.outcome.status, "no_changes")
        self.assertEqual(result.outcome.reading_revision, 0)
        self.assertEqual(current.chapters, before.chapters)
        self.assertEqual(current.document_task_artifacts, before.document_task_artifacts)
        self.assertIsNone(state_for(current).get("undo"))
        self.assertIsNotNone(self.lookup())

    def test_partial_and_fully_rejected_plans(self):
        result = self.repo.commit_cleanup(self.prepare([item(quote="@@??"), item(quote="missing")]))
        self.assertEqual((result.outcome.status, result.outcome.applied_count, result.outcome.skipped_count),
                         ("partial", 1, 1))
        before = self.snapshot()
        result = self.repo.commit_cleanup(self.prepare([item(quote="missing")], operation="second"))
        self.assertEqual(result.outcome.status, "skipped")
        self.assertEqual(result.outcome.reading_revision, 1)
        self.assertEqual(self.snapshot()[0].document_task_artifacts, before[0].document_task_artifacts)

    def test_replay_and_lost_ack_no_second_write(self):
        command = self.prepare()
        first = self.repo.commit_cleanup(command)
        before = self.snapshot()
        with self.fail_write():
            second = self.repo.commit_cleanup(command)
        self.assertTrue(second.replayed)
        self.assertEqual(second.outcome, first.outcome)
        self.assertEqual(self.lookup(), second)
        self.assertEqual(before, self.snapshot())

    def test_same_operation_different_plan_conflicts(self):
        command = self.prepare()
        changed = self.prepare([])
        self.repo.commit_cleanup(command)
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_operation_conflict"):
            self.repo.commit_cleanup(changed)

    def test_actual_lost_ack_is_reconciled_without_reapplication(self):
        command = self.prepare()
        def lose_ack(binding, stage, started):
            if stage == "committed":
                raise RuntimeError("lost acknowledgement")
        with patch("document_structure.cleanup_commit.log_cleanup_commit", side_effect=lose_ack):
            with self.assertRaisesRegex(RuntimeError, "lost acknowledgement"):
                self.repo.commit_cleanup(command)
        receipt = self.lookup()
        self.assertEqual(receipt.outcome.reading_revision, 1)
        with self.fail_write():
            self.assertEqual(self.repo.commit_cleanup(command), receipt)

    def test_stale_layout_stamp_is_not_blessed(self):
        def setup(document):
            artifacts = document.document_task_artifacts
            return replace(document, parse_provenance={**document.parse_provenance, "reading_content": {"revision": 2}},
                document_task_artifacts=replace(artifacts, metadata={**artifacts.metadata,
                    "task_layout": {"reading_revision": 0}})), None
        self.repo.mutate_reading_content("book", setup)
        result = self.repo.commit_cleanup(self.prepare())
        self.assertTrue(result.reading_layout_required)
        self.assertEqual(self.repo.load_document("book").document_task_artifacts.metadata["task_layout"]["reading_revision"], 0)

    def test_undo_supersedes_commit_without_resurrecting_it(self):
        command = self.prepare()
        self.repo.commit_cleanup(command)
        self.assertEqual(self.service.read("book")["regions"], [])
        self.change("undo")
        before = self.snapshot()
        self.assertIn("@@??", self.content())
        self.assertEqual(len(self.service.read("book")["regions"]), 1)
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_operation_superseded"):
            self.repo.commit_cleanup(command)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.lookup().outcome.reading_revision, 1)  # historical, not current state

    def test_final_write_failure_rolls_back_receipt_content_and_invalidation(self):
        command = self.prepare()
        before = self.snapshot()
        with self.fail_write(), self.assertRaisesRegex(RuntimeError, "injected"):
            self.repo.commit_cleanup(command)
        self.assertEqual(before, self.snapshot())
        self.assertIsNone(self.lookup())

    def test_concurrent_duplicate_commits_apply_once(self):
        command = self.prepare()
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.repo.commit_cleanup(command), range(2)))
        self.assertEqual(sorted(r.replayed for r in results), [False, True])
        state = state_for(self.repo.load_document("book"))
        self.assertEqual(state["revision"], 1)
        self.assertEqual(len(state["audit"]), 1)
        self.assertEqual(len(state[RECEIPTS]), 1)

    def test_competing_new_operations_require_new_snapshot(self):
        commands = [self.prepare(operation="first"), self.prepare(operation="second")]
        def commit(command):
            try:
                return self.repo.commit_cleanup(command)
            except ReadingContentConflict as error:
                self.assertEqual(str(error), "cleanup_confirmation_stale")
                return None
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(commit, commands))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(self.service.read("book")["revision"], 1)

    def test_late_manual_edit_rejects_pending_cleanup(self):
        command = self.prepare()
        block = self.service.read("book")["blocks"][1]
        self.change("edit", block_id=block["block_id"], text="Manual annotation correction")
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_confirmation_stale"):
            self.repo.commit_cleanup(command)
        self.assertEqual(before, self.snapshot())
        result = self.repo.commit_cleanup(self.prepare(operation="fresh"))
        self.assertEqual(result.outcome.status, "completed")
        self.assertIn("Manual annotation correction", self.content())

    def test_late_generated_save_cannot_restore_invalidated_content(self):
        before = self.repo.load_document("book")
        self.repo.commit_cleanup(self.prepare())
        self.repo.save_document(before, "book")
        self.assertNotIn("@@??", self.content())
        self.assertNotIn("reading_interaction_artifacts", self.repo.load_document("book").document_task_artifacts.metadata)

    def test_pending_cleanup_conflicts_with_reparse(self):
        command = self.prepare()
        self.repo.save_reparsed_document(self.repo.load_document("book"), "book")
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_confirmation_stale"):
            self.repo.commit_cleanup(command)
        self.assertEqual(before, self.snapshot())

    def test_policy_and_forged_staging_rejected_under_lock(self):
        command = self.prepare()
        bad = command.model_copy(update={"staged": command.staged.model_copy(update={
            "binding": command.staged.binding.model_copy(update={"policy_version": "obsolete"})})})
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_policy_changed"):
            self.repo.commit_cleanup(bad)
        bad = command.model_copy(update={"staged": command.staged.model_copy(update={
            "deletions": (replace(command.staged.deletions[0], section_id="invented"),)})})
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_stage_changed"):
            self.repo.commit_cleanup(bad)
        self.assertIsNone(self.lookup())

    def test_namespace_and_source_changes_rejected(self):
        command = self.prepare()
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_identity_changed"):
            self.repo.lookup_cleanup(namespace="wrong", document_id="book", operation_id="standalone-1")
        self.path.write_bytes(b"changed raw source")
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_source_changed"):
            self.repo.commit_cleanup(command)
        self.assertIsNone(self.lookup())

    def test_layout_stamp_only_advances_if_current(self):
        document = self.repo.load_document("book")
        artifacts = document.document_task_artifacts
        self.repo.save_document(replace(document, document_task_artifacts=replace(artifacts,
            metadata={**artifacts.metadata, "task_layout": {"reading_revision": 0, "resolver_version": "existing"}})), "book")
        result = self.repo.commit_cleanup(self.prepare())
        self.assertFalse(result.reading_layout_required)
        layout = self.repo.load_document("book").document_task_artifacts.metadata["task_layout"]
        self.assertEqual(layout, {"reading_revision": 1, "resolver_version": "existing"})

    def test_missing_units_not_generated_during_commit(self):
        document = self.repo.load_document("book")
        self.repo.save_document(replace(document, chapters=[replace(c, sections=[replace(s, task_units=[])
            for s in c.sections]) for c in document.chapters]), "book")
        result = self.repo.commit_cleanup(self.prepare())
        self.assertTrue(result.reading_layout_required)
        self.assertEqual(result.outcome.status, "completed")
        self.assertFalse(self.repo.load_document("book").chapters[0].sections[0].task_units)

    def test_five_levels_do_not_resurrect_deleted_text(self):
        self.repo.commit_cleanup(self.prepare())
        document = self.repo.load_document("book")
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = section.task_units[0]
        block = self.service.read("book")["blocks"][0]
        for kwargs in ({"target_level": "document"}, {"target_level": "chapter", "chapter_id": chapter.chapter_id},
                       {"target_level": "section", "section_id": section.section_id},
                       {"target_level": "task_unit", "task_unit_id": unit.unit_id},
                       {"target_level": "content_block", "task_unit_id": unit.unit_id, "content_block_id": block["block_id"]}):
            if kwargs["target_level"] == "content_block":
                with self.assertRaisesRegex(ValueError, "not found"):
                    ReadingTargetResolver().resolve(document=document, **kwargs)
            else:
                result = ReadingTargetResolver().resolve(document=document, **kwargs)
                self.assertNotIn("@@??", result.content)

    def test_history_limit_does_not_evict_receipts_or_partially_apply(self):
        state = {RECEIPTS: {str(i): {} for i in range(1000)}}
        # Explicit setup, not a runtime write through the cleanup port.
        self.repo.mutate_reading_content("book", lambda d: (
            replace(d, parse_provenance={**d.parse_provenance, "reading_content": state}), None))
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, "cleanup_history_limit_reached"):
            self.repo.commit_cleanup(self.prepare())
        self.assertEqual(before, self.snapshot())

    def test_receipt_is_minimal_and_logs_private_safe(self):
        command = self.prepare()
        with self.assertLogs("document_structure.cleanup_commit", level="INFO") as logs:
            self.repo.commit_cleanup(command)
        state = state_for(self.repo.load_document("book"))
        receipts = json.dumps(state[RECEIPTS])
        for private in ("@@??", self.text, "hierarchy_json", "canonical_text"):
            self.assertNotIn(private, receipts)
            self.assertNotIn(private, "\n".join(logs.output))
        self.assertIn("stage=committed", "\n".join(logs.output))

    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"), "opt-in immutable backups")
    def test_real_books_commit_reload_undo_without_backup_mutation(self):
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                folder = root / name
                paths = [folder / p for p in ("book.pdf", "document.json", "manifest.json")]
                before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                self.assertEqual(before[paths[0]], json.loads(paths[2].read_text())["source_sha256"])
                self.path.write_bytes(paths[0].read_bytes())
                document = StructuredDocument.from_dict(json.loads(paths[1].read_text()))
                self.repo.save_reparsed_document(replace(document, document_id="book", source_path=str(self.path)), "book")
                source = canonical_text(self.repo.load_document("book"))
                items = []
                if name == "sample2":
                    record, = json.loads((folder / "20261023T030347436965Z/deletions.json").read_text())
                    self.assertEqual(source[record["char_start"]:record["char_end"]], record["quote"])
                    items = [item(quote=record["quote"])]
                command = self.prepare(items, operation="real-" + name)
                with patch("document_structure.ocr_cleanup_planner.plan_cleanup_input", side_effect=AssertionError("no LLM in commit")):
                    result = self.repo.commit_cleanup(command)
                self.assertEqual(result.outcome.applied_count, len(items))
                self.assertEqual(canonical_text(self.repo.load_document("book")), source)
                self.assertTrue(self.repo.commit_cleanup(command).replayed)
                if items:
                    self.assertNotIn(record["quote"], self.content())
                    self.change("undo")
                    self.assertIn(record["quote"], self.content())
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__ == "__main__":
    unittest.main()
