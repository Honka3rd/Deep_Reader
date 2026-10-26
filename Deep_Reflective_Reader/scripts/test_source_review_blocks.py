"""Region editing and note ownership on canonical source coordinates."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from document_structure.reading_content import ReadingContentConflict, project_document, original_blocks
from section_tasks.note_interaction_service import NoteInteractionService, NoteConflict
from scripts import test_ocr_cleanup_overlay as cleanup_tests


class SourceReviewBlockTests(unittest.TestCase):
    setUp = cleanup_tests.CleanupOverlayTests.setUp
    repository = cleanup_tests.CleanupOverlayTests.repository

    def assert_manual_change(self, region, status):
        change = region["manual_change"]
        self.assertEqual(change["status"], status)
        at = datetime.fromisoformat(change["at"].replace("Z", "+00:00"))
        self.assertEqual(at.utcoffset(), timezone.utc.utcoffset(at))
        self.assertLessEqual(at, datetime.now(timezone.utc))

    def remove_persisted_mapping(self):
        path = self.repo.base_dir / "book.structured.json"
        payload = json.loads(path.read_text())
        payload["parse_provenance"].pop("source_review_mapping", None)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def replace_persisted_mapping(self, mapping):
        path = self.repo.base_dir / "book.structured.json"
        payload = json.loads(path.read_text())
        payload["parse_provenance"]["source_review_mapping"] = mapping
        path.write_text(json.dumps(payload), encoding="utf-8")

    def storage_snapshot(self):
        path = self.repo.base_dir / "book.structured.json"
        return path.read_bytes(), path.stat().st_mtime_ns

    def change(self, action="edit", **overrides):
        state = self.service.read("book")
        args = dict(action=action, expected_revision=state["revision"], source_token=state["source_token"],
                    mutation_id=uuid4().hex, block_id=state["blocks"][0]["block_id"])
        if action == "edit":
            args.update(text="Corrected text", discard_interactions=False)
        return self.service.mutate("book", **(args | overrides))

    def note_target(self, index=0):
        block = self.service.read("book")["blocks"][index]
        return dict(target_level="content_block", task_unit_id=block["task_unit_id"], content_block_id=block["block_id"])

    def add_note(self, index=0):
        service = NoteInteractionService(self.repo)
        target = self.note_target(index)
        token = service.read("book", **target)["target_token"]
        result = service.save("book", **target, text="User note", expected_revision=0, mutation_id=uuid4().hex, target_token=token)
        return service, target, result

    def test_mapping_and_region_scoped_pagination(self):
        state = self.service.read("book", region_id="r1")
        self.assertEqual(state["regions"][0]["mapping_status"], "complete")
        self.assertEqual(state["regions"][0]["body_status"], "linked")
        self.assertTrue(state["regions"][0]["mapping_reason"])
        self.assertEqual(state["regions"][0]["related_deleted_count"], 0)
        self.assertIsNone(state["regions"][0].get("manual_change"))
        self.assertEqual(len(state["blocks"]), 1)
        self.assertEqual(state["blocks"][0]["content"], "@@??")
        self.assertEqual(state["block_page"], {"offset": 0, "limit": 20, "total": 1})
        self.assertEqual(self.service.read("book", region_id="r1", block_offset=20)["blocks"], [])
        with self.assertRaises(FileNotFoundError):
            self.service.read("book", region_id="other")

    def test_durable_mapping_uses_final_ids_and_retains_deleted_links(self):
        document = self.repo.load_document("book")
        persisted = deepcopy(document.parse_provenance["source_review_mapping"])
        region = self.service.read("book")["regions"][0]
        self.assertEqual(persisted["regions"]["r1"]["block_ids"], region["block_ids"])
        self.change("delete")
        self.assertEqual(self.repo.load_document("book").parse_provenance["source_review_mapping"], persisted)
        self.change("undo")
        self.assertEqual(self.repo.load_document("book").parse_provenance["source_review_mapping"], persisted)

    def test_region_ownership_checked_at_commit(self):
        with self.assertRaisesRegex(ReadingContentConflict, "mapping_changed"):
            self.change(region_id="r1", block_id=self.service.read("book")["blocks"][1]["block_id"])
        self.assertEqual(self.service.read("book")["revision"], 0)

    def test_source_evidence_change_rejects_old_editor_token(self):
        before = self.service.read("book")
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_review"]["source_sha256"] = "0" * 64
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        self.assertNotEqual(self.service.read("book")["source_token"], before["source_token"])
        with self.assertRaises(ReadingContentConflict):
            self.change(source_token=before["source_token"])
        self.assertEqual(self.service.read("book")["revision"], before["revision"])

    def test_legacy_exact_page_quote_and_excluded_candidate(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        region = provenance["source_review"]["regions"][0]
        region.pop("canonical_span")
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        self.assertEqual(self.service.read("book")["regions"][0]["mapping_status"], "complete")
        region["raw_candidate"] = "OCR candidate excluded from canonical text"
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        self.assertEqual(self.service.read("book")["regions"][0]["block_ids"], [])

    def test_last_block_delete_retains_region_and_undo_restores_text_not_note(self):
        service, target, before = self.add_note()
        original = self.repo.load_document("book")
        result = self.change("delete", region_id="r1")
        region, = result["regions"]
        self.assertEqual(region["body_status"], "linked")
        self.assertEqual(region["related_deleted_count"], 1)
        self.assertEqual(self.service.read("book", region_id="r1")["blocks"], [])
        self.assert_manual_change(region, "deleted")
        self.assertEqual(result["disposition"], "needs_review")
        self.assertEqual(self.repo.load_document("book").raw_text, original.raw_text)
        with self.assertRaises(NoteConflict):
            service.read("book", **target)
        self.change("undo")
        restored, = self.service.read("book")["regions"]
        self.assertEqual(restored["related_deleted_count"], 0)
        self.assert_manual_change(restored, "restored")
        self.assertIsNone(service.read("book", **target)["note"])
        with self.assertRaises(NoteConflict):
            service.save("book", **target, text="Delayed note", expected_revision=0,
                         mutation_id="late", target_token=before["target_token"])

    def test_only_explicit_ignore_removes_deleted_region(self):
        deleted = self.change("delete")
        self.assertEqual(len(deleted["regions"]), 1)
        ignored = self.change("discard_region", region_id="r1")
        self.assertEqual(ignored["regions"], [])
        self.assertEqual(ignored["disposition"], "completed")
        restored = self.change("undo")
        self.assertEqual(restored["regions"], deleted["regions"])

    def test_shared_block_changes_every_related_region_and_survives_restart(self):
        from document_structure.reading_content import ReadingContentService
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_review"]["regions"].append({
            **provenance["source_review"]["regions"][0], "id": "shared"})
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        result = self.change()
        self.assertEqual({r["id"] for r in result["regions"]}, {"r1", "shared"})
        for region in result["regions"]:
            self.assert_manual_change(region, "modified")
        self.assertEqual(ReadingContentService(self.repo).read("book")["regions"],
                         result["regions"])

    def test_repeated_reads_do_not_write_mapping_or_manual_audit(self):
        self.remove_persisted_mapping()
        before = self.repo.load_document("book").to_dict()
        stored = self.storage_snapshot()
        for _ in range(3):
            self.service.read("book", region_id="r1", include_cleanup_audit=True)
            self.repo.list_documents()
        self.assertEqual(self.repo.load_document("book").to_dict(), before)
        self.assertEqual(self.storage_snapshot(), stored)

    def test_explicit_backfill_is_cas_checked_and_preserves_revision_artifacts(self):
        self.add_note()
        self.remove_persisted_mapping()
        before = self.repo.load_document("book").to_dict()
        state = self.service.read("book")
        self.assertNotIn("source_review_mapping", before["parse_provenance"])
        for changes in ({"expected_revision": state["revision"] + 1}, {"source_token": "stale"}):
            with self.assertRaises(ReadingContentConflict):
                self.service.rebuild_review_mapping("book", **({"expected_revision": state["revision"],
                    "source_token": state["source_token"]} | changes))
            self.assertEqual(self.repo.load_document("book").to_dict(), before)
        result = self.service.rebuild_review_mapping("book", expected_revision=state["revision"],
                                                   source_token=state["source_token"])
        after = self.repo.load_document("book").to_dict()
        self.assertTrue(after["parse_provenance"].pop("source_review_mapping"))
        self.assertEqual(after, before)
        self.assertEqual(result, state)
        persisted = self.repo.load_document("book").to_dict()
        self.service.rebuild_review_mapping("book", expected_revision=state["revision"],
                                           source_token=state["source_token"])
        self.assertEqual(self.repo.load_document("book").to_dict(), persisted)

    def test_legacy_omission_stays_reviewable_after_explicit_backfill(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_review"]["regions"] = [{"id": "r1", "page_index": 1,
                                                    "raw_candidate": "Rejected OCR candidate"}]
        provenance["source_page_spans"].append({"page_index": 1,
            "char_start": len(self.text), "char_end": len(self.text)})
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        self.remove_persisted_mapping()
        stored = self.storage_snapshot()
        before = self.service.read("book")
        self.assertEqual(before["regions"][0]["body_status"], "unknown")
        self.assertTrue(self.repo.list_documents()[0].review_required)
        self.assertEqual(self.storage_snapshot(), stored)
        after = self.service.rebuild_review_mapping("book", expected_revision=before["revision"],
                                                    source_token=before["source_token"])
        self.assertEqual(after["regions"], before["regions"])
        self.assertEqual(after["regions"][0]["body_status"], "unknown")
        self.assertEqual(after["regions"][0]["block_ids"], [])
        self.assertEqual(after["disposition"], "needs_review")
        self.assertEqual(after["revision"], before["revision"])
        self.assertTrue(self.repo.list_documents()[0].review_required)
        self.assertFalse(self.repo.load_document("book").parse_provenance.get("reading_content"))

    def test_v1_exclusion_cache_reads_keep_gap_visible_without_rewriting_storage(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_page_spans"].append({"page_index": 1,
            "char_start": len(self.text), "char_end": len(self.text)})
        provenance["source_review"]["regions"] = [{"id": "r1", "page_index": 1,
            "raw_candidate": "Missing important body", "reason": "page_quality_rejected",
            "body_inclusion": "excluded"}]
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        mapping = deepcopy(self.repo.load_document("book").parse_provenance["source_review_mapping"])
        mapping["policy"] = "source-review-blocks-v1"
        mapping["regions"]["r1"] = {
            "block_ids": [], "mapping_status": "unavailable", "body_status": "excluded",
            "mapping_reason": "never_in_body"}
        self.replace_persisted_mapping(mapping)
        before = self.storage_snapshot()
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("read attempted mapping write")):
            for _ in range(2):
                state = self.service.read("book", region_id="r1")
                self.assertEqual(state["disposition"], "needs_review")
                self.assertEqual(state["regions"][0]["body_status"], "unknown")
                self.assertEqual(state["regions"][0]["block_ids"], [])
                self.assertEqual(state["blocks"], [])
                self.assertTrue(self.repo.list_documents()[0].review_required)
        self.assertEqual(self.storage_snapshot(), before)

    def test_delete_only_removes_direct_block_notes(self):
        service, target, _ = self.add_note()
        _, sibling, before = self.add_note(1)
        self.change("delete")
        self.assertEqual(service.read("book", **sibling)["note"], before["note"])
        entries = self.repo.load_document("book").document_task_artifacts.metadata["note"]
        self.assertFalse(any(isinstance(v, dict) and v.get("target", [])[-1:] == [target["content_block_id"]] for v in entries.values()))

    def test_note_save_racing_block_delete_cannot_leave_orphan(self):
        from concurrent.futures import ThreadPoolExecutor
        service, target, before = self.add_note()
        def save_note():
            try:
                service.save("book", **target, text="Concurrent note", expected_revision=1,
                             mutation_id="racing-note", target_token=before["target_token"])
            except NoteConflict:
                pass
        with ThreadPoolExecutor(2) as pool:
            save = pool.submit(save_note)
            delete = pool.submit(self.change, "delete")
            save.result()
            delete.result()
        entries = self.repo.load_document("book").document_task_artifacts.metadata["note"]
        self.assertFalse(any(isinstance(v, dict) and v.get("target", [])[-1:] == [target["content_block_id"]] for v in entries.values()))

    def test_kept_generated_artifacts_are_stale_not_fresh(self):
        from section_tasks.reading_target_resolver import ReadingTargetResolver
        from section_tasks.reading_interaction_service_contracts import ReadingInteractionArtifact, ReadingInteractionRequest
        from section_tasks.reading_interaction_artifact_store import DocumentReadingInteractionArtifactStore
        from section_tasks.interaction_policy_identity import policy_metadata, stale_projection
        # Resolve the review explicitly before creating generated data.
        self.change("discard_region", region_id="r1")
        target = ReadingTargetResolver().resolve(document=self.repo.load_document("book"), **self.note_target())
        request = ReadingInteractionRequest(target=target, interaction_type="analysis", context_metadata={"doc_name": "book"})
        artifact = ReadingInteractionArtifact.from_target(target=target, interaction_type="analysis", status="completed",
            payload={"interpretation": "Stored analysis"}, metadata={**policy_metadata(request), "context": {"doc_name": "book"}})
        store = DocumentReadingInteractionArtifactStore(self.repo)
        store.save_analysis_artifact(artifact)
        self.change(discard_interactions=False)
        current = replace(request, target=ReadingTargetResolver().resolve(document=self.repo.load_document("book"), **self.note_target()))
        retained = store.get_analysis_artifact(current)
        self.assertIsNotNone(retained)
        self.assertEqual(stale_projection(retained, current).status, "stale_target")

    def test_keep_then_clear_and_late_empty_note_writer(self):
        service, target, before = self.add_note()
        _, sibling, sibling_note = self.add_note(1)
        self.change(discard_interactions=False)
        self.assertEqual(service.read("book", **target)["note"], before["note"])
        projected = original_blocks(project_document(self.repo.load_document("book")))
        updated_hash = next(b.metadata["source_hash"] for _, _, _, b in projected if b.block_id == target["content_block_id"])
        self.assertEqual(service.read("book", **target, source_hash=updated_hash)["note"], before["note"])
        with self.assertRaises(NoteConflict):
            service.read("book", **target, source_hash="invalid")
        stored = self.repo.load_document("book").document_task_artifacts.metadata
        self.assertIn("reading_interaction_artifacts", stored)
        self.change(discard_interactions=True, text="Corrected text again")
        response = service.read("book", **target)
        self.assertIsNone(response["note"])
        self.assertNotEqual(response["target_token"], before["target_token"])
        self.assertEqual(service.read("book", **sibling)["note"], sibling_note["note"])
        self.assertNotIn("reading_interaction_artifacts", self.repo.load_document("book").document_task_artifacts.metadata)
        self.change(discard_interactions=True)
        with self.assertRaises(NoteConflict):
            service.save("book", **target, text="Delayed create", expected_revision=0,
                         mutation_id="late-empty", target_token=response["target_token"])

    def test_unmapped_region_never_automatically_ignored(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_review"]["regions"][0]["canonical_span"]["text_sha256"] = "0" * 64
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        self.assertEqual(self.service.read("book")["regions"][0]["mapping_status"], "unavailable")
        self.change("delete")
        self.assertEqual(len(self.service.read("book")["regions"]), 1)

    def test_region_multiple_blocks_partial_then_final_deletion(self):
        from document_structure.ocr_cleanup_overlay import canonical_text
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        region = provenance["source_review"]["regions"][0]
        text = canonical_text(document)
        region["raw_candidate"] = text
        region["canonical_span"] = {"char_start": 0, "char_end": len(text),
                                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        blocks = self.service.read("book", region_id="r1")["blocks"]
        self.assertGreater(len(blocks), 1)
        for i, block in enumerate(blocks):
            result = self.change("delete", region_id="r1", block_id=block["block_id"])
            region, = result["regions"]
            self.assertEqual(region["related_deleted_count"], i + 1)
            self.assertEqual(self.service.read("book", region_id="r1")["block_page"]["total"],
                             len(blocks) - i - 1)
            self.assert_manual_change(region, "deleted")
        self.assertFalse(original_blocks(project_document(self.repo.load_document("book"))))

    def test_no_manual_mutation_quota_and_policy_in_idempotency(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["reading_content"] = {"receipts": {str(i): "old" for i in range(1001)}}
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        before = self.service.read("book")
        command = dict(action="edit", expected_revision=0, source_token=before["source_token"],
            mutation_id="repeat", block_id=before["blocks"][0]["block_id"], text="Corrected", discard_interactions=False)
        result = self.service.mutate("book", **command)
        self.assertEqual(self.service.mutate("book", **command), result)
        with self.assertRaisesRegex(ReadingContentConflict, "mutation_conflict"):
            self.service.mutate("book", **(command | {"discard_interactions": True}))

    def test_failed_delete_keeps_note_region_content(self):
        service, target, before = self.add_note()
        initial = self.service.read("book")
        persisted = self.storage_snapshot()
        with patch.object(self.repo, "_atomic_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.change("delete", region_id="r1")
        self.assertEqual(self.service.read("book"), initial)
        self.assertEqual(service.read("book", **target)["note"], before["note"])
        self.assertEqual(self.storage_snapshot(), persisted)


class SourceReviewManualVolumeTests(unittest.TestCase):
    """Actual mutation volume in temporary file storage, without PG inheritance."""

    setUp = cleanup_tests.CleanupOverlayTests.setUp
    repository = cleanup_tests.CleanupOverlayTests.repository

    def test_1002_real_edits_preserve_audit_and_unresolved_review(self):
        from document_structure.reading_content import ReadingContentService

        state = self.service.read("book")
        self.assertEqual(state["revision"], 0)
        block_id = state["blocks"][0]["block_id"]
        for revision in range(1, 1003):
            state = self.service.mutate(
                "book", action="edit", expected_revision=revision - 1,
                source_token=state["source_token"], mutation_id=f"volume-edit-{revision}",
                block_id=block_id, text=f"Reviewed text after edit {revision}",
                discard_interactions=False,
            )

        current = ReadingContentService(self.repo).read("book", region_id="r1")
        stored = self.repo.load_document("book").parse_provenance["reading_content"]
        self.assertEqual(current["revision"], 1002)
        self.assertEqual(stored["revision"], 1002)
        self.assertEqual(len(stored["audit"]), 1002)
        self.assertEqual([event["revision"] for event in stored["audit"]], list(range(1, 1003)))
        self.assertTrue(all(event["action"] == "edit" and not event["ignored_regions"]
                            for event in stored["audit"]))
        self.assertEqual(set(stored["receipts"]), {f"volume-edit-{i}" for i in range(1, 1003)})
        block, = current["blocks"]
        self.assertEqual(block["block_id"], block_id)
        self.assertEqual(block["content"], "Reviewed text after edit 1002")
        self.assertFalse(block["deleted"])
        region, = current["regions"]
        self.assertEqual(region["id"], "r1")
        self.assertEqual(region["status"], "unresolved")
        self.assertEqual(region["body_status"], "linked")
        self.assertEqual(region["related_deleted_count"], 0)
        self.assertEqual(region["manual_change"]["status"], "modified")
        self.assertEqual(current["disposition"], "needs_review")
        self.assertFalse(stored.get("regions"))


if __name__ == "__main__":
    unittest.main()
