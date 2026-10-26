"""Cleanup record recovery on real file/SQL repositories; no live book writes."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from uuid import uuid4
import unittest

from document_structure.reading_content import ReadingContentConflict, state_for
from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests
from scripts.test_ocr_cleanup_overlay_postgres import PostgresCleanupOverlayTests
from scripts.test_ocr_cleanup_contract import item


class HistoryCases:
    def test_delete_history_keeps_content_masks_and_undo_without_resurrection(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.assertTrue(row["can_delete"])
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Saved draft"))
        before = self.repo.load_document("book")
        content = self.content()
        command = self.command(row, "cleanup_delete")
        result = self.service.mutate("book", **command)
        after = self.repo.load_document("book")
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.content(), content)
        self.assertEqual(state_for(before)["cleanup_spans"], state_for(after)["cleanup_spans"])
        self.assertEqual(state_for(before)["undo"], state_for(after)["undo"])
        self.assertEqual(before.document_task_artifacts, after.document_task_artifacts)
        self.assertNotIn(row["record_id"], state_for(after).get("cleanup_record_edits", {}))
        self.assertNotIn(row["record_id"], state_for(after).get("cleanup_record_links", {}))
        self.assertFalse(result["refresh"]["generated_invalidated"])
        self.assertTrue(result["refresh"]["cleanup_change"]["deleted"])
        saved = self.snapshot()
        self.assertEqual(self.service.mutate("book", **command), result)
        self.assertEqual(self.snapshot(), saved)
        for action in ("cleanup_restore", "cleanup_edit", "cleanup_delete"):
            with self.assertRaisesRegex(ReadingContentConflict, "cleanup_record_missing"):
                self.service.mutate("book", **self.command(row, action, **({"text": "Draft"} if action == "cleanup_edit" else {})))
        self.change("undo")
        self.assertIn("@@??", self.content())
        self.assertEqual(self.rows(), [])

    def test_delete_restored_history_keeps_body_and_manual_flag_unchanged(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="Recovered"))
        before = self.content()
        self.service.mutate("book", **self.command(row, "cleanup_delete"))
        self.assertEqual(self.content(), before)
        self.assertFalse(self.service.read("book")["blocks"][0]["manually_edited"])
        self.change("undo")
        self.assertNotIn("Recovered", self.content())
        self.assertEqual(self.rows(), [])
        self.assertNotIn(row["record_id"], state_for(self.repo.load_document("book")).get("cleanup_record_applications", {}))

    def test_history_delete_rollback_conflict_and_invalid_body_inputs(self):
        self.commit(*self.prepare())
        row, = self.rows()
        before = self.snapshot()
        for extra in ({"text": "Not a content edit"}, {"block_id": "foreign"}, {"discard_interactions": True}):
            with self.assertRaisesRegex(ValueError, "invalid_cleanup_record_command"):
                self.service.mutate("book", **self.command(row, "cleanup_delete", **extra))
        with self.assertRaisesRegex(ReadingContentConflict, "changed_reload"):
            self.service.mutate("book", **self.command(row, "cleanup_delete", expected_revision=0))
        with patch("document_structure.cleanup_history.record_change", side_effect=RuntimeError("delta failure")):
            with self.assertRaisesRegex(RuntimeError, "delta failure"):
                self.service.mutate("book", **self.command(row, "cleanup_delete"))
        self.assertEqual(before, self.snapshot())

    def rows(self):
        return self.service.read("book", include_cleanup_audit=True)["cleanup_audit"]["items"][0]["deletions"]

    def command(self, row, action="cleanup_restore", **kwargs):
        current = self.service.read("book")
        return {"action": action, "cleanup_record_id": row["record_id"],
                "expected_revision": current["revision"], "source_token": current["source_token"],
                "mutation_id": uuid4().hex, **kwargs}

    def test_restore_hidden_block_replay_and_undo(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.assertTrue(row["can_restore"])
        self.assertFalse(row["links"][0]["can_navigate"])
        self.assertEqual(row["links"][0]["char_from"], 0)
        command = self.command(row)
        self.service.mutate("book", **command)
        saved = self.snapshot()
        self.service.mutate("book", **command)
        self.assertEqual(saved, self.snapshot())
        self.assertIn("@@??", self.content())
        row, = self.rows()
        self.assertEqual(row["status"], "restored")
        self.assertTrue(row["links"][0]["can_navigate"])
        self.assertTrue(row["can_restore"])
        self.change("undo")
        self.assertTrue(self.rows()[0]["can_restore"])
        self.assertNotIn("@@??", self.content())

    def test_saved_draft_preserves_original_and_restores_edited_text(self):
        self.commit(*self.prepare())
        row, = self.rows()
        before = self.repo.load_document("book")
        command = self.command(row, "cleanup_edit", text="Corrected text")
        self.service.mutate("book", **command)
        after = self.repo.load_document("book")
        self.assertEqual(before.document_task_artifacts, after.document_task_artifacts)
        self.assertEqual(state_for(before)["undo"], state_for(after)["undo"])
        row, = self.rows()
        self.assertEqual(row["quote"], "@@??")
        self.assertEqual(row["text"], "Corrected text")
        self.service.mutate("book", **self.command(row))
        self.assertIn("Corrected text", self.content())
        self.assertNotIn("@@??", self.content())
        self.assertEqual(self.repo.load_document("book").raw_text, before.raw_text)
        self.assertEqual(self.service.read("book")["blocks"][0]["manually_edited"], False)
        self.change("undo")
        self.assertNotIn("Corrected text", self.content())
        self.assertEqual(self.rows()[0]["text"], "Corrected text")

    def test_two_restores_in_same_block_do_not_set_edit_flag(self):
        quotes = ["@@", "??"]
        self.commit(*self.prepare(items=[item(quote=q) for q in quotes]))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.service.mutate("book", **self.command(rows[0], text="First"))
        self.assertTrue(self.rows()[1]["can_restore"])
        self.service.mutate("book", **self.command(self.rows()[1], text="Second"))
        self.assertIn("FirstSecond", self.content())
        self.assertFalse(self.service.read("book")["blocks"][0]["manually_edited"])

    def test_repeated_draft_apply_and_undo_replace_only_owned_text(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Second"))
        self.assertIn("First", self.content())
        self.assertNotIn("Second", self.content())
        self.assertTrue(self.rows()[0]["can_restore"])
        self.service.mutate("book", **self.command(row))
        self.assertNotIn("First", self.content())
        self.assertEqual(self.content().count("Second"), 1)
        state = state_for(self.repo.load_document("book"))
        self.assertEqual(len(state["source_edits"]), 1)
        self.assertEqual(state["source_edits"][0]["cleanup_record_id"], row["record_id"])
        self.change("undo")
        self.assertIn("First", self.content())
        self.assertEqual(self.rows()[0]["text"], "Second")
        self.assertTrue(self.rows()[0]["can_restore"])
        self.service.mutate("book", **self.command(row))
        self.assertEqual(self.content().count("Second"), 1)
        self.service.mutate("book", **self.command(row, text=row["quote"]))
        self.assertIn(row["quote"], self.content())
        self.assertFalse(state_for(self.repo.load_document("book"))["source_edits"])
        self.service.mutate("book", **self.command(row, text="Third"))
        for _ in range(2):
            self.service.mutate("book", **self.command(row))
        self.assertEqual(self.content().count("Third"), 1)
        self.assertFalse(self.service.read("book")["blocks"][0]["manually_edited"])
        self.assertNotIn(row["links"][0]["block_id"],
                         state_for(self.repo.load_document("book")).get("block_edit_flags", {}))

    def rewrite_state(self, change):
        from copy import deepcopy
        def mutate(document):
            updated = deepcopy(document)
            change(state_for(updated))
            return updated, None
        self.repo.mutate_reading_content("book", mutate)

    def test_legacy_incident_save_after_restore_does_not_lose_applied_text(self):
        self.commit(*self.prepare())
        row, = self.rows()
        old_text, new_text = "\u652f\u652f\u543e\u767e,", "\u652f\u652f\u543e\u543e,"
        self.service.mutate("book", **self.command(row, text=old_text))
        def legacy(state):
            state.pop("cleanup_record_applications")
            state["source_edits"][0].pop("cleanup_record_id")
        self.rewrite_state(legacy)
        self.service.mutate("book", **self.command(row, "cleanup_edit", text=new_text))
        before = self.snapshot()
        self.assertTrue(self.rows()[0]["can_restore"])
        self.assertEqual(before, self.snapshot())
        self.service.mutate("book", **self.command(row))
        self.assertIn(new_text, self.content())
        self.assertNotIn(old_text, self.content())
        self.assertEqual(self.content().count(new_text), 1)
        self.assertFalse(self.service.read("book")["blocks"][0]["manually_edited"])
        self.change("undo")
        self.assertIn(old_text, self.content())
        self.assertTrue(self.rows()[0]["can_restore"])
        self.service.mutate("book", **self.command(row))
        self.assertEqual(self.content().count(new_text), 1)

    def test_legacy_original_restore_can_be_reapplied(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row))
        self.rewrite_state(lambda state: state.pop("cleanup_record_applications"))
        self.service.mutate("book", **self.command(row, text="Recovered"))
        self.assertIn("Recovered", self.content())

    def test_missing_legacy_proof_or_foreign_replacement_is_not_claimed(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        def foreign(state):
            state["source_edits"][0]["cleanup_record_id"] = "foreign-record"
        self.rewrite_state(foreign)
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, "application_changed"):
            self.service.mutate("book", **self.command(row, text="Second"))
        self.assertEqual(before, self.snapshot())
        def unproven(state):
            state.pop("cleanup_record_applications")
            state["source_edits"] = []
            state["undo"] = None
        self.rewrite_state(unproven)
        self.assertEqual(self.rows()[0]["restore_reason"], "restore_ownership_unavailable")

    def test_reapply_after_body_edit_even_undone_is_rejected(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        block_id = row["links"][0]["block_id"]
        self.change("edit", block_id=block_id, text="Manual", discard_interactions=False)
        self.change("undo")
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Second"))
        with self.assertRaisesRegex(ReadingContentConflict, "block_edited"):
            self.service.mutate("book", **self.command(row))
        self.assertIn("First", self.content())
        self.assertTrue(self.service.read("book")["blocks"][0]["manually_edited"])

    def test_manual_edit_delete_and_undo_prevent_restore(self):
        self.commit(*self.prepare(items=[item(quote="@@")]))
        row, = self.rows()
        block = row["links"][0]["block_id"]
        self.change("edit", block_id=block, text="Manual", discard_interactions=False)
        self.assertTrue(self.service.read("book")["blocks"][0]["manually_edited"])
        self.assertTrue(state_for(self.repo.load_document("book"))["block_edit_flags"][block])
        self.change("undo")
        self.assertTrue(self.service.read("book")["blocks"][0]["manually_edited"])
        self.assertEqual(self.rows()[0]["restore_reason"], "block_edited")
        with self.assertRaisesRegex(ReadingContentConflict, "block_edited"):
            self.service.mutate("book", **self.command(row))
        self.change("delete", block_id=block)
        self.assertEqual(self.rows()[0]["restore_reason"], "block_deleted")

    def test_history_reads_are_pure_and_stale_commands_fail(self):
        self.commit(*self.prepare())
        before = self.snapshot()
        row, = self.rows()
        self.assertEqual(before, self.snapshot())
        stale = self.command(row)
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Draft"))
        with self.assertRaisesRegex(ReadingContentConflict, "changed_reload"):
            self.service.mutate("book", **stale)
        before = self.snapshot()
        for bad in ({"text": ""}, {"text": "x" * 10001}, {"block_id": "injected"}):
            with self.assertRaises(ValueError):
                self.service.mutate("book", **self.command(row, **bad))
        self.assertEqual(before, self.snapshot())

    def test_cross_block_restore_uses_canonical_not_visible_offsets(self):
        from document_structure.ocr_cleanup_overlay import canonical_text
        quote = "@@??\n\nMeaningful"
        self.commit(*self.prepare(items=[item(quote=quote)]))
        row, = self.rows()
        self.assertEqual(len(row["links"]), 2)
        self.assertEqual([link["char_from"] for link in row["links"]], [0, 0])
        self.assertTrue(row["can_restore"])
        self.service.mutate("book", **self.command(row, text="Recovered heading"))
        self.assertEqual(self.content().count("Recovered heading"), 1)
        self.service.mutate("book", **self.command(row, text="Updated heading"))
        self.assertNotIn("Recovered heading", self.content())
        self.assertEqual(self.content().count("Updated heading"), 1)
        self.assertEqual(canonical_text(self.repo.load_document("book")), self.text)

    def test_pinned_mapping_change_disables_restore(self):
        self.commit(*self.prepare())
        row, = self.rows()
        def alter(document):
            from copy import deepcopy
            updated = deepcopy(document)
            state_for(updated)["cleanup_record_links"][row["record_id"]][0]["block_id"] = "obsolete"
            return updated, None
        self.repo.mutate_reading_content("book", alter)
        current, = self.rows()
        self.assertEqual(current["restore_reason"], "mapping_changed")
        self.assertFalse(current["links"])
        with self.assertRaisesRegex(ReadingContentConflict, "mapping_changed"):
            self.service.mutate("book", **self.command(row))

    def test_concurrent_restore_inserts_once(self):
        self.commit(*self.prepare())
        row, = self.rows()
        commands = [self.command(row, text="Recovered") for _ in range(2)]
        def restore(command):
            try:
                self.service.mutate("book", **command)
                return True
            except ReadingContentConflict:
                return False
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(restore, commands))
        self.assertEqual(sum(results), 1)
        self.assertEqual(self.content().count("Recovered"), 1)

    def test_reapply_failure_rolls_back_application_and_body(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        before = self.snapshot()
        command = self.command(row, text="Second")
        with patch.object(self.service, "snapshot", side_effect=RuntimeError("reapply failed")):
            with self.assertRaisesRegex(RuntimeError, "reapply failed"):
                self.service.mutate("book", **command)
        self.assertEqual(before, self.snapshot())

    def test_concurrent_reapply_does_not_overwrite_a_newer_application(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        commands = [self.command(row, text=value) for value in ("Second", "Third")]
        def apply(command):
            try:
                self.service.mutate("book", **command)
                return command["text"]
            except ReadingContentConflict:
                return None
        with ThreadPoolExecutor(2) as executor:
            winners = [value for value in executor.map(apply, commands) if value]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.content().count(winners[0]), 1)
        self.assertNotIn("First", self.content())

    def test_failed_mutation_rolls_back_everything(self):
        self.commit(*self.prepare())
        row, = self.rows()
        before = self.snapshot()
        original = self.service.snapshot
        def failure(document):
            if state_for(document).get("revision", 0) > before[0].parse_provenance["reading_content"]["revision"]:
                raise RuntimeError("injected snapshot failure")
            return original(document)
        command = self.command(row)
        with patch.object(self.service, "snapshot", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.service.mutate("book", **command)
        self.assertEqual(before, self.snapshot())

    def test_http_projection_and_mutation(self):
        from fastapi.testclient import TestClient
        import main
        self.commit(*self.prepare())
        with patch.object(main, "_reading_content_service", return_value=self.service), TestClient(main.app) as client:
            response = client.get("/documents/book/reading-content?include_cleanup_audit=true")
            self.assertEqual(response.status_code, 200)
            row = response.json()["cleanup_audit"]["items"][0]["deletions"][0]
            command = self.command(row)
            self.assertEqual(client.post("/documents/book/reading-content", json=command).status_code, 200)
            result = client.get("/documents/book/reading-content?block_id=" + row["links"][0]["block_id"])
            self.assertFalse(result.json()["blocks"][0]["manually_edited"])
            self.assertEqual(client.post("/documents/book/reading-content", json={**command, "char_from": 0}).status_code, 422)


class FileHistoryTests(HistoryCases, CleanupOverlayTests):
    pass


class SqlHistoryTests(HistoryCases, PostgresCleanupOverlayTests):
    def test_reapply_rolls_back_after_database_writes(self):
        self.commit(*self.prepare())
        row, = self.rows()
        self.service.mutate("book", **self.command(row, text="First"))
        before = self.snapshot()
        command = self.command(row, text="Second")
        with patch.object(self.store, "_snapshot_payload", side_effect=RuntimeError("after SQL updates")):
            with self.assertRaisesRegex(RuntimeError, "after SQL updates"):
                self.service.mutate("book", **command)
        self.assertEqual(before, self.snapshot())

    def test_application_ownership_is_in_both_database_payloads(self):
        import psycopg
        self.commit(*self.prepare())
        row, = self.rows()
        for value in ("First", "Second"):
            self.service.mutate("book", **self.command(row, text=value))
        with psycopg.connect(self.scoped) as connection:
            document_id, payload = connection.execute(
                "SELECT id,metadata_payload FROM documents WHERE namespace=%s AND document_name='book'",
                (self.namespace,)).fetchone()
            snapshot = connection.execute(
                "SELECT structured_document_payload FROM structured_document_snapshots WHERE document_id=%s",
                (document_id,)).fetchone()[0]
        for value in (payload, snapshot):
            state = value["parse_provenance"]["reading_content"]
            self.assertEqual(state["cleanup_record_applications"][row["record_id"]]["text"], "Second")
            self.assertEqual(len(state["source_edits"]), 1)
            self.assertEqual(state["source_edits"][0]["cleanup_record_id"], row["record_id"])
            self.assertFalse(state.get("block_edit_flags", {}))

    def test_draft_keeps_generated_rows_restore_invalidates_and_retains_notes(self):
        import psycopg
        self.commit(*self.prepare())
        row, = self.rows()
        with psycopg.connect(self.scoped) as connection:
            doc_id = connection.execute("SELECT id FROM documents WHERE namespace=%s", (self.namespace,)).fetchone()[0]
            for kind in ("analysis", "note"):
                connection.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,"
                    "source_structure_version,payload) VALUES (%s,%s,'document',%s,1,'{}')", (doc_id, kind, doc_id))
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Restoration draft"))
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM artifacts WHERE document_id=%s", (doc_id,)).fetchone()[0], 2)
        self.service.mutate("book", **self.command(self.rows()[0]))
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT artifact_type FROM artifacts WHERE document_id=%s", (doc_id,)).fetchall(), [("note",)])

    def test_edit_flag_is_in_both_database_payloads(self):
        import psycopg
        self.commit(*self.prepare(items=[item(quote="@@")]))
        row, = self.rows()
        block_id = row["links"][0]["block_id"]
        self.change("edit", block_id=block_id, text="Manual database edit", discard_interactions=False)
        self.change("undo")
        with psycopg.connect(self.scoped) as connection:
            document_id, payload = connection.execute(
                "SELECT id,metadata_payload FROM documents WHERE namespace=%s AND document_name='book'",
                (self.namespace,)).fetchone()
            snapshot = connection.execute(
                "SELECT structured_document_payload FROM structured_document_snapshots WHERE document_id=%s",
                (document_id,)).fetchone()[0]
        self.assertTrue(payload["parse_provenance"]["reading_content"]["block_edit_flags"][block_id])
        self.assertTrue(snapshot["parse_provenance"]["reading_content"]["block_edit_flags"][block_id])


if __name__ == "__main__":
    unittest.main()
