"""Single-block edits and confirmation, including real SQL atomicity."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from document_structure.reading_content import ReadingContentConflict, state_for
from document_structure.ocr_cleanup_overlay import canonical_text
from scripts import test_source_region_mutations as fixtures


class ReviewBlockRepairTests(unittest.TestCase):
    setUp = fixtures.RegionMutations.setUp
    repository = fixtures.RegionMutations.repository
    change = fixtures.RegionMutations.change
    content = fixtures.RegionMutations.content
    note_target = fixtures.RegionMutations.note_target
    add_note = fixtures.RegionMutations.add_note

    def expand_region(self):
        def expand(document):
            provenance = deepcopy(document.parse_provenance)
            region = provenance["source_review"]["regions"][0]
            region["raw_candidate"] = canonical_text(document)
            region["canonical_span"]["char_end"] = len(region["raw_candidate"])
            second = {**deepcopy(region), "id": "r2"}
            provenance["source_review"]["regions"].append(second)
            return replace(document, parse_provenance=provenance), None
        self.repo.mutate_reading_content("book", expand)
        return self.service.read("book")["blocks"]

    def edit(self, key, text="Review correction", discard=False):
        return self.change("review_block_edit", region_id="r1", block_id=key, text=text,
                           discard_interactions=discard)

    def test_independent_blocks_repeat_and_main_edit_protection(self):
        blocks = self.expand_region()
        key, sibling = [b["block_id"] for b in blocks]
        self.change("edit", block_id=sibling, text="Protected sibling", discard_interactions=False)
        original = self.repo.load_document("book")
        for text in ("First", "Second", "Third"):
            result = self.edit(key, text)
            self.assertEqual(result["refresh"]["block_page"]["total"], 1)
            self.assertEqual([b["content"] for b in result["blocks"]], [text, "Protected sibling"])
            self.assertFalse(result["blocks"][0]["manually_edited"])
            self.assertTrue(result["blocks"][0]["review_repair"]["can_edit"])
            self.assertFalse(result["blocks"][1]["review_repair"]["can_edit"])
        self.assertEqual(self.repo.load_document("book").chapters, original.chapters)
        self.change("edit", block_id=key, text="Main reader correction", discard_interactions=False)
        self.change("undo")
        for action in ("review_block_edit", "review_block_delete"):
            with self.assertRaisesRegex(ReadingContentConflict, "block_manually_edited"):
                self.change(action, region_id="r1", block_id=key,
                    **({"text": "Rejected", "discard_interactions": False} if action.endswith("edit") else {}))

    def test_keep_removes_review_only_and_survives_undo_and_reload(self):
        blocks = self.expand_region()
        key, sibling = [b["block_id"] for b in blocks]
        notes, target, _ = self.add_note()
        self.edit(key)
        before = self.repo.load_document("book")
        result = self.change("review_block_keep", region_id="r1", block_id=key)
        after = self.repo.load_document("book")
        self.assertEqual(after.document_task_artifacts, before.document_task_artifacts)
        self.assertEqual(result["refresh"]["blocks"], [])
        self.assertFalse(result["refresh"]["generated_invalidated"])
        self.assertNotIn(key, result["regions"][0]["block_ids"])
        self.assertNotIn(key, result["regions"][1]["block_ids"])
        self.assertEqual(result["blocks"][0]["content"], "Review correction")
        self.assertEqual(notes.read("book", **target)["note"]["text"], "User note")
        self.change("undo")
        self.assertNotIn(key, self.service.read("book")["regions"][0]["block_ids"])
        with self.assertRaisesRegex(ReadingContentConflict, "review_block_kept"):
            self.edit(key)
        self.change("review_block_keep", region_id="r1", block_id=sibling)
        self.assertEqual(self.service.read("book")["regions"], [])
        self.assertEqual(len(self.service.read("book")["blocks"]), 2)
        self.change("edit", block_id=key, text="Main editing still allowed", discard_interactions=False)
        self.assertEqual(self.service.read("book")["regions"], [])

    def test_delete_only_selected_block_and_purge_note(self):
        blocks = self.expand_region()
        notes, target, _ = self.add_note()
        key = blocks[0]["block_id"]
        result = self.change("review_block_delete", region_id="r1", block_id=key)
        self.assertEqual(result["refresh"]["cleared_block_id"], key)
        self.assertTrue(result["refresh"]["blocks"][0]["deleted"])
        self.assertEqual(result["blocks"][1]["content"], blocks[1]["content"])
        self.change("undo")
        self.assertIsNone(notes.read("book", **target)["note"])
        self.assertTrue(self.service.read("book")["blocks"][0]["review_repair"]["can_edit"])

    def test_main_delete_undo_protects_but_keep_is_allowed(self):
        key = self.service.read("book")["blocks"][0]["block_id"]
        self.change("delete", block_id=key)
        self.change("undo")
        with self.assertRaisesRegex(ReadingContentConflict, "block_manually_deleted"):
            self.edit(key)
        self.change("review_block_keep", region_id="r1", block_id=key)

    def test_unrelated_id_rejected_without_writes(self):
        before = self.repo.load_document("book")
        key = self.service.read("book")["blocks"][1]["block_id"]
        for action in ("review_block_keep", "review_block_delete"):
            with self.assertRaisesRegex(ReadingContentConflict, "review_block_mapping_changed"):
                self.change(action, region_id="r1", block_id=key)
        self.assertEqual(self.repo.load_document("book"), before)

    def test_keep_replay_and_concurrent_commands(self):
        state = self.service.read("book")
        command = dict(action="review_block_keep", region_id="r1", block_id=state["blocks"][0]["block_id"],
            expected_revision=state["revision"], source_token=state["source_token"], mutation_id=uuid4().hex)
        first = self.service.mutate("book", **command)
        self.assertEqual(self.service.mutate("book", **command), first)
        def attempt(_):
            try:
                self.service.mutate("book", **{**command, "mutation_id": uuid4().hex})
                return "committed"
            except ReadingContentConflict:
                return "conflict"
        with ThreadPoolExecutor(2) as executor:
            self.assertEqual(list(executor.map(attempt, range(2))), ["conflict", "conflict"])

    def test_schema_requires_ids_consent_and_forbids_keep_text(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "_reading_content_service", return_value=self.service), TestClient(main.app) as client:
            state = client.get("/documents/book/reading-content?include_blocks=true").json()
            block = state["blocks"][0]
            self.assertTrue(block["review_repair"]["can_edit"])
            command = dict(action="review_block_edit", region_id="r1", block_id=block["block_id"],
                text="HTTP correction", discard_interactions=False, expected_revision=state["revision"],
                source_token=state["source_token"], mutation_id=uuid4().hex)
            for removed in ("region_id", "block_id", "discard_interactions", "text"):
                invalid = {k: v for k, v in command.items() if k != removed}
                self.assertEqual(client.post("/documents/book/reading-content", json=invalid).status_code, 422)
            self.assertEqual(client.post("/documents/book/reading-content", json={**command, "action": "review_block_keep"}).status_code, 422)
            response = client.post("/documents/book/reading-content", json=command)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["blocks"][0]["review_repair"]["can_edit"])


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "Dedicated test DSN required")
class SqlReviewBlockRepairTests(ReviewBlockRepairTests):
    setUpClass = classmethod(fixtures.SqlRegionMutations.setUpClass.__func__)
    drop_schema = classmethod(fixtures.SqlRegionMutations.drop_schema.__func__)
    repository = fixtures.SqlRegionMutations.repository

    def test_keep_and_edit_failure_roll_back_after_sql(self):
        import psycopg
        before = self.repo.load_document("book")
        key = self.service.read("book")["blocks"][0]["block_id"]
        with psycopg.connect(self.scoped) as connection:
            connection.execute("CREATE FUNCTION reject_block_repair() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'injected'; END $$")
            connection.execute("CREATE TRIGGER reject_block_repair BEFORE UPDATE ON structured_document_snapshots "
                "FOR EACH ROW EXECUTE FUNCTION reject_block_repair()")
        try:
            for action in ("review_block_edit", "review_block_keep", "review_block_delete"):
                with self.assertRaises(psycopg.Error):
                    self.change(action, region_id="r1", block_id=key,
                        **({"text": "Rollback", "discard_interactions": True} if action.endswith("edit") else {}))
                self.assertEqual(self.repo.load_document("book"), before)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("DROP TRIGGER reject_block_repair ON structured_document_snapshots")


if __name__ == "__main__":
    unittest.main()
