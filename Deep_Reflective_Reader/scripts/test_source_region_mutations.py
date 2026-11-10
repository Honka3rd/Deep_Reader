"""Region repair writes through real file/SQL ports and HTTP serialization."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from document_structure.reading_content import ReadingContentConflict, ReadingContentService, state_for
from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.source_region_repair import region_repair_targets
from scripts import test_ocr_cleanup_overlay as fixtures
from scripts import test_source_review_blocks as review


class RegionMutations(unittest.TestCase):
    setUp = fixtures.CleanupOverlayTests.setUp
    repository = fixtures.CleanupOverlayTests.repository
    change = fixtures.CleanupOverlayTests.change
    content = fixtures.CleanupOverlayTests.content
    note_target = review.SourceReviewBlockTests.note_target
    add_note = review.SourceReviewBlockTests.add_note

    def region(self):
        return self.service.read("book")["regions"][0]

    def apply(self, text=None, discard=False):
        return self.change("region_apply", region_id="r1", text=text, discard_interactions=discard)

    def test_drafts_reapply_delete_restore_and_undo(self):
        original = self.repo.load_document("book")
        self.change("region_edit", region_id="r1", text="First correction")
        self.assertIn("@@??", self.content())
        self.apply()
        self.assertIn("First correction", self.content())
        for text in ("Second correction", "Third correction", "@@??"):
            self.change("region_edit", region_id="r1", text=text)
            self.apply()
            self.assertTrue(self.content().startswith(text))
            self.assertEqual(self.content().count(text), 1)
            self.assertTrue(self.region()["repair"]["can_apply"])
        deleted = self.change("region_delete", region_id="r1")
        self.assertTrue(deleted["refresh"]["blocks"][0]["deleted"])
        self.assertEqual(self.region()["repair"]["applied_text"], "")
        self.assertTrue(self.region()["repair"]["can_apply"])
        self.assertEqual(len(deleted["regions"]), 1)
        self.apply("Restored")
        self.assertFalse(self.service.read("book")["blocks"][0]["manually_edited"])
        self.change("undo")
        self.assertNotIn("Restored", self.content())
        self.assertTrue(self.region()["repair"]["can_apply"])
        current = self.repo.load_document("book")
        self.assertEqual(current.chapters, original.chapters)
        self.assertEqual(canonical_text(current), canonical_text(original))

    def test_manual_edit_and_delete_are_permanent_protections(self):
        self.apply("Region correction")
        key = self.service.read("book")["blocks"][0]["block_id"]
        self.change("edit", block_id=key, text="Body correction", discard_interactions=False)
        self.change("undo")
        before = self.repo.load_document("book")
        for action in ("region_apply", "region_delete", "region_edit"):
            with self.assertRaisesRegex(ReadingContentConflict, "block_manually_edited"):
                self.change(action, region_id="r1", **({"discard_interactions": False} if action == "region_apply"
                            else {"text": "Draft"} if action == "region_edit" else {}))
            self.assertEqual(self.repo.load_document("book"), before)

    def test_manual_delete_undo_cannot_allow_region_writes(self):
        key = self.service.read("book")["blocks"][0]["block_id"]
        self.change("delete", block_id=key)
        self.change("undo")
        with self.assertRaisesRegex(ReadingContentConflict, "block_manually_deleted"):
            self.apply("No overwrite")

    def test_note_keep_clear_and_delete_epochs(self):
        service, target, note = self.add_note()
        self.apply("Keep note")
        self.assertEqual(service.read("book", **target)["note"]["text"], "User note")
        self.apply("Clear note", discard=True)
        self.assertIsNone(service.read("book", **target)["note"])
        self.assertIn(target["content_block_id"], state_for(self.repo.load_document("book"))["block_interaction_epochs"])
        self.add_note()
        self.change("region_delete", region_id="r1")
        self.change("undo")
        self.assertIsNone(service.read("book", **target)["note"])

    def test_foreign_overlap_and_changed_application_are_rejected(self):
        self.apply("Own repair")
        def corrupt(document):
            state = deepcopy(state_for(document))
            state["source_edits"][0]["replacement"] = "Foreign"
            return replace(document, parse_provenance={**document.parse_provenance, "reading_content": state}), None
        self.repo.mutate_reading_content("book", corrupt)
        with self.assertRaisesRegex(ReadingContentConflict, "region_application_changed"):
            self.apply("Must fail")

    def test_other_region_cannot_replace_owned_range(self):
        def duplicate(document):
            provenance = deepcopy(document.parse_provenance)
            region = deepcopy(provenance["source_review"]["regions"][0])
            region["id"] = "r2"
            provenance["source_review"]["regions"].append(region)
            return replace(document, parse_provenance=provenance), None
        self.repo.mutate_reading_content("book", duplicate)
        self.apply("Owner repair")
        before = self.repo.load_document("book")
        for action in ("region_apply", "region_delete"):
            with self.assertRaisesRegex(ReadingContentConflict, "reading_range_changed"):
                self.change(action, region_id="r2", **({"discard_interactions": False} if action == "region_apply" else {}))
        self.assertEqual(self.repo.load_document("book"), before)

    def test_one_protected_cross_block_target_rolls_back_everything(self):
        def expand(document):
            provenance = deepcopy(document.parse_provenance)
            region = provenance["source_review"]["regions"][0]
            region["raw_candidate"] = canonical_text(document)
            region["canonical_span"]["char_end"] = len(region["raw_candidate"])
            return replace(document, parse_provenance=provenance), None
        self.repo.mutate_reading_content("book", expand)
        blocks = self.service.read("book")["blocks"]
        self.change("edit", block_id=blocks[1]["block_id"], text="Protected", discard_interactions=False)
        before = self.repo.load_document("book")
        with self.assertRaisesRegex(ReadingContentConflict, "block_manually_edited"):
            self.change("region_delete", region_id="r1")
        self.assertEqual(self.repo.load_document("book"), before)

    def test_cross_block_delete_clears_all_notes_and_refreshes_all(self):
        original = self.repo.load_document("book")
        text = canonical_text(original)
        def expand(document):
            provenance = deepcopy(document.parse_provenance)
            region = provenance["source_review"]["regions"][0]
            region["raw_candidate"] = text
            region["canonical_span"]["char_end"] = len(text)
            return replace(document, parse_provenance=provenance), None
        self.repo.mutate_reading_content("book", expand)
        self.add_note(0)
        self.add_note(1)
        result = self.change("region_delete", region_id="r1")
        self.assertEqual(len(result["refresh"]["cleared_block_ids"]), 2)
        self.assertTrue(all(row["deleted"] for row in result["refresh"]["blocks"]))
        self.assertEqual(self.repo.load_document("book").document_task_artifacts.metadata["note"], {"keep": "annotation"})
        self.apply("Cross-block recovery")
        self.assertEqual(self.content(), "Cross-block recovery")
        self.assertFalse(any(b["manually_edited"] for b in self.service.read("book")["blocks"]))

    def test_idempotent_replay_conflict_and_concurrent_commit(self):
        state = self.service.read("book")
        command = dict(action="region_apply", region_id="r1", text="Only once", discard_interactions=False,
                       source_token=state["source_token"], expected_revision=state["revision"], mutation_id=uuid4().hex)
        first = self.service.mutate("book", **command)
        self.assertEqual(self.service.mutate("book", **command), first)
        with self.assertRaises(ReadingContentConflict):
            self.service.mutate("book", **{**command, "text": "Different"})
        # Race a real body change: identical reapply is intentionally a no-op
        # and does not consume the source revision.
        command.update(expected_revision=first["revision"], text="Concurrent correction")
        def attempt(_):
            try:
                self.service.mutate("book", **{**command, "mutation_id": uuid4().hex})
                return "ok"
            except ReadingContentConflict:
                return "conflict"
        with ThreadPoolExecutor(2) as executor:
            self.assertEqual(sorted(executor.map(attempt, range(2))), ["conflict", "ok"])

    def test_projection_failure_rolls_back(self):
        before = self.repo.load_document("book")
        with patch("document_structure.source_region_repair.project_document", side_effect=ReadingContentConflict("injected")):
            with self.assertRaises(ReadingContentConflict):
                self.apply("Must roll back")
        self.assertEqual(self.repo.load_document("book"), before)

    def test_http_contract_and_invalid_commands(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "_reading_content_service", return_value=self.service), TestClient(main.app) as client:
            state = client.get("/documents/book/reading-content").json()
            self.assertTrue(state["regions"][0]["repair"]["can_apply"])
            self.assertNotIn("block_spans", state["regions"][0])
            command = dict(action="region_apply", region_id="r1", text="HTTP correction", discard_interactions=False,
                           expected_revision=state["revision"], source_token=state["source_token"], mutation_id=uuid4().hex)
            for extra in ({"block_id": "wrong"}, {"discard_interactions": None}, {"text": ""}):
                response = client.post("/documents/book/reading-content", json={**command, **extra})
                self.assertEqual(response.status_code, 422, response.text)
            response = client.post("/documents/book/reading-content", json=command)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["regions"][0]["repair"]["applied_text"], "HTTP correction")
            self.assertFalse(result["refresh"]["blocks"][0]["manually_edited"])
            self.assertEqual(result["blocks"], [])

    @unittest.skipUnless(os.environ.get("SOURCE_REGION_REPAIR_REAL"), "Private backup acceptance opt-in")
    def test_real_page_repairs_preserve_backup_and_canonical_text(self):
        import hashlib
        from scripts.accept_source_review_mapping import load_cases, real_page_case
        cases, hashes = load_cases()
        for label, _, source in cases:
            with self.subTest(book=label):
                document, evidence = real_page_case(source)
                self.assertFalse(evidence["natural_loader_region"])
                self.assertFalse(evidence["canonical_span_injected"])
                name = "region-acceptance-" + label
                repo = self.repository()
                repo.save_reparsed_document(document, name)
                service = ReadingContentService(repo)
                canonical = canonical_text(repo.load_document(name))
                for replacement in ("Verified real-page repair", "Repeated real-page repair"):
                    state = service.read(name)
                    result = service.mutate(name, action="region_apply", region_id="real-subquote", text=replacement,
                        discard_interactions=False, expected_revision=state["revision"], source_token=state["source_token"],
                        mutation_id=uuid4().hex)
                    self.assertTrue(result["refresh"]["scope_current"])
                    self.assertFalse(any(row["manually_edited"] for row in result["refresh"]["blocks"]))
                    self.assertEqual(canonical_text(repo.load_document(name)), canonical)
                state = service.read(name)
                service.mutate(name, action="region_delete", region_id="real-subquote", expected_revision=state["revision"],
                    source_token=state["source_token"], mutation_id=uuid4().hex)
                self.assertEqual(canonical_text(repo.load_document(name)), canonical)
        for path, expected in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "Dedicated test DSN required")
class SqlRegionMutations(RegionMutations):
    from scripts.test_summary_note_postgres import PostgresNotes
    setUpClass = classmethod(PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(PostgresNotes.drop_schema.__func__)

    def repository(self):
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        return PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))

    def test_cross_block_physical_notes_are_purged_atomically(self):
        import psycopg
        from psycopg.types.json import Jsonb
        blocks = self.service.read("book")["blocks"][:2]
        with psycopg.connect(self.scoped) as conn:
            document_id = conn.execute("SELECT document_id FROM task_units WHERE id=%s",
                (int(blocks[0]["task_unit_id"]),)).fetchone()[0]
            for index, block in enumerate(blocks):
                row_id = conn.execute("INSERT INTO content_blocks(document_id,task_unit_id,block_order,content,source_structure_version,metadata_payload) "
                    "VALUES (%s,%s,%s,%s,1,%s) RETURNING id", (document_id, int(block["task_unit_id"]), index,
                    block["content"], Jsonb({"block_id": block["block_id"]}))).fetchone()[0]
                conn.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,source_structure_version,payload) "
                    "VALUES (%s,'note','content_block',%s,1,%s)", (document_id, row_id, Jsonb({"text": "Physical note"})))
        self.test_cross_block_delete_clears_all_notes_and_refreshes_all()
        with psycopg.connect(self.scoped) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM artifacts WHERE document_id=%s AND artifact_type='note'",
                (document_id,)).fetchone()[0], 0)

    def test_post_sql_failure_rolls_back_and_snapshot_matches(self):
        import psycopg
        before = self.repo.load_document("book")
        with psycopg.connect(self.scoped) as connection:
            connection.execute("CREATE FUNCTION reject_region_write() RETURNS trigger LANGUAGE plpgsql AS "
                               "$$ BEGIN RAISE EXCEPTION 'injected'; END $$")
            connection.execute("CREATE TRIGGER reject_region_write BEFORE UPDATE ON structured_document_snapshots "
                               "FOR EACH ROW EXECUTE FUNCTION reject_region_write()")
        try:
            with self.assertRaises(psycopg.Error):
                self.apply("Rollback after metadata and artifact SQL")
            self.assertEqual(self.repo.load_document("book"), before)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("DROP TRIGGER reject_region_write ON structured_document_snapshots")
        self.apply("Committed")
        with psycopg.connect(self.scoped) as connection:
            rows = connection.execute("SELECT d.metadata_payload->'parse_provenance'->'reading_content', "
                "s.structured_document_payload->'parse_provenance'->'reading_content' FROM documents d "
                "JOIN structured_document_snapshots s ON s.document_id=d.id WHERE d.namespace=%s",
                (self.repo.store._namespace,)).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], rows[0][1])


if __name__ == "__main__":
    unittest.main()
