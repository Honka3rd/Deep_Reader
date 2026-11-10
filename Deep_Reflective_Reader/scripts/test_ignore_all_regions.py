"""Irreversible bulk decisions on file and disposable PostgreSQL repositories."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from doc_loaders.extraction_identity import ExtractionIdentity
from doc_loaders.ocr_review import region_record, review_record
from document_structure.ocr_cleanup_overlay import canonical_text
from document_structure.reading_content import ReadingContentService, ReadingContentConflict, state_for, review_regions
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from scripts.test_reading_interaction_artifact_store import _document
from scripts import test_summary_note_postgres as sql_tests


class IgnoreAllCases:
    def initialize(self):
        document = _document()
        document = replace(document, raw_text="".join(s.content for c in document.chapters for s in c.sections))
        identity = ExtractionIdentity.create(source_sha256="a" * 64, source_format="pdf",
            extractor_id="test", extraction_version="v1", mode="ocr", settings={})
        regions = [region_record(i, 0, [0, 0, 100, 100], 200, "recognition_uncertain") for i in range(25)]
        document = replace(document, parse_provenance={"extraction": identity.record(canonical_text(document)),
            "source_review": review_record(identity.source_sha256, regions)})
        self.repo.save_reparsed_document(document, "book")
        self.service = ReadingContentService(self.repo)
        self.initial = self.service.read("book")
        self.block = self.initial["blocks"][0]["block_id"]

    def command(self, action="discard_all_regions", **values):
        state = self.service.read("book")
        return dict(action=action, expected_revision=state["revision"], source_token=state["source_token"],
                    mutation_id=uuid4().hex, **values)

    def change(self, action="discard_all_regions", **values):
        return self.service.mutate("book", **self.command(action, **values))

    def test_all_pages_immutable_body_audit_replay_and_no_undo(self):
        before = self.repo.load_document("book")
        command = self.command()
        result = self.service.mutate("book", **command)
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["blocks"], self.initial["blocks"])
        self.assertEqual(result["disposition"], "completed")
        self.assertFalse(result["can_undo"])
        self.assertFalse(result["refresh"]["body_changed"])
        self.assertEqual(self.service.mutate("book", **command), result)
        after = self.repo.load_document("book")
        self.assertEqual(after.raw_text, before.raw_text)
        self.assertEqual(after.chapters, before.chapters)
        self.assertEqual(after.parse_provenance["source_review"], before.parse_provenance["source_review"])
        state = state_for(after)
        self.assertEqual(len(state["permanently_ignored_regions"]), 25)
        self.assertEqual(len(state["audit"][-1]["ignored_regions"]), 25)
        self.assertEqual(len(state["audit"]), 1)
        self.assertTrue(all(r["status"] == "discarded" for r in review_regions(after)))
        with self.assertRaisesRegex(ReadingContentConflict, "nothing_to_undo"): self.change("undo")
        with self.assertRaisesRegex(ReadingContentConflict, "review_regions_missing"): self.change()

    def test_earlier_and_later_body_undo_cannot_reopen_bulk_flags(self):
        self.change("edit", block_id=self.block, text="Edited before ignore", discard_interactions=False)
        result = self.change()
        self.assertTrue(result["can_undo"])
        self.assertEqual(result["undo_action"], "edit")
        restored = self.change("undo")
        self.assertEqual(restored["regions"], [])
        self.assertEqual(restored["blocks"][0]["content"], self.initial["blocks"][0]["content"])
        self.change("edit", block_id=self.block, text="Edited after ignore", discard_interactions=False)
        self.assertEqual(self.change("undo")["regions"], [])

    def test_single_ignore_remains_reversible_but_bulk_is_permanent(self):
        first = self.initial["regions"][0]["id"]
        self.change("discard_region", region_id=first)
        self.change()
        self.assertEqual([r["id"] for r in self.change("undo")["regions"]], [first])

    def test_stale_conflict_and_concurrent_commands_are_atomic(self):
        stale = self.command()
        self.change("edit", block_id=self.block, text="Concurrent edit", discard_interactions=False)
        with self.assertRaisesRegex(ReadingContentConflict, "changed"):
            self.service.mutate("book", **stale)
        self.assertEqual(len(self.service.read("book")["regions"]), 25)
        commands = [self.command(), self.command()]
        def send(command):
            try:
                self.service.mutate("book", **command)
                return True
            except ReadingContentConflict:
                return False
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(sum(pool.map(send, commands)), 1)
        self.assertEqual(len(state_for(self.repo.load_document("book"))["permanently_ignored_regions"]), 25)

    def test_api_rejects_unexpected_fields_and_preserves_idempotency(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "_reading_content_service", return_value=self.service):
            client = TestClient(main.app)
            command = self.command()
            for key, value in {"block_id": "b", "region_id": "r", "text": "x", "discard_interactions": False,
                               "cleanup_record_id": "a" * 64}.items():
                self.assertEqual(client.post("/documents/book/reading-content", json={**command, key: value}).status_code, 422)
                with self.assertRaises(ValueError): self.service.mutate("book", **{**command, key: value})
            response = client.post("/documents/book/reading-content", json=command)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["regions"], [])
            self.assertEqual(client.post("/documents/book/reading-content", json=command).json(), response.json())
            self.assertEqual(client.post("/documents/book/reading-content", json={**command, "mutation_id": "stale"}).status_code, 409)

    def test_non_ocr_rejected_without_mutation(self):
        document = self.repo.load_document("book")
        self.repo.save_reparsed_document(replace(document, parse_provenance={}), "book")
        before = self.repo.load_document("book")
        with self.assertRaisesRegex(ReadingContentConflict, "ocr_review_unavailable"): self.change()
        self.assertEqual(self.repo.load_document("book"), before)


class FileIgnoreAll(IgnoreAllCases, unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=temp.name)
        self.initialize()

    def test_failed_commit_preserves_all_regions_and_receipts(self):
        with patch.object(self.repo, "_atomic_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError): self.change()
        self.assertEqual(self.service.read("book"), self.initial)


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "Disposable PostgreSQL required")
class PostgresIgnoreAll(IgnoreAllCases, unittest.TestCase):
    setUpClass = classmethod(sql_tests.PostgresNotes.setUpClass.__func__)
    drop_schema = classmethod(sql_tests.PostgresNotes.drop_schema.__func__)

    def setUp(self):
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        self.repo = PostgresStructuredDocumentArtifactRepository(PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))
        self.initialize()

    def test_failed_commit_after_document_write_rolls_back_every_decision(self):
        import psycopg
        with psycopg.connect(self.scoped) as conn:
            conn.execute("CREATE OR REPLACE FUNCTION reject_bulk_snapshot() RETURNS trigger LANGUAGE plpgsql "
                         "AS $$ BEGIN RAISE EXCEPTION 'injected failure'; END $$")
            conn.execute("CREATE TRIGGER reject_bulk_snapshot BEFORE UPDATE ON structured_document_snapshots "
                         "FOR EACH ROW EXECUTE FUNCTION reject_bulk_snapshot()")
        try:
            with self.assertRaises(psycopg.Error): self.change()
            self.assertEqual(self.service.read("book"), self.initial)
            self.assertNotIn("permanently_ignored_regions", state_for(self.repo.load_document("book")))
        finally:
            with psycopg.connect(self.scoped) as conn:
                conn.execute("DROP TRIGGER reject_bulk_snapshot ON structured_document_snapshots")


if __name__ == "__main__":
    unittest.main()
