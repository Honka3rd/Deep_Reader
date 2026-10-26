"""Version-bound partial reading refresh on file/SQL repositories and real HTTP DTOs."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest

from fastapi.testclient import TestClient

from document_structure.reading_content import ReadingContentConflict, state_for
from scripts.test_cleanup_history import HistoryCases
from scripts.test_ocr_cleanup_contract import item
from scripts.test_ocr_cleanup_overlay import CleanupOverlayTests
from scripts.test_ocr_cleanup_overlay_postgres import PostgresCleanupOverlayTests


class RefreshCases:
    rows = HistoryCases.rows
    command = HistoryCases.command

    def test_history_save_delete_http_delta_and_stale_receipt(self):
        self.commit(*self.prepare())
        row, = self.rows()
        client = self.client()
        saved = client.post("/documents/book/reading-content", json=self.command(row, "cleanup_edit", text="Draft"))
        self.assertEqual(saved.status_code, 200, saved.text)
        delta = saved.json()["refresh"]["cleanup_change"]
        self.assertEqual(delta["record"]["text"], "Draft")
        self.assertTrue(delta["record"]["can_delete"])
        command = self.command(row, "cleanup_delete")
        removed = client.post("/documents/book/reading-content", json=command)
        self.assertEqual(removed.status_code, 200, removed.text)
        refresh = removed.json()["refresh"]
        self.assertEqual(refresh["blocks"], [])
        self.assertEqual(refresh["cleanup_change"], {"batch_revision": 1, "record_id": row["record_id"],
            "deleted": True, "applied_count": 0, "remaining_count": 0})
        self.assertEqual(client.post("/documents/book/reading-content", json=command).json(), removed.json())
        self.change("undo")
        stale = client.post("/documents/book/reading-content", json=command).json()["refresh"]
        self.assertFalse(stale["scope_current"])
        self.assertNotIn("cleanup_change", stale)

    def client(self):
        import main
        service = patch.object(main, "_reading_content_service", return_value=self.service)
        service.start()
        self.addCleanup(service.stop)
        client = TestClient(main.app)
        self.addCleanup(client.close)
        return client

    def refresh_query(self, result, offset=0):
        return {"refresh_mutation_id": result["mutation_id"], "expected_revision": result["revision"],
                "source_token": result["source_token"], "block_offset": offset}

    def test_post_and_read_only_refresh_return_only_affected_block(self):
        self.commit(*self.prepare())
        row, = self.rows()
        client = self.client()
        command = self.command(row, text="Recovered")
        response = client.post("/documents/book/reading-content", json=command)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["blocks"], [])  # Old response remains compatible.
        refresh = data["refresh"]
        self.assertTrue(refresh["scope_current"])
        self.assertTrue(refresh["generated_invalidated"])
        self.assertEqual(refresh["cleanup_record_id"], row["record_id"])
        self.assertEqual(refresh["block_page"], {"total": 1, "offset": 0, "limit": 20})
        block, = refresh["blocks"]
        self.assertEqual(block["block_id"], row["links"][0]["block_id"])
        self.assertEqual(block["content_block"]["content"], "Recovered")
        self.assertEqual(block["block_order"], 0)
        self.assertFalse(block["deleted"])
        self.assertFalse(block["manually_edited"])
        self.assertEqual(block["unit_source_hash"], block["content_block"]["metadata"]["source_hash"])
        before = self.snapshot()
        with patch.object(self.service, "snapshot", side_effect=AssertionError("full snapshot on refresh")):
            response = client.get("/documents/book/reading-content", params=self.refresh_query(refresh))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), refresh)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(client.post("/documents/book/reading-content", json=command).json(), data)
        for private in ("cleanup_record_applications", "cleanup_record_links", "refresh_source_token", "audit", "chapters"):
            self.assertNotIn(private, response.text)

    def test_draft_has_no_body_scope_and_reapply_updates_same_target(self):
        self.commit(*self.prepare())
        row, = self.rows()
        original = self.service.mutate("book", **self.command(row, text="First"))["refresh"]
        saved = self.service.mutate("book", **self.command(row, "cleanup_edit", text="Second"))["refresh"]
        self.assertFalse(saved["generated_invalidated"])
        self.assertEqual(saved["block_page"]["total"], 0)
        self.assertEqual(saved["blocks"], [])
        applied = self.service.mutate("book", **self.command(row))["refresh"]
        self.assertEqual(applied["blocks"][0]["block_id"], original["blocks"][0]["block_id"])
        self.assertEqual(applied["blocks"][0]["content_block"]["content"], "Second")
        self.assertNotEqual(applied["blocks"][0]["unit_source_hash"], original["blocks"][0]["unit_source_hash"])

    def test_stale_replay_acknowledges_commit_without_stale_body_patch(self):
        self.commit(*self.prepare())
        row, = self.rows()
        client = self.client()
        command = self.command(row, text="First")
        first = client.post("/documents/book/reading-content", json=command).json()["refresh"]
        self.service.mutate("book", **self.command(row, "cleanup_edit", text="Second"))
        before = self.snapshot()
        replay = client.post("/documents/book/reading-content", json=command)
        self.assertEqual(replay.status_code, 200, replay.text)
        latest = replay.json()["refresh"]
        self.assertFalse(latest["scope_current"])
        self.assertEqual(latest["mutation_revision"], first["revision"])
        self.assertGreater(latest["revision"], latest["mutation_revision"])
        self.assertEqual(latest["blocks"], [])
        for params in (self.refresh_query(first), self.refresh_query(latest)):
            self.assertEqual(client.get("/documents/book/reading-content", params=params).status_code, 409)
        self.assertEqual(self.snapshot(), before)

    def test_cross_block_restore_and_undo_report_every_visibility_change(self):
        self.commit(*self.prepare(items=[item(quote="@@??\n\nMeaningful")]))
        row, = self.rows()
        restored = self.service.mutate("book", **self.command(row, text="Recovered"))["refresh"]
        self.assertEqual({b["block_id"] for b in restored["blocks"]}, {l["block_id"] for l in row["links"]})
        self.assertEqual(restored["blocks"][0]["content_block"]["content"], "Recovered")
        undone = self.change("undo")["refresh"]
        self.assertEqual(undone["blocks"][0]["block_id"], row["links"][0]["block_id"])
        self.assertTrue(undone["blocks"][0]["deleted"])
        self.assertIsNone(undone["blocks"][0]["content_block"])

    def test_undo_cleanup_and_manual_delete_have_precise_scope(self):
        self.commit(*self.prepare())
        restored = self.change("undo")["refresh"]
        block, = restored["blocks"]
        self.assertEqual(block["content_block"]["content"], "@@??")
        removed = self.change("delete", block_id=block["block_id"])["refresh"]
        self.assertEqual(removed["cleared_block_id"], block["block_id"])
        self.assertTrue(removed["blocks"][0]["deleted"])
        self.assertEqual(self.change("undo")["refresh"]["blocks"][0]["content_block"]["content"], "@@??")

    def test_legacy_receipt_has_no_invented_refresh_scope(self):
        self.commit(*self.prepare())
        row, = self.rows()
        command = self.command(row)
        first = self.service.mutate("book", **command)["refresh"]
        def legacy(document):
            updated = deepcopy(document)
            state_for(updated)["audit"][-1].pop("refresh_source_token")
            return updated, None
        self.repo.mutate_reading_content("book", legacy)
        before = self.snapshot()
        self.assertIsNone(self.service.mutate("book", **command)["refresh"])
        with self.assertRaisesRegex(ReadingContentConflict, "refresh_unavailable"):
            self.service.read_refresh("book", mutation_id=first["mutation_id"],
                expected_revision=first["revision"], source_token=first["source_token"])
        self.assertEqual(self.snapshot(), before)

    def test_source_change_invalidates_refresh_even_at_same_revision(self):
        self.commit(*self.prepare())
        row, = self.rows()
        command = self.command(row)
        first = self.service.mutate("book", **command)["refresh"]
        document = self.repo.load_document("book")
        self.repo.save_document(replace(document, parse_provenance={
            **document.parse_provenance, "interaction_epoch": "changed-source-binding"}), "book")
        self.assertEqual(self.repo.load_document("book").parse_provenance["interaction_epoch"],
            "changed-source-binding")
        self.assertEqual(state_for(self.repo.load_document("book"))["revision"], first["revision"])
        before = self.snapshot()
        with self.assertRaisesRegex(ReadingContentConflict, "changed_reload"):
            self.service.read_refresh("book", mutation_id=first["mutation_id"],
                expected_revision=first["revision"], source_token=first["source_token"])
        result = self.service.mutate("book", **command)["refresh"]
        self.assertFalse(result["scope_current"])
        self.assertEqual(result["blocks"], [])
        self.assertEqual(self.snapshot(), before)

    def test_refresh_construction_failure_rolls_back_mutation(self):
        self.commit(*self.prepare())
        row, = self.rows()
        before = self.snapshot()
        with patch("document_structure.reading_refresh.mutation_refresh", side_effect=RuntimeError("refresh failed")):
            with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                self.service.mutate("book", **self.command(row, text="Changed"))
        self.assertEqual(self.snapshot(), before)

    def test_response_schema_rejects_stale_or_misidentified_patches(self):
        from api_schemas import ReadingContentRefreshResponse
        from pydantic import ValidationError
        self.commit(*self.prepare())
        row, = self.rows()
        first = self.service.mutate("book", **self.command(row))["refresh"]
        ReadingContentRefreshResponse.model_validate(first)
        broken = []
        value = deepcopy(first); value["scope_current"] = False; broken.append(value)
        value = deepcopy(first); value["mutation_revision"] += 1; broken.append(value)
        value = deepcopy(first); value["blocks"][0]["deleted"] = True; broken.append(value)
        value = deepcopy(first); value["blocks"][0]["content_block"]["block_id"] = "other"; broken.append(value)
        value = deepcopy(first); value["blocks"] *= 2; broken.append(value)
        value = deepcopy(first); value["block_page"]["total"] += 1; broken.append(value)
        for value in broken:
            with self.assertRaises(ValidationError):
                ReadingContentRefreshResponse.model_validate(value)

    def test_query_validation_and_unknown_mutation_do_not_write(self):
        self.commit(*self.prepare())
        row, = self.rows()
        first = self.service.mutate("book", **self.command(row))["refresh"]
        query = self.refresh_query(first)
        client = self.client()
        before = self.snapshot()
        for changes in ({"region_id": "r1"}, {"block_id": "injected"}, {"include_blocks": True},
                        {"include_cleanup_audit": True}, {"cleanup_offset": 10}, {"source_token": "0"*64},
                        {"refresh_mutation_id": "unknown"}):
            response = client.get("/documents/book/reading-content", params={**query, **changes})
            self.assertEqual(response.status_code, 409, response.text)
        for key in ("source_token", "expected_revision"):
            response = client.get("/documents/book/reading-content", params={k:v for k,v in query.items() if k != key})
            self.assertEqual(response.status_code, 409)
        self.assertEqual(client.get("/documents/book/reading-content", params={**query, "block_offset": -1}).status_code, 422)
        self.assertEqual(self.snapshot(), before)

    def test_large_cross_block_refresh_pages_and_anchor_without_full_read(self):
        from doc_loaders.extraction_identity import read_extraction_record
        import document_structure.reading_refresh as projection
        document = self.repo.load_document("book")
        paragraphs = [f"Paragraph {index}: a complete sentence for bounded reading." for index in range(26)]
        text = "\n\n".join(paragraphs)
        chapter, section = document.chapters[0], document.chapters[0].sections[0]
        unit = replace(section.task_units[0], content=text, content_blocks=[])
        section = replace(section, content=text, char_start=0, char_end=len(text), task_units=[unit])
        identity, _ = read_extraction_record(document.parse_provenance)
        provenance = {**document.parse_provenance, "extraction": identity.record(text),
            "source_page_spans": [{"page_index": 0, "char_start": 0, "char_end": len(text)}],
            "source_review": {"regions": []}}
        self.repo.save_document(replace(document, raw_text=text,
            chapters=[replace(chapter, sections=[section])], parse_provenance=provenance), "book")
        blocks = self.service.read("book")["blocks"]
        self.assertEqual(len(blocks), 26)
        self.commit(*self.prepare(items=[item(quote="\n\n".join(paragraphs[:25]))]))
        row, = self.rows()
        first = self.service.mutate("book", **self.command(row))["refresh"]
        self.assertEqual(len(first["blocks"]), 20)
        self.assertEqual(first["block_page"]["total"], 25)
        client = self.client()
        found = []
        before = self.snapshot()
        for offset, size in ((0, 20), (20, 5), (40, 0)):
            with patch.object(self.service, "snapshot", side_effect=AssertionError("full snapshot")), \
                    patch.object(projection, "project_task_unit", wraps=projection.project_task_unit) as project:
                response = client.get("/documents/book/reading-content", params=self.refresh_query(first, offset))
            self.assertEqual(response.status_code, 200, response.text)
            page = response.json()
            self.assertEqual(len(page["blocks"]), size)
            self.assertLessEqual(project.call_count, 1)
            for block in page["blocks"]:
                self.assertEqual(block["insert_before_block_id"], blocks[25]["block_id"])
                found.append(block["block_id"])
        self.assertEqual(found, [block["block_id"] for block in blocks[:25]])
        self.assertEqual(self.snapshot(), before)


class FileRefreshTests(RefreshCases, CleanupOverlayTests):
    pass


class SqlRefreshTests(RefreshCases, PostgresCleanupOverlayTests):
    pass


if __name__ == "__main__":
    unittest.main()
