"""Narrow source-review transport and generation eligibility, without provider calls."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts import test_ocr_cleanup_overlay as fixtures
from section_tasks.reading_target_resolver import ReadingTargetResolver


class SourceReviewApiTests(unittest.TestCase):
    def setUp(self):
        import main
        self.fixture = fixtures.CleanupOverlayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo
        self.patch = patch.object(main, "document_artifact_repository", self.repo)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_response_excludes_internal_geometry_and_preserves_verified_targets(self):
        before = self.repo.load_document("book").to_dict()
        response = self.client.get("/documents/book/reading-content", params={"region_id": "r1"})
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        region = state["regions"][0]
        self.assertEqual(region["body_status"], "linked")
        self.assertEqual(region["mapping_status"], "complete")
        self.assertNotIn("canonical_span", region)
        self.assertNotIn("source_review_mapping", state)
        self.assertEqual(state["block_page"]["total"], 1)
        block = state["blocks"][0]
        for key in ("chapter_id", "section_id", "task_unit_id", "block_id"):
            self.assertTrue(block[key])
        self.assertEqual(before, self.repo.load_document("book").to_dict())

    def test_manual_delete_keeps_review_until_explicit_ignore(self):
        state = self.fixture.service.read("book", region_id="r1")
        command = dict(action="delete", expected_revision=state["revision"], source_token=state["source_token"],
                       mutation_id="delete", block_id=state["blocks"][0]["block_id"])
        response = self.client.post("/documents/book/reading-content", json=command)
        self.assertEqual(response.status_code, 200, response.text)
        state = response.json()
        self.assertEqual(state["regions"][0]["manual_change"]["status"], "deleted")
        self.assertEqual(state["regions"][0]["related_deleted_count"], 1)
        self.assertEqual(state["disposition"], "needs_review")
        replay = self.client.post("/documents/book/reading-content", json=command)
        self.assertEqual(replay.json(), state)
        response = self.client.post("/documents/book/reading-content", json=dict(action="discard_region",
            expected_revision=state["revision"], source_token=state["source_token"], mutation_id="ignore", region_id="r1"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["regions"], [])

    def test_quality_omission_keeps_consistent_list_and_all_level_guards(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_page_spans"].append({"page_index": 1,
            "char_start": len(document.raw_text), "char_end": len(document.raw_text)})
        provenance["source_review"]["regions"] = [{"id": "excluded", "page_index": 1,
            "raw_candidate": "not in canonical body", "reason": "page_quality_rejected"}]
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        before = self.repo.load_document("book")
        result = self.client.get("/documents/book/reading-content").json()
        self.assertEqual(result["regions"][0]["body_status"], "unknown")
        self.assertEqual(result["regions"][0]["block_ids"], [])
        self.assertEqual(result["disposition"], "needs_review")
        self.assertTrue(self.repo.list_documents()[0].review_required)
        block = self.fixture.service.read("book")["blocks"][0]
        resolver = ReadingTargetResolver()
        for level, args in (("document", {}), ("chapter", {"chapter_id": block["chapter_id"]}),
                            ("section", {"section_id": block["section_id"]}),
                            ("task_unit", {"task_unit_id": block["task_unit_id"]}),
                            ("content_block", {"task_unit_id": block["task_unit_id"],
                                               "content_block_id": block["block_id"]})):
            with self.subTest(level=level):
                target = resolver.resolve(document=before, target_level=level, **args)
                self.assertTrue(target.source_needs_review)
                self.assertNotIn("not in canonical body", target.content)
        self.assertEqual(before.to_dict(), self.repo.load_document("book").to_dict())
        self.assertNotIn("regions", before.parse_provenance.get("reading_content", {}))

    def test_confirmed_blank_exclusion_has_consistent_reader_and_library_state(self):
        from scripts.test_source_review_mapping import blank_document
        document = blank_document()
        self.repo.save_document(document, "book")
        before = self.repo.load_document("book")
        result = self.client.get("/documents/book/reading-content").json()
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["disposition"], "completed")
        self.assertFalse(self.repo.list_documents()[0].review_required)
        self.assertFalse(ReadingTargetResolver().resolve(document=before, target_level="document").source_needs_review)
        self.assertEqual(before.to_dict(), self.repo.load_document("book").to_dict())

    def test_unknown_region_remains_visible_and_invalid_queries_are_rejected(self):
        document = self.repo.load_document("book")
        provenance = deepcopy(document.parse_provenance)
        provenance["source_review"]["regions"][0]["canonical_span"]["text_sha256"] = "0" * 64
        self.repo.save_document(replace(document, parse_provenance=provenance), "book")
        result = self.client.get("/documents/book/reading-content").json()
        self.assertEqual(result["regions"][0]["body_status"], "unknown")
        self.assertEqual(result["disposition"], "needs_review")
        self.assertTrue(self.repo.list_documents()[0].review_required)
        self.assertEqual(self.client.get("/documents/book/reading-content?region_id=missing").status_code, 404)
        self.assertEqual(self.client.get("/documents/book/reading-content?block_offset=-1").status_code, 422)

    def test_body_and_region_queries_are_bounded_without_losing_targets(self):
        from doc_loaders.extraction_identity import read_extraction_record
        document = self.repo.load_document("book")
        text = "\n\n".join(f"Paragraph {index}: a complete sentence for bounded reading." for index in range(25))
        chapter = document.chapters[0]
        section = chapter.sections[0]
        unit = replace(section.task_units[0], content=text, content_blocks=[])
        section = replace(section, content=text, char_start=0, char_end=len(text), task_units=[unit])
        identity, _ = read_extraction_record(document.parse_provenance)
        provenance = deepcopy(document.parse_provenance)
        provenance["extraction"] = identity.record(text)
        provenance["source_page_spans"] = [{"page_index": 0, "char_start": 0, "char_end": len(text)}]
        provenance["source_review"]["regions"] = [{"id": "all", "page_index": 0, "raw_candidate": text}]
        self.repo.save_document(replace(document, raw_text=text,
            chapters=[replace(chapter, sections=[section])], parse_provenance=provenance), "book")
        before = self.repo.load_document("book").to_dict()
        expected = [b["block_id"] for b in self.fixture.service.read("book")["blocks"]]
        self.assertEqual(len(expected), 25)
        for params in ({"include_blocks": "true"}, {"region_id": "all"}):
            found = []
            for offset, size in ((0, 20), (20, 5), (40, 0)):
                response = self.client.get("/documents/book/reading-content", params={**params, "block_offset": offset})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["block_page"], {"offset": offset, "limit": 20, "total": 25})
                self.assertEqual(len(result["blocks"]), size)
                found.extend(b["block_id"] for b in result["blocks"])
            self.assertEqual(found, expected)
        self.assertEqual(before, self.repo.load_document("book").to_dict())


if __name__ == "__main__":
    unittest.main()
