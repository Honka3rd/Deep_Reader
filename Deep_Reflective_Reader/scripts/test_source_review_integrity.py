"""Source-review integrity regressions using memory and temporary storage only."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from document_structure.reading_content import (
    ReadingContentConflict, ReadingContentService, needs_review, original_blocks,
    pending_review_regions, source_token_for,
)
from document_structure.source_review_blocks import (
    needs_review_from_provenance, persist_region_mapping, region_block_mapping,
)
from document_structure.structured_document_store import StructuredDocumentStore
from scripts.test_source_review_mapping import exact_region, mapped_document, with_regions
from shared.task_unit_model import TaskUnitContentBlock


class MemoryRepository:
    def __init__(self, document):
        self.document = document

    def mutate_reading_content(self, name, mutation):
        self.document, result = mutation(self.document)
        return result


def excluded_document():
    document = mapped_document()
    document = with_regions(document, {
        "id": "excluded", "page_index": 1, "raw_candidate": "", "blank_render_confirmed": True,
    })
    document.parse_provenance["source_page_spans"].append({
        "page_index": 1, "char_start": len(document.raw_text), "char_end": len(document.raw_text),
    })
    return document


def title_change():
    document = mapped_document("Heading\nBody paragraph.\n\nTail paragraph.")
    chapter = document.chapters[0]
    section = chapter.sections[0]
    document = replace(document, chapters=[replace(chapter, title="Other", sections=[
        replace(section, title="Other"),
    ])])
    before = persist_region_mapping(with_regions(document, exact_region(document, "r", 0, 7)))
    chapter = before.chapters[0]
    after = replace(before, chapters=[replace(chapter, sections=[
        replace(chapter.sections[0], title="Heading"),
    ])])
    return before, after


class SourceReviewIntegrityTests(unittest.TestCase):
    def test_exclusion_requires_explicit_proof_and_reads_are_pure(self):
        document = excluded_document()
        before = deepcopy(document.to_dict())
        self.assertTrue(needs_review_from_provenance(document.parse_provenance))
        self.assertEqual(region_block_mapping(document)["excluded"]["body_status"], "unknown")
        staged = persist_region_mapping(document)
        self.assertFalse(needs_review_from_provenance(staged.parse_provenance))
        self.assertEqual(region_block_mapping(staged)["excluded"]["body_status"], "excluded")
        self.assertEqual(document.to_dict(), before)

    def test_canonical_mismatch_cannot_be_blessed_by_explicit_write(self):
        for field, value in (("text_sha256", "0" * 64), ("text_length", 99999)):
            with self.subTest(field=field):
                document = persist_region_mapping(excluded_document())
                document.parse_provenance["extraction"][field] = value
                self.assertTrue(needs_review_from_provenance(document.parse_provenance))
                staged = persist_region_mapping(document)
                self.assertTrue(needs_review_from_provenance(staged.parse_provenance))

    def test_initial_store_save_persists_proof_without_mutating_input(self):
        document = excluded_document()
        before = deepcopy(document.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "book.json")
            StructuredDocumentStore.save(document, path)
            loaded = StructuredDocumentStore.load(path)
        self.assertEqual(document.to_dict(), before)
        self.assertFalse(needs_review_from_provenance(loaded.parse_provenance))
        self.assertEqual(region_block_mapping(loaded)["excluded"]["body_status"], "excluded")

    def test_initial_store_roundtrip_preserves_custom_blocks_and_cached_links(self):
        document = mapped_document()
        section = document.chapters[0].sections[0]
        unit = section.task_units[0]
        quote = "Beta paragraph."
        start = unit.content.index(quote)
        block = TaskUnitContentBlock(block_id="custom-beta", content=quote, metadata={
            "source_hash": hashlib.sha256(unit.content.encode()).hexdigest(),
            "quote_span_start": start, "quote_span_end": start + len(quote),
        })
        section = replace(section, task_units=[replace(unit, content_blocks=[block])])
        document = replace(document, chapters=[replace(document.chapters[0], sections=[section])])
        document = with_regions(document, exact_region(document, "r", start, start + len(quote)))
        before = deepcopy(document)
        expected = region_block_mapping(document)
        self.assertIn(block.block_id, expected["r"]["block_ids"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "book.json")
            StructuredDocumentStore.save(document, path)
            loaded = StructuredDocumentStore.load(path)
        self.assertEqual(document, before)
        self.assertEqual(loaded.chapters[0].sections[0].task_units[0].content_blocks, [block])
        with patch("document_structure.source_review_blocks._build_mapping",
                   side_effect=AssertionError("Roundtrip unexpectedly invalidated mapping")):
            self.assertEqual(region_block_mapping(loaded), expected)

    def test_changed_in_memory_body_drops_exclusion_proof_without_mutation(self):
        document = persist_region_mapping(excluded_document())
        changed = replace(document, raw_text=document.raw_text + " changed source")
        before = deepcopy(changed)
        self.assertTrue(needs_review(changed))
        self.assertEqual([r["id"] for r in pending_review_regions(changed)], ["excluded"])
        self.assertEqual(region_block_mapping(changed)["excluded"]["body_status"], "unknown")
        self.assertEqual(changed, before)

    def test_title_change_invalidates_cached_mapping(self):
        before, after = title_change()
        fresh = replace(after, parse_provenance={
            key: value for key, value in after.parse_provenance.items()
            if key != "source_review_mapping"
        })
        self.assertTrue(region_block_mapping(before)["r"]["block_ids"])
        self.assertEqual(region_block_mapping(after), region_block_mapping(fresh))
        self.assertEqual(region_block_mapping(after)["r"]["block_ids"], [])

    def test_conflicting_custom_block_metadata_never_links_unrelated_text(self):
        for source_hash in (None, "0" * 64):
            with self.subTest(source_hash=source_hash):
                document = mapped_document()
                section = document.chapters[0].sections[0]
                unit = section.task_units[0]
                metadata = {"quote_span_start": 0, "quote_span_end": 16}
                if source_hash is not None:
                    metadata["source_hash"] = source_hash
                block = TaskUnitContentBlock(
                    block_id="custom", content="Beta paragraph.", metadata=metadata,
                )
                section = replace(section, task_units=[replace(unit, content_blocks=[block])])
                document = replace(document, chapters=[replace(document.chapters[0], sections=[section])])
                document = with_regions(document, exact_region(document, "r", 0, 5))
                self.assertNotIn("custom", region_block_mapping(document)["r"]["block_ids"])

    def test_changed_extraction_rejects_old_mutation_token(self):
        before = mapped_document()
        after = deepcopy(before)
        after.parse_provenance["extraction"]["identity"]["source_sha256"] = "1" * 64
        after.parse_provenance["source_review"]["source_sha256"] = "1" * 64
        repository = MemoryRepository(after)
        with self.assertRaises(ReadingContentConflict):
            ReadingContentService(repository).mutate(
                "book", action="edit", expected_revision=0,
                source_token=source_token_for(before), mutation_id="stale-source",
                block_id=original_blocks(before)[0][3].block_id, text="Updated text",
            )
        self.assertEqual(repository.document, after)

    def test_legacy_pdf_out_of_bounds_page_stays_reviewable_after_backfill(self):
        document = excluded_document()
        record = document.parse_provenance.pop("extraction")
        document.parse_provenance["pdf_extraction"] = {
            "identity": {key: value for key, value in record["identity"].items()
                         if key not in {"source_format", "extractor_id"}},
            "text_sha256": record["text_sha256"],
        }
        document.parse_provenance["source_page_spans"][-1].update(
            char_start=99999, char_end=99999,
        )
        staged = persist_region_mapping(document)
        self.assertTrue(needs_review_from_provenance(staged.parse_provenance))
        self.assertEqual(region_block_mapping(staged)["excluded"]["body_status"], "unknown")

    def test_title_change_rejects_old_block_mutation_token(self):
        before, after = title_change()
        old_block = original_blocks(before)[0][3]
        new_block = original_blocks(after)[0][3]
        self.assertEqual(old_block.block_id, new_block.block_id)
        self.assertNotEqual(old_block.content, new_block.content)
        repository = MemoryRepository(after)
        with self.assertRaises(ReadingContentConflict):
            ReadingContentService(repository).mutate(
                "book", action="delete", expected_revision=0,
                source_token=source_token_for(before), mutation_id="stale-title",
                block_id=old_block.block_id,
            )
        self.assertEqual(repository.document, after)


if __name__ == "__main__":
    unittest.main()
