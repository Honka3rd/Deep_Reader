"""Pure adapters and backed-up real input; no live generation or persistence."""

import ast
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from doc_loaders.extraction_identity import ExtractionResult
from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot, resolve_accepted_source_snapshot
from document_preparation.cleanup_handoff import accepted_cleanup_context
from document_preparation.cleanup_input_adapter import (
    CleanupInputUnavailable, adapt_candidate_cleanup_input, adapt_document_cleanup_input,
)
from document_preparation.structure_enhancement_candidate import StructureEnhancementCandidate
from document_structure.ocr_cleanup_overlay import reading_slice
from document_structure.reading_content import original_blocks
from document_structure.structured_document import StructuredDocument
from shared.task_unit_model import TaskUnit
from scripts.test_cleanup_operation_contract import snapshot
from scripts.test_ocr_cleanup_contract import evidence


def accepted(document, value):
    return AcceptedSourceSnapshot("test", document.document_id, 1, value.identity,
        value.canonical_text, value.text_sha256, value.text_length, document.language,
        accepted_cleanup_context(value, document.parse_provenance))


def fixture(mode="ocr", state=None):
    initial = snapshot(mode=mode)
    document = initial.document()
    if state is not None:
        document.parse_provenance["reading_content"] = {
            "revision": 2, "cleanup_source_sha256": initial.source.text_sha256, **state}
    return document, accepted(document, initial.source)


def adapt(document, source, revision=None):
    if revision is None:
        revision = document.parse_provenance.get("reading_content", {}).get("revision", 0)
    return adapt_document_cleanup_input(document=document, snapshot=source, expected_reading_revision=revision)


def candidate(document, source):
    provenance = {**document.parse_provenance, "effective_parser_mode": "llm_enhanced",
                  "source": "validated_v2_plan", "fallback_used": False, "source_coverage": "complete"}
    provenance.pop("reading_content", None)
    changed = replace(document, raw_text=source.canonical_text, parse_provenance=provenance,
        document_task_artifacts=None, chapters=[replace(c, task_artifacts=None, sections=[
            replace(s, task_artifacts=None, task_units=[]) for s in c.sections]) for c in document.chapters])
    return StructureEnhancementCandidate(source, changed)


def block_document():
    document, source = fixture()
    chapter, = document.chapters
    section, = chapter.sections
    unit = TaskUnit(unit_id="unit-1", title=None, container_title=None, content=section.content,
        source_section_ids=[section.section_id], is_fallback_generated=False, parent_section_id=section.section_id)
    document = replace(document, chapters=[replace(chapter, sections=[replace(section, task_units=[unit])])])
    block = original_blocks(document)[0][3]
    document.parse_provenance["reading_content"] = {
        "revision": 2, "blocks": {block.block_id: {"content": "Manual correction", "deleted": False}}}
    return document, source, block


class CleanupAdapterTests(unittest.TestCase):
    def test_common_before_units_and_candidate_have_equivalent_views(self):
        document, source = fixture()
        before = deepcopy(document.to_dict())
        stored = adapt(document, source)
        split = candidate(document, source)
        split_before = deepcopy(split.document.to_dict())
        enhanced = adapt_candidate_cleanup_input(candidate=split, baseline=stored)
        for field in ("source", "reading_revision", "evidence", "pages", "parts", "protected_ranges", "prior_deletions"):
            self.assertEqual(getattr(stored, field), getattr(enhanced, field))
        self.assertEqual(stored.document().chapters[0].sections[0].task_units, [])
        self.assertEqual(document.to_dict(), before)
        self.assertEqual(split.document.to_dict(), split_before)

    def test_missing_raw_text_uses_accepted_canonical_without_reextraction(self):
        document, source = fixture()
        value = adapt(replace(document, raw_text=""), source)
        self.assertEqual(value.source.canonical_text, source.canonical_text)
        self.assertEqual(value.document().raw_text, "")

    def test_missing_and_stale_source_rejected(self):
        for change in (lambda d: d.parse_provenance.pop("extraction"),
                       lambda d: d.parse_provenance["extraction"].update(text_sha256="f" * 64),
                       lambda d: d.parse_provenance.update(reading_projected=True),
                       lambda d: d.parse_provenance.update(storage_structure_version=2)):
            document, source = fixture()
            change(document)
            with self.assertRaises(CleanupInputUnavailable):
                adapt(document, source)
        document, source = fixture()
        for stale in (replace(source, text_sha256="a" * 64), replace(source, text_length=0),
                      replace(source, canonical_text="different"), replace(source, document_id="other")):
            with self.assertRaises(CleanupInputUnavailable):
                adapt(document, stale)

    def test_reading_revision_and_profile_language_guards(self):
        document, source = fixture()
        for revision in (True, -1, 1, "0"):
            with self.subTest(revision=revision), self.assertRaises(CleanupInputUnavailable):
                adapt(document, source, revision)
        for language in (None, "unknown", "en"):
            with self.subTest(language=language), self.assertRaises(CleanupInputUnavailable):
                adapt(replace(document, language=language), source)

    def test_native_unknown_and_auto_do_not_guess_ocr(self):
        for mode, reason in (("fast", "native_source"), ("native", "native_source"),
                             ("unknown", "extraction_mode_unsupported"), ("auto", "ocr_evidence_missing")):
            document, source = fixture(mode)
            self.assertEqual(adapt(document, source).eligibility().reason, reason)
        document, source = fixture("auto")
        value = ExtractionResult(source.extraction_identity, source.canonical_text)
        document.parse_provenance["actual_ocr_evidence"] = evidence(value).model_dump(mode="json")
        source = accepted(document, value)
        self.assertEqual(adapt(document, source).eligibility().status, "eligible")

    def test_corrupt_or_stale_evidence_not_silently_dropped(self):
        document, source = fixture()
        for raw in ({"invalid": True}, {"spans": []}):
            document.parse_provenance["actual_ocr_evidence"] = raw
            with self.assertRaisesRegex(CleanupInputUnavailable, "ocr_evidence_invalid"):
                adapt(document, source)
        document, source = fixture("auto")
        document.parse_provenance["actual_ocr_evidence"] = evidence(snapshot().source).model_dump(mode="json")
        with self.assertRaises(CleanupInputUnavailable):
            adapt(document, source)

    def test_invalid_page_evidence_rejected_and_valid_pages_preserved(self):
        document, source = fixture()
        for pages in ("invalid", [{}], [{"page_index": 0, "char_start": 2, "char_end": source.text_length}]):
            document.parse_provenance["source_page_spans"] = pages
            with self.assertRaisesRegex(CleanupInputUnavailable, "cleanup_page_evidence_invalid"):
                adapt(document, source)
        document.parse_provenance["source_page_spans"] = [
            {"page_index": 0, "char_start": 0, "char_end": source.text_length}]
        with self.assertRaisesRegex(CleanupInputUnavailable, "cleanup_evidence_changed"):
            adapt(document, source)
        source = accepted(document, ExtractionResult(source.extraction_identity, source.canonical_text))
        self.assertEqual(adapt(document, source).pages, source.cleanup_context.pages)

    def test_prior_deletions_remain_gaps(self):
        document, source = fixture(state={"cleanup_spans": [{"char_start": 0, "char_end": 4, "quote": "@@??"}]})
        value = adapt(document, source)
        self.assertEqual([(p.char_start, p.char_end) for p in value.prior_deletions], [(0, 4)])
        self.assertEqual("".join(p.text for p in value.parts), source.canonical_text[4:])
        self.assertEqual(adapt_candidate_cleanup_input(candidate=candidate(document, source), baseline=value).parts,
                         value.parts)

    def test_source_manual_replacement_and_empty_deletion_are_protected(self):
        for replacement in ("Correction", ""):
            document, source = fixture(state={"source_edits": [
                {"char_start": 0, "char_end": 4, "quote": "@@??", "replacement": replacement}]})
            value = adapt(document, source)
            self.assertTrue(value.parts[0].manual)
            self.assertEqual(value.parts[0].text, replacement)
            self.assertEqual(value.protected_ranges[0].char_end, 4)

    def test_manual_block_mapping_survives_candidate_new_ids(self):
        document, source, block = block_document()
        before = deepcopy(document.to_dict())
        baseline = adapt(document, source)
        split = candidate(document, source)
        c = split.document.chapters[0]
        split = replace(split, document=replace(split.document, chapters=[replace(c, chapter_id="new-chapter",
            sections=[replace(s, section_id="new-section", parent_chapter_id="new-chapter") for s in c.sections])]))
        converted = adapt_candidate_cleanup_input(candidate=split, baseline=baseline)
        self.assertEqual(converted.parts, baseline.parts)
        self.assertEqual(converted.parts[0].text, "Manual correction")
        self.assertNotIn("blocks", converted.document().parse_provenance["reading_content"])
        self.assertTrue(converted.document().parse_provenance["reading_content"]["source_edits"])
        self.assertIn(block.block_id, document.parse_provenance["reading_content"]["blocks"])
        self.assertEqual(before, document.to_dict())

    def test_manual_override_of_cleanup_matches_reading_projection(self):
        document, source = fixture(state={
            "cleanup_spans": [{"char_start": 0, "char_end": 4, "quote": "@@??"}],
            "source_edits": [{"char_start": 1, "char_end": 3, "quote": "@?", "replacement": "fixed"}]})
        value = adapt(document, source)
        self.assertEqual(value.prior_deletions, ())
        self.assertEqual("".join(p.text for p in value.parts),
                         reading_slice(source.canonical_text, 0, source.text_length,
                                       document.parse_provenance["reading_content"]))

    def test_unknown_block_stale_hash_and_bad_spans_rejected(self):
        states = [
            {"blocks": {"missing": {"content": "changed"}}},
            {"cleanup_spans": [{"char_start": 0, "char_end": 4, "quote": "@@??"}],
             "cleanup_source_sha256": "0" * 64},
            {"source_edits": [{"char_start": 0, "char_end": 4, "quote": "wrong", "replacement": "x"}]},
            {"cleanup_spans": [{"char_start": True, "char_end": 4, "quote": "@??"}]},
            {"cleanup_spans": [{"char_start": 0, "char_end": 3, "quote": "@@?"},
                               {"char_start": 2, "char_end": 4, "quote": "??"}]}]
        for state in states:
            with self.subTest(state=state), self.assertRaises(CleanupInputUnavailable):
                adapt(*fixture(state=state))

    def test_pending_review_is_preserved_without_blocking_or_resolving(self):
        document, source = fixture()
        review = {"regions": [{"id": "unmapped", "page_index": 0, "raw_candidate": ""}]}
        document.parse_provenance["source_review"] = review
        baseline = adapt(document, source)
        value = adapt_candidate_cleanup_input(candidate=candidate(document, source), baseline=baseline)
        self.assertEqual(value.eligibility().status, "eligible")
        self.assertEqual(value.document().parse_provenance["source_review"], review)

    def test_candidate_source_or_validation_failure_rejected(self):
        document, source = fixture()
        baseline = adapt(document, source)
        split = candidate(document, source)
        for changed in (replace(split, source=replace(source, namespace="other")),
                        replace(split, source=replace(source, structure_version=2)),
                        replace(split, source=replace(source, language="en"),
                                document=replace(split.document, language="en")),
                        replace(split, document=replace(split.document, raw_text="changed"))):
            with self.assertRaises(CleanupInputUnavailable):
                adapt_candidate_cleanup_input(candidate=changed, baseline=baseline)

    def test_candidate_stale_evidence_is_not_overwritten(self):
        document, source = fixture()
        baseline = adapt(document, source)
        split = candidate(document, source)
        split.document.parse_provenance["source_page_spans"] = [
            {"page_index": 0, "char_start": 0, "char_end": source.text_length}]
        with self.assertRaisesRegex(CleanupInputUnavailable, "cleanup_evidence_changed"):
            adapt_candidate_cleanup_input(candidate=split, baseline=baseline)

    def test_fully_deleted_effective_text_does_not_restore_original(self):
        document, source = fixture()
        document.parse_provenance["reading_content"] = {"revision": 1,
            "cleanup_source_sha256": source.text_sha256, "cleanup_spans": [
                {"char_start": 0, "char_end": source.text_length, "quote": source.canonical_text}]}
        value = adapt(document, source)
        self.assertEqual(value.parts, ())
        self.assertEqual(value.prior_deletions[0].char_end, source.text_length)

    def test_zero_split_ocr_profile_read_or_write_dependencies(self):
        document, source = fixture()
        split = candidate(document, source)
        targets = (
            "document_preparation.structure_enhancement_candidate.build_structure_enhancement_candidate",
            "document_structure.structured_document_builder.StructuredDocumentBuilder.build",
            "document_preparation.cleanup_handoff.plan_ocr_cleanup",
            "doc_loaders.source_session.SourceSession.extract",
            "profile.document_profile_builder.DocumentProfileBuilder.build",
            "document_structure.structured_document_artifact_repository.StructuredDocumentArtifactRepository.load_document",
            "document_structure.structured_document_artifact_repository.StructuredDocumentArtifactRepository.save_document",
            "document_preparation.accepted_source_snapshot.source_file_sha256",
        )
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch(target, side_effect=AssertionError(target))) for target in targets]
            stack.enter_context(patch("builtins.open", side_effect=AssertionError("unexpected I/O")))
            baseline = adapt(document, source)
            adapt_candidate_cleanup_input(candidate=split, baseline=baseline)
            for mock in mocks:
                mock.assert_not_called()

    def test_logs_contain_counts_not_book_content(self):
        document, source = fixture()
        with self.assertLogs("document_preparation.cleanup_input_adapter", level="INFO") as captured:
            adapt(document, source)
            with self.assertRaises(CleanupInputUnavailable):
                adapt(document, source, 9)
        joined = "\n".join(captured.output)
        self.assertIn("origin=stored outcome=ready", joined)
        self.assertIn("cleanup_reading_revision_changed", joined)
        self.assertNotIn("@@??", joined)
        self.assertNotIn(source.canonical_text, joined)

    def test_dependency_boundary_has_no_concrete_loader_or_provider(self):
        path = Path(__file__).resolve().parents[1] / "document_preparation" / "cleanup_input_adapter.py"
        tree = ast.parse(path.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(m and m.split(".")[0] in {"llm", "profile", "db", "app", "main"} for m in imports))
        self.assertFalse(any(m and ("pdf" in m or "txt" in m) for m in imports))


class PrivateCleanupAdapterTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"), "opt-in immutable backups")
    def test_real_common_sources_and_accepted_source_boundary(self):
        root = Path(os.environ["DEEP_READER_TEST_CLEANUP_INPUT_BOOKS"])
        for name in ("sample", "sample2", "noise"):
            with self.subTest(book=name):
                folder = root / name
                files = [folder / name for name in ("book.pdf", "document.json", "manifest.json")]
                before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
                self.assertEqual(before[files[0]], json.loads(files[2].read_text())["source_sha256"])
                document = StructuredDocument.from_dict(json.loads(files[1].read_text()))
                self.assertEqual(document.parse_provenance["effective_parser_mode"], "common")
                source = resolve_accepted_source_snapshot(document=document, namespace="test",
                    expected_document_id=document.document_id, structure_version=1,
                    source_path=files[0], accepted=True)
                value = adapt(document, source)
                self.assertEqual(value.eligibility().status, "eligible")
                self.assertEqual(value.source.canonical_text, document.raw_text)
                self.assertEqual(value.document().to_dict(), document.to_dict())
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == "__main__":
    unittest.main()
