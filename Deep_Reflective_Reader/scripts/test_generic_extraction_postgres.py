"""Real PostgreSQL provenance-only migration and generic TXT commit tests."""
from dataclasses import replace
from pathlib import Path

from doc_loaders.text_document_loader import TextDocumentLoader
from test_enhancement_conditional_commit import ConditionalCommitTests


class GenericBackfillTests(ConditionalCommitTests):
    def prepare_txt(self):
        root = Path(self.fixture.temp.name)
        (root / "doc.pdf").unlink()
        source = root / "doc.txt"
        source.write_text(self.fixture.text, encoding="utf-8")
        self.repository.save_reparsed_document(replace(self.fixture.document,
            parse_provenance={}, source_path=str(source)), "doc")
        return source, TextDocumentLoader(root).extract("doc")

    def test_backfill_idempotent_and_preserves_hierarchy_artifacts_profile(self):
        source, result = self.prepare_txt()
        self.add_artifact("note", source_version=self.snapshot().structure_version)
        tables = ("chapters", "sections", "task_units", "content_blocks", "artifacts", "document_profile", "parse_events")
        before = {table: self.rows(table) for table in tables}
        version = self.snapshot().structure_version
        self.assertTrue(self.repository.backfill_txt_extraction("doc", result=result, source_path=source))
        self.assertFalse(self.repository.backfill_txt_extraction("doc", result=result, source_path=source))
        self.assertEqual(before, {table: self.rows(table) for table in tables})
        self.assertEqual(version, self.snapshot().structure_version)
        state = self.coordinator.read_enhancement_state("doc")
        self.assertEqual(state.source.canonical_text, result.canonical_text)
        self.assertEqual(state.source.extraction_identity.source_format, "txt")
        state, candidate = self.make_candidate()
        committed = self.repository.commit_enhancement("doc", candidate=candidate,
            interaction_revision=state.interaction_revision, source_path=source)
        self.assertEqual(committed.change_kind, "structure_changed")
        self.assertEqual(self.coordinator.read_enhancement_state("doc").source.canonical_text, result.canonical_text)

    def test_invalid_backfill_rolls_back(self):
        source, result = self.prepare_txt()
        before = {table: self.rows(table) for table in ("documents", "structured_document_snapshots")}
        source.write_text("different", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.repository.backfill_txt_extraction("doc", result=result, source_path=source)
        self.assertEqual(before, {table: self.rows(table) for table in before})
