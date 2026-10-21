"""Conditional commit plus inherited guard regressions in disposable PostgreSQL."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import psycopg
from psycopg.types.json import Jsonb
import test_enhancement_postgres_preflight as base
from db.postgres_structured_document_store import EnhancementCommitConflict
from shared.task_artifacts import DocumentTaskArtifacts


class ConditionalCommitTests(base.PostgresPreflightTests):
    def matching_candidate(self, title=None):
        state, candidate = self.make_candidate()
        old = self.snapshot().document.chapters[0]
        chapter = candidate.document.chapters[0]
        section = replace(chapter.sections[0], level=old.sections[0].level,
            section_role=old.sections[0].section_role, title=title,
            container_title=title)
        chapter = replace(chapter, level=old.level, chapter_role=old.chapter_role,
                          title=title, sections=[section])
        return state, replace(candidate, document=replace(candidate.document, chapters=[chapter]))

    def rows(self, table):
        assert table in {"documents", "chapters", "sections", "task_units", "content_blocks",
                         "artifacts", "structured_document_snapshots", "parse_events", "document_profile"}
        column = "id" if table == "documents" else "document_id"
        with psycopg.connect(self.scoped) as connection:
            return connection.execute(f"SELECT row_to_json(t), xmin::text FROM {table} t "
                f"WHERE {column}=%s ORDER BY id", (self.doc_id,)).fetchall()

    def seed_interactions(self):
        self.add_artifact("note")
        with psycopg.connect(self.scoped) as connection:
            section_id = connection.execute("SELECT id FROM sections WHERE document_id=%s",
                                            (self.doc_id,)).fetchone()[0]
            unit = connection.execute("INSERT INTO task_units (document_id,section_id,task_unit_order,"
                "content_payload,metadata_payload) VALUES (%s,%s,0,'body',%s) RETURNING id",
                (self.doc_id, section_id, Jsonb({"notes": ["keep unit note"]}))).fetchone()[0]
            connection.execute("INSERT INTO content_blocks (document_id,task_unit_id,block_order,"
                "content,source_structure_version) VALUES (%s,%s,0,'body',1)", (self.doc_id, unit))
            connection.execute("UPDATE sections SET metadata_payload=coalesce(metadata_payload,'{}') || %s "
                "WHERE id=%s", (Jsonb({"notes": ["keep section note"]}), section_id))

    def test_noop_performs_no_row_writes(self):
        self.seed_interactions()
        tables = ["documents", "chapters", "sections", "task_units", "content_blocks",
                  "artifacts", "structured_document_snapshots", "parse_events"]
        before = {table: self.rows(table) for table in tables}
        state, candidate = self.matching_candidate()
        result = self.commit_candidate(state, candidate)
        self.assertEqual(result.change_kind, "unchanged")
        self.assertEqual(result.structure_version, 1)
        self.assertEqual(result.document, self.snapshot().document)
        self.assertEqual(before, {table: self.rows(table) for table in tables})

    def test_title_only_preserves_ids_artifacts_units_and_snapshot(self):
        self.seed_interactions()
        tables = ["documents", "task_units", "content_blocks", "artifacts", "parse_events"]
        before = {table: self.rows(table) for table in tables}
        old = self.snapshot().document
        state, candidate = self.matching_candidate("New navigation")
        result = self.commit_candidate(state, candidate)
        self.assertEqual(result.change_kind, "titles_only")
        self.assertEqual(result.structure_version, 1)
        chapter = result.document.chapters[0]
        self.assertEqual(chapter.chapter_id, old.chapters[0].chapter_id)
        self.assertEqual(chapter.sections[0].section_id, old.chapters[0].sections[0].section_id)
        self.assertEqual(chapter.title, "New navigation")
        self.assertEqual(chapter.sections[0].title, "New navigation")
        snapshot = self.rows("structured_document_snapshots")[0][0]["structured_document_payload"]
        self.assertEqual(snapshot["chapters"][0]["title"], chapter.title)
        self.assertEqual(snapshot["chapters"][0]["sections"][0]["title"], chapter.sections[0].title)
        self.assertEqual(before, {table: self.rows(table) for table in tables})
        self.assertEqual(self.rows("sections")[0][0]["metadata_payload"]["notes"], ["keep section note"])

    def test_delayed_metadata_writer_preserves_latest_labels(self):
        delayed = self.repository.load_document("doc")
        state, candidate = self.matching_candidate("Latest navigation")
        self.commit_candidate(state, candidate)
        delayed = replace(delayed, document_task_artifacts=DocumentTaskArtifacts(metadata={
            "reading_interaction_artifacts": {"analysis": {"new": {
                "source_structure_version": 1, "payload": {"analysis": "late valid answer"}}}}}))
        self.repository.save_document(delayed, "doc")
        current = self.repository.load_document("doc")
        self.assertEqual(current.chapters[0].title, "Latest navigation")
        self.assertEqual(current.chapters[0].sections[0].container_title, "Latest navigation")
        self.assertIsNotNone(current.document_task_artifacts)
        snapshot = self.rows("structured_document_snapshots")[0][0]["structured_document_payload"]
        self.assertEqual(snapshot["chapters"][0]["title"], "Latest navigation")
        self.add_artifact("quiz", source_version=1)
        self.assertEqual(self.snapshot().structure_version, 1)

    def test_title_commit_invalidates_old_confirmation_without_version_change(self):
        state, candidate = self.matching_candidate("New")
        self.commit_candidate(state, candidate)
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "confirmation_stale"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(before.structure_version, 1)

    def test_title_update_rolls_back_labels_and_snapshot_on_failure(self):
        state, candidate = self.matching_candidate("Rejected")
        before = self.snapshot()
        snapshot_before = self.rows("structured_document_snapshots")
        update = self.store._update_enhancement_labels
        def fail(*args):
            update(*args)
            raise RuntimeError("after label update")
        with patch.object(self.store, "_update_enhancement_labels", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "after label update"):
                self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(snapshot_before, self.rows("structured_document_snapshots"))

    def test_label_writer_during_generation_invalidates_confirmation(self):
        state, candidate = self.matching_candidate("Candidate")
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE chapters SET title='Concurrent title' WHERE document_id=%s",
                               (self.doc_id,))
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "confirmation_stale"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_concurrent_title_writer_lock_rejects_commit(self):
        state, candidate = self.matching_candidate("Candidate")
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE chapters SET title='Concurrent' WHERE document_id=%s", (self.doc_id,))
            with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_commit_busy"):
                self.commit_candidate(state, candidate)

    def test_structural_result_is_explicit(self):
        state, candidate = self.make_candidate()
        self.assertEqual(self.commit_candidate(state, candidate).change_kind, "structure_changed")

    def test_new_artifact_invalidates_title_candidate(self):
        state, candidate = self.matching_candidate("Candidate")
        self.add_artifact("quiz")
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "confirmation_stale"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_snapshot_mismatch_rolls_back_title_edits(self):
        state, candidate = self.matching_candidate("Candidate")
        with psycopg.connect(self.scoped) as connection:
            connection.execute("UPDATE structured_document_snapshots SET structured_document_payload="
                "jsonb_set(structured_document_payload, '{chapters,0,sections,0,content}', '\"corrupt\"') "
                "WHERE document_id=%s", (self.doc_id,))
        before = self.snapshot()
        with self.assertRaisesRegex(EnhancementCommitConflict, "enhancement_snapshot_mismatch"):
            self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())

    def test_source_changed_during_title_commit_rolls_back(self):
        state, candidate = self.matching_candidate("Candidate")
        before = self.snapshot()
        update = self.store._update_enhancement_labels
        def change_source(*args):
            update(*args)
            from pathlib import Path
            (Path(self.fixture.temp.name) / "doc.pdf").write_bytes(b"changed")
        with patch.object(self.store, "_update_enhancement_labels", side_effect=change_source):
            with self.assertRaisesRegex(EnhancementCommitConflict, "source_changed"):
                self.commit_candidate(state, candidate)
        self.assertEqual(before, self.snapshot())


if __name__ == "__main__":
    unittest.main()
