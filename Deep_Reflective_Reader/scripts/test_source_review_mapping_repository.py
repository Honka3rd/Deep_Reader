"""SRMAP repository writes, final identities and atomic backfill in disposable storage."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from document_structure.reading_content import ReadingContentConflict, original_blocks
from document_structure.source_review_blocks import persist_region_mapping
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from shared.task_artifacts import DocumentTaskArtifacts, SummaryArtifact, TaskArtifacts
from shared.task_unit_model import TaskUnit
from scripts.test_ocr_cleanup_validation import fixture
from scripts import test_ocr_cleanup_overlay as cleanup_fixture


def document_fixture(*, materialized=True):
    source, _, document = fixture("Uncertain text.\n\nBody remains intact.")
    section = document.chapters[0].sections[0]
    unit = TaskUnit(unit_id="candidate-unit", title=None, container_title=None,
                    content=section.content, source_section_ids=[section.section_id],
                    parent_section_id=section.section_id, is_fallback_generated=False)
    unit = replace(unit, content_blocks=unit.segment_content_blocks())
    return replace(document,
        chapters=[replace(document.chapters[0], sections=[replace(section,
            task_units=[unit] if materialized else [],
            task_artifacts=TaskArtifacts(summary=SummaryArtifact(content="Saved summary")))])],
        document_task_artifacts=DocumentTaskArtifacts(metadata={"note": {"sentinel": "keep"}}),
        parse_provenance={"extraction": source.record(),
            "source_page_spans": [{"page_index": 0, "char_start": 0, "char_end": source.text_length}],
            "source_review": {"source_sha256": source.identity.source_sha256, "regions": [
                {"id": "r1", "page_index": 0, "raw_candidate": "Uncertain text.",
                 "canonical_span": {"char_start": 0, "char_end": 15, "text_sha256": source.text_sha256}}]}})


class MappingRepositoryContract:
    def mapping(self):
        return self.repo.load_document("book").parse_provenance["source_review_mapping"]

    def test_save_uses_final_ids_without_mutating_candidate(self):
        candidate = document_fixture()
        before = deepcopy(candidate)
        self.repo.save_document(candidate, "book")
        self.assertEqual(candidate, before)
        current = self.repo.load_document("book")
        mapping = self.mapping()
        self.assertEqual(mapping["regions"]["r1"]["mapping_status"], "complete")
        self.assertEqual(mapping["regions"]["r1"]["block_ids"],
                         [original_blocks(current)[0][3].block_id])
        self.assertEqual(persist_region_mapping(current), current)

    def test_pending_mapping_materializes_in_explicit_layout_write(self):
        self.repo.save_document(document_fixture(materialized=False), "book")
        self.assertEqual(self.mapping()["regions"]["r1"]["body_status"], "pending")
        section = self.repo.load_document("book").chapters[0].sections[0]
        unit = document_fixture().chapters[0].sections[0].task_units[0]
        self.repo.update_task_layout("book", {section.section_id: [unit]}, {"reading_revision": 0})
        self.assertEqual(self.mapping()["regions"]["r1"]["body_status"], "linked")
        current = self.repo.load_document("book")
        self.assertEqual(self.mapping()["regions"]["r1"]["block_ids"],
                         [original_blocks(current)[0][3].block_id])

    def test_reads_never_persist_and_listing_uses_shared_policy(self):
        self.repo.save_document(document_fixture(), "book")
        before = self.repo.load_document("book")
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("read attempted persistence")), \
             patch("document_structure.source_review_blocks.needs_review_from_provenance",
                   return_value=False) as policy:
            self.assertEqual(self.repo.load_document("book"), before)
            self.assertFalse(self.repo.list_documents()[0].review_required)
            policy.assert_called_once_with(before.parse_provenance)
        self.assertEqual(self.repo.load_document("book"), before)

    def test_overlay_and_deleted_links_survive_metadata_save(self):
        self.repo.save_document(document_fixture(), "book")
        original = self.repo.load_document("book")
        mapping = self.mapping()
        block_id = mapping["regions"]["r1"]["block_ids"][0]
        state = {"revision": 1, "blocks": {block_id: {"deleted": True}},
                 "audit": [{"action": "delete", "block_id": block_id, "invalidate_generated": False}]}
        self.repo.mutate_reading_content("book", lambda current: (replace(current,
            parse_provenance={**current.parse_provenance, "reading_content": state}), None))
        # Mutation persists a fresh block epoch. Metadata save must preserve it too.
        epochs = self.repo.load_document("book").parse_provenance["reading_content"]["block_content_epochs"]
        self.assertTrue(epochs[block_id])
        state = {**state, "block_content_epochs": dict(epochs)}
        self.repo.save_document(original, "book")
        current = self.repo.load_document("book")
        self.assertEqual(current.parse_provenance["reading_content"], state)
        self.assertEqual(self.mapping(), mapping)
        self.assertEqual(current.document_task_artifacts.metadata["note"], {"sentinel": "keep"})

    def test_backfill_callback_is_atomic_idempotent_and_keeps_artifacts(self):
        with patch("document_structure.source_review_blocks.persist_region_mapping", side_effect=lambda d: d):
            self.repo.save_document(document_fixture(), "book")
        before = self.repo.load_document("book")
        self.assertNotIn("source_review_mapping", before.parse_provenance)

        def backfill(current):
            if current.parse_provenance.get("reading_content", {}).get("revision", 0) != 0:
                raise ReadingContentConflict("reading_revision_changed")
            updated = persist_region_mapping(current)
            return updated, updated != current

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.repo.mutate_reading_content("book", backfill), range(2)))
        self.assertEqual(sorted(results), [False, True])
        current = self.repo.load_document("book")
        self.assertEqual(current.chapters, before.chapters)
        self.assertEqual(current.document_task_artifacts, before.document_task_artifacts)
        self.assertEqual(current.parse_provenance.get("reading_content"), None)
        self.assertEqual(persist_region_mapping(current), current)

    def test_mapping_failure_rolls_back_write(self):
        self.repo.save_document(document_fixture(), "book")
        before = self.repo.load_document("book")
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=RuntimeError("mapping failed")):
            with self.assertRaisesRegex(RuntimeError, "mapping failed"):
                self.repo.save_reparsed_document(document_fixture(materialized=False), "book")
        self.assertEqual(self.repo.load_document("book"), before)

    def test_hard_reparse_replaces_epoch_mapping(self):
        self.repo.save_document(document_fixture(), "book")
        old_mapping = self.mapping()
        self.repo.save_reparsed_document(document_fixture(materialized=False), "book")
        current = self.mapping()
        self.assertNotEqual(current["source_key"], old_mapping["source_key"])
        self.assertEqual(current["regions"]["r1"]["block_ids"], [])
        self.assertEqual(current["regions"]["r1"]["body_status"], "pending")


class FileMappingRepositoryTests(MappingRepositoryContract, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = StructuredDocumentArtifactRepository(base_dir=directory.name)

    def test_failed_replace_preserves_mapping_and_cleans_temporary_file(self):
        self.repo.save_document(document_fixture(), "book")
        before = self.repo.load_document("book")
        with patch("document_structure.structured_document_artifact_repository.os.replace",
                   side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                self.repo.save_reparsed_document(document_fixture(materialized=False), "book")
        self.assertEqual(self.repo.load_document("book"), before)
        self.assertEqual(list(self.repo.base_dir.glob("*.tmp")), [])


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "Dedicated test DSN required")
class PostgresMappingRepositoryTests(MappingRepositoryContract, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo
        cls.dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
        cls.schema = "srmap_repository_" + uuid4().hex
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.scoped = make_conninfo(cls.dsn, options=f"-c search_path={cls.schema}")
        with psycopg.connect(cls.scoped) as connection:
            for path in sorted((Path(__file__).resolve().parents[1] / "db/migrations").glob("*.sql")):
                connection.execute(path.read_text())

    @classmethod
    def drop_schema(cls):
        import psycopg
        from psycopg import sql
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        self.repo = PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex))

    def stored_rows(self):
        import psycopg
        with psycopg.connect(self.scoped) as connection:
            return connection.execute(
                "SELECT d.id,d.metadata_payload,s.structured_document_payload "
                "FROM documents d JOIN structured_document_snapshots s ON s.document_id=d.id "
                "AND s.source_structure_version=d.current_structure_version WHERE d.namespace=%s",
                (self.repo.store._namespace,)).fetchone()

    def test_snapshot_and_metadata_use_allocated_ids_and_same_mapping(self):
        self.repo.save_document(document_fixture(), "book")
        current = self.repo.load_document("book")
        _, metadata, snapshot = self.stored_rows()
        self.assertEqual(metadata["parse_provenance"], snapshot["parse_provenance"])
        self.assertEqual(metadata["parse_provenance"], current.parse_provenance)
        unit = current.chapters[0].sections[0].task_units[0]
        self.assertTrue(unit.unit_id.isdigit())
        self.assertEqual(snapshot["chapters"][0]["sections"][0]["task_units"][0]["unit_id"], unit.unit_id)
        self.assertTrue(self.mapping()["regions"]["r1"]["block_ids"][0].startswith(unit.unit_id + ":content:"))

    def test_backfill_preserves_physical_artifacts_and_updates_snapshot(self):
        import psycopg
        with patch("document_structure.source_review_blocks.persist_region_mapping", side_effect=lambda d: d):
            self.repo.save_document(document_fixture(), "book")
        document_id, _, _ = self.stored_rows()
        with psycopg.connect(self.scoped) as connection:
            for kind in ("note", "analysis"):
                connection.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,"
                    "source_structure_version,payload) VALUES (%s,%s,'document',%s,1,'{}')",
                    (document_id, kind, document_id))
        self.repo.mutate_reading_content("book", lambda current: (persist_region_mapping(current), None))
        _, metadata, snapshot = self.stored_rows()
        self.assertEqual(metadata["parse_provenance"], snapshot["parse_provenance"])
        self.assertIn("source_review_mapping", metadata["parse_provenance"])
        with psycopg.connect(self.scoped) as connection:
            self.assertEqual(connection.execute("SELECT artifact_type FROM artifacts WHERE document_id=%s "
                "ORDER BY artifact_type", (document_id,)).fetchall(), [("analysis",), ("note",)])

    def test_snapshot_failure_rolls_back_hierarchy_and_mapping(self):
        import psycopg
        self.repo.save_document(document_fixture(), "book")
        before = self.repo.load_document("book"), self.stored_rows()
        with psycopg.connect(self.scoped) as connection:
            connection.execute("CREATE FUNCTION fail_srmap_snapshot() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'injected snapshot failure'; END $$")
            connection.execute("CREATE TRIGGER fail_srmap_snapshot BEFORE INSERT OR UPDATE ON "
                "structured_document_snapshots FOR EACH ROW EXECUTE FUNCTION fail_srmap_snapshot()")
        try:
            with self.assertRaises(psycopg.Error):
                self.repo.save_reparsed_document(document_fixture(materialized=False), "book")
            self.assertEqual((self.repo.load_document("book"), self.stored_rows()), before)
        finally:
            with psycopg.connect(self.scoped) as connection:
                connection.execute("DROP TRIGGER fail_srmap_snapshot ON structured_document_snapshots")
                connection.execute("DROP FUNCTION fail_srmap_snapshot()")


class TitleMappingFixture:
    # Reuse the accepted-source/cleanup fixture without inheriting its test suite.
    setUp = cleanup_fixture.CleanupOverlayTests.setUp
    repository = cleanup_fixture.CleanupOverlayTests.repository
    snapshot = cleanup_fixture.CleanupOverlayTests.snapshot
    prepare = cleanup_fixture.CleanupOverlayTests.prepare
    commit = cleanup_fixture.CleanupOverlayTests.commit


class FileTitleMappingTests(TitleMappingFixture, unittest.TestCase):
    def test_title_commit_returns_persisted_mapping(self):
        before = self.snapshot()[0]
        result = self.commit(*self.prepare("titles_only"))
        current = self.snapshot()[0]
        self.assertEqual(result.document.parse_provenance["source_review_mapping"],
                         current.parse_provenance["source_review_mapping"])
        self.assertEqual(result.document.chapters, current.chapters)
        self.assertEqual(persist_region_mapping(current), current)
        self.assertNotEqual(before.parse_provenance["source_review_mapping"]["hierarchy_key"],
                            current.parse_provenance["source_review_mapping"]["hierarchy_key"])

    def test_title_replay_does_not_write_or_refresh_mapping(self):
        command = self.prepare("titles_only")
        self.commit(*command)
        before = self.snapshot()
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("replay refreshed mapping")), \
             patch.object(self.repo, "_atomic_save", side_effect=AssertionError("replay wrote")):
            result = self.commit(*command)
        self.assertEqual(result.document, before[0])
        self.assertEqual(self.snapshot(), before)


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "Dedicated test DSN required")
class PostgresTitleMappingTests(TitleMappingFixture, unittest.TestCase):
    setUpClass = classmethod(PostgresMappingRepositoryTests.setUpClass.__func__)
    drop_schema = classmethod(PostgresMappingRepositoryTests.drop_schema.__func__)
    stored_rows = PostgresMappingRepositoryTests.stored_rows

    def repository(self):
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        return PostgresStructuredDocumentArtifactRepository(
            PostgresStructuredDocumentStore(self.scoped, namespace=self.namespace))

    def snapshot(self):
        snapshot = self.repo.load_enhancement_snapshot("book")
        return snapshot.document, snapshot.structure_version, snapshot.interaction_revision

    def commit(self, candidate, application=None, *, operation=None):
        fingerprint = (application.plan.binding.structure_fingerprint if application is not None
                       else self.snapshot()[2])
        return self.repo.commit_enhancement("book", candidate=candidate, cleanup=application,
            interaction_revision=fingerprint, source_path=self.path, operation=operation)

    def physical_rows(self):
        import psycopg
        from psycopg import sql
        document_id = self.stored_rows()[0]
        with psycopg.connect(self.scoped) as connection:
            return {table: connection.execute(sql.SQL(
                "SELECT row_to_json(t), xmin::text FROM {} t WHERE {}=%s ORDER BY id"
            ).format(sql.Identifier(table), sql.Identifier("id" if table == "documents" else "document_id")),
                (document_id,)).fetchall() for table in (
                    "documents", "chapters", "sections", "task_units", "content_blocks", "artifacts",
                    "structured_document_snapshots", "parse_events", "document_profile")}

    def seed_note(self):
        import psycopg
        document_id = self.stored_rows()[0]
        with psycopg.connect(self.scoped) as connection:
            connection.execute("INSERT INTO artifacts(document_id,artifact_type,target_type,target_id,"
                "source_structure_version,payload) VALUES (%s,'note','document',%s,1,'{\"text\":\"keep\"}')",
                (document_id, document_id))

    def assert_title_commit(self, *, cleanup):
        from document_structure.reading_content import state_for
        from document_structure.structured_document import StructuredDocument

        self.seed_note()
        before, version, fingerprint = self.snapshot()
        rows = self.physical_rows()
        candidate, application = self.prepare("titles_only", items=[] if cleanup == "empty" else None)
        result = self.commit(candidate, application if cleanup else None)
        current, after_version, after_fingerprint = self.snapshot()
        mapping = current.parse_provenance["source_review_mapping"]
        self.assertEqual(result.change_kind, "titles_only")
        self.assertEqual(result.document, current)
        self.assertEqual(after_version, version)
        self.assertEqual(state_for(current).get("revision", 0), 1 if cleanup == "applied" else 0)
        self.assertNotEqual(fingerprint, after_fingerprint)
        self.assertEqual(result.interaction_revision, after_fingerprint)
        self.assertEqual(persist_region_mapping(current), current)
        self.assertNotEqual(mapping["hierarchy_key"], before.parse_provenance["source_review_mapping"]["hierarchy_key"])
        self.assertEqual(mapping["canonical_sha256"], before.parse_provenance["source_review_mapping"]["canonical_sha256"])
        old_chapter, chapter = before.chapters[0], current.chapters[0]
        old_section, section = old_chapter.sections[0], chapter.sections[0]
        self.assertEqual(chapter.chapter_id, old_chapter.chapter_id)
        self.assertEqual(section.section_id, old_section.section_id)
        self.assertEqual(section.task_units, old_section.task_units)
        self.assertEqual((section.char_start, section.char_end, section.content),
                         (old_section.char_start, old_section.char_end, old_section.content))
        self.assertEqual(section.title, "New heading")
        self.assertEqual(mapping["regions"]["r1"]["block_ids"], [original_blocks(current)[0][3].block_id])
        self.assertTrue(section.task_units[0].unit_id.isdigit())
        self.assertEqual(current.document_task_artifacts.metadata["note"],
                         before.document_task_artifacts.metadata["note"])
        after_rows = self.physical_rows()
        for table in ("artifacts", "content_blocks", "parse_events", "document_profile"):
            self.assertEqual(after_rows[table], rows[table], table)
        if cleanup != "applied":
            self.assertEqual(after_rows["task_units"], rows["task_units"])
            self.assertEqual(current.document_task_artifacts, before.document_task_artifacts)
        if not cleanup:
            self.assertEqual(current.parse_provenance.get("reading_content"),
                             before.parse_provenance.get("reading_content"))
        _, metadata, snapshot = self.stored_rows()
        self.assertEqual(metadata["parse_provenance"], current.parse_provenance)
        self.assertEqual(snapshot["parse_provenance"], current.parse_provenance)
        snapshot_document = StructuredDocument.from_dict(snapshot)
        self.assertEqual(persist_region_mapping(snapshot_document), snapshot_document)
        self.assertEqual(snapshot_document.chapters, current.chapters)
        return result

    def test_title_only_refreshes_mapping_without_cleanup(self):
        self.assert_title_commit(cleanup=None)

    def test_title_only_refreshes_mapping_with_noop_cleanup(self):
        self.assert_title_commit(cleanup="empty")

    def test_title_only_refreshes_mapping_with_applied_cleanup(self):
        self.assert_title_commit(cleanup="applied")

    def test_mapping_failure_rolls_back_labels_cleanup_revisions_and_notes(self):
        self.seed_note()
        candidate, application = self.prepare("titles_only")
        before, rows = self.snapshot(), self.physical_rows()
        for cleanup in (None, application):
            with self.subTest(cleanup=cleanup is not None), \
                 patch("document_structure.source_review_blocks.persist_region_mapping",
                       side_effect=RuntimeError("mapping failed")) as mapping:
                with self.assertRaisesRegex(RuntimeError, "mapping failed"):
                    self.commit(candidate, cleanup)
                mapping.assert_called_once()
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.physical_rows(), rows)

    def test_unchanged_and_cleanup_replay_do_not_refresh_mapping_or_write(self):
        candidate, _ = self.prepare()
        before = self.physical_rows()
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("unchanged refreshed mapping")):
            self.assertEqual(self.commit(candidate).change_kind, "unchanged")
        self.assertEqual(self.physical_rows(), before)
        command = self.prepare("titles_only")
        first = self.commit(*command)
        before = self.physical_rows()
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("replay refreshed mapping")):
            replay = self.commit(*command)
        self.assertEqual(replay, first)
        self.assertEqual(self.physical_rows(), before)

    def test_operation_receipt_parity_and_replay_are_preserved(self):
        candidate, application = self.prepare("titles_only")
        operation = {"id": "a" * 64, "cleanup": None}
        first = self.commit(candidate, application, operation=operation)
        _, metadata, snapshot = self.stored_rows()
        self.assertEqual(metadata["parse_provenance"], snapshot["parse_provenance"])
        self.assertIn(operation["id"], metadata["parse_provenance"]["enhancement_receipts"])
        self.assertEqual(persist_region_mapping(first.document), first.document)
        before = self.physical_rows()
        with patch("document_structure.source_review_blocks.persist_region_mapping",
                   side_effect=AssertionError("receipt replay refreshed mapping")):
            self.assertEqual(self.commit(candidate, application, operation=operation), first)
        self.assertEqual(self.physical_rows(), before)


if __name__ == "__main__":
    unittest.main()
