"""Backups survive Library deletion and reject accidental replacement/corruption."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_accepted_source_snapshot as source_fixture
import private_book_corpus as corpus


class CorpusTests(unittest.TestCase):
    def setUp(self):
        fixture = source_fixture.AcceptedSourceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.snapshot = fixture.resolve()
        fixture.source.rename(Path(fixture.temp.name) / "doc.pdf")
        self.coordinator = Mock()
        self.coordinator.read_enhancement_state.return_value = SimpleNamespace(source=self.snapshot)
        self.coordinator.document_artifact_repository.load_enhancement_snapshot.return_value = SimpleNamespace(document=fixture.document)
        self.coordinator.document_preparation_pipeline.loader_factory.base_dir = fixture.temp.name
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        patched = patch.object(corpus, "CORPUS_ROOT", Path(self.root.name) / "test-books")
        patched.start()
        self.addCleanup(patched.stop)

    def test_survives_deleted_original_and_repository(self):
        case_id = corpus.backup_book(self.coordinator, "doc")
        (Path(self.fixture.temp.name) / "doc.pdf").unlink()
        self.coordinator.read_enhancement_state.side_effect = AssertionError("Deleted Library")
        self.coordinator.document_artifact_repository.load_enhancement_snapshot.side_effect = AssertionError("Deleted Library")
        path, snapshot, document = corpus.load_book(case_id)
        self.assertEqual(snapshot, self.snapshot)
        self.assertEqual(document.document_id, "doc")
        self.assertEqual((path / "source.pdf").read_bytes(), b"accepted source fixture")
        self.assertEqual((path / "source.pdf").stat().st_mode & 0o777, 0o600)

    def test_repeat_backup_is_idempotent_and_conflicts_do_not_overwrite(self):
        case_id = corpus.backup_book(self.coordinator, "doc")
        self.assertEqual(corpus.backup_book(self.coordinator, "doc"), case_id)
        path = corpus.fixture_path(case_id) / "source.pdf"
        with self.assertRaisesRegex(ValueError, "backup_conflict"):
            corpus.preserve(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"accepted source fixture")

    def test_corruption_and_missing_backups_fail_closed(self):
        case_id = corpus.backup_book(self.coordinator, "doc")
        path = corpus.fixture_path(case_id) / "source.pdf"
        path.write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "backup_hash_mismatch"):
            corpus.load_book(case_id)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            corpus.load_book(case_id)

    def test_case_cannot_escape_corpus(self):
        with self.assertRaises(ValueError):
            corpus.fixture_path("../raw/doc")


if __name__ == "__main__":
    unittest.main()
