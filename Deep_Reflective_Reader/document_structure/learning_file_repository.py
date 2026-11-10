"""POSIX learning adapter: one atomic sidecar per workspace/document.

Lock order: preferences -> canonical document -> learning sidecar. Only the
sidecar is written, so source/settings validation and result commit serialize
against cooperating writers without claiming cross-file write atomicity.
Reads create nothing. The constructor does not wire this into the application.
"""
from contextlib import contextmanager, suppress
from hashlib import sha256
import fcntl
import logging
import os
from pathlib import Path
import tempfile
from time import perf_counter

from document_structure import learning_file_codec as codec
from document_structure.learning_repository_behavior import LearningRepositoryBehavior
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from reading_preferences.file_repository import FileReadingPreferencesRepository
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError
from document_structure.learning_source_resolver import learning_blocks
from reading_preferences.models import WorkspaceScope, identifier
from shared.learning_source import require



logger = logging.getLogger(__name__)


class FileLearningRepository(LearningRepositoryBehavior):
    def __init__(self, base_dir: Path, documents: StructuredDocumentArtifactRepository,
                 preferences: FileReadingPreferencesRepository, *, admission=None):
        require(type(preferences.scope) is WorkspaceScope and documents.namespace == preferences.scope.namespace,
                "learning_scope_mismatch")
        self._base = Path(base_dir)
        self._documents = documents
        self._preferences = preferences
        self._scope = preferences.scope
        self._admission = admission

    @property
    def scope(self):
        return self._scope

    @property
    def document_repository(self):
        return self._documents

    @property
    def preference_repository(self):
        return self._preferences

    def lifecycle_path(self, document_id):
        return self._path(document_id)

    def lifecycle_lock(self, document_id):
        return self._lock(document_id)

    def lifecycle_state(self, document_id):
        """Explicit lifecycle owner must hold preferences/document/learning locks."""
        return self._load(document_id)

    def _path(self, document_id):
        identifier(document_id)
        digest = sha256(codec.canonical([self.scope.kind, self.scope.namespace, document_id]).encode()).hexdigest()
        return self._base / (digest + '.learning.json')

    def _load(self, document_id):
        self._preferences.assert_lifecycle_ready()
        path = self._path(document_id)
        try:
            raw = path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return codec.empty_state(self.scope, document_id)
        except (OSError, UnicodeError):
            raise LearningStorageError('learning_read_failed') from None
        try:
            return codec.loads(raw, self.scope, document_id)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise LearningStorageError('learning_data_invalid_or_unsupported') from None

    @contextmanager
    def _lock(self, document_id):
        path = self._path(document_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.with_suffix('.lock').open('a')
        except OSError:
            raise LearningStorageError('learning_lock_failed') from None
        with stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX)
            except OSError:
                raise LearningStorageError('learning_lock_failed') from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def _sync_directory(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _commit(self, document_id, state):
        codec.validate_state(state, self.scope, document_id)
        payload = codec.dumps(state)
        path = self._path(document_id)
        temporary = None
        replacing = False
        started = perf_counter()
        outcome = "failed"
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                    prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            replacing = True
            os.replace(temporary, path)
            self._sync_directory(path.parent)
            outcome = "committed"
        except OSError:
            if replacing:
                outcome = "unknown"
                raise LearningCommitUnknown('learning_commit_unknown') from None
            raise LearningStorageError('learning_write_failed') from None
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
            logger.info("learning_file_commit", extra={"stage": "snapshot_commit", "outcome": outcome,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "entry_count": len(state['entries']), "operation_count": len(state['operations'])})

    def _document(self, document_id):
        identifier(document_id)
        document = self._documents.load_document(document_id)
        require(document.document_id == document_id, 'learning_document_identity_mismatch')
        return document

    @contextmanager
    def _read_context(self, document_id, *, include_document=True, source_only=False, **selection):
        document = self._document(document_id) if include_document else None
        state = codec.empty_state(self.scope, document_id) if source_only else self._load(document_id)
        yield document, state

    @contextmanager
    def _write_context(self, document_id, **selection):
        identifier(document_id)
        with self._preferences.locked_snapshot(document_id) as (workspace, book):
            # Rename/delete won the shared preference lock: the old identity is
            # tombstoned. Reject late writers before opening its removed file.
            require(not book.deleted, 'learning_document_missing')
            with self._documents.locked_document(document_id) as document:
                with self._lock(document_id):
                    blocks = {b.source.content_block_id: b
                              for b in learning_blocks(document, self.scope, document_id)}
                    yield workspace, book, blocks, self._load(document_id), lambda state: self._commit(document_id, state)

    def recover(self, document_id):
        """Explicit orphan-temp cleanup only; never adopt a temp or retry a model.

        Atomic snapshot is the sole truth, even if a larger/newer temp survived a
        crash. A corrupt canonical file fails closed and remains for diagnosis.
        """
        with self._lock(document_id):
            self._load(document_id)
            path = self._path(document_id)
            removed = 0
            try:
                for temporary in path.parent.glob(path.name + '.*.tmp'):
                    temporary.unlink()
                    removed += 1
                self._sync_directory(path.parent)
            except OSError:
                raise LearningStorageError('learning_recovery_failed') from None
            return removed
