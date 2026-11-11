"""Active-repository rename with explicit forward recovery of optional files.

The journal contains identities/fingerprints, never canonical document payloads.
Repository transactions/lifecycle journals remain owned by their repositories.
"""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import fcntl
import json
import os
import re
from pathlib import Path
from uuid import uuid4

from document_structure.learning_file_lifecycle import atomic_write
from reading_preferences.models import PreferenceConflict
from section_tasks.interaction_rename import renamed_interactions


class RenameRecoveryRequired(RuntimeError):
    def __init__(self, operation_id):
        super().__init__('document_rename_recovery_required')
        self.operation_id = operation_id


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fingerprint(path):
    """Reject links/special files; hash bounded chunks, not a whole FAISS in RAM."""
    if path.is_symlink():
        raise ValueError('storage_failure')
    if not path.exists():
        return None
    digest = sha256()
    paths = sorted(path.rglob('*')) if path.is_dir() else [path]
    for item in paths:
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError('storage_failure')
        digest.update(str(item.relative_to(path) if item != path else '.').encode())
        digest.update(b'D' if item.is_dir() else b'F')
        if item.is_file():
            with item.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def _document(repository, name):
    try:
        return repository.load_document(name)
    except FileNotFoundError:
        return None


def _epoch(document):
    if document is None:
        return None
    token = document.parse_provenance.get('interaction_epoch')
    if not token:
        # Legacy documents remain renameable; a new import always gets an epoch.
        # Exclude identity/title/path, which this operation is allowed to change.
        from document_structure.learning_lifecycle import effective_sources
        source = [document.raw_text, document.parse_provenance, effective_sources(document)]
        return 'legacy:' + sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return token


class DocumentRename:
    def __init__(self, repository, raw_root, derived_root):
        self.repository = repository
        self.raw_root, self.derived_root = Path(raw_root), Path(derived_root)
        identity = [repository.lifecycle_storage_identity,
                    str(self.raw_root.resolve()), str(self.derived_root.resolve())]
        self.scope = sha256(json.dumps(identity).encode()).hexdigest()
        self.directory = self.raw_root.parent / '.document-renames' / self.scope
        self.pending = self.directory / 'pending.json'

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Persist newly created journal directories before a repository commit.
        _sync_directory(self.directory.parent.parent)
        _sync_directory(self.directory.parent)
        with (self.directory/'rename.lock').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _save(self, path, record):
        atomic_write(path, json.dumps(record, sort_keys=True))

    def _paths(self, move):
        root = self.raw_root if move['kind'] == 'raw' else self.derived_root
        names = (move['source'], move['target'])
        if any(not name or Path(name).name != name or name in ('.','..') or '\\' in name for name in names):
            raise ValueError('storage_failure')
        return root/names[0], root/names[1]

    def _plan_move(self, kind, source, target):
        if target is not None and (target.exists() or target.is_symlink()):
            raise ValueError('conflict')
        if source is None:
            return None
        if source.is_symlink() or (source.exists() and (not source.is_file() if kind == 'raw' else not source.is_dir())):
            raise ValueError('storage_failure')
        if not source.exists():
            return None
        if not os.access(source.parent, os.W_OK | os.X_OK) or not os.access(target.parent, os.W_OK | os.X_OK):
            raise OSError('rename_directory_unwritable')
        return dict(kind=kind, source=source.name, target=target.name, fingerprint=_fingerprint(source))

    def _validate_record(self, record, old, new):
        if not isinstance(record, dict) or set(record) != {
                'version', 'scope', 'old', 'new', 'epoch', 'moves', 'operation_id'}:
            raise RenameRecoveryRequired('unreadable-journal')
        if (record['version'] != 1 or record['scope'] != self.scope or
                not isinstance(record['operation_id'], str) or
                re.fullmatch('[0-9a-f]{32}', record['operation_id']) is None or
                not isinstance(record['moves'], list) or len(record['moves']) > 2 or
                (record['epoch'] is not None and not isinstance(record['epoch'], str))):
            raise RenameRecoveryRequired('unreadable-journal')
        if record['old'] != old or record['new'] != new:
            raise ValueError('conflict')
        kinds = set()
        for move in record['moves']:
            if (not isinstance(move, dict) or set(move) != {'kind','source','target','fingerprint'} or
                    move['kind'] not in ('raw','derived') or move['kind'] in kinds or
                    not all(isinstance(move[key], str) for key in ('source','target','fingerprint')) or
                    re.fullmatch('[0-9a-f]{64}', move['fingerprint']) is None):
                raise RenameRecoveryRequired(record['operation_id'])
            kinds.add(move['kind'])
            names = (move['source'], move['target'])
            if move['kind'] == 'raw':
                if any(Path(name).suffix.lower() not in ('.txt','.pdf') for name in names):
                    raise RenameRecoveryRequired(record['operation_id'])
                names = tuple(Path(name).stem for name in names)
            if names != (old,new):
                raise RenameRecoveryRequired(record['operation_id'])
            self._paths(move)

    def _check_moves(self, record):
        for move in record['moves']:
            source, target = self._paths(move)
            before, after = _fingerprint(source), _fingerprint(target)
            if not ((before == move['fingerprint'] and after is None) or
                    (before is None and after == move['fingerprint'])):
                raise ValueError('rename_resource_changed')

    def _finish(self, record, receipt):
        old, new = record['old'], record['new']
        self.repository.recover_document_rename(old, new)
        source = _document(self.repository, old)
        target = _document(self.repository, new)
        expected = record['epoch']
        self._check_moves(record)
        if expected is not None:
            if source is not None and _epoch(source) == expected and target is None:
                # Revalidate epoch while the repository holds its source lock.
                def transform(document, name):
                    if _epoch(document) != expected:
                        raise ValueError('rename_source_changed')
                    renamed = renamed_interactions(document, name)
                    for move in record['moves']:
                        if move['kind'] == 'raw' and document.source_path:
                            source_path, target_path = self._paths(move)
                            if Path(document.source_path).resolve() == source_path.resolve():
                                renamed = replace(renamed, source_path=str(target_path))
                    return renamed
                if not self.repository.rename_document(old, new, transform):
                    raise ValueError('rename_source_changed')
            elif source is not None or target is None or _epoch(target) != expected:
                raise ValueError('rename_source_changed')
        elif source is not None or target is not None:
            raise ValueError('rename_source_changed')
        for move in record['moves']:
            source_path, target_path = self._paths(move)
            # Recheck just before every move; a retry never overwrites a destination.
            self._check_moves(record)
            if source_path.exists():
                source_path.rename(target_path)
                _sync_directory(source_path.parent)
        self._save(receipt, record)
        atomic_write(self.pending, None)
        return self._effects(record)

    @staticmethod
    def _effects(record):
        kinds = {move['kind'] for move in record['moves']}
        return dict(new_doc_name=record['new'], source_affected='raw' in kinds,
                    structured_artifacts_affected=record['epoch'] is not None,
                    derived_resources_affected='derived' in kinds,
                    operation_id=record['operation_id'])

    def _start(self, old, new, raw_source, raw_target):
        source = _document(self.repository, old)
        if _document(self.repository, new) is not None:
            raise ValueError('conflict')
        # A repository-only source must not adopt another raw candidate/profile.
        # Check both supported extensions, even when there is no raw file to move.
        if self.raw_root.exists():
            for candidate in self.raw_root.iterdir():
                if candidate.stem == new and candidate.suffix.lower() in ('.txt', '.pdf'):
                    raise ValueError('conflict')
        moves = [self._plan_move('raw', raw_source, raw_target),
                 self._plan_move('derived', self.derived_root/old, self.derived_root/new)]
        moves = [move for move in moves if move is not None]
        # A stale derived-only directory is not a document candidate.
        if source is None and not any(move['kind']=='raw' for move in moves):
            raise FileNotFoundError('missing_document')
        if source is not None:
            renamed_interactions(source, new)  # Predictable transform errors before journal/commit.
        record = dict(version=1, scope=self.scope, old=old, new=new,
                      epoch=_epoch(source), moves=moves, operation_id=uuid4().hex)
        try:
            self._save(self.pending, record)
        except Exception as error:
            if self.pending.exists():
                raise RenameRecoveryRequired(record['operation_id']) from error
            raise
        return record

    def rename(self, old, new, *, raw_source=None, raw_target=None):
        if old == new:
            raise ValueError('conflict')
        receipt = self.directory / (sha256(json.dumps([old,new]).encode()).hexdigest()+'.json')
        with self._locked():
            if self.pending.exists():
                try:
                    record = json.loads(self.pending.read_text())
                except (OSError, ValueError) as error:
                    raise RenameRecoveryRequired('unreadable-journal') from error
                self._validate_record(record, old, new)
            elif receipt.exists():
                record = json.loads(receipt.read_text())
                self._validate_record(record, old, new)
                # Permit a fresh rename after this same incarnation was explicitly
                # renamed back. Recreated books have a different epoch and are fenced.
                source = _document(self.repository, old)
                target = _document(self.repository, new)
                if source is not None:
                    if target is not None or _epoch(source) != record['epoch']:
                        raise ValueError('conflict')
                    record = self._start(old, new, raw_source, raw_target)
                else:
                    if _epoch(target) != record['epoch']:
                        raise ValueError('conflict')
                    self._check_moves(record)
                    return self._effects(record)
            else:
                record = self._start(old, new, raw_source, raw_target)
            try:
                return self._finish(record, receipt)
            except (PreferenceConflict, ValueError) as error:
                # These repository conflicts occur before its first write. Other
                # failures may have committed and MUST retain recovery evidence.
                if str(error) in ('conflict', 'preference_destination_exists', 'learning_rename_destination_exists'):
                    atomic_write(self.pending, None)
                    raise ValueError('conflict') from None
                raise RenameRecoveryRequired(record['operation_id']) from error
            except Exception as error:
                raise RenameRecoveryRequired(record['operation_id']) from error
