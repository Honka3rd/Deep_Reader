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
from app.document_file_identity import DocumentFileIdentity, sync_directory as _sync_directory
from reading_preferences.models import PreferenceConflict
from section_tasks.interaction_rename import renamed_interactions


class RenameRecoveryRequired(RuntimeError):
    def __init__(self, operation_id):
        super().__init__('document_rename_recovery_required')
        self.operation_id = operation_id


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
        self.file_identity = DocumentFileIdentity(self.raw_root)

    @contextmanager
    def _locked(self):
        with self.file_identity.locked():
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
        identity = self._file_stat(source)
        fingerprint = _fingerprint(source)
        if self._file_stat(source) != identity:
            raise ValueError('rename_resource_changed')
        return dict(kind=kind, source=source.name, target=target.name, fingerprint=fingerprint,
                    identity=identity, stage='planned', destination_stat=None)

    def source_token(self, old, raw_source=None, *, version=3):
        return self._source_context(old, raw_source, version=version)[0]

    def _source_context(self, old, raw_source, *, version=3):
        """Pure context read; the mutation rechecks this projection under its lock."""
        source = _document(self.repository, old)
        raw_identity = None
        if raw_source is not None:
            if raw_source.is_symlink() or not raw_source.is_file():
                raise ValueError('storage_failure')
            stat = raw_source.stat()
            raw_identity = [raw_source.name, stat.st_dev, stat.st_ino, stat.st_size,
                            stat.st_mtime_ns, stat.st_ctime_ns]
        if source is None and raw_identity is None:
            raise FileNotFoundError('missing_document')
        identity = [self.scope, old, _epoch(source), raw_identity]
        if version >= 3:
            identity.append(self.file_identity.read(old))
        return sha256(json.dumps(identity).encode()).hexdigest(), source

    def _validate_record(self, record, old, new):
        fields = {'version', 'scope', 'old', 'new', 'epoch', 'moves', 'operation_id',
                  'command_id', 'source_token', 'result_token'}
        if isinstance(record, dict) and record.get('version') == 3:
            fields.add('file_generations')
        if not isinstance(record, dict) or set(record) != fields:
            raise RenameRecoveryRequired('unreadable-journal')
        if (record['version'] not in (2, 3) or record['scope'] != self.scope or
                not isinstance(record['operation_id'], str) or
                re.fullmatch('[0-9a-f]{32}', record['operation_id']) is None or
                (record['result_token'] is not None and
                 (not isinstance(record['result_token'], str) or
                  re.fullmatch('[0-9a-f]{64}', record['result_token']) is None)) or
                not isinstance(record['command_id'], str) or
                re.fullmatch('[0-9a-f]{32}', record['command_id']) is None or
                not isinstance(record['source_token'], str) or
                re.fullmatch('[0-9a-f]{64}', record['source_token']) is None or
                not isinstance(record['moves'], list) or len(record['moves']) > 2 or
                (record['epoch'] is not None and not isinstance(record['epoch'], str))):
            raise RenameRecoveryRequired('unreadable-journal')
        if record['old'] != old or record['new'] != new:
            raise ValueError('conflict')
        if record['version'] == 3:
            generations = record['file_generations']
            if (not isinstance(generations, list) or len(generations) != 2 or
                    any(g is not None and (not isinstance(g, str) or re.fullmatch('[0-9a-f]{32}', g) is None)
                        for g in generations)):
                raise RenameRecoveryRequired(record['operation_id'])
        kinds = set()
        for move in record['moves']:
            move_fields = {'kind','source','target','fingerprint'}
            if record['version'] == 3:
                move_fields.update(('identity', 'stage', 'destination_stat'))
            if (not isinstance(move, dict) or set(move) != move_fields or
                    move['kind'] not in ('raw','derived') or move['kind'] in kinds or
                    not all(isinstance(move[key], str) for key in ('source','target','fingerprint')) or
                    re.fullmatch('[0-9a-f]{64}', move['fingerprint']) is None):
                raise RenameRecoveryRequired(record['operation_id'])
            if record['version'] == 3:
                def valid_stat(value):
                    return isinstance(value, list) and len(value) == 5 and all(type(n) is int and n >= 0 for n in value)
                if (not valid_stat(move['identity']) or move['stage'] not in ('planned', 'moved') or
                        (move['stage'] == 'planned' and move['destination_stat'] is not None) or
                        (move['stage'] == 'moved' and not valid_stat(move['destination_stat']))):
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

    @staticmethod
    def _file_stat(path):
        if path.is_symlink():
            raise ValueError('rename_resource_changed')
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

    def _check_moves(self, record):
        if record['version'] != 3:
            raise ValueError('rename_identity_evidence_missing')
        generations = [self.file_identity.read(name) for name in (record['old'], record['new'])]
        if generations != record['file_generations']:
            raise ValueError('rename_resource_changed')
        for move in record['moves']:
            source, target = self._paths(move)
            before, after = self._file_stat(source), self._file_stat(target)
            if before is not None and after is None:
                valid = move['stage'] == 'planned' and before == move['identity']
                path = source
            elif before is None and after is not None:
                # Rename can change ctime. A lost acknowledgement is proven by
                # the original inode/mtime plus unchanged durable generations.
                valid = after[:4] == move['identity'][:4]
                if move['stage'] == 'moved':
                    valid = valid and after == move['destination_stat']
                path = target
            else:
                raise ValueError('rename_resource_changed')
            if not valid or _fingerprint(path) != move['fingerprint']:
                raise ValueError('rename_resource_changed')

    def _finish(self, record, receipt):
        old, new = record['old'], record['new']
        self._check_moves(record)  # Fence replacements before even repository recovery.
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
            if move['stage'] == 'planned':
                # Also covers a completed rename whose acknowledgement was lost.
                self._check_moves(record)
                move['stage'] = 'moved'
                move['destination_stat'] = self._file_stat(target_path)
                self._save(self.pending, record)
        raw_target = next((self._paths(move)[1] for move in record['moves'] if move['kind'] == 'raw'), None)
        record = {**record, 'result_token': self.source_token(new, raw_target)}
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

    def _start(self, old, new, raw_source, raw_target, command_id, source_token):
        actual_token, source = self._source_context(old, raw_source)
        if actual_token != source_token:
            raise ValueError('conflict')
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
        record = dict(version=3, scope=self.scope, old=old, new=new,
                      file_generations=[self.file_identity.read(name) for name in (old, new)],
                      command_id=command_id, source_token=source_token, result_token=None,
                      epoch=_epoch(source), moves=moves, operation_id=uuid4().hex)
        try:
            self._save(self.pending, record)
        except Exception as error:
            if self.pending.exists():
                raise RenameRecoveryRequired(record['operation_id']) from error
            raise
        return record

    def rename(self, old, new, *, command_id, source_token, raw_source=None, raw_target=None):
        from uuid import UUID
        command_id = UUID(str(command_id)).hex
        if old == new or re.fullmatch('[0-9a-f]{64}', source_token) is None:
            raise ValueError('conflict')
        receipt = self.directory / ('command-' + command_id + '.json')
        with self._locked():
            if self.pending.exists() or receipt.exists():
                path = self.pending if self.pending.exists() else receipt
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError) as error:
                    raise RenameRecoveryRequired('unreadable-journal') from error
                self._validate_record(record, old, new)
                if record['command_id'] != command_id or record['source_token'] != source_token:
                    raise ValueError('conflict')
                if path == self.pending and record['version'] != 3:
                    raise RenameRecoveryRequired(record['operation_id'])
                if path == receipt:
                    if record['version'] == 3:
                        try:
                            self._check_moves(record)
                        except ValueError:
                            raise ValueError('conflict') from None
                    # A receipt is evidence of THIS command, never authorization
                    # to start another mutation after a reverse rename/reimport.
                    source = _document(self.repository, old)
                    target = _document(self.repository, new)
                    if source is not None or _epoch(target) != record['epoch']:
                        raise ValueError('conflict')
                    for move in record['moves']:
                        before, after = self._paths(move)
                        if before.exists() or before.is_symlink() or _fingerprint(after) != move['fingerprint']:
                            raise ValueError('conflict')
                    # A newly uploaded raw source also supersedes SQL-only receipts.
                    if raw_source is not None and (raw_source.exists() or raw_source.is_symlink()):
                        raise ValueError('conflict')
                    raw_target = next((self._paths(move)[1] for move in record['moves'] if move['kind'] == 'raw'), None)
                    if self.source_token(new, raw_target, version=record['version']) != record['result_token']:
                        raise ValueError('conflict')
                    return self._effects(record)
            else:
                record = self._start(old, new, raw_source, raw_target, command_id, source_token)
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
