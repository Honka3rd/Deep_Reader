"""Explicit file rename/delete coordinator with durable forward recovery.

Composition installs this on the canonical document repository. Content edits use
source epochs in that repository's one-file commit; rename/delete coordinate
source, learning sidecar and preference metadata. Reads fail closed while the
journal exists. Recovery is an explicit maintenance command, never a read repair.
"""
from contextlib import ExitStack
from hashlib import sha256
import json
import os
import re
from pathlib import Path
import tempfile
from time import time
from uuid import uuid4

from document_structure import learning_file_codec as codec
from document_structure.learning_lifecycle import rename_learning_state
from document_structure.learning_repository import LearningCommitUnknown, LearningStorageError
from reading_preferences.models import identifier
from shared.learning_source import require


def _hash(payload):
    return None if payload is None else sha256(payload.encode('utf-8')).hexdigest()


def _read(path):
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, payload):
    if payload is None:
        if path.exists():
            path.unlink()
            _sync(path.parent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FileLearningDocumentLifecycle:
    def __init__(self, documents, learning, preferences):
        require(documents.namespace == learning.scope.namespace == preferences.scope.namespace,
                'learning_scope_mismatch')
        require(learning.document_repository is documents and learning.preference_repository is preferences,
                'learning_lifecycle_composition_mismatch')
        self.documents, self.learning, self.preferences = documents, learning, preferences
        documents.install_learning_lifecycle(self)

    def assert_ready(self):
        self.preferences.assert_lifecycle_ready()

    def _paths(self, document_id, new_document_id):
        paths = {'document': self.documents.lifecycle_path(document_id),
                 'learning': self.learning.lifecycle_path(document_id),
                 'preferences': self.preferences.lifecycle_data_path}
        # The owner supplies its exact location; never accept paths from a journal.
        if new_document_id is not None:
            paths.update(destination=self.documents.lifecycle_path(new_document_id),
                         learning_destination=self.learning.lifecycle_path(new_document_id))
        return paths

    def _locked(self, stack, document_id, new_document_id):
        stack.enter_context(self.preferences.lifecycle_lock())
        for name in sorted({document_id, new_document_id} - {None}):
            stack.enter_context(self.documents.lifecycle_lock(name))
        for name in sorted({document_id, new_document_id} - {None}):
            stack.enter_context(self.learning.lifecycle_lock(name))

    def _apply(self, journal):
        require(type(journal) is dict and set(journal) == {'version','document_id','new_document_id','writes'}
                and type(journal['version']) is int and journal['version'] == 1, 'learning_lifecycle_journal_invalid')
        paths = self._paths(journal['document_id'], journal['new_document_id'])
        require(type(journal['writes']) is dict and set(journal['writes']) == set(paths), 'learning_lifecycle_journal_invalid')
        require(journal['document_id'] != journal['new_document_id'], 'learning_lifecycle_journal_invalid')
        for owner, path in paths.items():
            value = journal['writes'][owner]
            require(type(value) is dict and set(value) == {'before','payload'}
                    and (value['payload'] is None or type(value['payload']) is str), 'learning_lifecycle_journal_invalid')
            require(value['before'] is None or (type(value['before']) is str and
                    re.fullmatch(r'[0-9a-f]{64}', value['before']) is not None), 'learning_lifecycle_journal_invalid')
            payload = value['payload']
            if owner in ('document', 'learning'):
                require(payload is None, 'learning_lifecycle_journal_invalid')
            elif owner == 'preferences':
                from reading_preferences.serialization import validate_state
                require(type(payload) is str, 'learning_lifecycle_journal_invalid')
                data = json.loads(payload, object_pairs_hook=codec.strict_object, parse_constant=codec.reject_constant)
                validate_state(self.preferences.scope, data)
            elif owner == 'destination':
                from document_structure.structured_document import StructuredDocument
                require(type(payload) is str, 'learning_lifecycle_journal_invalid')
                data = json.loads(payload, object_pairs_hook=codec.strict_object, parse_constant=codec.reject_constant)
                document = StructuredDocument.from_dict(data)
                require(document.document_id == journal['new_document_id'], 'learning_lifecycle_journal_invalid')
            elif payload is not None:
                codec.loads(payload, self.learning.scope, journal['new_document_id'])
            current = _hash(_read(path))
            require(current in (value['before'], _hash(value['payload'])), 'learning_lifecycle_recovery_conflict')
        # Each replacement is idempotent. No rollback guesses after a durable
        # journal; all readers/writers stay gated until forward recovery finishes.
        for owner, path in paths.items():
            payload = journal['writes'][owner]['payload']
            if _read(path) != payload:
                atomic_write(path, payload)
        atomic_write(self.preferences.lifecycle_journal_path, None)

    def _change(self, document_id, new_document_id, transform):
        identifier(document_id)
        if new_document_id is not None:
            identifier(new_document_id)
            require(document_id != new_document_id, 'learning_rename_same_identity')
        with ExitStack() as stack:
            self._locked(stack, document_id, new_document_id)
            self.assert_ready()
            paths = self._paths(document_id, new_document_id)
            if not paths['document'].exists():
                return False
            payloads = {owner: None for owner in paths}
            if new_document_id is not None:
                require(not paths['destination'].exists() and not paths['learning_destination'].exists(),
                        'learning_rename_destination_exists')
                before = self.documents.load_document(document_id)
                renamed = transform(before, new_document_id)
                require(renamed.document_id == new_document_id, 'learning_rename_identity_mismatch')
                from document_structure.learning_lifecycle import effective_sources
                require(effective_sources(before) == effective_sources(renamed)
                        and before.parse_provenance == renamed.parse_provenance, 'learning_rename_source_changed')
                payloads['destination'] = self.documents.lifecycle_payload(renamed)
                if paths['learning'].exists():
                    state = self.learning.lifecycle_state(document_id)
                    updated = rename_learning_state(state, self.learning.scope, document_id, new_document_id,
                                                    now_ms=int(time()*1000))
                    payloads['learning_destination'] = codec.dumps(updated)
            staged = self.preferences.stage_document_lifecycle(document_id,
                new_document_id=new_document_id, mutation_id=uuid4().hex)
            require(staged.path == paths['preferences'], 'learning_lifecycle_composition_mismatch')
            payloads['preferences'] = staged.payload
            journal = {'version':1, 'document_id':document_id, 'new_document_id':new_document_id,
                'writes':{owner:{'before':_hash(_read(path)), 'payload':payloads[owner]} for owner,path in paths.items()}}
            # Detect known path/access failures before publishing any lifecycle write.
            # This cannot promise future disk capacity; post-journal IO failures still
            # require forward recovery under the existing owner locks.
            for path in (*paths.values(), self.preferences.lifecycle_journal_path):
                parent = path.parent
                while not parent.exists():
                    parent = parent.parent
                if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                    raise LearningStorageError('document_lifecycle_path_unwritable')
            try:
                atomic_write(self.preferences.lifecycle_journal_path, codec.canonical(journal))
                self._apply(journal)
            except Exception:
                # Journal acknowledgement or any participating file may be durable.
                raise LearningCommitUnknown('document_lifecycle_commit_unknown') from None
            return True

    def rename(self, document_id, new_document_id, transform):
        return self._change(document_id, new_document_id, transform)

    def delete(self, document_id):
        return self._change(document_id, None, None)

    def recover(self, *, expected_rename=None):
        """Forward-complete one interrupted workspace operation under owner locks."""
        with self.preferences.lifecycle_lock():
            raw = _read(self.preferences.lifecycle_journal_path)
            if raw is None:
                return False
            try:
                journal = json.loads(raw, object_pairs_hook=codec.strict_object, parse_constant=codec.reject_constant)
                if expected_rename is not None:
                    require((journal['document_id'], journal['new_document_id']) == expected_rename,
                            'learning_lifecycle_recovery_conflict')
                identifier(journal['document_id'])
                if journal['new_document_id'] is not None:
                    identifier(journal['new_document_id'])
                with ExitStack() as stack:
                    for name in sorted({journal['document_id'],journal['new_document_id']} - {None}):
                        stack.enter_context(self.documents.lifecycle_lock(name))
                    for name in sorted({journal['document_id'],journal['new_document_id']} - {None}):
                        stack.enter_context(self.learning.lifecycle_lock(name))
                    self._apply(journal)
            except (ValueError, KeyError, TypeError, OSError):
                raise LearningStorageError('document_lifecycle_recovery_failed') from None
            return True
