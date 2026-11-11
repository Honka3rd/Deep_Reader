"""Durable file-namespace generations for cooperating upload/delete/rename writers.

Reads are pure. Mutations publish a new generation before touching raw/derived
files, under the shared raw-root lock. This deliberately fails closed on an
uncertain generation acknowledgement instead of adopting a replacement file.
"""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import fcntl
import json
import os
import re

from document_structure.learning_file_lifecycle import atomic_write


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DocumentFileIdentity:
    def __init__(self, raw_root):
        root = Path(raw_root).resolve()
        scope = sha256(str(root).encode()).hexdigest()
        self.directory = root.parent / '.document-file-identities' / scope

    @contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        sync_directory(self.directory.parent.parent)
        sync_directory(self.directory.parent)
        with (self.directory / 'identity.lock').open('a') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _path(self, name):
        return self.directory / (sha256(name.encode()).hexdigest() + '.json')

    def read(self, name):
        path = self._path(name)
        try:
            record = json.loads(path.read_text())
        except FileNotFoundError:
            return None  # Legacy untouched namespace, never repaired by a read.
        except ValueError as error:
            raise OSError('file_identity_unreadable') from error
        if (not isinstance(record, dict) or set(record) != {'version', 'generation'} or
                record['version'] != 1 or not isinstance(record['generation'], str) or
                re.fullmatch('[0-9a-f]{32}', record['generation']) is None):
            raise OSError('file_identity_unreadable')
        return record['generation']

    def advance(self, name):
        """Caller holds locked(); durable invalidation MUST precede file effects."""
        self.read(name)  # Corruption is not permission to reset history.
        generation = uuid4().hex
        atomic_write(self._path(name), json.dumps({'version': 1, 'generation': generation}))
        return generation
