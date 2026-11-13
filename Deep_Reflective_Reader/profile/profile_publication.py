"""Source-bound profile publication; model work never runs inside this boundary."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path


class ProfilePublicationConflict(ValueError):
    pass


def document_binding(document):
    if document is None:
        return None
    provenance = document.parse_provenance
    return (document.document_id, provenance.get("interaction_epoch"),
            provenance.get("artifact_generation_epoch"),
            provenance.get("storage_structure_version", 1))


class ProfilePublicationGuard:
    def __init__(self, *, name, source, lifecycle, snapshot, profile_path):
        self.name, self.source, self.lifecycle = name, source, lifecycle
        self.snapshot, self.path = snapshot, Path(profile_path)
        self.generation = self.binding = self.digest = None
        self.captured = False

    def capture(self):
        if self.captured:
            raise ProfilePublicationConflict("profile_binding_already_captured")
        with self.lifecycle.locked(), self.snapshot() as document:
            try:
                self.source.verify_source()
            except Exception as error:
                raise ProfilePublicationConflict("profile_source_changed") from error
            self.generation = self.lifecycle.read(self.name)
            self.binding = document_binding(document)
            self.digest = self._digest()
            self.captured = True
        return self

    def _digest(self):
        try:
            return sha256(self.path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None

    def _verify(self, document):
        try:
            if not self.captured:
                raise ValueError("profile_binding_not_captured")
            self.source.verify_source()
            if (self.lifecycle.read(self.name) != self.generation
                    or document_binding(document) != self.binding
                    or self._digest() != self.digest):
                raise ValueError("identity_or_profile_changed")
        except Exception as error:
            raise ProfilePublicationConflict("profile_source_changed") from error

    @contextmanager
    def publication(self):
        with self.lifecycle.locked(), self.snapshot() as document:
            self._verify(document)
            yield lambda: self._verify(document)
            self.digest = self._digest()

    def verify_source(self):
        with self.lifecycle.locked(), self.snapshot() as document:
            self._verify(document)

    def accept_prepared_structure(self, document):
        """Only the explicit structure writer may hand over its saved snapshot."""
        if not self.captured:
            raise ProfilePublicationConflict("profile_binding_not_captured")
        self.binding = document_binding(document)
