"""Invalidate file-backed derived profile fields using committed hierarchy truth."""
from dataclasses import replace

from profile.document_profile_store import DocumentProfileStore


class VersionedDocumentProfileStore(DocumentProfileStore):
    def __init__(self, repository=None):
        self.repository = repository

    def supports_enhancement_invalidation(self, repository):
        return (self.repository is repository
                and callable(getattr(repository, "profile_structure_invalidated", None)))

    def load(self, config):
        profile = super().load(config)
        reader = getattr(self.repository, "profile_structure_invalidated", None)
        # Do not serve old structure advice when authoritative storage is unavailable.
        # Source-level profile fields and the file itself remain unchanged.
        if reader is not None and reader(config.doc_name):
            return replace(profile, post_structure_metadata=None, structure_profile=None)
        return profile
