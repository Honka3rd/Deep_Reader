"""Read-only source and accepted-version checks across derived work."""
from pathlib import Path

from document_preparation.accepted_source_snapshot import (
    SourceNotReady, resolve_accepted_source_snapshot,
)


def resolve_snapshot(document, doc_name, source_path):
    # File repository's initial version is 1; PostgreSQL supplies its persisted
    # storage_structure_version. This never adds a version to stored metadata.
    return resolve_accepted_source_snapshot(
        document=document, namespace=doc_name, expected_document_id=document.document_id,
        structure_version=document.parse_provenance.get("storage_structure_version", 1),
        source_path=Path(source_path), accepted=True)


def source_revision(path):
    stat = path.stat()
    return (path.resolve(), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


class AcceptedSourceGuard:
    def __init__(self, *, document, doc_name, source_path, current_document):
        self.doc_name = doc_name
        self.source_path = source_path
        self.current_document = current_document
        self.incarnation = document.parse_provenance.get("interaction_epoch")
        path = Path(source_path())
        self.revision = source_revision(path)
        self.snapshot = resolve_snapshot(document, doc_name, path)
        self.verify_source()

    def verify_source(self):
        self.verify_document(self.current_document())

    def verify_document(self, document):
        """Validate a caller-owned consistent snapshot without another storage read."""
        path = Path(self.source_path())
        if source_revision(path) != self.revision:
            raise SourceNotReady("source_changed")
        current = resolve_snapshot(document, self.doc_name, path)
        if (current.document_id != self.snapshot.document_id
                or document.parse_provenance.get("interaction_epoch") != self.incarnation
                or current.structure_version != self.snapshot.structure_version
                or current.extraction_identity != self.snapshot.extraction_identity
                or current.text_sha256 != self.snapshot.text_sha256
                or source_revision(Path(self.source_path())) != self.revision):
            raise SourceNotReady("accepted_source_changed")
        return current
