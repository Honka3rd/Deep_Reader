"""Read-only validation of accepted text; never performs document extraction."""

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import re

from doc_loaders.extraction_identity import ExtractionIdentity, read_extraction_record
from document_structure.structured_document import StructuredDocument
from language.language_code import LanguageCode, LanguageCodeResolver


class SourceNotReady(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        self.recovery = "explicit_prepare_required"
        super().__init__(reason)


@dataclass(frozen=True)
class AcceptedSourceSnapshot:
    namespace: str
    document_id: str
    structure_version: int
    extraction_identity: ExtractionIdentity
    canonical_text: str
    text_sha256: str
    text_length: int
    language: str


def source_file_sha256(path: Path) -> str:
    """Hash bytes with bounded memory, without selecting or invoking a loader."""
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(source.fileno())
            current = path.stat()
            def revision(stat):
                return (stat.st_dev, stat.st_ino, stat.st_size,
                        stat.st_mtime_ns, stat.st_ctime_ns)
            if revision(before) != revision(after) or revision(after) != revision(current):
                raise SourceNotReady("source_changed")
            return digest.hexdigest()
    except FileNotFoundError as error:
        raise SourceNotReady("source_missing") from error
    except OSError as error:
        raise SourceNotReady("source_unreadable") from error


def resolve_accepted_source_snapshot(
    *,
    document: StructuredDocument,
    namespace: str,
    expected_document_id: str,
    structure_version: int,
    source_path: Path,
    accepted: bool,
) -> AcceptedSourceSnapshot:
    """Inputs must come from the storage/source owner, never client assertions.

    Caller reads document, identity and version consistently. This is a readiness
    snapshot, not authorization or a commit lock. No storage config is constructed
    here: some legacy config constructors migrate files even on a read path.
    """
    if not namespace or not expected_document_id or document.document_id != expected_document_id:
        raise SourceNotReady("document_identity_mismatch")
    if type(structure_version) is not int or structure_version < 1:
        raise SourceNotReady("structure_version_unknown")
    if accepted is not True or document.structure_error_code:
        raise SourceNotReady("source_not_accepted")
    provenance = document.parse_provenance
    try:
        identity, record = read_extraction_record(provenance)
        text_hash = record["text_sha256"]
        if not isinstance(text_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", text_hash):
            raise ValueError("invalid text digest")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise SourceNotReady("extraction_identity_unknown") from error

    if source_path.suffix.lower() != "." + identity.source_format:
        raise SourceNotReady("source_format_mismatch")
    if source_file_sha256(source_path) != identity.source_sha256:
        raise SourceNotReady("source_changed")
    language = LanguageCodeResolver.resolve(document.language)
    if language == LanguageCode.UNKNOWN:
        raise SourceNotReady("language_unavailable")

    text = document.raw_text
    if not isinstance(text, str):
        raise SourceNotReady("canonical_text_invalid")
    if not text:
        # PostgreSQL omits raw_text. Only exact, contiguous canonical section
        # slices plus the accepted full-text digest can prove reconstruction.
        parts = []
        cursor = 0
        for chapter in document.chapters:
            for section in chapter.sections:
                if (type(section.char_start) is not int
                        or type(section.char_end) is not int
                        or section.char_start != cursor
                        or section.char_end <= cursor
                        or not isinstance(section.content, str)
                        or len(section.content) != section.char_end - cursor):
                    raise SourceNotReady("canonical_coverage_invalid")
                parts.append(section.content)
                cursor = section.char_end
        text = "".join(parts)
    if not text.strip():
        raise SourceNotReady("canonical_text_empty")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_hash:
        raise SourceNotReady("canonical_hash_mismatch")
    if record.get("text_length", len(text)) != len(text):
        raise SourceNotReady("canonical_length_mismatch")

    logging.getLogger(__name__).info(
        "accepted_source outcome=ready format=%s schema=%s chars=%s extraction=reused",
        identity.source_format, identity.schema_version, len(text))

    return AcceptedSourceSnapshot(
        namespace=namespace, document_id=document.document_id,
        structure_version=structure_version, extraction_identity=identity,
        canonical_text=text, text_sha256=text_hash, text_length=len(text),
        language=language.value,
    )
