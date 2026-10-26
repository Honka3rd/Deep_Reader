"""Immutable, internal cleanup input; no extraction, planning or persistence."""

import json
from typing import Literal

from pydantic import Field, model_validator

from doc_loaders.extraction_identity import ExtractionResult, extraction_matches, text_digest
from document_structure.enhancement_change import compare_enhancement_documents
from document_structure.ocr_cleanup_contract import (
    ActualOcrEvidence, CanonicalOcrSpan, CleanupContract, Digest, Identifier,
    cleanup_eligibility,
)
from document_structure.source_mapped_text import PageRange


class CleanupViewPart(CleanupContract):
    """One contiguous canonical interval, or protected manual replacement context."""

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str
    manual: bool = False

    @model_validator(mode="after")
    def ordered(self):
        if self.char_end <= self.char_start:
            raise ValueError("cleanup_view_range_invalid")
        return self


class CleanupInput(CleanupContract):
    """Adapter-produced snapshot, never an HTTP request or authorization proof.

    hierarchy_json freezes the existing DTO's mutable nested lists/dictionaries.
    Consumers receive a detached document, not a second persistence authority.
    The adapter must prove accepted source and coherent storage revisions; commit
    must recheck them. No parser mode or newly successful split is required here.
    """

    schema_version: Literal["cleanup_input_v1"] = "cleanup_input_v1"
    namespace: Identifier
    document_id: Identifier
    source: ExtractionResult
    # Trusted source-owner location, not persisted hierarchy metadata or an HTTP field.
    accepted_source_path: str | None = Field(default=None, min_length=1, repr=False)
    structure_version: int = Field(ge=1)
    hierarchy_json: str = Field(repr=False)
    hierarchy_fingerprint: Digest
    reading_revision: int = Field(ge=0)
    evidence: ActualOcrEvidence | None
    pages: tuple[PageRange, ...] = ()
    protected_ranges: tuple[CanonicalOcrSpan, ...] = ()
    prior_deletions: tuple[CanonicalOcrSpan, ...] = ()
    parts: tuple[CleanupViewPart, ...]

    def document(self):
        from document_structure.structured_document import StructuredDocument

        return StructuredDocument.from_dict(json.loads(self.hierarchy_json))

    @model_validator(mode="after")
    def coherent(self):
        try:
            payload = json.loads(self.hierarchy_json)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False)
            if self.hierarchy_json != encoded or text_digest(encoded) != self.hierarchy_fingerprint:
                raise ValueError("hierarchy_fingerprint_mismatch")
            if "sections" in payload or "structure_nodes" in payload:
                raise ValueError("legacy_hierarchy")
            document = self.document()
            if document.document_id != self.document_id:
                raise ValueError("document_mismatch")
            compare_enhancement_documents(document, document, canonical_text=self.source.canonical_text)
            if not extraction_matches(document.parse_provenance, self.source.identity,
                                      self.source.canonical_text):
                raise ValueError("source_mismatch")
            revision = document.parse_provenance.get("reading_content", {}).get("revision", 0)
            if type(revision) is not int or revision != self.reading_revision:
                raise ValueError("reading_revision_mismatch")
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ValueError("cleanup_snapshot_invalid") from None

        text = self.source.canonical_text
        for spans in (self.protected_ranges, self.prior_deletions):
            end = 0
            for span in spans:
                if not end <= span.char_start < span.char_end <= len(text):
                    raise ValueError("cleanup_ranges_invalid")
                end = span.char_end
        page_end, page_index = 0, -1
        for page in self.pages:
            if (any(type(v) is not int for v in (page.page_index, page.start, page.end))
                    or page.page_index <= page_index
                    or not page_end <= page.start <= page.end <= len(text)
                    or text[page_end:page.start].strip()):
                raise ValueError("cleanup_pages_invalid")
            page_end, page_index = page.end, page.page_index
        if self.pages and text[page_end:].strip():
            raise ValueError("cleanup_pages_invalid")

        # Gaps must be explicit prior deletions, never fabricated contiguous text.
        intervals = [(p.char_start, p.char_end) for p in self.parts]
        if intervals != sorted(intervals):
            raise ValueError("cleanup_view_unordered")
        intervals += [(p.char_start, p.char_end) for p in self.prior_deletions]
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor or end > len(text):
                raise ValueError("cleanup_view_coverage_invalid")
            cursor = end
        if cursor != len(text):
            raise ValueError("cleanup_view_coverage_invalid")
        for part in self.parts:
            protected = any(p.char_start <= part.char_start < part.char_end <= p.char_end
                            for p in self.protected_ranges)
            if part.manual and not protected:
                raise ValueError("cleanup_manual_context_unprotected")
            if not part.manual and part.text != text[part.char_start:part.char_end]:
                raise ValueError("cleanup_view_text_mismatch")
        if any(a.char_start < b.char_end and b.char_start < a.char_end
               for a in self.prior_deletions for b in self.protected_ranges):
            raise ValueError("cleanup_deleted_protected_overlap")

        # Explicit OCR identity is evidence itself. Supplied evidence must still
        # match, including in OCR mode where the legacy eligibility helper skips it.
        if self.evidence is not None:
            if (self.evidence.extraction_identity != self.source.identity
                    or self.evidence.text_sha256 != self.source.text_sha256):
                raise ValueError("ocr_evidence_mismatch")
            end = 0
            for span in self.evidence.spans:
                if not end <= span.char_start < span.char_end <= len(text):
                    raise ValueError("ocr_evidence_mismatch")
                end = span.char_end
        return self

    def eligibility(self):
        return cleanup_eligibility(self.source, self.evidence)

    def fingerprint(self) -> str:
        """Bind consent to the complete view, evidence and all source revisions."""
        return text_digest(json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                      separators=(",", ":"), ensure_ascii=False))

    def source_structure_fingerprint(self) -> str:
        """Stable across content-only cleanup and explicit task-layout preparation."""
        return text_digest(json.dumps([self.namespace, self.document_id, self.structure_version,
            self.source.identity.to_dict(), self.source.text_sha256], sort_keys=True))
