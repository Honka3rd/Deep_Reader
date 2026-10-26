"""Read-only adapters for accepted Common documents and in-memory split results."""

from copy import deepcopy
from dataclasses import replace
import json
import logging
from time import perf_counter

from doc_loaders.extraction_identity import ExtractionResult, extraction_matches, text_digest
from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot
from document_preparation.cleanup_handoff import CleanupSourceContext, accepted_cleanup_context
from document_preparation.structure_enhancement_candidate import (
    StructureEnhancementCandidate, validate_enhancement_document,
)
from document_structure.cleanup_input import CleanupInput
from document_structure.cleanup_reading_view import cleanup_reading_view
from document_structure.enhancement_change import compare_enhancement_documents
from document_structure.ocr_cleanup_overlay import manual_source_edits
from document_structure.reading_content import ReadingContentConflict, state_for
from document_structure.structured_document import StructuredDocument
from language.language_code import LanguageCode, LanguageCodeResolver


LOGGER = logging.getLogger(__name__)


class CleanupInputUnavailable(ValueError):
    """Safe reason only; the caller decides recovery, never implicit Prepare."""


def _context(source, provenance):
    context = accepted_cleanup_context(source, provenance)
    if provenance.get("actual_ocr_evidence") is not None and context.evidence is None:
        raise CleanupInputUnavailable("ocr_evidence_invalid")
    pages = provenance.get("source_page_spans", ())
    if not isinstance(pages, (list, tuple)) or pages and not context.pages:
        raise CleanupInputUnavailable("cleanup_page_evidence_invalid")
    return context


def _assemble(document, snapshot, expected_reading_revision, accepted_source_path=None):
    source = ExtractionResult(snapshot.extraction_identity, snapshot.canonical_text)
    if (source.text_sha256 != snapshot.text_sha256 or source.text_length != snapshot.text_length
            or type(snapshot.text_length) is not int):
        raise CleanupInputUnavailable("cleanup_source_changed")
    if (document.document_id != snapshot.document_id or document.parse_provenance.get("reading_projected")
            or not extraction_matches(document.parse_provenance, source.identity, source.canonical_text)):
        raise CleanupInputUnavailable("cleanup_source_changed")
    language = LanguageCodeResolver.resolve(document.language)
    if language == LanguageCode.UNKNOWN or language.value != snapshot.language:
        raise CleanupInputUnavailable("cleanup_language_unavailable")
    version = document.parse_provenance.get("storage_structure_version", snapshot.structure_version)
    if type(version) is not int or version != snapshot.structure_version:
        raise CleanupInputUnavailable("cleanup_structure_changed")
    revision = state_for(document).get("revision", 0)
    if (type(expected_reading_revision) is not int or type(revision) is not int
            or revision != expected_reading_revision):
        raise CleanupInputUnavailable("cleanup_reading_revision_changed")
    try:
        compare_enhancement_documents(document, document, canonical_text=source.canonical_text)
    except ValueError:
        raise CleanupInputUnavailable("cleanup_hierarchy_invalid") from None
    context = _context(source, document.parse_provenance)
    if snapshot.cleanup_context is not None and context != snapshot.cleanup_context:
        raise CleanupInputUnavailable("cleanup_evidence_changed")
    try:
        view = cleanup_reading_view(document)
    except ReadingContentConflict:
        raise CleanupInputUnavailable("cleanup_reading_mapping_invalid") from None
    encoded = json.dumps(document.to_dict(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return CleanupInput(namespace=snapshot.namespace, document_id=snapshot.document_id,
        source=source, accepted_source_path=accepted_source_path,
        structure_version=snapshot.structure_version, hierarchy_json=encoded,
        hierarchy_fingerprint=text_digest(encoded), reading_revision=revision,
        evidence=context.evidence, pages=context.pages, parts=view.parts,
        protected_ranges=view.protected_ranges, prior_deletions=view.prior_deletions)


def _adapt(origin, build):
    started = perf_counter()
    try:
        result = build()
    except CleanupInputUnavailable as error:
        reason = str(error)
    except (ValueError, TypeError, KeyError, AttributeError):
        reason = "cleanup_input_invalid"
    else:
        LOGGER.info("cleanup_input origin=%s outcome=ready eligibility=%s parts=%s protected=%s gaps=%s elapsed_ms=%.1f",
                    origin, result.eligibility().status, len(result.parts), len(result.protected_ranges),
                    len(result.prior_deletions), (perf_counter() - started) * 1000)
        return result
    LOGGER.info("cleanup_input origin=%s outcome=rejected reason=%s elapsed_ms=%.1f",
                origin, reason, (perf_counter() - started) * 1000)
    raise CleanupInputUnavailable(reason) from None


def adapt_document_cleanup_input(*, document: StructuredDocument, snapshot: AcceptedSourceSnapshot,
                                 expected_reading_revision: int,
                                 accepted_source_path: str | None = None) -> CleanupInput:
    """Caller supplies one coherent accepted storage/source snapshot.

    Source-file acceptance is established by resolve_accepted_source_snapshot before
    entry. This adapter performs no I/O, source extraction or profile generation.
    Commit still rechecks current source/hierarchy/reading state under its lock.
    """
    return _adapt("stored", lambda: _assemble(document, snapshot, expected_reading_revision,
                                             accepted_source_path))


def adapt_candidate_cleanup_input(*, candidate: StructureEnhancementCandidate,
                                  baseline: CleanupInput) -> CleanupInput:
    """Reuse the same accepted reading view after splitting, before any save.

    Old block edits are represented in canonical coordinates in a detached input
    copy. Neither the original candidate nor the accepted baseline is modified.
    """
    def build():
        snapshot = candidate.source
        previous = baseline.document()
        if (snapshot.namespace != baseline.namespace or snapshot.document_id != baseline.document_id
                or snapshot.structure_version != baseline.structure_version
                or snapshot.extraction_identity != baseline.source.identity
                or snapshot.canonical_text != baseline.source.canonical_text
                or snapshot.language != LanguageCodeResolver.resolve(previous.language).value):
            raise CleanupInputUnavailable("cleanup_candidate_source_changed")
        context = CleanupSourceContext(baseline.evidence, baseline.pages)
        if snapshot.cleanup_context is not None and snapshot.cleanup_context != context:
            raise CleanupInputUnavailable("cleanup_evidence_changed")
        validate_enhancement_document(candidate.document, snapshot)
        provenance = deepcopy(candidate.document.parse_provenance)
        supplied = _context(baseline.source, provenance)
        if ("actual_ocr_evidence" in provenance and supplied.evidence != context.evidence
                or "source_page_spans" in provenance and supplied.pages != context.pages):
            raise CleanupInputUnavailable("cleanup_evidence_changed")
        # Unit IDs may change after splitting; preserve manual intent via its
        # existing canonical mapping, not old block IDs in the new hierarchy.
        state = deepcopy(state_for(previous))
        state["source_edits"] = manual_source_edits(previous)
        state.pop("blocks", None)
        state["revision"] = baseline.reading_revision
        if state["source_edits"] or state.get("cleanup_spans"):
            state["cleanup_source_sha256"] = baseline.source.text_sha256
        provenance["reading_content"] = state
        provenance["source_page_spans"] = [dict(page_index=p.page_index, char_start=p.start, char_end=p.end)
                                            for p in context.pages]
        if context.evidence is not None:
            provenance["actual_ocr_evidence"] = context.evidence.model_dump(mode="json")
        if "source_review" in previous.parse_provenance:
            provenance["source_review"] = deepcopy(previous.parse_provenance["source_review"])
        document = replace(candidate.document, parse_provenance=provenance)
        return _assemble(document, replace(snapshot, cleanup_context=context), baseline.reading_revision,
                         baseline.accepted_source_path)

    return _adapt("candidate", build)
