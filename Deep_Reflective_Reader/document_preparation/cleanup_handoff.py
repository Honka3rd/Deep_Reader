"""Accepted-source capabilities for optional cleanup; no loader or storage calls."""

from dataclasses import asdict, dataclass
import logging
import json

from doc_loaders.extraction_identity import ExtractionResult
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, CanonicalOcrSpan, CleanupOutcome
from document_structure.ocr_cleanup_overlay import CleanupApplication
from document_structure.ocr_cleanup_planner import plan_ocr_cleanup
from document_structure.source_mapped_text import PageRange


@dataclass(frozen=True)
class CleanupSourceContext:
    evidence: ActualOcrEvidence | None = None
    pages: tuple[PageRange, ...] = ()


def extraction_cleanup_metadata(artifact):
    """Bind adapter-owned OCR coordinates to the exact accepted extraction."""
    if artifact.ocr_spans is None:
        return {}
    evidence = ActualOcrEvidence(schema_version="cleanup_ocr_evidence_v1",
        extraction_identity=artifact.result.identity, text_sha256=artifact.result.text_sha256,
        spans=tuple(CanonicalOcrSpan(**asdict(span)) for span in artifact.ocr_spans))
    return {"actual_ocr_evidence": evidence.model_dump(mode="json")}


def accepted_cleanup_context(source, provenance):
    """Read only owner-produced evidence, never infer OCR from text or file type."""
    evidence = None
    raw = provenance.get("actual_ocr_evidence")
    if raw is not None:
        try:
            evidence = ActualOcrEvidence.model_validate_json(json.dumps(raw))
        except (ValueError, TypeError):
            logging.getLogger(__name__).warning("cleanup_handoff evidence=invalid")
    pages = ()
    raw_pages = provenance.get("source_page_spans", ())
    try:
        pages = tuple(PageRange(p["page_index"], p["char_start"], p["char_end"]) for p in raw_pages)
        end, index = 0, -1
        for page in pages:
            if (any(type(v) is not int for v in (page.start, page.end, page.page_index))
                    or not end <= page.start <= page.end <= source.text_length
                    or page.page_index <= index or source.canonical_text[end:page.start].strip()):
                raise ValueError("invalid_source_pages")
            end, index = page.end, page.page_index
        if pages and source.canonical_text[end:].strip():
            raise ValueError("incomplete_source_pages")
    except (ValueError, TypeError, KeyError):
        pages = ()
        logging.getLogger(__name__).warning("cleanup_handoff pages=invalid")
    return CleanupSourceContext(evidence=evidence, pages=pages)


@dataclass(frozen=True)
class PreparedCleanup:
    application: CleanupApplication | None
    outcome: CleanupOutcome | None


def prepare_enhancement_cleanup(*, candidate, binding, provider, baseline=None, is_cancelled=None):
    """Plan against the validated in-memory split, before its sole final commit.

    The caller owns consent and commit. An application is not a success receipt;
    storage revalidates it under its lock. Optional failures leave candidate intact.
    """
    from document_preparation.structure_enhancement_candidate import validate_enhancement_document

    validate_enhancement_document(candidate.document, candidate.source)
    if baseline is not None:
        from document_preparation.cleanup_input_adapter import adapt_candidate_cleanup_input
        from document_structure.ocr_cleanup_contract import BoundCleanupProposal, CleanupProposal
        from document_structure.ocr_cleanup_planner import plan_cleanup_input
        from document_structure.reading_content import digest
        snapshot = adapt_candidate_cleanup_input(candidate=candidate, baseline=baseline)
        planned = plan_cleanup_input(snapshot=snapshot, provider=provider,
            operation_id=binding.operation_id, is_cancelled=is_cancelled)
        # An empty failed plan still carries prior overlays through a structural
        # commit. The separate outcome retains the failure, not a false no-op.
        proposal = planned.proposal or CleanupProposal(schema_version="ocr_cleanup_plan_v1", items=())
        application = CleanupApplication(BoundCleanupProposal(binding=binding, proposal=proposal),
            evidence=snapshot.evidence, pages=snapshot.pages, cleanup_input=snapshot,
            baseline_digest=digest(baseline.document().to_dict()),
            candidate_digest=digest(candidate.document.to_dict()))
        return PreparedCleanup(application=application, outcome=planned.outcome)
    snapshot = candidate.source
    source = ExtractionResult(snapshot.extraction_identity, snapshot.canonical_text)
    context = snapshot.cleanup_context or CleanupSourceContext()
    planned = plan_ocr_cleanup(source=source, binding=binding, provider=provider,
        structure_ready=True, evidence=context.evidence, is_cancelled=is_cancelled)
    application = (CleanupApplication(planned.proposal, evidence=context.evidence, pages=context.pages)
                   if planned.proposal is not None else None)
    return PreparedCleanup(application=application, outcome=planned.outcome)
