"""Read-only review visibility from accepted extraction evidence."""

import json

from doc_loaders.extraction_identity import ExtractionResult, extraction_matches, read_extraction_record
from document_structure.ocr_cleanup_contract import ActualOcrEvidence, cleanup_eligibility


def ocr_review_available(document):
    from document_structure.ocr_cleanup_overlay import canonical_text

    provenance = document.parse_provenance
    try:
        identity, _ = read_extraction_record(provenance)
        text = canonical_text(document)
        if identity.source_format != "pdf" or not extraction_matches(provenance, identity, text):
            return False
        source = ExtractionResult(identity, text)
        if identity.mode == "ocr":
            return True
        if identity.mode != "auto":
            return False
        raw = provenance.get("actual_ocr_evidence")
        if raw is not None:
            evidence = ActualOcrEvidence.model_validate_json(json.dumps(raw))
            if cleanup_eligibility(source, evidence).status == "eligible":
                return True
        # Empty OCR output can have review regions without any recognized-text span.
        review = provenance.get("source_review", {})
        return (review.get("source_sha256") == identity.source_sha256
                and bool(review.get("regions")))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
