"""Proof for explicitly requested legacy TXT provenance migration."""
from dataclasses import replace

from doc_loaders.extraction_identity import extraction_matches
from document_preparation.accepted_source_snapshot import resolve_accepted_source_snapshot


def prove_txt_backfill(stored, result, source_path):
    if result.identity.source_format != "txt":
        raise ValueError("backfill_requires_txt")
    document = stored.document
    if "extraction" in document.parse_provenance or "pdf_extraction" in document.parse_provenance:
        if not extraction_matches(document.parse_provenance, result.identity, result.canonical_text):
            raise ValueError("backfill_identity_conflict")
    # Require complete ordered coverage even if a redundant raw_text is present.
    candidate = replace(document, raw_text="", parse_provenance={
        **document.parse_provenance, "extraction": result.record()})
    snapshot = resolve_accepted_source_snapshot(
        document=candidate, namespace=stored.namespace,
        expected_document_id=document.document_id, structure_version=stored.structure_version,
        source_path=source_path, accepted=stored.accepted)
    if snapshot.canonical_text != result.canonical_text:
        raise ValueError("backfill_text_mismatch")
    if document.raw_text and document.raw_text != result.canonical_text:
        raise ValueError("backfill_raw_text_mismatch")
    return result.record()
