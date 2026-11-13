"""Manual replacement policy; storage owners supply accepted-source verification."""
from copy import deepcopy
from dataclasses import replace

from document_structure.manual_structure_document_builder import validate_manual_coverage


# These records describe the accepted source, not an old hierarchy or interactions.
_SOURCE_KEYS = ("extraction", "pdf_extraction", "source_page_spans", "pdf_page_spans",
                "actual_ocr_evidence", "source_review")


def prepare_manual_replacement(current, draft, validate_current):
    canonical_text = validate_current(current)
    if draft.raw_text != canonical_text:
        raise ValueError("manual_source_binding_mismatch")
    validate_manual_coverage(draft, canonical_text)
    provenance = {key: deepcopy(draft.parse_provenance[key]) for key in (
        "requested_parser_mode", "effective_parser_mode", "fallback_used",
        "fallback_reason", "source", "manual_structure") if key in draft.parse_provenance}
    provenance.update({key: deepcopy(current.parse_provenance[key]) for key in _SOURCE_KEYS
                       if key in current.parse_provenance})
    return replace(draft, document_id=current.document_id, source_path=current.source_path, language=current.language,
                   title=current.title, parse_provenance=provenance,
                   sections=[], structure_nodes=[], document_task_artifacts=None,
                   chapters=[replace(chapter, task_artifacts=None, sections=[
                       replace(section, task_artifacts=None, task_units=[])
                       for section in chapter.sections]) for chapter in draft.chapters])
