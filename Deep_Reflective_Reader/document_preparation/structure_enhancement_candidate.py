"""Build a source-preserving candidate without extraction or persistence."""
from dataclasses import dataclass, replace
import hashlib

from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot
from document_structure.common_parse_policy import TocHint
from document_structure.section_splitter_selector import SectionSplitterMode
from document_structure.structure_plan_contract import EnhancementIntent
from document_structure.structured_document import StructuredDocument
from language.language_code import LanguageCode


class EnhancementCandidateRejected(ValueError):
    """Stable failure reason; never include source or provider output."""


@dataclass(frozen=True)
class StructureEnhancementCandidate:
    source: AcceptedSourceSnapshot
    document: StructuredDocument


def build_structure_enhancement_candidate(*, snapshot, builder, toc_hint="auto",
                                          enhancement_intent="restore_source"):
    """Internal candidate only. Caller owns consent, lease and atomic commit."""
    hint = TocHint(toc_hint)
    intent = EnhancementIntent(enhancement_intent)
    language = LanguageCode(snapshot.language)
    text = snapshot.canonical_text
    if (language == LanguageCode.UNKNOWN or not text.strip()
            or len(text) != snapshot.text_length
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != snapshot.text_sha256):
        raise EnhancementCandidateRejected("invalid_enhancement_snapshot")
    try:
        document = builder.build(
            document_id=snapshot.document_id, title=snapshot.document_id,
            raw_text=text, language=language, parser_mode=SectionSplitterMode.LLM_ENHANCED,
            toc_hint=hint, enhancement_intent=intent.value,
            page_evidence=None, page_boundaries=None, outline=None,
            page_evidence_provider=None,
        )
    except Exception as error:
        raise EnhancementCandidateRejected("enhancement_builder_failed") from error
    validate_enhancement_document(document, snapshot)
    # Preserve accepted extraction evidence without copying old interaction data.
    provenance = dict(document.parse_provenance)
    from doc_loaders.extraction_identity import ExtractionIdentity, normalize_pdf_identity
    identity = snapshot.extraction_identity
    if not isinstance(identity, ExtractionIdentity):
        identity = normalize_pdf_identity(identity)
    provenance.pop("pdf_extraction", None)
    provenance["extraction"] = identity.record(text)
    provenance["enhancement_source"] = {
        "structure_version": snapshot.structure_version,
        "text_sha256": snapshot.text_sha256, "text_length": snapshot.text_length,
    }
    document = replace(document, parse_provenance=provenance, sections=[], structure_nodes=[])
    return StructureEnhancementCandidate(source=snapshot, document=document)


def validate_enhancement_document(document, snapshot):
    """Recheck mutable candidate containers before persistence, without an LLM."""
    text = snapshot.canonical_text
    if (len(text) != snapshot.text_length
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != snapshot.text_sha256):
        raise EnhancementCandidateRejected("invalid_enhancement_snapshot")
    if document.structure_error_code:
        raise EnhancementCandidateRejected(document.structure_error_code)
    provenance = document.parse_provenance
    if (document.document_id != snapshot.document_id or document.raw_text != text
            or document.language != snapshot.language
            or provenance.get("effective_parser_mode") != "llm_enhanced"
            or provenance.get("source") != "validated_v2_plan"
            or provenance.get("fallback_used") is not False
            or provenance.get("source_coverage") != "complete"):
        raise EnhancementCandidateRejected("enhancement_candidate_identity_invalid")
    cursor = 0
    chapter_ids, section_ids = set(), set()
    origins = provenance.get("boundary_origins", [])
    seen_origins = []
    for chapter in document.chapters:
        if chapter.chapter_id in chapter_ids or not chapter.sections or chapter.task_artifacts:
            raise EnhancementCandidateRejected("enhancement_hierarchy_invalid")
        chapter_ids.add(chapter.chapter_id)
        for section in chapter.sections:
            if (section.section_id in section_ids or section.parent_chapter_id != chapter.chapter_id
                    or type(section.char_start) is not int or type(section.char_end) is not int
                    or section.char_start != cursor or not cursor < section.char_end <= len(text)
                    or section.content != text[cursor:section.char_end]
                    or section.task_artifacts or section.task_units):
                raise EnhancementCandidateRejected("enhancement_coverage_invalid")
            section_ids.add(section.section_id)
            matches = [origin for origin in origins if origin.get("char_start") == cursor
                       and origin.get("char_end") == section.char_end]
            if len(matches) != 1 or matches[0].get("title_origin") not in {"source", "generated", "implicit"}:
                raise EnhancementCandidateRejected("enhancement_origin_invalid")
            seen_origins.append(matches[0])
            cursor = section.char_end
    if cursor != len(text) or len(seen_origins) != len(origins) or document.document_task_artifacts:
        raise EnhancementCandidateRejected("enhancement_coverage_invalid")
