"""Atomic candidate selection: never merge common boundaries into a v2 plan."""
from bisect import bisect_left

from document_structure.common_parse_policy import common_analysis_view
from document_structure.heading_acceptance import POLICY_VERSION, heading_decision
from document_structure.section_role import SectionRole
from document_structure.structured_document import StructuredSection
from document_structure.structure_plan_contract import (
    PLAN_VERSION, PROMPT_VERSION, StructurePlanCandidate, resolve_candidate_positions, is_semantic_boundary,
)


def apply_candidate(raw_text: str, candidate: StructurePlanCandidate, preview: dict):
    positions = resolve_candidate_positions(candidate, preview)
    view = common_analysis_view(raw_text)
    accepted = []
    rejected = 0
    for boundary, position in zip(candidate.boundaries, positions):
        if boundary.origin == "source":
            if boundary.role == "toc" and (boundary.title or "").strip().lower() not in {
                    "目录", "目錄", "contents", "table of contents"}:
                rejected += 1
                continue
            analysis_start = bisect_left(view.source_offsets, position)
            if heading_decision(view.text, analysis_start, boundary.title) != "accepted":
                rejected += 1
                continue
        elif boundary.origin == "generated" and position:
            if not is_semantic_boundary(view, position):
                raise ValueError("generated_boundary_inside_prose")
        accepted.append((position, boundary))
    # An entirely unsupported plan is a failed enhancement, not a successful fallback.
    if not accepted:
        raise ValueError("no_supported_plan_boundaries")
    if accepted[0][0] != 0:
        accepted.insert(0, (0, None))
    sections, origins = [], []
    for index, (start, boundary) in enumerate(accepted):
        end = accepted[index + 1][0] if index + 1 < len(accepted) else len(raw_text)
        if start >= end:
            raise ValueError("empty_candidate_interval")
        title = boundary.title if boundary is not None else None
        origin = boundary.origin if boundary is not None else "implicit"
        sections.append(StructuredSection(
            section_id=f"section-{index}", section_index=index, title=title,
            level=boundary.level if boundary is not None else 1,
            content=raw_text[start:end], char_start=start, char_end=end,
            section_role=SectionRole(boundary.role) if boundary is not None else SectionRole.MAIN_BODY,
            is_implicit_section=origin == "implicit"))
        origins.append({"char_start": start, "char_end": end, "title_origin": origin})
    if "".join(section.content for section in sections) != raw_text:
        raise ValueError("candidate_coverage_failed")
    return sections, {
        "requested_parser_mode": "llm_enhanced", "effective_parser_mode": "llm_enhanced",
        "fallback_used": False, "fallback_reason": None, "source": "validated_v2_plan",
        "schema_version": PLAN_VERSION, "prompt_instruction_version": PROMPT_VERSION,
        "heading_policy_version": POLICY_VERSION, "enhancement_intent": candidate.intent,
        "proposed_boundary_count": len(candidate.boundaries),
        "resolved_boundary_count": len(positions), "rejected_boundary_count": rejected,
        "accepted_boundary_count": len(sections), "source_coverage": "complete",
        "boundary_origins": origins,
    }
