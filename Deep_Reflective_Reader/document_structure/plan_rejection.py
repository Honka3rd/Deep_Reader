"""Bounded diagnostics for untrusted model plans; never expose field values."""
import json
import re
from pydantic import ValidationError

SCHEMA_CODES = frozenset({"generated_navigation_is_not_source_region",
    "single_body_requires_implicit_body", "implicit_title_must_be_null",
    "implicit_region_requires_source_evidence", "explicit_title_required", "blank_anchor",
    "integer_level_required", "exactly_one_boundary_reference_required", "anchor_requires_segment_id"})
REFERENCE_CODES = frozenset({"unknown_source_segment", "anchor_occurrence_unresolved",
    "duplicate_boundary_position", "unknown_boundary_reference"})
APPLICATION_CODES = frozenset({"plan_response_over_budget", "plan_intent_mismatch",
    "single_body_requires_full_source_visibility", "single_body_requires_source_start",
    "source_title_not_at_boundary", "generated_boundary_inside_prose",
    "no_supported_plan_boundaries", "empty_candidate_interval", "candidate_coverage_failed"})
FIELDS = frozenset({"boundaries", "boundary_id", "segment_id", "anchor", "occurrence",
    "title", "origin", "level", "role", "schema_version", "intent", "outcome", "start_quote", "preceding_quote"})


class EnhancementPlanRejected(ValueError):
    pass


class BoundaryReferenceRejected(ValueError):
    """Preserve the public reason while carrying source-free validation evidence."""
    def __init__(self, *, boundary_index, reference_count, boundary_count,
                 reference_found, occurrence, locator_form):
        super().__init__("unknown_boundary_reference")
        self.diagnostic = {
            "stage": "reference", "reason": "unknown_boundary_reference",
            "issues": [{"path": ["boundaries", boundary_index, "boundary_id"],
                "code": "reference_not_in_map" if not reference_found else "reference_occurrence_not_one",
                "reference_found": reference_found, "occurrence_is_one": occurrence == 1,
                "locator_form": locator_form}],
            "reference_count": reference_count, "boundary_count": boundary_count,
        }


def rejection_diagnostic(error, response):
    from document_structure.semantic_structure_plan import QuoteResolutionError
    if isinstance(error, QuoteResolutionError):
        return error.diagnostic
    if isinstance(error, BoundaryReferenceRejected):
        return error.diagnostic
    if not isinstance(error, ValidationError):
        reason = str(error)
        if reason not in REFERENCE_CODES | APPLICATION_CODES:
            reason = "unexpected_validation_error"
        stage = "reference" if reason in REFERENCE_CODES else "application"
        return {"stage": stage, "reason": reason, "issues": []}
    errors = error.errors(include_input=False, include_url=False)
    issues = []
    payload = None
    try:
        text = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        payload = json.loads(fenced.group(1) if fenced else text)
    except (ValueError, TypeError):
        pass
    for item in errors[:16]:
        custom = str(item.get("ctx", {}).get("error"))
        code = custom if custom in SCHEMA_CODES else item["type"]
        path = [part if type(part) is int or part in FIELDS else "unknown_field"
                for part in item["loc"][:8]]
        issue = {"code": code, "path": path}
        if (len(path) >= 2 and path[0] == "boundaries" and type(path[1]) is int
                and isinstance(payload, dict) and isinstance(payload.get("boundaries"), list)
                and 0 <= path[1] < len(payload["boundaries"])):
            boundary = payload["boundaries"][path[1]]
            if isinstance(boundary, dict):
                issue["locator_present"] = {key: boundary.get(key) is not None
                                            for key in ("boundary_id", "anchor", "segment_id")}
        issues.append(issue)
    reason = "schema:" + ",".join(sorted({issue["code"] for issue in issues}))
    return {"stage": "json" if any(e["type"] == "json_invalid" for e in errors) else "schema",
            "reason": reason, "issues": issues, "issue_count": len(errors)}
