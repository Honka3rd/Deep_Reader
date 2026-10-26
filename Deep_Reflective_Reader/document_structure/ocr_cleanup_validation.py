"""Pure deletion validation on an immutable source and an in-memory hierarchy."""

from dataclasses import dataclass
import logging
import unicodedata

from doc_loaders.extraction_identity import ExtractionResult
from document_structure.cleanup_input import CleanupInput
from document_structure.enhancement_change import compare_enhancement_documents
from document_structure.ocr_cleanup_contract import (
    ActualOcrEvidence, BoundCleanupProposal, CleanupBinding, CleanupProposal, cleanup_eligibility,
)
from document_structure.source_mapped_text import (
    PageRange, SourceMappedText, SourceRange, build_source_mapped_text,
)
from document_structure.structured_document import StructuredDocument


LOGGER = logging.getLogger(__name__)


class CleanupValidationError(ValueError):
    """Whole-plan rejection containing only a safe code."""


@dataclass(frozen=True)
class ValidatedDeletion:
    item_index: int
    char_start: int
    char_end: int
    chapter_id: str
    section_id: str
    quote: str
    reason: str
    pages: tuple[PageRange, ...]


@dataclass(frozen=True)
class CleanupItemRejection:
    item_index: int
    reason: str


@dataclass(frozen=True)
class CleanupResolution:
    deletions: tuple[ValidatedDeletion, ...]
    rejections: tuple[CleanupItemRejection, ...]
    proposed_count: int

    # Deliberately not CleanupOutcome: validation is neither application nor commit.
    def diagnostics(self) -> dict[str, object]:
        return {
            "proposed_count": self.proposed_count,
            "validated_count": len(self.deletions),
            "rejected_count": len(self.rejections),
            "reasons": sorted({item.reason for item in self.rejections}),
        }


@dataclass(frozen=True)
class CleanupValidationResult:
    binding: CleanupBinding
    deletions: tuple[ValidatedDeletion, ...]
    rejections: tuple[CleanupItemRejection, ...]
    proposed_count: int

    def diagnostics(self):
        return CleanupResolution(self.deletions, self.rejections, self.proposed_count).diagnostics()


def validate_cleanup_input(proposal: CleanupProposal, *, snapshot: CleanupInput,
                           expected_fingerprint: str) -> CleanupResolution:
    """Shared pure validation for accepted Common and in-memory split snapshots."""
    if expected_fingerprint != snapshot.fingerprint():
        raise CleanupValidationError("cleanup_input_binding_mismatch")
    eligibility = snapshot.eligibility()
    if eligibility.status != "eligible":
        raise CleanupValidationError(eligibility.reason)

    def resolve(item):
        matches = []
        for part in snapshot.parts:
            start, reason = _quote_position(part.text, item)
            if reason == "ambiguous_quote":
                return None, reason
            if start is not None:
                matches.append((part, start))
        if len(matches) != 1:
            if not matches:
                LOGGER.info("ocr_cleanup stage=quote_resolution input=%s quote_chars=%s quote_present=%s context_supplied=%s",
                    expected_fingerprint[:16], len(item.quote),
                    any(item.quote in part.text for part in snapshot.parts),
                    bool(item.prefix_quote or item.suffix_quote))
            return None, "ambiguous_quote" if matches else "quote_not_found"
        part, offset = matches[0]
        if part.manual:
            return None, "user_edited_content"
        start = part.char_start + offset
        # Context must obey the same prompt boundaries as the deletion itself.
        prefix_start = start - len(item.prefix_quote or "")
        suffix_end = start + len(item.quote) + len(item.suffix_quote or "")
        edges = {edge for span in snapshot.protected_ranges
                 for edge in (span.char_start, span.char_end)}
        if snapshot.source.identity.mode == "auto" and snapshot.evidence:
            edges.update(edge for span in snapshot.evidence.spans
                         for edge in (span.char_start, span.char_end))
        if any(prefix_start < edge < suffix_end for edge in edges):
            return None, "cross_context_boundary"
        return start, None

    result = _validate_items(
        proposal=proposal, source=snapshot.source, candidate=snapshot.document(),
        resolve=resolve, page_ranges=snapshot.pages, evidence=snapshot.evidence,
        protected_ranges=tuple(SourceRange(s.char_start, s.char_end) for s in snapshot.protected_ranges),
        prior_deletions=tuple(SourceRange(s.char_start, s.char_end) for s in snapshot.prior_deletions),
        allow_common=True)
    LOGGER.info("ocr_cleanup stage=validation input=%s proposed=%s validated=%s rejected=%s reasons=%s",
                expected_fingerprint[:16], result.proposed_count, len(result.deletions),
                len(result.rejections), ",".join(result.diagnostics()["reasons"]))
    return result


def _overlap(start, end, span):
    return start < span.end and span.start < end


def _checked_ranges(ranges, length):
    for span in ranges:
        if (type(span.start) is not int or type(span.end) is not int
                or not 0 <= span.start < span.end <= length):
            raise CleanupValidationError("invalid_protected_ranges")


def _quote_position(text, item):
    prefix, suffix = item.prefix_quote or "", item.suffix_quote or ""
    needle = prefix + item.quote + suffix
    position = text.find(needle)
    if position < 0:
        return None, "quote_not_found"
    if text.find(needle, position + 1) >= 0:
        return None, "ambiguous_quote"
    return position + len(prefix), None


def _check_pages(text, pages):
    cursor, previous_index = 0, -1
    for page in pages:
        if (type(page.page_index) is not int or page.page_index <= previous_index
                or type(page.start) is not int or type(page.end) is not int
                or not cursor <= page.start <= page.end <= len(text)
                or text[cursor:page.start].strip()):
            raise CleanupValidationError("invalid_page_mapping")
        cursor, previous_index = page.end, page.page_index
    if pages and text[cursor:].strip():
        raise CleanupValidationError("invalid_page_mapping")


def _unsafe_boundary(text, start, end):
    # Never split an alphanumeric token, CRLF, combining sequence or joined glyph.
    for offset in (start, end):
        if not 0 < offset < len(text):
            continue
        left, right = text[offset - 1], text[offset]
        if (left.isalnum() and right.isalnum()
                or left == "\r" and right == "\n"
                or unicodedata.category(right).startswith("M")
                or left == "\u200d" or right == "\u200d"):
            return True
    return False


def validate_cleanup_proposal(
    bound: BoundCleanupProposal,
    *,
    source: ExtractionResult,
    expected_binding: CleanupBinding,
    candidate: StructuredDocument,
    page_ranges: tuple[PageRange, ...] = (),
    user_edit_ranges: tuple[SourceRange, ...] = (),
    protected_ranges: tuple[SourceRange, ...] = (),
    readable_view: SourceMappedText | None = None,
    evidence: ActualOcrEvidence | None = None,
) -> CleanupValidationResult:
    """Resolve exact locations, not semantic truth or permission to persist.

    expected_binding comes from the accepted pre-split snapshot/consent, NOT from
    model output. candidate is the derived source-preserving in-memory hierarchy.
    Caller supplies trusted user-edit/retention ranges and rechecks CAS at commit.
    """
    if not bound.binding.matches_source(source):
        raise CleanupValidationError("cleanup_source_mismatch")
    if bound.binding != expected_binding:
        raise CleanupValidationError("cleanup_binding_mismatch")
    eligibility = cleanup_eligibility(source, evidence)
    if eligibility.status != "eligible":
        raise CleanupValidationError(eligibility.reason)
    text = source.canonical_text
    if candidate.document_id != bound.binding.document_id:
        raise CleanupValidationError("candidate_identity_mismatch")
    try:
        compare_enhancement_documents(candidate, candidate, canonical_text=text)
    except ValueError:
        raise CleanupValidationError("invalid_candidate") from None
    _checked_ranges(user_edit_ranges, len(text))
    _checked_ranges(protected_ranges, len(text))
    _check_pages(text, page_ranges)
    lookup = text
    if bound.binding.source_representation == "readable_full_source_v1":
        if readable_view is None or readable_view.source != text:
            raise CleanupValidationError("invalid_readable_mapping")
        try:
            rebuilt = build_source_mapped_text(
                text, confirmed_fragment_ranges=readable_view.confirmed_fragment_ranges,
                page_ranges=readable_view.page_ranges)
        except ValueError:
            raise CleanupValidationError("invalid_readable_mapping") from None
        if rebuilt != readable_view:
            raise CleanupValidationError("invalid_readable_mapping")
        lookup = readable_view.text

    def resolve(item):
        start, reason = _quote_position(lookup, item)
        if reason:
            return None, reason
        if bound.binding.source_representation == "readable_full_source_v1":
            mapped = readable_view.source_range(start, start + len(item.quote))
            if text[mapped.start:mapped.end] != item.quote:
                return None, "nonexact_canonical_mapping"
            start = mapped.start
        return start, None

    result = _validate_items(
        proposal=bound.proposal, source=source, candidate=candidate, resolve=resolve,
        page_ranges=page_ranges, user_edit_ranges=user_edit_ranges,
        protected_ranges=protected_ranges, evidence=evidence)
    return CleanupValidationResult(bound.binding, result.deletions, result.rejections, result.proposed_count)


def _validate_items(*, proposal, source, candidate, resolve, page_ranges=(),
                    user_edit_ranges=(), protected_ranges=(), evidence=None,
                    prior_deletions=(), allow_common=False):
    text = source.canonical_text
    sections = [(chapter.chapter_id, section) for chapter in candidate.chapters
                for section in chapter.sections]
    origins = candidate.parse_provenance.get("boundary_origins", [])
    headings = []
    if allow_common and "boundary_origins" not in candidate.parse_provenance:
        # Common need not carry LLM boundary evidence. Protect its first nonempty
        # line conservatively; never invent generated-title authority to delete it.
        origins = [{"char_start": section.char_start, "char_end": section.char_end,
                    "title_origin": "source"} for _, section in sections]
    if not isinstance(origins, (list, tuple)) or len(origins) != len(sections):
        raise CleanupValidationError("missing_boundary_evidence")
    for (_, section), origin in zip(sections, origins):
        if (not isinstance(origin, dict) or origin.get("char_start") != section.char_start
                or origin.get("char_end") != section.char_end
                or origin.get("title_origin") not in {"source", "generated", "implicit"}):
            raise CleanupValidationError("invalid_boundary_evidence")
        if origin["title_origin"] == "source":
            # Protect the source heading line, including indentation/line terminator.
            end = section.char_start
            for line in text[section.char_start:section.char_end].splitlines(keepends=True):
                end += len(line)
                if line.strip():
                    break
            headings.append(SourceRange(section.char_start, end))

    accepted, rejected = [], []
    for index, item in enumerate(proposal.items):
        start, reason = resolve(item)
        if reason:
            rejected.append(CleanupItemRejection(index, reason))
            continue
        end = start + len(item.quote)
        owner = next(((chapter_id, section) for chapter_id, section in sections
                      if section.char_start <= start < end <= section.char_end), None)
        if owner is None:
            reason = "cross_section"
        elif any(_overlap(start, end, span) for span in headings):
            reason = "heading_anchor"
        elif any(_overlap(start, end, span) for span in user_edit_ranges):
            reason = "user_edited_content"
        elif any(_overlap(start, end, span) for span in protected_ranges):
            reason = "protected_content"
        elif _unsafe_boundary(text, start, end):
            reason = "unsafe_text_boundary"
        elif source.identity.mode == "auto" and not _ocr_covers(start, end, evidence):
            reason = "outside_ocr_scope"
        if reason:
            rejected.append(CleanupItemRejection(index, reason))
            continue
        accepted.append(ValidatedDeletion(
            index, start, end, owner[0], owner[1].section_id, item.quote, item.reason,
            tuple(PageRange(page.page_index, max(start, page.start), min(end, page.end))
                  for page in page_ranges if _overlap(start, end, page)),
        ))

    # Reject every participant in a nonidentical overlap, regardless of model order.
    overlapping = {item.item_index for item in accepted for other in accepted
                   if (item.char_start, item.char_end) != (other.char_start, other.char_end)
                   and item.char_start < other.char_end and other.char_start < item.char_end}
    unique = {}
    for item in accepted:
        key = (item.char_start, item.char_end)
        if item.item_index in overlapping:
            rejected.append(CleanupItemRejection(item.item_index, "overlapping_deletion"))
        elif key in unique:
            rejected.append(CleanupItemRejection(item.item_index, "duplicate_deletion"))
        else:
            unique[key] = item
    deletions = sorted(unique.values(), key=lambda item: item.char_start)
    active_ranges = [*prior_deletions, *(SourceRange(item.char_start, item.char_end) for item in deletions)]
    nonspace = sum(not char.isspace() for char in text)
    deleted_nonspace = sum(sum(not char.isspace() for char in text[span.start:span.end])
                           for span in active_ranges)
    body_ids = {section.section_id for chapter in candidate.chapters for section in chapter.sections
                if (section.section_role or chapter.chapter_role) == "main_body"}
    body_size = sum(sum(not char.isspace() for char in section.content)
                    for _, section in sections if section.section_id in body_ids)
    deleted_body = sum(sum(not char.isspace() for char in
                          text[max(span.start, section.char_start):min(span.end, section.char_end)])
                       for span in active_ranges for _, section in sections
                       if section.section_id in body_ids and
                       span.start < section.char_end and section.char_start < span.end)
    # Earlier cleanup still counts when checking whether usable text would vanish.
    limit_reason = None
    if deletions and deleted_nonspace >= nonspace:
        limit_reason = "all_content_removal"
    elif body_size and deleted_body >= body_size:
        limit_reason = "all_body_removal"
    if limit_reason:
        rejected.extend(CleanupItemRejection(item.item_index, limit_reason) for item in deletions)
        deletions = []
    return CleanupResolution(
        tuple(deletions), tuple(sorted(rejected, key=lambda item: item.item_index)),
        len(proposal.items),
    )


def _ocr_covers(start, end, evidence):
    cursor = start
    for span in evidence.spans:
        if span.char_start > cursor:
            break
        if span.char_end > cursor:
            cursor = min(end, span.char_end)
        if cursor == end:
            return True
    return False
