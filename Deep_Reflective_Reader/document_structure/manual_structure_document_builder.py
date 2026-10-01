from __future__ import annotations

from dataclasses import dataclass, field

from doc_loaders.pdf_page_evidence import PdfPageTextBoundary
from document_structure.manual_structure_projection import (
    ManualStructureAnchor,
    ManualStructureEntry,
    ManualStructureIssue,
    ManualStructurePreviewChapter,
    ManualStructurePreviewSection,
    project_manual_structure_plan,
)
from document_structure.section_role import SectionRole
from document_structure.structured_document import (
    StructuredChapter,
    StructuredDocument,
    StructuredSection,
)


@dataclass(frozen=True)
class ManualStructureDocumentDraftResult:
    valid: bool
    issues: list[ManualStructureIssue] = field(default_factory=list)
    document: StructuredDocument | None = None


def build_manual_structure_document_draft(
    *,
    document_id: str,
    title: str,
    raw_text: str,
    entries: list[ManualStructureEntry],
    source_path: str | None = None,
    language: str | None = None,
    source_hash: str | None = None,
    page_boundaries: list[PdfPageTextBoundary] | None = None,
) -> ManualStructureDocumentDraftResult:
    """Build an in-memory StructuredDocument from a validated manual plan.

    This function is deliberately side-effect free: it does not save files,
    create task units, mutate task-layout state, or fallback to another parser.
    """
    projection = project_manual_structure_plan(entries, source_hash=source_hash)
    if not projection.valid:
        return ManualStructureDocumentDraftResult(
            valid=False,
            issues=list(projection.issues),
        )
    if not projection.preview_chapters:
        return ManualStructureDocumentDraftResult(
            valid=False,
            issues=[
                ManualStructureIssue(
                    code="malformed_payload",
                    message="manual structure requires at least one chapter",
                )
            ],
        )

    issues = _validate_char_backed_preview(
        raw_text=raw_text,
        chapters=projection.preview_chapters,
        page_boundaries=page_boundaries or [],
    )
    if issues:
        return ManualStructureDocumentDraftResult(valid=False, issues=issues)

    chapters = _build_chapters(
        raw_text=raw_text,
        preview_chapters=projection.preview_chapters,
        page_boundaries=page_boundaries or [],
    )
    document = StructuredDocument(
        document_id=document_id,
        title=title,
        source_path=source_path,
        language=language,
        raw_text=raw_text,
        sections=[],
        chapters=chapters,
        structure_nodes=[],
        structure_error_code=None,
        structure_error_message=None,
        parse_provenance={
            "requested_parser_mode": "manual_structure",
            "effective_parser_mode": "manual_structure_projection",
            "fallback_used": False,
            "fallback_reason": None,
            "source": "user_supplied_structure",
            "manual_structure": {
                "entry_count": len(entries),
                "chapter_count": len(chapters),
                "section_count": sum(len(chapter.sections) for chapter in chapters),
                "source_hash": projection.source_hash,
                "anchor_type": _manual_structure_anchor_type(
                    projection.preview_chapters
                ),
                "validation_summary": {
                    "valid": True,
                    "issue_count": 0,
                },
            },
        },
    )
    return ManualStructureDocumentDraftResult(valid=True, document=document)


def _validate_char_backed_preview(
    *,
    raw_text: str,
    chapters: list[ManualStructurePreviewChapter],
    page_boundaries: list[PdfPageTextBoundary],
) -> list[ManualStructureIssue]:
    issues: list[ManualStructureIssue] = []
    issues.extend(
        _validate_page_boundary_evidence(
            chapters=chapters,
            page_boundaries=page_boundaries,
        )
    )
    if issues:
        return issues
    for chapter_index, chapter in enumerate(chapters):
        chapter_start = _resolve_anchor_start(
            anchor=chapter.anchor,
            page_boundaries=page_boundaries,
        )
        next_chapter_start = _next_chapter_start(
            chapters=chapters,
            chapter_index=chapter_index,
            page_boundaries=page_boundaries,
        )
        chapter_end = _resolve_entry_end(
            explicit_end=_resolve_anchor_explicit_end(
                anchor=chapter.anchor,
                page_boundaries=page_boundaries,
            ),
            next_start=next_chapter_start,
            default_end=len(raw_text),
        )
        issues.extend(
            _validate_anchor_range(
                anchor=chapter.anchor,
                page_boundaries=page_boundaries,
                raw_text=raw_text,
                start=chapter_start,
                end=chapter_end,
                entry_index=chapter.source_entry_index,
            )
        )
        if chapter_start is None or chapter_end is None:
            continue

        for section_index, section in enumerate(chapter.sections):
            section_start = _resolve_anchor_start(
                anchor=section.anchor,
                page_boundaries=page_boundaries,
            )
            next_section_start = _next_section_start(
                sections=chapter.sections,
                section_index=section_index,
                page_boundaries=page_boundaries,
            )
            section_end = _resolve_entry_end(
                explicit_end=_resolve_anchor_explicit_end(
                    anchor=section.anchor,
                    page_boundaries=page_boundaries,
                ),
                next_start=next_section_start,
                default_end=chapter_end,
            )
            issues.extend(
                _validate_anchor_range(
                    anchor=section.anchor,
                    page_boundaries=page_boundaries,
                    raw_text=raw_text,
                    start=section_start,
                    end=section_end,
                    entry_index=section.source_entry_index,
                )
            )
            if section_start is None or section_end is None:
                continue
            if section_start < chapter_start or section_end > chapter_end:
                issues.append(
                    _issue(
                        code="out_of_range_anchor",
                        message="manual section anchor must be inside its chapter range",
                        entry_index=section.source_entry_index,
                    )
                )
    return issues


def _validate_page_boundary_evidence(
    *,
    chapters: list[ManualStructurePreviewChapter],
    page_boundaries: list[PdfPageTextBoundary],
) -> list[ManualStructureIssue]:
    first_page_range_entry_index = _first_page_range_entry_index(chapters)
    if first_page_range_entry_index is None:
        return []

    seen_page_indexes: set[int] = set()
    duplicate_page_indexes: set[int] = set()
    for boundary in page_boundaries:
        if boundary.page_index in seen_page_indexes:
            duplicate_page_indexes.add(boundary.page_index)
        seen_page_indexes.add(boundary.page_index)
    if not duplicate_page_indexes:
        return []

    duplicates = ", ".join(str(page_index) for page_index in sorted(duplicate_page_indexes))
    return [
        _issue(
            code="ambiguous_page_boundary",
            message=(
                "manual page_range anchors require unique page boundary evidence; "
                f"duplicate page_index values: {duplicates}"
            ),
            entry_index=first_page_range_entry_index,
        )
    ]


def _first_page_range_entry_index(
    chapters: list[ManualStructurePreviewChapter],
) -> int | None:
    for chapter in chapters:
        if chapter.anchor.anchor_type == "page_range":
            return chapter.source_entry_index
        for section in chapter.sections:
            if section.anchor.anchor_type == "page_range":
                return section.source_entry_index
    return None


def _build_chapters(
    *,
    raw_text: str,
    preview_chapters: list[ManualStructurePreviewChapter],
    page_boundaries: list[PdfPageTextBoundary],
) -> list[StructuredChapter]:
    chapters: list[StructuredChapter] = []
    section_index = 0
    for chapter_index, chapter in enumerate(preview_chapters):
        chapter_end = _resolve_entry_end(
            explicit_end=_resolve_anchor_explicit_end(
                anchor=chapter.anchor,
                page_boundaries=page_boundaries,
            ),
            next_start=_next_chapter_start(
                chapters=preview_chapters,
                chapter_index=chapter_index,
                page_boundaries=page_boundaries,
            ),
            default_end=len(raw_text),
        )
        sections: list[StructuredSection] = []
        for local_section_index, section in enumerate(chapter.sections):
            section_start = _resolve_anchor_start(
                anchor=section.anchor,
                page_boundaries=page_boundaries,
            )
            section_end = _resolve_entry_end(
                explicit_end=_resolve_anchor_explicit_end(
                    anchor=section.anchor,
                    page_boundaries=page_boundaries,
                ),
                next_start=_next_section_start(
                    sections=chapter.sections,
                    section_index=local_section_index,
                    page_boundaries=page_boundaries,
                ),
                default_end=chapter_end,
            )
            if section_start is None or section_end is None:
                continue
            sections.append(
                StructuredSection(
                    section_id=section.section_id,
                    section_index=section_index,
                    title=section.title,
                    level=2,
                    content=raw_text[section_start:section_end],
                    char_start=section_start,
                    char_end=section_end,
                    container_title=chapter.title,
                    section_role=SectionRole.MAIN_BODY,
                    parent_chapter_id=chapter.chapter_id,
                    section_kind="manual_structure_section",
                    is_implicit_section=(
                        section.source_entry_index == chapter.source_entry_index
                    ),
                )
            )
            section_index += 1
        chapters.append(
            StructuredChapter(
                chapter_id=chapter.chapter_id,
                title=chapter.title,
                level=1,
                chapter_role=None,
                sections=sections,
                metadata={
                    "source": "manual_structure",
                    "source_entry_index": chapter.source_entry_index,
                    "anchor_type": chapter.anchor.anchor_type,
                },
            )
        )
    return chapters


def _resolve_entry_end(
    *,
    explicit_end: int | None,
    next_start: int | None,
    default_end: int,
) -> int | None:
    if explicit_end is not None:
        return explicit_end
    if next_start is not None:
        return next_start
    return default_end


def _next_chapter_start(
    *,
    chapters: list[ManualStructurePreviewChapter],
    chapter_index: int,
    page_boundaries: list[PdfPageTextBoundary],
) -> int | None:
    if chapter_index + 1 >= len(chapters):
        return None
    return _resolve_anchor_start(
        anchor=chapters[chapter_index + 1].anchor,
        page_boundaries=page_boundaries,
    )


def _next_section_start(
    *,
    sections: list[ManualStructurePreviewSection],
    section_index: int,
    page_boundaries: list[PdfPageTextBoundary],
) -> int | None:
    if section_index + 1 >= len(sections):
        return None
    return _resolve_anchor_start(
        anchor=sections[section_index + 1].anchor,
        page_boundaries=page_boundaries,
    )


def _validate_anchor_range(
    *,
    anchor: ManualStructureAnchor,
    page_boundaries: list[PdfPageTextBoundary],
    raw_text: str,
    start: int | None,
    end: int | None,
    entry_index: int,
) -> list[ManualStructureIssue]:
    if anchor.anchor_type == "page_range" and not page_boundaries:
        return [
            _issue(
                code="page_boundaries_required",
                message="manual page_range anchors require page boundary evidence",
                entry_index=entry_index,
            )
        ]
    if start is None or end is None:
        return [
            _issue(
                code="malformed_payload",
                message="manual anchors require resolvable start and end offsets",
                entry_index=entry_index,
            )
        ]
    if start < 0 or end > len(raw_text):
        return [
            _issue(
                code="out_of_range_anchor",
                message="manual char_range anchor must fit inside raw text",
                entry_index=entry_index,
            )
        ]
    if end <= start:
        return [
            _issue(
                code="empty_projected_range",
                message="manual char_range anchor resolves to an empty range",
                entry_index=entry_index,
            )
        ]
    return []


def _resolve_anchor_start(
    *,
    anchor: ManualStructureAnchor,
    page_boundaries: list[PdfPageTextBoundary],
) -> int | None:
    if anchor.anchor_type == "char_range":
        return anchor.char_start
    if anchor.anchor_type != "page_range" or anchor.page_start_index is None:
        return None
    boundary = _page_boundary_by_index(
        page_boundaries=page_boundaries,
        page_index=anchor.page_start_index,
    )
    return None if boundary is None else boundary.char_start


def _resolve_anchor_explicit_end(
    *,
    anchor: ManualStructureAnchor,
    page_boundaries: list[PdfPageTextBoundary],
) -> int | None:
    if anchor.anchor_type == "char_range":
        return anchor.char_end
    if anchor.anchor_type != "page_range" or anchor.page_end_index is None:
        return None
    boundary = _page_boundary_by_index(
        page_boundaries=page_boundaries,
        page_index=anchor.page_end_index,
    )
    return None if boundary is None else boundary.char_end


def _page_boundary_by_index(
    *,
    page_boundaries: list[PdfPageTextBoundary],
    page_index: int,
) -> PdfPageTextBoundary | None:
    for boundary in page_boundaries:
        if boundary.page_index == page_index:
            return boundary
    return None


def _manual_structure_anchor_type(
    chapters: list[ManualStructurePreviewChapter],
) -> str:
    anchor_types = {chapter.anchor.anchor_type for chapter in chapters}
    for chapter in chapters:
        anchor_types.update(section.anchor.anchor_type for section in chapter.sections)
    if len(anchor_types) == 1:
        return next(iter(anchor_types))
    return "mixed"


def _issue(*, code: str, message: str, entry_index: int) -> ManualStructureIssue:
    return ManualStructureIssue(
        code=code,
        message=message,
        entry_index=entry_index,
    )


__all__ = [
    "ManualStructureDocumentDraftResult",
    "build_manual_structure_document_draft",
]
