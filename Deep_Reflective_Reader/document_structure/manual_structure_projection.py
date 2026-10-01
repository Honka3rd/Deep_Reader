from __future__ import annotations

from dataclasses import dataclass, field


SUPPORTED_MANUAL_ANCHOR_TYPES = frozenset({"char_range", "page_range"})


@dataclass(frozen=True)
class ManualStructureAnchor:
    anchor_type: str
    char_start: int | None = None
    char_end: int | None = None
    page_start_index: int | None = None
    page_end_index: int | None = None


@dataclass(frozen=True)
class ManualStructureEntry:
    title: str
    level: int
    anchor: ManualStructureAnchor
    external_id: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ManualStructureIssue:
    code: str
    message: str
    severity: str = "error"
    entry_index: int | None = None


@dataclass(frozen=True)
class ManualStructureNormalizedEntry:
    title: str
    level: int
    anchor: ManualStructureAnchor
    entry_index: int
    chapter_index: int | None = None
    section_index: int | None = None
    external_id: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ManualStructurePreviewSection:
    section_id: str
    title: str
    source_entry_index: int
    anchor: ManualStructureAnchor


@dataclass(frozen=True)
class ManualStructurePreviewChapter:
    chapter_id: str
    title: str
    source_entry_index: int
    anchor: ManualStructureAnchor
    sections: list[ManualStructurePreviewSection] = field(default_factory=list)


@dataclass(frozen=True)
class ManualStructureProjectionResult:
    valid: bool
    issues: list[ManualStructureIssue]
    normalized_entries: list[ManualStructureNormalizedEntry]
    preview_chapters: list[ManualStructurePreviewChapter]
    source_hash: str | None = None


def project_manual_structure_plan(
    entries: list[ManualStructureEntry],
    *,
    source_hash: str | None = None,
) -> ManualStructureProjectionResult:
    """Validate and preview a user-supplied chapter->section structure plan.

    This is a deterministic planning boundary. It does not read source text,
    create task units, instantiate StructuredDocument, or write persistence.
    """
    issues: list[ManualStructureIssue] = []
    normalized_entries: list[ManualStructureNormalizedEntry] = []
    current_chapter_index: int | None = None
    current_section_index = 0

    for entry_index, entry in enumerate(entries):
        title = entry.title.strip()
        anchor = _normalize_anchor(entry.anchor)
        entry_issues = _validate_entry(entry_index, title, entry.level, anchor)
        issues.extend(entry_issues)

        if entry.level == 2 and current_chapter_index is None:
            issues.append(
                ManualStructureIssue(
                    code="invalid_level_sequence",
                    message="manual section entry requires a preceding chapter",
                    entry_index=entry_index,
                )
            )

        if entry.level == 1:
            current_chapter_index = (
                1 if current_chapter_index is None else current_chapter_index + 1
            )
            current_section_index = 0
        elif entry.level == 2 and current_chapter_index is not None:
            current_section_index += 1

        if not entry_issues and entry.level in (1, 2):
            normalized_entries.append(
                ManualStructureNormalizedEntry(
                    title=title,
                    level=entry.level,
                    anchor=anchor,
                    entry_index=entry_index,
                    chapter_index=current_chapter_index,
                    section_index=(
                        current_section_index if entry.level == 2 else None
                    ),
                    external_id=_normalize_optional_text(entry.external_id),
                    notes=_normalize_optional_text(entry.notes),
                )
            )

    issues.extend(_find_overlapping_ranges(normalized_entries))

    has_error = any(issue.severity == "error" for issue in issues)
    preview_chapters = [] if has_error else _build_preview(normalized_entries)
    return ManualStructureProjectionResult(
        valid=not has_error,
        issues=issues,
        normalized_entries=[] if has_error else normalized_entries,
        preview_chapters=preview_chapters,
        source_hash=_normalize_optional_text(source_hash),
    )


def _normalize_anchor(anchor: ManualStructureAnchor) -> ManualStructureAnchor:
    return ManualStructureAnchor(
        anchor_type=anchor.anchor_type.strip().replace("-", "_"),
        char_start=anchor.char_start,
        char_end=anchor.char_end,
        page_start_index=anchor.page_start_index,
        page_end_index=anchor.page_end_index,
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_entry(
    entry_index: int,
    title: str,
    level: int,
    anchor: ManualStructureAnchor,
) -> list[ManualStructureIssue]:
    issues: list[ManualStructureIssue] = []
    if not title:
        issues.append(
            ManualStructureIssue(
                code="malformed_payload",
                message="manual structure title cannot be empty",
                entry_index=entry_index,
            )
        )
    if level < 1:
        issues.append(
            ManualStructureIssue(
                code="invalid_level_sequence",
                message="manual structure level must be at least 1",
                entry_index=entry_index,
            )
        )
    if level > 2:
        issues.append(
            ManualStructureIssue(
                code="unsupported_depth",
                message="manual structure supports chapter->section depth only",
                entry_index=entry_index,
            )
        )
    issues.extend(_validate_anchor(entry_index, anchor))
    return issues


def _validate_anchor(
    entry_index: int,
    anchor: ManualStructureAnchor,
) -> list[ManualStructureIssue]:
    issues: list[ManualStructureIssue] = []
    if anchor.anchor_type not in SUPPORTED_MANUAL_ANCHOR_TYPES:
        return [
            ManualStructureIssue(
                code="unsupported_anchor_type",
                message=f"unsupported manual anchor type: {anchor.anchor_type}",
                entry_index=entry_index,
            )
        ]

    has_char = anchor.char_start is not None or anchor.char_end is not None
    has_page = anchor.page_start_index is not None or anchor.page_end_index is not None
    if has_char and has_page:
        issues.append(
            ManualStructureIssue(
                code="malformed_payload",
                message="manual structure anchor must not mix char and page fields",
                entry_index=entry_index,
            )
        )

    if anchor.anchor_type == "char_range":
        if anchor.char_start is None or has_page:
            issues.append(
                ManualStructureIssue(
                    code="malformed_payload",
                    message="char_range anchor requires char_start and no page fields",
                    entry_index=entry_index,
                )
            )
        if anchor.char_start is not None and anchor.char_start < 0:
            issues.append(
                ManualStructureIssue(
                    code="out_of_range_anchor",
                    message="char_start must be non-negative",
                    entry_index=entry_index,
                )
            )
        if (
            anchor.char_start is not None
            and anchor.char_end is not None
            and anchor.char_end <= anchor.char_start
        ):
            issues.append(
                ManualStructureIssue(
                    code="empty_projected_range",
                    message="char_end must be greater than char_start",
                    entry_index=entry_index,
                )
            )

    if anchor.anchor_type == "page_range":
        if anchor.page_start_index is None or has_char:
            issues.append(
                ManualStructureIssue(
                    code="malformed_payload",
                    message="page_range anchor requires page_start_index and no char fields",
                    entry_index=entry_index,
                )
            )
        if anchor.page_start_index is not None and anchor.page_start_index < 0:
            issues.append(
                ManualStructureIssue(
                    code="out_of_range_anchor",
                    message="page_start_index must be non-negative",
                    entry_index=entry_index,
                )
            )
        if (
            anchor.page_start_index is not None
            and anchor.page_end_index is not None
            and anchor.page_end_index < anchor.page_start_index
        ):
            issues.append(
                ManualStructureIssue(
                    code="empty_projected_range",
                    message="page_end_index must be greater than or equal to page_start_index",
                    entry_index=entry_index,
                )
            )
    return issues


def _find_overlapping_ranges(
    entries: list[ManualStructureNormalizedEntry],
) -> list[ManualStructureIssue]:
    issues: list[ManualStructureIssue] = []
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            if left.level != right.level:
                continue
            if left.level == 2 and left.chapter_index != right.chapter_index:
                continue
            if _ranges_overlap(left.anchor, right.anchor):
                issues.append(
                    ManualStructureIssue(
                        code="overlapping_range",
                        message="manual structure sibling anchors must not overlap",
                        entry_index=right.entry_index,
                    )
                )
    return issues


def _ranges_overlap(
    left: ManualStructureAnchor,
    right: ManualStructureAnchor,
) -> bool:
    if left.anchor_type != right.anchor_type:
        return False
    if left.anchor_type == "char_range":
        if (
            left.char_start is None
            or left.char_end is None
            or right.char_start is None
            or right.char_end is None
        ):
            return False
        return max(left.char_start, right.char_start) < min(left.char_end, right.char_end)
    if left.anchor_type == "page_range":
        if left.page_start_index is None or right.page_start_index is None:
            return False
        left_end = left.page_end_index if left.page_end_index is not None else left.page_start_index
        right_end = (
            right.page_end_index
            if right.page_end_index is not None
            else right.page_start_index
        )
        return max(left.page_start_index, right.page_start_index) <= min(left_end, right_end)
    return False


def _build_preview(
    entries: list[ManualStructureNormalizedEntry],
) -> list[ManualStructurePreviewChapter]:
    chapters: list[ManualStructurePreviewChapter] = []
    current_chapter: ManualStructurePreviewChapter | None = None
    current_chapter_section_count = 0

    for entry in entries:
        if entry.level == 1:
            if current_chapter is not None and not current_chapter.sections:
                current_chapter.sections.append(
                    ManualStructurePreviewSection(
                        section_id=f"{current_chapter.chapter_id}_section_1",
                        title=current_chapter.title,
                        source_entry_index=current_chapter.source_entry_index,
                        anchor=current_chapter.anchor,
                    )
                )
            current_chapter_section_count = 0
            current_chapter = ManualStructurePreviewChapter(
                chapter_id=f"manual_chapter_{len(chapters) + 1}",
                title=entry.title,
                source_entry_index=entry.entry_index,
                anchor=entry.anchor,
            )
            chapters.append(current_chapter)
            continue

        if entry.level == 2 and current_chapter is not None:
            current_chapter_section_count += 1
            current_chapter.sections.append(
                ManualStructurePreviewSection(
                    section_id=(
                        f"{current_chapter.chapter_id}_section_"
                        f"{current_chapter_section_count}"
                    ),
                    title=entry.title,
                    source_entry_index=entry.entry_index,
                    anchor=entry.anchor,
                )
            )

    if current_chapter is not None and not current_chapter.sections:
        current_chapter.sections.append(
            ManualStructurePreviewSection(
                section_id=f"{current_chapter.chapter_id}_section_1",
                title=current_chapter.title,
                source_entry_index=current_chapter.source_entry_index,
                anchor=current_chapter.anchor,
            )
        )
    return chapters


__all__ = [
    "ManualStructureAnchor",
    "ManualStructureEntry",
    "ManualStructureIssue",
    "ManualStructureNormalizedEntry",
    "ManualStructurePreviewChapter",
    "ManualStructurePreviewSection",
    "ManualStructureProjectionResult",
    "SUPPORTED_MANUAL_ANCHOR_TYPES",
    "project_manual_structure_plan",
]
