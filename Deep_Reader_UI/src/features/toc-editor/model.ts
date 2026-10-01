import type {
  AnchorEvidence,
  DocumentTaskLayout,
  ManualStructureAnchorType,
  ManualStructureEntryRequest,
  ManualStructurePlanRequest,
  ManualStructureValidationIssue,
} from "../../types/api";

export type TocItemKind = "chapter" | "section";

export interface EditableTocRange {
  anchorType: ManualStructureAnchorType;
  charStart: string;
  charEnd: string;
  pageStart: string;
  pageEnd: string;
}

export interface EditableTocSection {
  id: string;
  title: string;
  range: EditableTocRange;
}

export interface EditableTocChapter {
  id: string;
  title: string;
  range: EditableTocRange;
  sections: EditableTocSection[];
}

export interface SelectedTocItem {
  kind: TocItemKind;
  chapterId: string;
  sectionId?: string;
}

export interface TocValidationResult {
  valid: boolean;
  issues: ManualStructureValidationIssue[];
}

export function buildEditableTocFromScratch(): EditableTocChapter[] {
  return [];
}

export function layoutHasPageAnchorEvidence(layout: DocumentTaskLayout | null): boolean {
  return Boolean(
    layout?.chapters?.some(
      (chapter) =>
        hasAvailablePageEvidence(chapter.anchor_evidence) ||
        chapter.sections?.some((section) => hasAvailablePageEvidence(section.anchor_evidence)),
    ),
  );
}

function emptyRange(anchorType: ManualStructureAnchorType): EditableTocRange {
  return { anchorType, charStart: "", charEnd: "", pageStart: "", pageEnd: "" };
}

function hasAvailablePageEvidence(evidence: AnchorEvidence | null | undefined): boolean {
  return (
    evidence?.status === "available" &&
    evidence.anchor_type === "page_range" &&
    typeof evidence.page_start_index === "number"
  );
}

function hasAvailableCharEvidence(evidence: AnchorEvidence | null | undefined): boolean {
  return (
    evidence?.status === "available" &&
    evidence.anchor_type === "char_range" &&
    typeof evidence.char_start === "number" &&
    typeof evidence.char_end === "number" &&
    evidence.char_end > evidence.char_start
  );
}

function rangeFromAnchorEvidence(
  evidence: AnchorEvidence | null | undefined,
  fallbackAnchorType: ManualStructureAnchorType,
): EditableTocRange {
  if (hasAvailablePageEvidence(evidence)) {
    const pageEvidence = evidence!;
    const pageStart = pageEvidence.page_start_index! + 1;
    const pageEnd = (pageEvidence.page_end_index ?? pageEvidence.page_start_index)! + 1;
    return {
      anchorType: "page_range",
      charStart: "",
      charEnd: "",
      pageStart: String(pageStart),
      pageEnd: pageEnd === pageStart ? "" : String(pageEnd),
    };
  }
  if (hasAvailableCharEvidence(evidence)) {
    const charEvidence = evidence!;
    return {
      anchorType: "char_range",
      charStart: String(charEvidence.char_start),
      charEnd: String(charEvidence.char_end),
      pageStart: "",
      pageEnd: "",
    };
  }
  return emptyRange(fallbackAnchorType);
}

export function buildEditableTocFromLayout(
  layout: DocumentTaskLayout,
  anchorType: ManualStructureAnchorType = "char_range",
): EditableTocChapter[] {
  return (layout.chapters || []).map((chapter, chapterIndex) => {
    const sections = chapter.sections || [];
    const chapterOnly = sections.length === 1;
    const displayTitle = chapterOnly
      ? sections[0]?.title?.trim() || chapter.title?.trim() || `Chapter ${chapterIndex + 1}`
      : chapter.title?.trim() || `Chapter ${chapterIndex + 1}`;

    return {
      id: chapter.chapter_id || `chapter-${chapterIndex}`,
      title: displayTitle,
      range: rangeFromAnchorEvidence(
        chapter.anchor_evidence || (chapterOnly ? sections[0]?.anchor_evidence : null),
        anchorType,
      ),
      sections: chapterOnly
        ? []
        : sections.map((section, sectionIndex) => ({
            id: section.section_id || `chapter-${chapterIndex}-section-${sectionIndex}`,
            title: section.title?.trim() || `Section ${sectionIndex + 1}`,
            range: rangeFromAnchorEvidence(section.anchor_evidence, anchorType),
          })),
    };
  });
}

export function createChapter(
  index: number,
  anchorType: ManualStructureAnchorType = "char_range",
): EditableTocChapter {
  return {
    id: `new-chapter-${Date.now()}-${index}`,
    title: `Chapter ${index + 1}`,
    range: emptyRange(anchorType),
    sections: [],
  };
}

export function createSection(
  index: number,
  anchorType: ManualStructureAnchorType = "char_range",
): EditableTocSection {
  return {
    id: `new-section-${Date.now()}-${index}`,
    title: `Section ${index + 1}`,
    range: emptyRange(anchorType),
  };
}

function parseRange(range: EditableTocRange): { start: number; end: number } | null {
  if (!range.charStart.trim() || !range.charEnd.trim()) {
    return null;
  }
  const start = Number(range.charStart);
  const end = Number(range.charEnd);
  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    return null;
  }
  return { start, end };
}

function parsePageRange(range: EditableTocRange): { start: number; end: number } | null {
  if (!range.pageStart.trim()) {
    return null;
  }
  const start = Number(range.pageStart);
  const end = range.pageEnd.trim() ? Number(range.pageEnd) : start;
  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    return null;
  }
  return { start, end };
}

function pushIssue(
  issues: ManualStructureValidationIssue[],
  message: string,
  externalId?: string,
) {
  issues.push({
    code: "malformed_payload",
    message,
    severity: "error",
    external_id: externalId,
  });
}

export function validateEditableToc(chapters: EditableTocChapter[]): TocValidationResult {
  const issues: ManualStructureValidationIssue[] = [];
  const chapterCharRanges: Array<{ start: number; end: number; id: string }> = [];
  const chapterPageRanges: Array<{ start: number; end: number; id: string }> = [];

  if (chapters.length === 0) {
    pushIssue(issues, "At least one chapter is required.");
  }

  chapters.forEach((chapter) => {
    if (!chapter.title.trim()) {
      pushIssue(issues, "Chapter title is required.", chapter.id);
    }
    validateItemRange(
      chapter.id,
      "Chapter",
      chapter.range,
      issues,
      chapter.range.anchorType === "page_range" ? chapterPageRanges : chapterCharRanges,
    );
    const sectionCharRanges: Array<{ start: number; end: number; id: string }> = [];
    const sectionPageRanges: Array<{ start: number; end: number; id: string }> = [];
    chapter.sections.forEach((section) => {
      if (!section.title.trim()) {
        pushIssue(issues, "Section title is required.", section.id);
      }
      validateItemRange(
        section.id,
        "Section",
        section.range,
        issues,
        section.range.anchorType === "page_range" ? sectionPageRanges : sectionCharRanges,
      );
    });
    validateNonOverlappingRanges(sectionCharRanges, issues);
    validateNonOverlappingPageRanges(sectionPageRanges, issues);
  });

  validateNonOverlappingRanges(chapterCharRanges, issues);
  validateNonOverlappingPageRanges(chapterPageRanges, issues);

  return { valid: issues.length === 0, issues };
}

function validateNonOverlappingRanges(
  ranges: Array<{ start: number; end: number; id: string }>,
  issues: ManualStructureValidationIssue[],
) {
  const orderedRanges = [...ranges].sort((left, right) => left.start - right.start);
  for (let index = 1; index < orderedRanges.length; index += 1) {
    const previous = orderedRanges[index - 1];
    const current = orderedRanges[index];
    if (current.start < previous.end) {
      pushIssue(issues, "Character ranges must not overlap.", current.id);
    }
  }
}

function validateNonOverlappingPageRanges(
  ranges: Array<{ start: number; end: number; id: string }>,
  issues: ManualStructureValidationIssue[],
) {
  const orderedRanges = [...ranges].sort((left, right) => left.start - right.start);
  for (let index = 1; index < orderedRanges.length; index += 1) {
    const previous = orderedRanges[index - 1];
    const current = orderedRanges[index];
    if (current.start <= previous.end) {
      pushIssue(issues, "Page ranges must not overlap.", current.id);
    }
  }
}

function validateItemRange(
  id: string,
  label: string,
  range: EditableTocRange,
  issues: ManualStructureValidationIssue[],
  seenRanges: Array<{ start: number; end: number; id: string }>,
) {
  if (range.anchorType === "page_range") {
    const parsedRange = parsePageRange(range);
    if (!parsedRange) {
      pushIssue(issues, `${label} range requires an integer start page.`, id);
      return;
    }
    if (parsedRange.start < 1 || parsedRange.end < parsedRange.start) {
      pushIssue(issues, `${label} page end must be greater than or equal to page start.`, id);
      return;
    }
    seenRanges.push({ ...parsedRange, id });
    return;
  }

  const parsedRange = parseRange(range);
  if (!parsedRange) {
    pushIssue(issues, `${label} range requires integer char_start and char_end.`, id);
    return;
  }
  if (parsedRange.start < 0 || parsedRange.end <= parsedRange.start) {
    pushIssue(issues, `${label} char_end must be greater than char_start.`, id);
    return;
  }
  seenRanges.push({ ...parsedRange, id });
}

export function buildManualStructurePlan(
  chapters: EditableTocChapter[],
): ManualStructurePlanRequest {
  const entries: ManualStructureEntryRequest[] = [];

  chapters.forEach((chapter) => {
    const chapterRange = parseRange(chapter.range);
    const chapterPageRange = parsePageRange(chapter.range);
    if (chapter.range.anchorType === "char_range" && !chapterRange) {
      return;
    }
    if (chapter.range.anchorType === "page_range" && !chapterPageRange) {
      return;
    }
    entries.push({
      title: chapter.title.trim(),
      level: 1,
      external_id: chapter.id,
      anchor:
        chapter.range.anchorType === "page_range"
          ? {
              anchor_type: "page_range",
              page_start_index: chapterPageRange!.start - 1,
              page_end_index: chapterPageRange!.end - 1,
            }
          : {
              anchor_type: "char_range",
              char_start: chapterRange!.start,
              char_end: chapterRange!.end,
            },
    });

    chapter.sections.forEach((section) => {
      const sectionRange = parseRange(section.range);
      const sectionPageRange = parsePageRange(section.range);
      if (section.range.anchorType === "char_range" && !sectionRange) {
        return;
      }
      if (section.range.anchorType === "page_range" && !sectionPageRange) {
        return;
      }
      entries.push({
        title: section.title.trim(),
        level: 2,
        external_id: section.id,
        anchor:
          section.range.anchorType === "page_range"
            ? {
                anchor_type: "page_range",
                page_start_index: sectionPageRange!.start - 1,
                page_end_index: sectionPageRange!.end - 1,
              }
            : {
                anchor_type: "char_range",
                char_start: sectionRange!.start,
                char_end: sectionRange!.end,
              },
      });
    });
  });

  return { entries };
}

export function getSelectedRange(
  chapters: EditableTocChapter[],
  selected: SelectedTocItem | null,
): EditableTocRange | null {
  if (!selected) {
    return null;
  }
  const chapter = chapters.find((item) => item.id === selected.chapterId);
  if (!chapter) {
    return null;
  }
  if (selected.kind === "chapter") {
    return chapter.range;
  }
  return chapter.sections.find((section) => section.id === selected.sectionId)?.range || null;
}
