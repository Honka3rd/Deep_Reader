import type {
  ChapterLayout,
  DocumentTaskLayout,
  SectionLayout,
  StructureParserMode,
} from "../../types/api";

export function countTaskUnits(layout: DocumentTaskLayout | null): number {
  return (
    layout?.chapters?.reduce(
      (chapterTotal, chapter) =>
        chapterTotal +
        (chapter.sections || []).reduce(
          (sectionTotal, section) => sectionTotal + (section.task_units || []).length,
          0,
        ),
      0,
    ) || 0
  );
}

export function countSections(layout: DocumentTaskLayout | null): number {
  return (
    layout?.chapters?.reduce(
      (chapterTotal, chapter) => chapterTotal + (chapter.sections || []).length,
      0,
    ) || 0
  );
}

export function resolveLayoutParserMode(
  layout: DocumentTaskLayout,
): StructureParserMode {
  return layout.parse_provenance?.effective_parser_mode === "llm_enhanced"
    ? "llm_enhanced"
    : "common";
}

export function labelOrId(label: string | null | undefined, fallback: string) {
  return label?.trim() || fallback;
}

export function sameDisplayLabel(
  left: string | null | undefined,
  right: string | null | undefined,
) {
  return (
    Boolean(left?.trim()) &&
    left?.trim().toLocaleLowerCase() === right?.trim().toLocaleLowerCase()
  );
}

export function isGenericContainerTitle(title: string | null | undefined) {
  const normalizedTitle = title?.trim().toLocaleLowerCase();
  return (
    normalizedTitle === "front matter" ||
    normalizedTitle === "back matter" ||
    normalizedTitle === "table of contents"
  );
}

export function buildSectionDisplay(
  chapter: ChapterLayout,
  section: SectionLayout,
) {
  const chapterTitle = labelOrId(chapter.title, chapter.chapter_id);
  const sectionTitle = labelOrId(section.title, section.section_id);
  const useSectionTitle =
    Boolean(section.title?.trim()) && isGenericContainerTitle(chapter.title);
  const title = useSectionTitle ? sectionTitle : chapterTitle;
  const subtitle =
    useSectionTitle && !sameDisplayLabel(chapter.title, section.title)
      ? chapterTitle
      : !sameDisplayLabel(chapter.title, section.title)
      ? sectionTitle
      : section.container_title || null;

  return { title, subtitle };
}
