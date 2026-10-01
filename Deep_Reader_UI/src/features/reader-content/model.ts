import type {
  ContentBlock,
  SectionSelection,
  TaskUnitContent,
} from "../../types/api";

export interface ReaderContentGroup {
  taskUnitId: string;
  title?: string | null;
  blocks: ContentBlock[];
}

export interface ReaderContentPage {
  groupIndexes: number[];
  oversized: boolean;
}

export function headingFromSelection(selection: SectionSelection) {
  return selection.section.title?.trim() || selection.section.section_id;
}

export function buildContentGroups(taskUnitContents: TaskUnitContent[]): ReaderContentGroup[] {
  return taskUnitContents
    .map((taskUnitContent) => ({
      taskUnitId: taskUnitContent.task_unit_id,
      title: taskUnitContent.title,
      blocks: taskUnitContent.content_blocks || [],
    }))
    .filter((group) => group.blocks.length > 0);
}

export function aggregateContentBlocks(taskUnitContents: TaskUnitContent[]): ContentBlock[] {
  return taskUnitContents.flatMap((taskUnitContent) => taskUnitContent.content_blocks || []);
}

export function paginateMeasuredGroups(
  groupHeights: number[],
  pageHeight: number,
): ReaderContentPage[] {
  if (groupHeights.length === 0) {
    return [];
  }
  if (pageHeight <= 0) {
    return [{ groupIndexes: groupHeights.map((_height, index) => index), oversized: true }];
  }

  const pages: ReaderContentPage[] = [];
  let currentGroupIndexes: number[] = [];
  let currentHeight = 0;

  groupHeights.forEach((height, index) => {
    const normalizedHeight = Math.max(0, height);
    if (currentGroupIndexes.length === 0) {
      currentGroupIndexes = [index];
      currentHeight = normalizedHeight;
      if (normalizedHeight > pageHeight) {
        pages.push({ groupIndexes: currentGroupIndexes, oversized: true });
        currentGroupIndexes = [];
        currentHeight = 0;
      }
      return;
    }

    if (currentHeight + normalizedHeight <= pageHeight) {
      currentGroupIndexes.push(index);
      currentHeight += normalizedHeight;
      return;
    }

    pages.push({ groupIndexes: currentGroupIndexes, oversized: false });
    currentGroupIndexes = [index];
    currentHeight = normalizedHeight;
    if (normalizedHeight > pageHeight) {
      pages.push({ groupIndexes: currentGroupIndexes, oversized: true });
      currentGroupIndexes = [];
      currentHeight = 0;
    }
  });

  if (currentGroupIndexes.length > 0) {
    pages.push({ groupIndexes: currentGroupIndexes, oversized: false });
  }

  return pages;
}
