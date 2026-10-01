import type {
  ContentBlock,
  SectionSelection,
  TaskUnitContent,
} from "../../types/api";

export function headingFromSelection(selection: SectionSelection) {
  return selection.section.title?.trim() || selection.section.section_id;
}

export function aggregateContentBlocks(taskUnitContents: TaskUnitContent[]): ContentBlock[] {
  return taskUnitContents.flatMap((taskUnitContent) => taskUnitContent.content_blocks || []);
}
