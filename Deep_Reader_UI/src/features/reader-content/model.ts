import type {
  ContentBlock,
  ContentBlockInteractionAvailabilitySlot,
  SectionSelection,
  TaskUnitContent,
} from "../../types/api";
import {
  formatReadingInteractionStatusLabel,
  normalizeReadingInteractionAvailabilityStatus,
  type ReadingInteractionAvailabilityStatus,
  type ReadingInteractionKind,
} from "../reading-interactions/model";

export interface ReaderContentGroup {
  taskUnitId: string;
  chapterId?: string | null;
  sectionId?: string | null;
  title?: string | null;
  blocks: ReaderContentBlock[];
}

export interface ReaderContentBlockOwnerContext {
  documentId?: string;
  documentTitle?: string | null;
  taskUnitId: string;
  chapterId?: string | null;
  sectionId?: string | null;
  sourceHash?: string | null;
}

export interface ReaderContentBlock extends ContentBlock {
  ownerContext: ReaderContentBlockOwnerContext;
}

export type ContentBlockInteractionEligibilityReason =
  | "eligible"
  | "target_identity_available"
  | "missing_task_unit_id"
  | "missing_content_block_id";

export interface ContentBlockInteractionEligibility {
  eligible: boolean;
  reason: ContentBlockInteractionEligibilityReason;
}

export interface ContentBlockInteractionAvailabilityHint {
  kind: ReadingInteractionKind;
  status: ReadingInteractionAvailabilityStatus;
  label: string;
  reason?: string;
  artifactId?: string;
  sessionId?: string;
  generatedAt?: string;
  updatedAt?: string;
}

export interface ReaderContentPage {
  groupIndexes: number[];
  oversized: boolean;
}

export function headingFromSelection(selection: SectionSelection) {
  return selection.section.title?.trim() || selection.section.section_id;
}

function stringMetadataValue(
  metadata: Record<string, unknown> | undefined,
  key: string,
): string | null {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function withOwnerContext(
  block: ContentBlock,
  taskUnitContent: TaskUnitContent,
): ReaderContentBlock {
  return {
    ...block,
    ownerContext: {
      documentId: taskUnitContent.document_id,
      documentTitle: taskUnitContent.document_title,
      taskUnitId: taskUnitContent.task_unit_id,
      chapterId: taskUnitContent.chapter_id,
      sectionId: taskUnitContent.section_id,
      sourceHash: stringMetadataValue(block.metadata, "source_hash"),
    },
  };
}

function contentBlocksForTaskUnit(taskUnitContent: TaskUnitContent): ReaderContentBlock[] {
  const contentBlocks = taskUnitContent.content_blocks || [];
  if (contentBlocks.length > 0) {
    return contentBlocks.map((block) => withOwnerContext(block, taskUnitContent));
  }

  const compatibilityContent = taskUnitContent.content?.trim();
  if (!compatibilityContent) {
    return [];
  }

  return [
    {
      block_id: `${taskUnitContent.task_unit_id}:content`,
      block_type: "content",
      content: compatibilityContent,
    },
  ].map((block) => withOwnerContext(block, taskUnitContent));
}

export function buildContentGroups(taskUnitContents: TaskUnitContent[]): ReaderContentGroup[] {
  return taskUnitContents.map((taskUnitContent) => ({
    taskUnitId: taskUnitContent.task_unit_id,
    chapterId: taskUnitContent.chapter_id,
    sectionId: taskUnitContent.section_id,
    title: taskUnitContent.title,
    blocks: contentBlocksForTaskUnit(taskUnitContent),
  }));
}

export function aggregateContentBlocks(taskUnitContents: TaskUnitContent[]): ReaderContentBlock[] {
  return taskUnitContents.flatMap(contentBlocksForTaskUnit);
}

function hasInteractionAvailability(block: ReaderContentBlock): boolean {
  const availability = block.interaction_availability;
  return Boolean(
    availability?.insight ||
      availability?.quiz ||
      availability?.critical_thinking,
  );
}

function hasMatchingArtifactTargetRef(block: ReaderContentBlock): boolean {
  const taskUnitId = block.ownerContext.taskUnitId.trim();
  const contentBlockId = block.block_id.trim();
  return Boolean(
    block.artifact_target_refs?.some((targetRef) => {
      const targetLevel = targetRef.target_level?.trim();
      const refTaskUnitId = targetRef.task_unit_id?.trim();
      const refContentBlockId = (
        targetRef.content_block_id ||
        targetRef.target_id ||
        ""
      ).trim();

      return (
        targetLevel === "content_block" &&
        (!refTaskUnitId || refTaskUnitId === taskUnitId) &&
        refContentBlockId === contentBlockId
      );
    }),
  );
}

export function contentBlockInteractionEligibility(
  block: ReaderContentBlock,
): ContentBlockInteractionEligibility {
  if (!block.ownerContext.taskUnitId.trim()) {
    return { eligible: false, reason: "missing_task_unit_id" };
  }
  if (!block.block_id.trim()) {
    return { eligible: false, reason: "missing_content_block_id" };
  }
  if (hasInteractionAvailability(block) || hasMatchingArtifactTargetRef(block)) {
    return { eligible: true, reason: "eligible" };
  }

  return { eligible: true, reason: "target_identity_available" };
}

export function isContentBlockInteractionEligible(block: ReaderContentBlock): boolean {
  return contentBlockInteractionEligibility(block).eligible;
}

function availabilitySlotForKind(
  block: ReaderContentBlock,
  kind: ReadingInteractionKind,
): ContentBlockInteractionAvailabilitySlot | null | undefined {
  switch (kind) {
    case "insight":
      return block.interaction_availability?.insight;
    case "quiz":
      return block.interaction_availability?.quiz;
    case "critical_thinking":
      return block.interaction_availability?.critical_thinking;
    default:
      return null;
  }
}

export function contentBlockInteractionAvailabilityHint(
  block: ReaderContentBlock,
  kind: ReadingInteractionKind,
): ContentBlockInteractionAvailabilityHint | null {
  const slot = availabilitySlotForKind(block, kind);
  if (!slot?.status?.trim()) {
    return null;
  }

  const status = normalizeReadingInteractionAvailabilityStatus(slot.status);
  return {
    kind,
    status,
    label: formatReadingInteractionStatusLabel(status),
    reason: slot.reason || undefined,
    artifactId: slot.artifact_id || undefined,
    sessionId: slot.session_id || undefined,
    generatedAt: slot.generated_at || undefined,
    updatedAt: slot.updated_at || undefined,
  };
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
