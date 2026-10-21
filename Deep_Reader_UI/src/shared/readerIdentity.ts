import type {
  DocumentTaskLayout,
  ReadingInteractionTargetRequest,
  SectionSelection,
} from "../types/api";

export function requireBackendIdentity(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || !value.trim() || value !== value.trim()) {
    throw new Error(`${label} is missing or invalid. Select the document again.`);
  }
}

export function assertSectionInLayout(
  docName: string,
  layout: DocumentTaskLayout | null,
  selection: SectionSelection,
) {
  requireBackendIdentity(docName, "Document name");
  requireBackendIdentity(selection.chapter.chapter_id, "Chapter ID");
  requireBackendIdentity(selection.section.section_id, "Section ID");
  // Object membership rejects a stale layout snapshot even if IDs were reused.
  if (!layout?.chapters?.includes(selection.chapter)
    || !selection.chapter.sections?.includes(selection.section)) {
    throw new Error("The selected section is no longer in the current document. Select it again.");
  }
  const units = selection.section.task_units || [];
  if (units.length !== selection.taskUnits.length
    || selection.taskUnits.some((unit, index) => unit !== units[index])) {
    throw new Error("The selected task units are stale. Select the section again.");
  }
  const ids = selection.taskUnits.map((unit) => {
    requireBackendIdentity(unit.unit_id, "Task-unit ID");
    return unit.unit_id;
  });
  if (new Set(ids).size !== ids.length) throw new Error("Duplicate task-unit IDs in section.");
}

export function assertInteractionIdentity(target: ReadingInteractionTargetRequest) {
  requireBackendIdentity(target.doc_name, "Document name");
  const identityField = {
    document: "doc_name",
    chapter: "chapter_id",
    section: "section_id",
    task_unit: "task_unit_id",
    content_block: "content_block_id",
  } as const;
  const field = identityField[target.target_type];
  if (!field) throw new Error("Unknown interaction target type.");
  requireBackendIdentity(target[field], "Interaction target ID");
  if (target.target_type === "content_block") requireBackendIdentity(target.task_unit_id, "Task-unit ID");
  for (const key of ["chapter_id", "section_id", "task_unit_id", "content_block_id"] as const) {
    if (target[key] != null) requireBackendIdentity(target[key], key);
  }
}

export function assertTargetInLayout(
  target: ReadingInteractionTargetRequest,
  docName: string,
  layout: DocumentTaskLayout | null,
) {
  assertInteractionIdentity(target);
  if (!layout || target.doc_name !== docName) throw new Error("Interaction belongs to an inactive document.");
  if (target.target_type === "document") return;
  const chapter = layout.chapters?.find((item) => item.chapter_id === target.chapter_id);
  if (!chapter) throw new Error("Interaction chapter is no longer available.");
  if (target.target_type === "chapter") return;
  const section = chapter.sections?.find((item) => item.section_id === target.section_id);
  if (!section) throw new Error("Interaction section is no longer available.");
  if (target.target_type === "section") return;
  if (!section.task_units?.some((item) => item.unit_id === target.task_unit_id)) {
    throw new Error("Interaction task unit is no longer available.");
  }
}
