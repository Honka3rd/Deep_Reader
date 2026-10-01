import { useMemo, useState } from "react";
import type {
  DocumentTaskLayout,
  ManualStructureAnchorType,
  ManualStructureValidationIssue,
  ManualStructureValidationResponse,
  RequestStatus,
} from "../../types/api";
import type { ManualStructureService } from "../../services";
import { manualStructureService } from "../../services";
import type { AppNotificationState } from "../../shared/components/AppNotification";
import {
  buildEditableTocFromScratch,
  buildEditableTocFromLayout,
  buildManualStructurePlan,
  createChapter,
  createSection,
  layoutHasPageAnchorEvidence,
  type EditableTocChapter,
  type EditableTocRange,
  type SelectedTocItem,
  validateEditableToc,
} from "./model";

export type TocEditMode = "from_scratch" | "edit_existing";

interface UseTocEditorControllerOptions {
  docName: string;
  layout: DocumentTaskLayout | null;
  onCommitSuccess: () => Promise<void>;
  service?: ManualStructureService;
}

export function useTocEditorController({
  docName,
  layout,
  onCommitSuccess,
  service = manualStructureService,
}: UseTocEditorControllerOptions) {
  const preferredAnchorType: ManualStructureAnchorType = layoutHasPageAnchorEvidence(layout)
    ? "page_range"
    : "char_range";
  const existingLayoutChapters = useMemo(
    () => (layout ? buildEditableTocFromLayout(layout, preferredAnchorType) : []),
    [layout, preferredAnchorType],
  );
  const [mode, setModeState] = useState<TocEditMode>("from_scratch");
  const [chapters, setChapters] = useState<EditableTocChapter[]>(buildEditableTocFromScratch);
  const [selectedItem, setSelectedItem] = useState<SelectedTocItem | null>(null);
  const [frontendIssues, setFrontendIssues] = useState<ManualStructureValidationIssue[]>([]);
  const [backendValidation, setBackendValidation] =
    useState<ManualStructureValidationResponse | null>(null);
  const [validationStatus, setValidationStatus] = useState<RequestStatus>("initial");
  const [commitStatus, setCommitStatus] = useState<RequestStatus>("initial");
  const [notification, setNotification] = useState<AppNotificationState>({
    open: false,
    message: "",
    severity: "info",
  });
  const [confirmOpen, setConfirmOpen] = useState(false);

  function showNotification(
    message: string,
    severity: AppNotificationState["severity"],
    actionLabel?: string,
  ) {
    setNotification({
      open: true,
      message,
      severity,
      actionLabel,
    });
  }

  function closeNotification() {
    setNotification((current) => ({ ...current, open: false }));
  }

  function resetValidationState() {
    setFrontendIssues([]);
    setBackendValidation(null);
    setValidationStatus("initial");
    setCommitStatus("initial");
    closeNotification();
  }

  function selectFirstChapter(nextChapters: EditableTocChapter[]) {
    setSelectedItem(
      nextChapters[0] ? { kind: "chapter", chapterId: nextChapters[0].id } : null,
    );
  }

  function setMode(nextMode: TocEditMode) {
    if (nextMode === mode) {
      return;
    }
    const nextChapters =
      nextMode === "edit_existing" ? existingLayoutChapters : buildEditableTocFromScratch();
    setModeState(nextMode);
    setChapters(nextChapters);
    selectFirstChapter(nextChapters);
    resetValidationState();
  }

  function updateChapter(chapterId: string, patch: Partial<EditableTocChapter>) {
    setBackendValidation(null);
    setChapters((current) =>
      current.map((chapter) => (chapter.id === chapterId ? { ...chapter, ...patch } : chapter)),
    );
  }

  function updateSection(
    chapterId: string,
    sectionId: string,
    patch: Partial<EditableTocChapter["sections"][number]>,
  ) {
    setBackendValidation(null);
    setChapters((current) =>
      current.map((chapter) =>
        chapter.id === chapterId
          ? {
              ...chapter,
              sections: chapter.sections.map((section) =>
                section.id === sectionId ? { ...section, ...patch } : section,
              ),
            }
          : chapter,
      ),
    );
  }

  function updateSelectedRange(range: EditableTocRange) {
    if (!selectedItem) {
      return;
    }
    if (selectedItem.kind === "chapter") {
      updateChapter(selectedItem.chapterId, { range });
      return;
    }
    if (selectedItem.sectionId) {
      updateSection(selectedItem.chapterId, selectedItem.sectionId, { range });
    }
  }

  function addChapter() {
    setBackendValidation(null);
    const chapter = createChapter(chapters.length, preferredAnchorType);
    setChapters((current) => [...current, chapter]);
    setSelectedItem({ kind: "chapter", chapterId: chapter.id });
  }

  function addSection(chapterId: string) {
    setBackendValidation(null);
    setChapters((current) =>
      current.map((chapter) =>
        chapter.id === chapterId
          ? {
              ...chapter,
              sections: [
                ...chapter.sections,
                createSection(chapter.sections.length, preferredAnchorType),
              ],
            }
          : chapter,
      ),
    );
  }

  function removeChapter(chapterId: string) {
    setBackendValidation(null);
    setChapters((current) => current.filter((chapter) => chapter.id !== chapterId));
    setSelectedItem(null);
  }

  function removeSection(chapterId: string, sectionId: string) {
    setBackendValidation(null);
    setChapters((current) =>
      current.map((chapter) =>
        chapter.id === chapterId
          ? {
              ...chapter,
              sections: chapter.sections.filter((section) => section.id !== sectionId),
            }
          : chapter,
      ),
    );
    setSelectedItem({ kind: "chapter", chapterId });
  }

  function moveChapter(chapterId: string, direction: -1 | 1) {
    setBackendValidation(null);
    setChapters((current) => moveById(current, chapterId, direction));
  }

  function moveSection(chapterId: string, sectionId: string, direction: -1 | 1) {
    setBackendValidation(null);
    setChapters((current) =>
      current.map((chapter) =>
        chapter.id === chapterId
          ? { ...chapter, sections: moveById(chapter.sections, sectionId, direction) }
          : chapter,
      ),
    );
  }

  async function validateWithBackend() {
    const frontendValidation = validateEditableToc(chapters);
    setFrontendIssues(frontendValidation.issues);
    setBackendValidation(null);
    if (!frontendValidation.valid) {
      setValidationStatus("error");
      showNotification(
        `${frontendValidation.issues.length} frontend validation issue${
          frontendValidation.issues.length === 1 ? "" : "s"
        } found.`,
        "error",
        "Details",
      );
      return;
    }

    setValidationStatus("loading");
    try {
      const response = await service.validateManualStructure({
        doc_name: docName.trim(),
        manual_structure: buildManualStructurePlan(chapters),
      });
      setBackendValidation(response);
      setValidationStatus(response.valid ? "success" : "error");
      if (response.valid) {
        showNotification("Backend validation passed.", "success");
      } else {
        showNotification(
          `${response.errors.length} backend validation issue${
            response.errors.length === 1 ? "" : "s"
          } found.`,
          "error",
          "Details",
        );
      }
    } catch (error) {
      setValidationStatus("error");
      showNotification(error instanceof Error ? error.message : String(error), "error");
    }
  }

  function requestCommitConfirmation() {
    const frontendValidation = validateEditableToc(chapters);
    setFrontendIssues(frontendValidation.issues);
    if (!frontendValidation.valid || !backendValidation?.valid) {
      showNotification("Validate the TOC successfully before hard reparse.", "error");
      return;
    }
    setConfirmOpen(true);
  }

  async function commitHardReparse() {
    setConfirmOpen(false);
    setCommitStatus("loading");
    try {
      const result = await service.commitManualStructure({
        doc_name: docName.trim(),
        manual_structure: buildManualStructurePlan(chapters),
      });
      if (!result.success) {
        throw new Error(result.error || "Manual TOC hard reparse failed.");
      }
      setCommitStatus("success");
      showNotification("Manual TOC committed. Reloading layout.", "success");
      await onCommitSuccess();
    } catch (error) {
      setCommitStatus("error");
      showNotification(error instanceof Error ? error.message : String(error), "error");
    }
  }

  return {
    mode,
    chapters,
    selectedItem,
    frontendIssues,
    backendValidation,
    validationStatus,
    commitStatus,
    notification,
    confirmOpen,
    preferredAnchorType,
    setMode,
    setConfirmOpen,
    closeNotification,
    setSelectedItem,
    updateChapter,
    updateSection,
    updateSelectedRange,
    addChapter,
    addSection,
    removeChapter,
    removeSection,
    moveChapter,
    moveSection,
    validateWithBackend,
    requestCommitConfirmation,
    commitHardReparse,
  };
}

function moveById<T extends { id: string }>(items: T[], id: string, direction: -1 | 1): T[] {
  const index = items.findIndex((item) => item.id === id);
  const nextIndex = index + direction;
  if (index < 0 || nextIndex < 0 || nextIndex >= items.length) {
    return items;
  }
  const nextItems = [...items];
  const [item] = nextItems.splice(index, 1);
  nextItems.splice(nextIndex, 0, item);
  return nextItems;
}
