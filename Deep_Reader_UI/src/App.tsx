import BuildOutlinedIcon from "@mui/icons-material/BuildOutlined";
import { ReaderWorkspace } from "./shared/components/ReaderWorkspace";
import { ReaderContentPane } from "./shared/components/ReaderContentPane";
import CleaningServicesOutlinedIcon from "@mui/icons-material/CleaningServicesOutlined";
import { SmartCleanupDialog } from "./features/smart-cleanup";
import { ReadingContentActions, confirmReadingNavigation, type ContentEditSelection } from "./features/reader-content/ReadingContentActions";
import { SummaryNoteDrawer, type SummaryNoteSelection } from "./features/reading-interactions/SummaryNoteDrawer";
import CheckIcon from "@mui/icons-material/Check";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import LibraryBooksOutlinedIcon from "@mui/icons-material/LibraryBooksOutlined";
import {
  Box,
  Button,
  CircularProgress,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Paper,
} from "@mui/material";
import type { MouseEvent } from "react";
import { assertTargetInLayout } from "./shared/readerIdentity";
import { TocHintControl } from "./shared/TocHintControl";
import { EnhancementDialog } from "./features/structure-enhancement/view";
import type { TocHint } from "./types/api";
import { useEffect, useRef, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useMatch,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  BookSearchView,
  useBookSearchController,
} from "./features/book-search";
import {
  FileManagementView,
  useFileManagementController,
} from "./features/file-management";
import {
  HierarchyNavigationView,
  resolveLayoutParserMode,
} from "./features/hierarchy-navigation";
import {
  ReaderContentView,
  type ReaderContentBlock,
  type ReaderContentGroup,
  useReaderContentController,
} from "./features/reader-content";
import { TocEditorView } from "./features/toc-editor";
import {
  AppNotification,
  type AppNotificationState,
} from "./shared/components/AppNotification";
import {
  readingInteractionService,
  structureRepairService,
  taskLayoutService,
} from "./services";
import type {
  ChapterLayout,
  DocumentTaskLayout,
  RequestStatus,
  SectionLayout,
  SectionSelection,
  StructureParserMode,
} from "./types/api";
import {
  buildReadingInteractionTargetKey,
  CriticalThinkingDrawerCommands,
  CriticalThinkingDrawerContent,
  mapAnalysisInteractionResponseToInsightViewState,
  mapCriticalThinkingSessionResponseToViewState,
  mapQuizInteractionResponseToQuizViewState,
  QuizDrawerCommands,
  QuizDrawerContent,
  ReadingInteractionsView,
  toReadingInteractionTargetRequest as mapInteractionTargetRequest,
  type ReadingInteractionKind,
  type ReadingInteractionMenuSelection,
  type ReadingInteractionTarget,
  useReadingInteractionMenuController,
} from "./features/reading-interactions";

import { LearningSessionMemory } from "./features/learning-reader";
import { useReadingSettingsController } from "./features/reading-settings";

import { BlockTranslationView, useBlockTranslation } from "./features/block-translation";

export default function App() {
  const navigate = useNavigate();
  const routeMatch = useMatch("/documents/:routeDocName/*");
  const routeDocName = routeMatch?.params.routeDocName || "";
  const [docName, setDocName] = useState("");
  const activeDocumentRef = useRef("");
  const [summaryNoteSelection, setSummaryNoteSelection] = useState<SummaryNoteSelection | null>(null);
  const [sourceNeedsReview, setSourceNeedsReview] = useState(false);
  const [readingRefresh, setReadingRefresh] = useState(0);
  const [contentEditSelection,setContentEditSelection] = useState<ContentEditSelection>(null);
  const noteDirtyRef = useRef(false);
  function closeSummaryNote() {
    if (noteDirtyRef.current && !window.confirm("Discard unsaved note changes?")) return false;
    noteDirtyRef.current = false;
    setSummaryNoteSelection(null);
    return true;
  }
  const [layout, setLayout] = useState<DocumentTaskLayout | null>(null);
  const layoutScopeRef = useRef<{ docName: string; layout: DocumentTaskLayout } | null>(null);
  const [layoutStatus, setLayoutStatus] = useState<RequestStatus>("initial");
  const [repairStatus, setRepairStatus] = useState<RequestStatus>("initial");
  const [layoutError, setLayoutError] = useState("");
  const [notification, setNotification] = useState<AppNotificationState>({
    open: false,
    message: "",
    severity: "info",
  });
  const [currentRepairMode, setCurrentRepairMode] = useState<StructureParserMode | null>(null);
  const [activeRepairMode, setActiveRepairMode] = useState<StructureParserMode | null>(null);
  const [repairMenuAnchor, setRepairMenuAnchor] = useState<HTMLElement | null>(null);
  const [repairTocHint, setRepairTocHint] = useState<TocHint>("auto");
  const [enhancementOpen, setEnhancementOpen] = useState(false);
  const [cleanupOpen, setCleanupOpen] = useState(false);
  const [criticalThinkingDraftDirty, setCriticalThinkingDraftDirty] = useState(false);
  const [criticalThinkingDraftAnswersByTarget, setCriticalThinkingDraftAnswersByTarget] =
    useState<Record<string, string>>({});
  const layoutRequestIdRef = useRef(0);
  const {
    searchInput,
    setSearchInput,
    documentOptions,
    searching: documentSearchLoading,
    loadDocumentOptions,
    removeDocumentOption,
  } = useBookSearchController();
  const fileManagement = useFileManagementController({
    onDocumentPreparationSettled: (preparedDocName) => {
      if (activeDocumentRef.current === preparedDocName) void loadLayoutForDocument(preparedDocName);
    },
    onDocumentDeleted: (deletedDocName) => {
      removeDocumentOption(deletedDocName);
      if (activeDocumentRef.current === deletedDocName) {
        clearActiveDocument();
        navigate("/");
      }
    },
  });
  const readingSettings = useReadingSettingsController(fileManagement.open);
  const [readerModeSettingsOpen, setReaderModeSettingsOpen] = useState(false);
  const [learningMemory] = useState(() => new LearningSessionMemory());
  const {
    selectedSection,
    contentBlocks,
    contentGroups,
    contentStatus,
    contentError,
    navigationTarget,
    navigateToBlock,
    selectSection,
    retryContent,
    refreshChangedContent,
    refreshFromLayout,
    refreshAffectedBlocks,
    resetContent,
    refreshSelectionLabels,
  } = useReaderContentController({
    docName: layoutScopeRef.current?.docName || "",
    layout: layoutStatus === "success" ? layout : null,
  });
  const translations = useBlockTranslation({ docName, blocks: contentBlocks,
    scopeKey: JSON.stringify([docName, selectedSection?.section.section_id, layoutStatus]) });
  const {
    openInlineInsightsByTarget,
    quizStateByTarget,
    criticalThinkingStateByTarget,
    selectedInteraction,
    selectInteractionAction,
    activateInteraction,
    generationTasks,
    beginInlineInsightRequest,
    applyInlineInsightState,
    failInlineInsightRequest,
    beginQuizRequest,
    applyQuizState,
    failQuizRequest,
    beginCriticalThinkingRequest,
    applyCriticalThinkingState,
    failCriticalThinkingRequest,
    clearSelectedInteraction,
    resetInteractions,
    isLatestInteractionRequest,
    closeInlineInsight,
  } = useReadingInteractionMenuController();

  function showNotification(
    message: string,
    severity: AppNotificationState["severity"] = "info",
  ) {
    setNotification({
      open: true,
      message,
      severity,
    });
  }

  function closeNotification() {
    setNotification((current) => ({ ...current, open: false }));
  }

  useEffect(() => {
    if (contentStatus === "error" && contentError) {
      showNotification(contentError, "error");
    }
  }, [contentStatus, contentError]);

  async function reloadLayout(trimmedDocName: string) {
    const requestId = layoutRequestIdRef.current;
    const nextLayout = await taskLayoutService.prepareTaskLayout(trimmedDocName, { layoutOnly: true });
    if (requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== trimmedDocName) return false;
    setLayout(nextLayout);
    layoutScopeRef.current = { docName: trimmedDocName, layout: nextLayout };
    setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
    setLayoutStatus("success");
    return nextLayout;
  }

  async function loadLayoutForDocument(nextDocName: string, validatedLayout?: DocumentTaskLayout) {
    if (!confirmReadingNavigation()) {
      navigate(`/documents/${encodeURIComponent(activeDocumentRef.current)}`, { replace: true });
      return;
    }
    setContentEditSelection(null);
    if (nextDocName.trim() !== activeDocumentRef.current && !closeSummaryNote()) {
      navigate(`/documents/${encodeURIComponent(activeDocumentRef.current)}`, { replace: true });
      return;
    }
    setSummaryNoteSelection(null);
    setEnhancementOpen(false);
    setCleanupOpen(false);
    const trimmedDocName = nextDocName.trim();
    const requestId = layoutRequestIdRef.current + 1;
    layoutRequestIdRef.current = requestId;

    if (!trimmedDocName) {
      const message = "Document name is required.";
      setLayoutStatus("error");
      setLayoutError(message);
      showNotification(message, "error");
      return;
    }

    activeDocumentRef.current = trimmedDocName;
    setSourceNeedsReview(false);
    setDocName(trimmedDocName);
    setSearchInput(trimmedDocName);
    setRepairStatus("initial");
    setActiveRepairMode(null);
    resetInteractions();
    setRepairTocHint("auto");
    setCriticalThinkingDraftAnswersByTarget({});
    setLayout(null);
    layoutScopeRef.current = null;
    resetContent();
    setLayoutError("");
    setLayoutStatus("loading");

    try {
      const nextLayout = validatedLayout ?? await taskLayoutService.fetchTaskLayout(trimmedDocName);
      if (layoutRequestIdRef.current !== requestId) {
        return;
      }
      setLayout(nextLayout);
      layoutScopeRef.current = { docName: trimmedDocName, layout: nextLayout };
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
      setLayoutStatus("success");
      fileManagement.recordRecentDocument(trimmedDocName);
    } catch (error) {
      if (layoutRequestIdRef.current !== requestId) {
        return;
      }
      const message = error instanceof Error ? error.message : String(error);
      setLayoutStatus("error");
      setLayoutError(message);
      showNotification(message, "error");
    }
  }

  function selectDocument(nextDocName: string, validatedLayout?: DocumentTaskLayout) {
    if (!nextDocName.trim()) return;
    if (!closeSummaryNote()) return;
    navigate(`/documents/${encodeURIComponent(nextDocName.trim())}`);
    void loadLayoutForDocument(nextDocName, validatedLayout);
  }

  function clearActiveDocument() {
    if (!confirmReadingNavigation()) {
      navigate(`/documents/${encodeURIComponent(activeDocumentRef.current)}`, { replace: true });
      return;
    }
    setSourceNeedsReview(false);
    setContentEditSelection(null);
    setSummaryNoteSelection(null);
    setEnhancementOpen(false);
    setCleanupOpen(false);
    ++layoutRequestIdRef.current;
    activeDocumentRef.current = "";
    setDocName("");
    setSearchInput("");
    setRepairStatus("initial");
    setActiveRepairMode(null);
    setLayout(null);
    layoutScopeRef.current = null;
    resetContent();
    resetInteractions();
    setCriticalThinkingDraftAnswersByTarget({});
    setLayoutError("");
    setLayoutStatus("initial");
  }

  async function openManagedDocument(nextDocName: string) {
    const navigation = layoutRequestIdRef.current;
    const readyLayout = await fileManagement.validateDocumentOpen(nextDocName);
    if (!readyLayout || navigation !== layoutRequestIdRef.current) return;
    fileManagement.closeLibrary();
    selectDocument(nextDocName, readyLayout);
  }

  useEffect(() => {
    if (routeDocName && routeDocName !== activeDocumentRef.current) {
      void loadLayoutForDocument(routeDocName);
    } else if (!routeDocName && activeDocumentRef.current) {
      clearActiveDocument();
    }
  }, [routeDocName]);

  async function repairStructure(parserMode: StructureParserMode) {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || !layout) {
      return;
    }

    setRepairStatus("loading");
    const requestId = layoutRequestIdRef.current;
    setActiveRepairMode(parserMode);
    setLayoutError("");
    let applied = false;

    try {
      const result = await structureRepairService.reparseDocumentStructure(
        trimmedDocName,
        parserMode,
        { tocHint: repairTocHint },
      );
      if (requestId !== layoutRequestIdRef.current) return;
      if (!result.success) {
        throw new Error(result.error || "Structure repair failed");
      }
      applied = true;
      resetInteractions();
      resetContent();
      setCriticalThinkingDraftAnswersByTarget({});
      layoutScopeRef.current = null;
      setLayoutStatus("loading");
      const updatedLayout = await reloadLayout(trimmedDocName);
      if (!updatedLayout) return;
      setRepairStatus("success");
      showNotification(updatedLayout.parse_provenance?.fallback_used
        ? "Requested parser was not applied. The fallback structure is loaded."
        : "Structure updated. Layout reloaded.", updatedLayout.parse_provenance?.fallback_used ? "warning" : "success");
    } catch (error) {
      if (requestId !== layoutRequestIdRef.current) return;
      setRepairStatus("error");
      if (applied) setLayout(null);
      setLayoutStatus(!applied && layout ? "success" : "error");
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    } finally {
      if (requestId === layoutRequestIdRef.current) setActiveRepairMode(null);
    }
  }

  const canRepair = repairStatus !== "loading" && ((layoutStatus === "success" && Boolean(layout))
    || (layoutStatus === "error" && Boolean(docName.trim())));
  const repairMenuOpen = Boolean(repairMenuAnchor);
  const canCheckCleanup = Boolean(docName.trim()) && repairStatus !== "loading" && layoutStatus !== "loading";

  function openRepairMenu(event: MouseEvent<HTMLButtonElement>) {
    setRepairMenuAnchor(event.currentTarget);
  }

  function closeRepairMenu() {
    setRepairMenuAnchor(null);
  }

  function selectRepairMode(parserMode: StructureParserMode) {
    closeRepairMenu();
    if (parserMode === "llm_enhanced") {
      if (!confirmReadingNavigation()) return;
      setContentEditSelection(null);
      setEnhancementOpen(true);
    }
    else void repairStructure(parserMode);
  }

  function clearEnhancedDocument(name: string) {
    if (activeDocumentRef.current !== name) throw new Error("Return to this document to reload it.");
    ++layoutRequestIdRef.current;
    setSummaryNoteSelection(null);
    noteDirtyRef.current = false;
    resetInteractions();
    resetContent();
    setCriticalThinkingDraftAnswersByTarget({});
    setCriticalThinkingDraftDirty(false);
    layoutScopeRef.current = null;
    setLayout(null);
    setLayoutStatus("loading");
  }

  async function reloadEnhancedDocument(name: string, signal?: AbortSignal) {
    if (activeDocumentRef.current !== name) throw new Error("Return to this document to reload it.");
    const requestId = ++layoutRequestIdRef.current;
    try {
      const next = await taskLayoutService.fetchTaskLayout(name, { signal });
      if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== name) throw new Error("Document changed.");
      setLayout(next);
      layoutScopeRef.current = { docName: name, layout: next };
      setCurrentRepairMode(resolveLayoutParserMode(next));
      setLayoutStatus("success");
      setLayoutError("");
      closeNotification();
    } catch (error) {
      if (requestId === layoutRequestIdRef.current && activeDocumentRef.current === name) setLayoutStatus("error");
      throw error;
    }
  }

  async function refreshEnhancedTitles(name: string, signal?: AbortSignal) {
    const requestId = ++layoutRequestIdRef.current;
    const next = await taskLayoutService.fetchTaskLayout(name, { signal });
    if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== name) throw new Error("Document changed.");
    setLayout(next);
    layoutScopeRef.current = { docName: name, layout: next };
    refreshSelectionLabels(next);
    setCurrentRepairMode(resolveLayoutParserMode(next));
    setLayoutStatus("success");
    setLayoutError("");
    closeNotification();
  }

  async function refreshReadyReading(name: string, contentChanged: boolean, signal?: AbortSignal, reconcile = false) {
    if (signal?.aborted || activeDocumentRef.current !== name) throw new Error("Document changed.");
    if (contentChanged || reconcile) {
      resetInteractions();
      setSummaryNoteSelection(current => current?.kind === "summary" ? null : current);
      setReadingRefresh(value => value + 1);
    }
    if (reconcile) {
      const requestId = ++layoutRequestIdRef.current;
      const next = await taskLayoutService.fetchTaskLayout(name, { signal });
      if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== name) throw new Error("Document changed.");
      await refreshFromLayout(next, signal);
      if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== name) throw new Error("Document changed.");
      setLayout(next); layoutScopeRef.current = { docName: name, layout: next };
      setCurrentRepairMode(resolveLayoutParserMode(next)); setLayoutStatus("success"); setLayoutError("");
    }
    else if (layoutStatus !== "success" || !layout) await reloadEnhancedDocument(name, signal);
    else if (contentChanged) await refreshChangedContent(signal);
    if (!signal?.aborted && activeDocumentRef.current === name) closeNotification();
  }

  function interactionLabel(kind: ReadingInteractionKind) {
    return kind === "critical_thinking"
      ? "Critical thinking"
      : kind === "quiz"
      ? "Quiz"
      : "Insights";
  }

  function interactionTargetLevelLabel(target: ReadingInteractionTarget) {
    switch (target.targetLevel) {
      case "document":
        return "document";
      case "chapter":
        return "chapter";
      case "section":
        return "section";
      case "task_unit":
        return "task unit";
      case "content_block":
        return "content block";
      default:
        return "target";
    }
  }

  function notifyReadingInteractionSelection(selection: ReadingInteractionMenuSelection) {
    const label = interactionLabel(selection.kind);
    const targetLabel = interactionTargetLevelLabel(selection.target);
    const targetTitle = selection.target.displayTitle || selection.targetKey;
    const surface =
      selection.surface === "inline_insight" ? "inline insight" : "drawer";
    showNotification(
      `${label} for ${targetLabel} "${targetTitle}" is ready for ${surface} wiring. Backend routes exist; frontend service wiring is still pending for this interaction type.`,
      "info",
    );
  }

  function selectReadingInteraction(
    target: ReadingInteractionTarget | null,
    kind: ReadingInteractionKind,
    retryGeneration = false,
  ) {
    if (!target) {
      return;
    }
    if (sourceNeedsReview && kind !== "note") {
      showNotification("Review the unresolved OCR regions before generating interactions.", "warning");
      return;
    }

    try {
      const request = toReadingInteractionTargetRequest(target);
      if (!closeSummaryNote()) return;
      closeNotification();
      if (kind === "summary" || kind === "note") {
        closeReadingInteractionDrawer();
        setSummaryNoteSelection({ kind, target: request, breadcrumb: target.breadcrumb });
        return;
      }
      const selection = selectInteractionAction(kind, target);
      if (selection) {
        if (["insight", "quiz", "critical_thinking"].includes(selection.kind)) {
          void activateInteraction(selection, request, readingInteractionService,
            message => showNotification(message, "error"), retryGeneration, key => {
              setCriticalThinkingDraftAnswersByTarget(current => ({ ...current, [key]: "" }));
              setCriticalThinkingDraftDirty(false);
            });
        } else {
          notifyReadingInteractionSelection(selection);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    }
  }

  function toReadingInteractionTargetRequest(target: ReadingInteractionTarget) {
    const request = mapInteractionTargetRequest(target);
    const scope = layoutScopeRef.current;
    if (layoutStatus !== "success" || scope?.layout !== layout
      || scope.docName !== activeDocumentRef.current) {
      throw new Error("Document layout is not ready. Select the document again.");
    }
    assertTargetInLayout(request, scope.docName, scope.layout);
    if (target.targetLevel === "content_block" && !contentBlocks.some((block) =>
      block.block_id === target.contentBlockId
      && block.ownerContext.taskUnitId === target.taskUnitId)) {
      throw new Error("Content block is no longer available. Select the section again.");
    }
    return request;
  }

  function documentInteractionTarget(): ReadingInteractionTarget | null {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName) {
      return null;
    }

    const displayTitle = layout?.title?.trim() || trimmedDocName;
    return {
      targetLevel: "document",
      docName: trimmedDocName,
      documentId: layout?.document_id,
      displayTitle,
      breadcrumb: [displayTitle],
    };
  }

  function chapterInteractionTarget(chapter: ChapterLayout): ReadingInteractionTarget | null {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName) {
      return null;
    }

    const documentTitle = layout?.title?.trim() || trimmedDocName;
    const chapterTitle = chapter.title?.trim() || chapter.chapter_id;
    return {
      targetLevel: "chapter",
      docName: trimmedDocName,
      documentId: layout?.document_id,
      chapterId: chapter.chapter_id,
      displayTitle: chapterTitle,
      breadcrumb: [documentTitle, chapterTitle],
    };
  }

  function sectionInteractionTarget(
    chapter: ChapterLayout,
    section: SectionLayout,
  ): ReadingInteractionTarget | null {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName) {
      return null;
    }

    const documentTitle = layout?.title?.trim() || trimmedDocName;
    const chapterTitle = chapter.title?.trim() || chapter.chapter_id;
    const sectionTitle = section.title?.trim() || section.section_id;
    return {
      targetLevel: "section",
      docName: trimmedDocName,
      documentId: layout?.document_id,
      chapterId: chapter.chapter_id,
      parentChapterId: chapter.chapter_id,
      sectionId: section.section_id,
      displayTitle: sectionTitle,
      breadcrumb: [documentTitle, chapterTitle, sectionTitle],
    };
  }

  function taskUnitInteractionTarget(
    selection: SectionSelection,
    group: ReaderContentGroup,
  ): ReadingInteractionTarget | null {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName) {
      return null;
    }

    const documentTitle = layout?.title?.trim() || trimmedDocName;
    const chapterTitle = selection.chapter.title?.trim() || selection.chapter.chapter_id;
    const sectionTitle = selection.section.title?.trim() || selection.section.section_id;
    const taskUnitTitle = group.title?.trim() || group.taskUnitId;
    return {
      targetLevel: "task_unit",
      docName: trimmedDocName,
      documentId: layout?.document_id,
      chapterId: selection.chapter.chapter_id,
      sectionId: selection.section.section_id,
      parentChapterId: selection.chapter.chapter_id,
      parentSectionId: selection.section.section_id,
      taskUnitId: group.taskUnitId,
      displayTitle: taskUnitTitle,
      breadcrumb: [documentTitle, chapterTitle, sectionTitle, taskUnitTitle],
    };
  }

  function contentBlockInteractionTarget(
    selection: SectionSelection,
    block: ReaderContentBlock,
  ): ReadingInteractionTarget | null {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName) {
      return null;
    }

    const chapterId = block.ownerContext.chapterId || selection.chapter.chapter_id;
    const sectionId = block.ownerContext.sectionId || selection.section.section_id;
    const taskUnitId = block.ownerContext.taskUnitId;
    const contentBlockId = block.block_id;
    const documentTitle = layout?.title?.trim() || trimmedDocName;
    const chapterTitle = selection.chapter.title?.trim() || chapterId;
    const sectionTitle = selection.section.title?.trim() || sectionId;
    const taskUnitTitle = taskUnitId;
    const blockTitle = contentBlockId;

    return {
      targetLevel: "content_block",
      docName: trimmedDocName,
      documentId: layout?.document_id || block.ownerContext.documentId,
      chapterId,
      sectionId,
      parentChapterId: chapterId,
      parentSectionId: sectionId,
      taskUnitId,
      contentBlockId,
      displayTitle: blockTitle,
      breadcrumb: [
        documentTitle,
        chapterTitle,
        sectionTitle,
        taskUnitTitle,
        blockTitle,
      ],
      sourceHash: block.ownerContext.sourceHash || undefined,
    };
  }

  function selectDocumentInteraction(kind: ReadingInteractionKind) {
    selectReadingInteraction(documentInteractionTarget(), kind);
  }

  function selectChapterInteraction(chapter: ChapterLayout, kind: ReadingInteractionKind) {
    selectReadingInteraction(chapterInteractionTarget(chapter), kind);
  }

  function selectSectionInteraction(
    chapter: ChapterLayout,
    section: SectionLayout,
    kind: ReadingInteractionKind,
  ) {
    selectReadingInteraction(sectionInteractionTarget(chapter, section), kind);
  }

  function selectTaskUnitInteraction(
    selection: SectionSelection,
    group: ReaderContentGroup,
    kind: ReadingInteractionKind,
  ) {
    selectReadingInteraction(taskUnitInteractionTarget(selection, group), kind);
  }

  function selectContentBlockInteraction(
    selection: SectionSelection,
    block: ReaderContentBlock,
    kind: ReadingInteractionKind,
  ) {
    selectReadingInteraction(contentBlockInteractionTarget(selection, block), kind);
  }

  function inlineInsightForTarget(target: ReadingInteractionTarget | null) {
    if (!target) {
      return null;
    }

    try {
      return openInlineInsightsByTarget[buildReadingInteractionTargetKey(target)] || null;
    } catch {
      return null;
    }
  }

  function getChapterInlineInsight(chapter: ChapterLayout) {
    return inlineInsightForTarget(chapterInteractionTarget(chapter));
  }

  function getSectionInlineInsight(chapter: ChapterLayout, section: SectionLayout) {
    return inlineInsightForTarget(sectionInteractionTarget(chapter, section));
  }

  function getTaskUnitInlineInsight(
    selection: SectionSelection,
    group: ReaderContentGroup,
  ) {
    return inlineInsightForTarget(taskUnitInteractionTarget(selection, group));
  }

  function getContentBlockInlineInsight(
    selection: SectionSelection,
    block: ReaderContentBlock,
  ) {
    return inlineInsightForTarget(contentBlockInteractionTarget(selection, block));
  }

  function notifyInteractionResult(
    response: { envelope: { status: string; reason?: string | null } },
    requestId: number,
    successMessage: string,
  ) {
    if (!isLatestInteractionRequest(requestId)) return false;
    const successful = ["completed", "question_generated", "answer_submitted"].includes(response.envelope.status);
    showNotification(successful ? successMessage : response.envelope.reason || response.envelope.status,
      successful ? "success" : "warning");
    return successful;
  }

  function generateInlineInsight(targetKey: string) {
    const target = openInlineInsightsByTarget[targetKey]?.target;
    if (target) selectReadingInteraction(target, "insight", true);
  }

  async function refreshInlineInsight(targetKey: string) {
    const insight = openInlineInsightsByTarget[targetKey];
    if (!insight) {
      return;
    }

    const requestId = beginInlineInsightRequest(targetKey, "refreshing");
    try {
      const response = await readingInteractionService.refreshInsight(
        toReadingInteractionTargetRequest(insight.target),
      );
      applyInlineInsightState(
        targetKey,
        mapAnalysisInteractionResponseToInsightViewState(insight.target, response),
        requestId,
      );
      notifyInteractionResult(response, requestId, "Insight refreshed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failInlineInsightRequest(targetKey, requestId, message);
      showNotification(message, "error");
    }
  }

  const selectedQuiz =
    selectedInteraction?.kind === "quiz"
      ? quizStateByTarget[selectedInteraction.targetKey] || null
      : null;
  const selectedCriticalThinking =
    selectedInteraction?.kind === "critical_thinking"
      ? criticalThinkingStateByTarget[selectedInteraction.targetKey] || null
      : null;

  useEffect(() => {
    setCriticalThinkingDraftDirty(false);
  }, [selectedCriticalThinking?.targetKey]);

  function updateCriticalThinkingDraftAnswer(targetKey: string, draftAnswer: string) {
    setCriticalThinkingDraftAnswersByTarget((current) => ({
      ...current,
      [targetKey]: draftAnswer,
    }));
  }

  function generateQuiz(targetKey: string) {
    const target = quizStateByTarget[targetKey]?.target;
    if (target) selectReadingInteraction(target, "quiz", true);
  }

  async function refreshQuiz(targetKey: string) {
    const quiz = quizStateByTarget[targetKey];
    if (!quiz) {
      return;
    }

    const requestId = beginQuizRequest(targetKey, "refreshing");
    try {
      const response = await readingInteractionService.refreshQuiz(
        toReadingInteractionTargetRequest(quiz.target),
      );
      applyQuizState(
        targetKey,
        mapQuizInteractionResponseToQuizViewState(quiz.target, response),
        requestId,
      );
      notifyInteractionResult(response, requestId, "Quiz refreshed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failQuizRequest(targetKey, requestId, message);
      showNotification(message, "error");
    }
  }

  function generateCriticalThinkingQuestion(targetKey: string) {
    const target = criticalThinkingStateByTarget[targetKey]?.target;
    if (target) selectReadingInteraction(target, "critical_thinking", true);
  }

  async function submitCriticalThinkingAnswer(targetKey: string) {
    const session = criticalThinkingStateByTarget[targetKey];
    const answer = criticalThinkingDraftAnswersByTarget[targetKey]?.trim() || "";
    const sessionId = session?.metadata?.sessionId;
    if (!session || !sessionId) {
      showNotification("Generate a critical-thinking question before submitting.", "warning");
      return;
    }
    if (!answer) {
      showNotification("Write an answer before submitting.", "warning");
      return;
    }

    const requestId = beginCriticalThinkingRequest(targetKey, "submitting");
    try {
      const response = await readingInteractionService.submitCriticalThinkingAnswer(
        toReadingInteractionTargetRequest(session.target),
        {
          sessionId,
          answer,
        },
      );
      applyCriticalThinkingState(
        targetKey,
        mapCriticalThinkingSessionResponseToViewState(session.target, response),
        requestId,
      );
      if (!notifyInteractionResult(response, requestId, "Critical-thinking answer submitted.")) return;
      setCriticalThinkingDraftAnswersByTarget((current) => ({
        ...current,
        [targetKey]: "",
      }));
      setCriticalThinkingDraftDirty(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failCriticalThinkingRequest(targetKey, requestId, message);
      showNotification(message, "error");
    }
  }

  async function retryCriticalThinkingEvaluation(targetKey: string) {
    const session = criticalThinkingStateByTarget[targetKey];
    const sessionId = session?.metadata?.sessionId;
    if (!session || !sessionId) {
      showNotification("A submitted critical-thinking session is required for retry.", "warning");
      return;
    }

    const requestId = beginCriticalThinkingRequest(targetKey, "retrying");
    try {
      const response = await readingInteractionService.retryCriticalThinkingEvaluation(
        toReadingInteractionTargetRequest(session.target),
        {
          sessionId,
        },
      );
      applyCriticalThinkingState(
        targetKey,
        mapCriticalThinkingSessionResponseToViewState(session.target, response),
        requestId,
      );
      notifyInteractionResult(response, requestId, "Critical-thinking evaluation retried.");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failCriticalThinkingRequest(targetKey, requestId, message);
      showNotification(message, "error");
    }
  }

  function closeReadingInteractionDrawer() {
    // Closing the drawer must not discard a draft before non-destructive enhancement.
    setCriticalThinkingDraftDirty(false);
    clearSelectedInteraction();
  }

  async function editToc() {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || layoutStatus !== "success" || !layout) {
      return;
    }
    setLayoutStatus("loading");
    const requestId = ++layoutRequestIdRef.current;
    try {
      const nextLayout = await taskLayoutService.fetchTaskLayout(trimmedDocName, {
        includeAnchorPageEvidence: true,
      });
      if (requestId !== layoutRequestIdRef.current) return;
      setLayout(nextLayout);
      layoutScopeRef.current = { docName: trimmedDocName, layout: nextLayout };
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
      setLayoutStatus("success");
      navigate(`/documents/${encodeURIComponent(trimmedDocName)}/toc-edit`);
    } catch (error) {
      if (requestId !== layoutRequestIdRef.current) return;
      setLayoutStatus("success");
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    }
  }

  function backToReader(routeDocName: string) {
    navigate(`/documents/${encodeURIComponent(routeDocName)}`);
  }

  async function commitTocSuccess(routeDocName: string) {
    if (activeDocumentRef.current !== routeDocName) return;
    resetInteractions();
    setCriticalThinkingDraftAnswersByTarget({});
    resetContent();
    layoutScopeRef.current = null;
    if (!await reloadLayout(routeDocName)) return;
    navigate(`/documents/${encodeURIComponent(routeDocName)}`);
  }

  return (
    <Box className="app-shell reader-app-shell">
      <SmartCleanupDialog open={cleanupOpen} docName={docName} onClose={() => setCleanupOpen(false)}
        onReadingReady={refreshReadyReading} />
      <EnhancementDialog open={enhancementOpen} docName={docName} tocHint={repairTocHint}
        onSmartCleanup={() => { setEnhancementOpen(false); closeNotification(); setCleanupOpen(true); }}
        onReadingReady={refreshReadyReading}
        onStructureChanged={clearEnhancedDocument} onTitlesChanged={refreshEnhancedTitles}
        onClose={() => setEnhancementOpen(false)} onCommitted={reloadEnhancedDocument}
        onPrepare={fileManagement.openLibrary}
        onPrepareLayout={async (name, signal) => {
          if (activeDocumentRef.current !== name) throw new Error("Document changed.");
          const requestId = layoutRequestIdRef.current;
          try { await taskLayoutService.prepareTaskLayout(name, { layoutOnly: true, signal }); }
          catch (error) {
            if (signal?.aborted && requestId === layoutRequestIdRef.current && activeDocumentRef.current === name) setLayoutStatus("error");
            throw error;
          }
        }} />
      <Paper component="header" elevation={0} className="topbar reader-topbar">
        <BookSearchView
          value={docName}
          inputValue={searchInput}
          options={documentOptions}
          loading={layoutStatus === "loading"}
          searching={documentSearchLoading}
          onChange={setSearchInput}
          onOpen={loadDocumentOptions}
          onSelect={selectDocument}
        />
        <Box className="repair-controls structure-repair-controls">
          <Button
            className="file-management-trigger"
            startIcon={<LibraryBooksOutlinedIcon />}
            variant="outlined"
            onClick={fileManagement.openLibrary}
          >
            Library
          </Button>
          <Button
            className="structure-repair-trigger"
            id="repair-menu-button"
            aria-controls={repairMenuOpen ? "repair-menu" : undefined}
            aria-haspopup="menu"
            aria-expanded={repairMenuOpen ? "true" : undefined}
            disabled={(!canRepair && !canCheckCleanup) || cleanupOpen || enhancementOpen}
            startIcon={
              activeRepairMode ? (
                <CircularProgress color="inherit" size={16} />
              ) : (
                <BuildOutlinedIcon />
              )
            }
            endIcon={<ExpandMoreIcon />}
            variant="outlined"
            onClick={openRepairMenu}
          >
            Repairs: {currentRepairMode === "llm_enhanced" ? "LLM" : "Common"}
          </Button>
          <Menu
            className="structure-repair-menu"
            id="repair-menu"
            disableRestoreFocus
            anchorEl={repairMenuAnchor}
            open={repairMenuOpen}
            onClose={closeRepairMenu}
            MenuListProps={{ "aria-labelledby": "repair-menu-button" }}
          >
            <Box sx={{ px: 2, width: 280 }}><TocHintControl value={repairTocHint} disabled={!canRepair} onChange={setRepairTocHint} /></Box>
            <MenuItem
              className="structure-repair-option structure-repair-option-common"
              disabled={!canRepair}
              selected={currentRepairMode === "common"}
              onClick={() => selectRepairMode("common")}
            >
              <ListItemIcon className="structure-repair-option-icon">
                {currentRepairMode === "common" ? <CheckIcon fontSize="small" /> : null}
              </ListItemIcon>
              <ListItemText className="structure-repair-option-label">Common</ListItemText>
            </MenuItem>
            <MenuItem
              className="structure-repair-option structure-repair-option-llm"
              disabled={!canRepair}
              selected={currentRepairMode === "llm_enhanced"}
              onClick={() => selectRepairMode("llm_enhanced")}
            >
              <ListItemIcon className="structure-repair-option-icon">
                {currentRepairMode === "llm_enhanced" ? <CheckIcon fontSize="small" /> : null}
              </ListItemIcon>
              <ListItemText className="structure-repair-option-label">LLM enhance</ListItemText>
            </MenuItem>
            <MenuItem className="smart-cleanup-menu-item" disabled={!canCheckCleanup || Boolean(contentEditSelection)}
              title={contentEditSelection ? "Finish editing the block before cleanup." : "Clean OCR text without changing chapter structure"}
              onClick={() => { closeRepairMenu(); closeNotification(); setCleanupOpen(true); }}>
              <ListItemIcon><CleaningServicesOutlinedIcon fontSize="small" /></ListItemIcon>
              <ListItemText>Smart cleanup</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
      </Paper>
      <FileManagementView
        readingSettings={readingSettings}
        canOpenDocument={fileManagement.canOpenDocument}
        openingDocName={fileManagement.openingDocName}
        openError={fileManagement.openError}
        prepareState={fileManagement.prepareState}
        onChangeUploadPdfMode={fileManagement.changeUploadPdfMode}
        onChangeUploadTocHint={fileManagement.changeUploadTocHint}
        onChangePrepareTocHint={fileManagement.changePrepareTocHint}
        onChangePreparePdfMode={fileManagement.changePreparePdfMode}
        onRequestPrepare={fileManagement.requestPrepareDocument}
        onClosePrepare={fileManagement.closePrepareDocument}
        onPrepareExisting={() => { void fileManagement.prepareExistingDocument(); }}
        activeFilterCount={fileManagement.activeFilterCount}
        deleteState={fileManagement.deleteState}
        error={fileManagement.error}
        filterOptions={fileManagement.filterOptions}
        filters={fileManagement.filters}
        items={fileManagement.items}
        open={fileManagement.open}
        recentItems={fileManagement.recentItems}
        renameState={fileManagement.renameState}
        status={fileManagement.status}
        total={fileManagement.total}
        visibleTotal={fileManagement.visibleTotal}
        uploadConflict={fileManagement.uploadConflict}
        uploadDraft={fileManagement.uploadDraft}
        uploadError={fileManagement.uploadError}
        uploadResult={fileManagement.uploadResult}
        uploadStatus={fileManagement.uploadStatus}
        onCancelDelete={fileManagement.cancelDeleteDocument}
        onCancelDuplicateUpload={fileManagement.cancelDuplicateUpload}
        onCancelRename={fileManagement.cancelRenameDocument}
        onCancelUpload={fileManagement.cancelUploadDraft}
        onChangeFilter={fileManagement.changeFilter}
        onChangeRenameDraft={fileManagement.changeRenameDraft}
        onConfirmDelete={() => {
          void fileManagement.confirmDeleteDocument();
        }}
        onConfirmDuplicateUpload={fileManagement.confirmDuplicateUpload}
        onConfirmRename={() => {
          void fileManagement.confirmRenameDocument();
        }}
        onClose={fileManagement.closeLibrary}
        onOpenDocument={(nextDocName) => {
          void openManagedDocument(nextDocName);
        }}
        onPrepareUploadedDocument={() => {
          void fileManagement.prepareUploadedDocument();
        }}
        onCancelUploadedPreparation={fileManagement.cancelUploadedPreparation}
        onRefresh={() => {
          void fileManagement.loadDocuments();
        }}
        onRequestDelete={fileManagement.requestDeleteDocument}
        onRequestRename={fileManagement.requestRenameDocument}
        onResetFilters={fileManagement.resetFilters}
        onRetryUpload={fileManagement.retryUpload}
        onSelectUploadFile={fileManagement.selectUploadFile}
        onUpload={() => {
          void fileManagement.uploadSelectedFile();
        }}
      />

      <ReaderWorkspace>
        <Paper
          component="aside"
          id="document-hierarchy-pane"
          elevation={0}
          className="navigation-pane hierarchy-navigation-pane"
          aria-label="Document hierarchy"
          aria-busy={layoutStatus === "loading"}
        >
          <HierarchyNavigationView
            layout={layout}
            selectedSectionId={selectedSection?.section.section_id || null}
            status={layoutStatus}
            error={layoutError ? "Open the error notification for details." : ""}
            canEditToc={layoutStatus === "success" && Boolean(layout)}
            onSelectSection={selectSection}
            onEditToc={editToc}
            onSelectDocumentInteraction={selectDocumentInteraction}
            onSelectChapterInteraction={selectChapterInteraction}
            onSelectSectionInteraction={selectSectionInteraction}
            documentInlineInsight={inlineInsightForTarget(documentInteractionTarget())}
            getChapterInlineInsight={getChapterInlineInsight}
            getSectionInlineInsight={getSectionInlineInsight}
            onDismissInlineInsight={closeInlineInsight}
            onGenerateInlineInsight={generateInlineInsight}
            onRefreshInlineInsight={refreshInlineInsight}
          />
        </Paper>
        <ReaderContentPane busy={contentStatus === "loading"} scopeKey={docName}>
          {layoutStatus === "success" && <ReadingContentActions key={docName} docName={docName} refreshKey={readingRefresh} selection={contentEditSelection} onReviewState={setSourceNeedsReview}
            onNavigate={async (block, snapshot, signal, assertCurrent) => {
              if (!confirmReadingNavigation() || !closeSummaryNote()) throw new Error("Navigation cancelled.");
              await navigateToBlock(block, snapshot, signal, assertCurrent);
              if (!signal.aborted && activeDocumentRef.current === docName) navigate(`/documents/${encodeURIComponent(docName)}`);
            }}
            onRefreshBlocks={async (refresh, signal) => {
              if (signal.aborted || activeDocumentRef.current !== docName || layoutScopeRef.current?.layout !== layout) throw new Error("Document changed.");
              if (refresh.body_changed || refresh.generated_invalidated || refresh.summaries_invalidated) resetInteractions();
              translations.invalidate(refresh);
              setSummaryNoteSelection(current => ((refresh.body_changed || refresh.summaries_invalidated || refresh.generated_invalidated) && current?.kind === "summary")
                || (refresh.cleared_block_id && current?.target.content_block_id === refresh.cleared_block_id)
                || (current?.target.content_block_id && refresh.cleared_block_ids?.includes(current.target.content_block_id)) ? null : current);
              await refreshAffectedBlocks(refresh, signal);
              if (signal.aborted || activeDocumentRef.current !== docName || layoutScopeRef.current?.layout !== layout) throw new Error("Document changed.");
              closeNotification();
            }}
            onCloseSelection={() => setContentEditSelection(null)} onChanged={async (signal, clearedBlockId) => {
              if (signal?.aborted || activeDocumentRef.current !== docName) throw new Error("Document changed.");
              resetInteractions(); setSummaryNoteSelection(current => current?.kind === "summary"
                || (clearedBlockId && current?.target.content_block_id === clearedBlockId) ? null : current);
              const requestId = ++layoutRequestIdRef.current;
              const next = await taskLayoutService.fetchTaskLayout(docName, { signal });
              if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== docName) throw new Error("Document changed.");
              await refreshFromLayout(next, signal);
              if (signal?.aborted || requestId !== layoutRequestIdRef.current || activeDocumentRef.current !== docName) throw new Error("Document changed.");
              setLayout(next); layoutScopeRef.current = { docName, layout: next };
              closeNotification();
            }} />}
          <Routes>
            <Route
              path="/"
              element={
                <ReaderContentView
                  readingMode={{ docName, suspended: fileManagement.open || enhancementOpen || cleanupOpen || !!contentEditSelection, revision: readingRefresh, memory: learningMemory, onSettingsOpenChange: setReaderModeSettingsOpen, onBeforeSettingsOpen: () => confirmReadingNavigation() && closeSummaryNote() }}
                  getInlineTranslationState={translations.getInlineTranslationState}
                  onVisibleTranslationBlocksChange={translations.onVisibleTranslationBlocksChange}
                  navigationTarget={navigationTarget}
                  onTranslateBlock={(block, mode, trigger) => {
                    if (!confirmReadingNavigation() || !closeSummaryNote()) return;
                    closeNotification(); translations.open(block, mode, trigger);
                  }}
                  onEditContentBlock={(block,action) => setContentEditSelection({blockId: block.block_id,action})}
                  layoutStatus={layoutStatus}
                  contentStatus={contentStatus}
                  selectedSection={selectedSection}
                  contentBlocks={contentBlocks}
                  contentGroups={contentGroups}
                  error={contentError}
                  onRetryContent={() => { void retryContent(); }}
                  getContentBlockInlineInsight={getContentBlockInlineInsight}
                  getTaskUnitInlineInsight={getTaskUnitInlineInsight}
                  onDismissInlineInsight={closeInlineInsight}
                  onGenerateInlineInsight={generateInlineInsight}
                  onRefreshInlineInsight={refreshInlineInsight}
                  onSelectContentBlockInteraction={selectContentBlockInteraction}
                  onSelectTaskUnitInteraction={selectTaskUnitInteraction}
                />
              }
            />
            <Route
              path="/documents/:routeDocName"
              element={
                <ReaderContentView
                  readingMode={{ docName, suspended: fileManagement.open || enhancementOpen || cleanupOpen || !!contentEditSelection, revision: readingRefresh, memory: learningMemory, onSettingsOpenChange: setReaderModeSettingsOpen, onBeforeSettingsOpen: () => confirmReadingNavigation() && closeSummaryNote() }}
                  getInlineTranslationState={translations.getInlineTranslationState}
                  onVisibleTranslationBlocksChange={translations.onVisibleTranslationBlocksChange}
                  navigationTarget={navigationTarget}
                  onTranslateBlock={(block, mode, trigger) => {
                    if (!confirmReadingNavigation() || !closeSummaryNote()) return;
                    closeNotification(); translations.open(block, mode, trigger);
                  }}
                  onEditContentBlock={(block,action) => setContentEditSelection({blockId: block.block_id,action})}
                  layoutStatus={layoutStatus}
                  contentStatus={contentStatus}
                  selectedSection={selectedSection}
                  contentBlocks={contentBlocks}
                  contentGroups={contentGroups}
                  error={contentError}
                  onRetryContent={() => { void retryContent(); }}
                  getContentBlockInlineInsight={getContentBlockInlineInsight}
                  getTaskUnitInlineInsight={getTaskUnitInlineInsight}
                  onDismissInlineInsight={closeInlineInsight}
                  onGenerateInlineInsight={generateInlineInsight}
                  onRefreshInlineInsight={refreshInlineInsight}
                  onSelectContentBlockInteraction={selectContentBlockInteraction}
                  onSelectTaskUnitInteraction={selectTaskUnitInteraction}
                />
              }
            />
            <Route
              path="/documents/:routeDocName/toc-edit"
              element={
                <TocEditorRoutePane
                  layout={layout}
                  onBackToReader={backToReader}
                  onCommitSuccess={commitTocSuccess}
                />
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ReaderContentPane>
      </ReaderWorkspace>
      <ReadingInteractionsView
        drawerSelection={selectedInteraction?.surface === "drawer" ? selectedInteraction : null}
        commandSlot={
          selectedQuiz ? (
            <QuizDrawerCommands
              quiz={selectedQuiz}
              onGenerate={generateQuiz}
              onRefresh={refreshQuiz}
            />
          ) : selectedCriticalThinking ? (
            <CriticalThinkingDrawerCommands
              canSubmitAnswer={Boolean(
                criticalThinkingDraftAnswersByTarget[
                  selectedCriticalThinking.targetKey
                ]?.trim(),
              )}
              session={selectedCriticalThinking}
              onGenerateQuestion={generateCriticalThinkingQuestion}
              onRetryEvaluation={retryCriticalThinkingEvaluation}
              onSubmitAnswer={submitCriticalThinkingAnswer}
            />
          ) : undefined
        }
        contentSlot={
          selectedQuiz ? (
            <QuizDrawerContent quiz={selectedQuiz} />
          ) : selectedCriticalThinking ? (
            <CriticalThinkingDrawerContent
              draftAnswer={criticalThinkingDraftAnswersByTarget[selectedCriticalThinking.targetKey] || ""}
              session={selectedCriticalThinking}
              onDraftAnswerChange={updateCriticalThinkingDraftAnswer}
              onDraftDirtyChange={setCriticalThinkingDraftDirty}
            />
          ) : undefined
        }
        onCloseDrawer={closeReadingInteractionDrawer}
      />
      {summaryNoteSelection && <SummaryNoteDrawer
        generationTasks={generationTasks}
        key={`${summaryNoteSelection.kind}:${JSON.stringify(summaryNoteSelection.target)}`}
        selection={summaryNoteSelection} dirtyRef={noteDirtyRef} onClose={() => { closeSummaryNote(); }} />}
      <BlockTranslationView controller={translations} />
      <AppNotification notification={{ ...notification, open: notification.open && !readerModeSettingsOpen && !readingSettings.target && !enhancementOpen && !cleanupOpen && !translations.notification.open && !translations.surface }} onClose={closeNotification} />
    </Box>
  );
}

function TocEditorRoutePane({
  layout,
  onBackToReader,
  onCommitSuccess,
}: {
  layout: DocumentTaskLayout | null;
  onBackToReader: (docName: string) => void;
  onCommitSuccess: (docName: string) => Promise<void>;
}) {
  const { routeDocName = "" } = useParams();
  const docName = decodeURIComponent(routeDocName);

  return (
    <TocEditorView
      docName={docName}
      layout={layout}
      onBackToReader={() => onBackToReader(docName)}
      onCommitSuccess={() => onCommitSuccess(docName)}
    />
  );
}
