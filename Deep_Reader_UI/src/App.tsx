import BuildOutlinedIcon from "@mui/icons-material/BuildOutlined";
import CheckIcon from "@mui/icons-material/Check";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
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
import { useEffect, useRef, useState } from "react";
import {
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  BookSearchView,
  useBookSearchController,
} from "./features/book-search";
import {
  HierarchyNavigationView,
  resolveLayoutParserMode,
} from "./features/hierarchy-navigation";
import {
  ReaderContentView,
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
  QuizDrawerCommands,
  QuizDrawerContent,
  ReadingInteractionsView,
  toReadingInteractionTargetRequest,
  type ReadingInteractionKind,
  type ReadingInteractionMenuSelection,
  type ReadingInteractionTarget,
  useReadingInteractionMenuController,
} from "./features/reading-interactions";

export default function App() {
  const navigate = useNavigate();
  const [layout, setLayout] = useState<DocumentTaskLayout | null>(null);
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
  const [criticalThinkingDraftDirty, setCriticalThinkingDraftDirty] = useState(false);
  const layoutRequestIdRef = useRef(0);
  const {
    docName,
    setDocName,
    documentOptions,
    searching: documentSearchLoading,
    loadDocumentOptions,
  } = useBookSearchController();
  const {
    selectedSection,
    contentBlocks,
    contentGroups,
    contentStatus,
    contentError,
    selectSection,
    resetContent,
  } = useReaderContentController({ docName });
  const {
    openInlineInsightsByTarget,
    quizStateByTarget,
    criticalThinkingStateByTarget,
    selectedInteraction,
    selectInteractionAction,
    beginInlineInsightRequest,
    applyInlineInsightState,
    failInlineInsightRequest,
    clearSelectedInteraction,
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
    const nextLayout = await taskLayoutService.fetchTaskLayout(trimmedDocName);
    setLayout(nextLayout);
    setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
    setLayoutStatus("success");
  }

  async function loadExistingOrPrepareTaskLayout(
    trimmedDocName: string,
  ): Promise<DocumentTaskLayout> {
    try {
      return await taskLayoutService.fetchTaskLayout(trimmedDocName);
    } catch {
      return taskLayoutService.prepareTaskLayout(trimmedDocName);
    }
  }

  async function loadLayoutForDocument(nextDocName: string) {
    const trimmedDocName = nextDocName.trim();
    const requestId = layoutRequestIdRef.current + 1;
    layoutRequestIdRef.current = requestId;

    if (!trimmedDocName || !documentOptions.includes(trimmedDocName)) {
      const message = "Select a document returned by the document list API";
      setLayoutStatus("error");
      setLayoutError(message);
      showNotification(message, "error");
      return;
    }

    setLayout(null);
    resetContent();
    setLayoutError("");
    setLayoutStatus("loading");

    try {
      const nextLayout = await loadExistingOrPrepareTaskLayout(trimmedDocName);
      if (layoutRequestIdRef.current !== requestId) {
        return;
      }
      setLayout(nextLayout);
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
      setLayoutStatus("success");
      navigate(`/documents/${encodeURIComponent(trimmedDocName)}`);
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

  function selectDocument(nextDocName: string) {
    setDocName(nextDocName);
    void loadLayoutForDocument(nextDocName);
  }

  async function repairStructure(parserMode: StructureParserMode) {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || !layout) {
      return;
    }

    setRepairStatus("loading");
    setActiveRepairMode(parserMode);
    setLayoutError("");
    resetContent();

    try {
      const result = await structureRepairService.reparseDocumentStructure(
        trimmedDocName,
        parserMode,
      );
      if (!result.success) {
        throw new Error(result.error || "Structure repair failed");
      }
      setLayoutStatus("loading");
      await reloadLayout(trimmedDocName);
      setRepairStatus("success");
      showNotification("Structure repair completed. Layout reloaded.", "success");
    } catch (error) {
      setRepairStatus("error");
      setLayoutStatus(layout ? "success" : "error");
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    } finally {
      setActiveRepairMode(null);
    }
  }

  const canRepair = layoutStatus === "success" && Boolean(layout) && repairStatus !== "loading";
  const repairMenuOpen = Boolean(repairMenuAnchor);

  function openRepairMenu(event: MouseEvent<HTMLButtonElement>) {
    setRepairMenuAnchor(event.currentTarget);
  }

  function closeRepairMenu() {
    setRepairMenuAnchor(null);
  }

  function selectRepairMode(parserMode: StructureParserMode) {
    closeRepairMenu();
    void repairStructure(parserMode);
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

  async function readInlineInsight(selection: ReadingInteractionMenuSelection) {
    const requestId = beginInlineInsightRequest(selection.targetKey, "loading");
    try {
      const response = await readingInteractionService.readInsight(
        toReadingInteractionTargetRequest(selection.target),
      );
      applyInlineInsightState(
        selection.targetKey,
        mapAnalysisInteractionResponseToInsightViewState(selection.target, response),
        requestId,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failInlineInsightRequest(selection.targetKey, requestId, message);
      showNotification(message, "error");
    }
  }

  function selectReadingInteraction(
    target: ReadingInteractionTarget | null,
    kind: ReadingInteractionKind,
  ) {
    if (!target) {
      return;
    }

    try {
      const selection = selectInteractionAction(kind, target);
      if (selection) {
        if (selection.kind === "insight") {
          void readInlineInsight(selection);
        } else {
          notifyReadingInteractionSelection(selection);
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    }
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

  async function generateInlineInsight(targetKey: string) {
    const insight = openInlineInsightsByTarget[targetKey];
    if (!insight) {
      return;
    }

    const requestId = beginInlineInsightRequest(targetKey, "generating");
    try {
      const response = await readingInteractionService.generateInsight(
        toReadingInteractionTargetRequest(insight.target),
      );
      applyInlineInsightState(
        targetKey,
        mapAnalysisInteractionResponseToInsightViewState(insight.target, response),
        requestId,
      );
      showNotification("Insight generated.", "success");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failInlineInsightRequest(targetKey, requestId, message);
      showNotification(message, "error");
    }
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
      showNotification("Insight refreshed.", "success");
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

  function generateQuiz(targetKey: string) {
    showNotification(
      `Generate quiz for ${targetKey} requires quiz service wiring in the frontend.`,
      "info",
    );
  }

  function refreshQuiz(targetKey: string) {
    showNotification(
      `Refresh quiz for ${targetKey} requires quiz service wiring in the frontend.`,
      "info",
    );
  }

  function generateCriticalThinkingQuestion(targetKey: string) {
    showNotification(
      `Generate critical-thinking question for ${targetKey} requires critical-thinking service wiring in the frontend.`,
      "info",
    );
  }

  function submitCriticalThinkingAnswer(targetKey: string) {
    showNotification(
      `Submit critical-thinking answer for ${targetKey} requires critical-thinking service wiring in the frontend.`,
      "info",
    );
  }

  function retryCriticalThinkingEvaluation(targetKey: string) {
    showNotification(
      `Retry critical-thinking evaluation for ${targetKey} requires critical-thinking service wiring in the frontend.`,
      "info",
    );
  }

  function closeReadingInteractionDrawer() {
    if (selectedCriticalThinking && criticalThinkingDraftDirty) {
      showNotification("Unsent critical-thinking draft was discarded.", "warning");
    }
    setCriticalThinkingDraftDirty(false);
    clearSelectedInteraction();
  }

  async function editToc() {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || layoutStatus !== "success" || !layout) {
      return;
    }
    setLayoutStatus("loading");
    try {
      const nextLayout = await taskLayoutService.fetchTaskLayout(trimmedDocName, {
        includeAnchorPageEvidence: true,
      });
      setLayout(nextLayout);
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
      setLayoutStatus("success");
      navigate(`/documents/${encodeURIComponent(trimmedDocName)}/toc-edit`);
    } catch (error) {
      setLayoutStatus("success");
      const message = error instanceof Error ? error.message : String(error);
      showNotification(message, "error");
    }
  }

  function backToReader(routeDocName: string) {
    navigate(`/documents/${encodeURIComponent(routeDocName)}`);
  }

  async function commitTocSuccess(routeDocName: string) {
    await reloadLayout(routeDocName);
    resetContent();
    navigate(`/documents/${encodeURIComponent(routeDocName)}`);
  }

  return (
    <Box className="app-shell reader-app-shell">
      <Paper component="header" elevation={0} className="topbar reader-topbar">
        <BookSearchView
          value={docName}
          options={documentOptions}
          loading={layoutStatus === "loading"}
          searching={documentSearchLoading}
          onChange={setDocName}
          onOpen={loadDocumentOptions}
          onSelect={selectDocument}
        />
        <Box className="repair-controls structure-repair-controls">
          <Button
            className="structure-repair-trigger"
            id="repair-menu-button"
            aria-controls={repairMenuOpen ? "repair-menu" : undefined}
            aria-haspopup="menu"
            aria-expanded={repairMenuOpen ? "true" : undefined}
            disabled={!canRepair}
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
            anchorEl={repairMenuAnchor}
            open={repairMenuOpen}
            onClose={closeRepairMenu}
            MenuListProps={{ "aria-labelledby": "repair-menu-button" }}
          >
            <MenuItem
              className="structure-repair-option structure-repair-option-common"
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
              selected={currentRepairMode === "llm_enhanced"}
              onClick={() => selectRepairMode("llm_enhanced")}
            >
              <ListItemIcon className="structure-repair-option-icon">
                {currentRepairMode === "llm_enhanced" ? <CheckIcon fontSize="small" /> : null}
              </ListItemIcon>
              <ListItemText className="structure-repair-option-label">LLM</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
      </Paper>

      <Box component="main" className="reader-layout reader-workspace">
        <Paper
          component="aside"
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
        <Paper
          component="section"
          elevation={0}
          className="content-pane reader-content-pane"
          aria-label="Reading content"
          aria-busy={contentStatus === "loading"}
        >
          <Routes>
            <Route
              path="/"
              element={
                <ReaderContentView
                  layoutStatus={layoutStatus}
                  contentStatus={contentStatus}
                  selectedSection={selectedSection}
                  contentBlocks={contentBlocks}
                  contentGroups={contentGroups}
                  error={contentError}
                  getTaskUnitInlineInsight={getTaskUnitInlineInsight}
                  onDismissInlineInsight={closeInlineInsight}
                  onGenerateInlineInsight={generateInlineInsight}
                  onRefreshInlineInsight={refreshInlineInsight}
                  onSelectTaskUnitInteraction={selectTaskUnitInteraction}
                />
              }
            />
            <Route
              path="/documents/:routeDocName"
              element={
                <ReaderContentView
                  layoutStatus={layoutStatus}
                  contentStatus={contentStatus}
                  selectedSection={selectedSection}
                  contentBlocks={contentBlocks}
                  contentGroups={contentGroups}
                  error={contentError}
                  getTaskUnitInlineInsight={getTaskUnitInlineInsight}
                  onDismissInlineInsight={closeInlineInsight}
                  onGenerateInlineInsight={generateInlineInsight}
                  onRefreshInlineInsight={refreshInlineInsight}
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
        </Paper>
      </Box>
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
              session={selectedCriticalThinking}
              onDraftDirtyChange={setCriticalThinkingDraftDirty}
            />
          ) : undefined
        }
        onCloseDrawer={closeReadingInteractionDrawer}
      />
      <AppNotification notification={notification} onClose={closeNotification} />
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
