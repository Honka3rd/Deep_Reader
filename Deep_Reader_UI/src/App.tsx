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
  Typography,
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
  useHierarchyNavigationController,
} from "./features/hierarchy-navigation";
import {
  ReaderContentView,
  useReaderContentController,
} from "./features/reader-content";
import { TocEditorView } from "./features/toc-editor";
import {
  AppNotification,
  type AppNotificationState,
} from "./shared/components/AppNotification";
import {
  structureRepairService,
  taskLayoutService,
} from "./services";
import type {
  DocumentTaskLayout,
  RequestStatus,
  StructureParserMode,
} from "./types/api";

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
    contentStatus,
    contentError,
    selectSection,
    resetContent,
  } = useReaderContentController({ docName });
  const { sectionCount, unitCount } = useHierarchyNavigationController(layout);

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
      const nextLayout = await taskLayoutService.prepareTaskLayout(trimmedDocName);
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

  function editToc() {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || layoutStatus !== "success" || !layout) {
      return;
    }
    navigate(`/documents/${encodeURIComponent(trimmedDocName)}/toc-edit`);
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
        <Typography
          aria-live="polite"
          className="status-region reader-status-region"
          color="text.secondary"
        >
          {repairStatus === "loading" ? "Repairing structure" : ""}
          {repairStatus === "error" ? "Repair failed" : ""}
          {repairStatus !== "loading" && repairStatus !== "error" && layoutStatus === "success"
            ? `${sectionCount} sections / ${unitCount} internal units`
            : ""}
          {layoutStatus === "loading" ? "Loading document" : ""}
          {layoutStatus === "error" ? "Document load failed" : ""}
        </Typography>
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
                  error={contentError}
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
                  error={contentError}
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
