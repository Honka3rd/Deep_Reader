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
import { useMemo, useRef, useState } from "react";
import {
  fetchDocumentList,
  prepareTaskLayout,
  fetchTaskLayout,
  fetchTaskUnitContent,
  reparseDocumentStructure,
} from "./api/client";
import { DocumentSearch } from "./components/DocumentSearch";
import { HierarchyNavigation } from "./components/HierarchyNavigation";
import { ReaderContent } from "./components/ReaderContent";
import type {
  ContentBlock,
  DocumentListItem,
  DocumentTaskLayout,
  RequestStatus,
  SectionSelection,
  StructureParserMode,
} from "./types/api";

export default function App() {
  const [docName, setDocName] = useState("");
  const [layout, setLayout] = useState<DocumentTaskLayout | null>(null);
  const [selectedSection, setSelectedSection] = useState<SectionSelection | null>(null);
  const [contentBlocks, setContentBlocks] = useState<ContentBlock[]>([]);
  const [layoutStatus, setLayoutStatus] = useState<RequestStatus>("initial");
  const [contentStatus, setContentStatus] = useState<RequestStatus>("initial");
  const [repairStatus, setRepairStatus] = useState<RequestStatus>("initial");
  const [layoutError, setLayoutError] = useState("");
  const [contentError, setContentError] = useState("");
  const [repairError, setRepairError] = useState("");
  const [currentRepairMode, setCurrentRepairMode] = useState<StructureParserMode | null>(null);
  const [activeRepairMode, setActiveRepairMode] = useState<StructureParserMode | null>(null);
  const [repairMenuAnchor, setRepairMenuAnchor] = useState<HTMLElement | null>(null);
  const [backendDocuments, setBackendDocuments] = useState<DocumentListItem[]>([]);
  const [documentSearchLoading, setDocumentSearchLoading] = useState(false);
  const contentRequestIdRef = useRef(0);

  const documentOptions = useMemo(
    () => backendDocuments.map((item) => item.doc_name),
    [backendDocuments],
  );

  function resolveLayoutParserMode(nextLayout: DocumentTaskLayout): StructureParserMode {
    return nextLayout.parse_provenance?.effective_parser_mode === "llm_enhanced"
      ? "llm_enhanced"
      : "common";
  }

  async function loadDocumentOptions() {
    if (documentSearchLoading) {
      return;
    }
    setDocumentSearchLoading(true);
    try {
      const response = await fetchDocumentList("", 200);
      setBackendDocuments(response.items);
    } catch {
      setBackendDocuments([]);
    } finally {
      setDocumentSearchLoading(false);
    }
  }

  async function loadLayout() {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || !documentOptions.includes(trimmedDocName)) {
      setLayoutStatus("error");
      setLayoutError("Select a document returned by the document list API");
      return;
    }

    setLayout(null);
    setSelectedSection(null);
    setContentBlocks([]);
    contentRequestIdRef.current += 1;
    setLayoutError("");
    setContentError("");
    setContentStatus("initial");
    setLayoutStatus("loading");

    try {
      const nextLayout = await prepareTaskLayout(trimmedDocName);
      setLayout(nextLayout);
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
      setLayoutStatus("success");
    } catch (error) {
      setLayoutStatus("error");
      setLayoutError(error instanceof Error ? error.message : String(error));
    }
  }

  async function selectSection(selection: SectionSelection) {
    const requestId = contentRequestIdRef.current + 1;
    contentRequestIdRef.current = requestId;
    setSelectedSection(selection);
    setContentBlocks([]);
    setContentError("");
    if (selection.taskUnits.length === 0) {
      setContentStatus("empty");
      return;
    }

    try {
      setContentStatus("loading");
      const taskUnitContents = await Promise.all(
        selection.taskUnits.map((taskUnit) =>
          fetchTaskUnitContent(docName.trim(), taskUnit.unit_id),
        ),
      );
      const nextBlocks = taskUnitContents.flatMap((taskUnitContent) =>
        taskUnitContent.content_blocks || [],
      );
      if (contentRequestIdRef.current !== requestId) {
        return;
      }
      setContentBlocks(nextBlocks);
      const hasBlocks = nextBlocks.length > 0;
      setContentStatus(hasBlocks ? "success" : "empty");
    } catch (error) {
      if (contentRequestIdRef.current !== requestId) {
        return;
      }
      setContentStatus("error");
      setContentError(error instanceof Error ? error.message : String(error));
    }
  }

  async function repairStructure(parserMode: StructureParserMode) {
    const trimmedDocName = docName.trim();
    if (!trimmedDocName || !layout) {
      return;
    }

    setRepairStatus("loading");
    setActiveRepairMode(parserMode);
    setRepairError("");
    setLayoutError("");
    setSelectedSection(null);
    setContentBlocks([]);
    setContentError("");
    setContentStatus("initial");
    contentRequestIdRef.current += 1;

    try {
      const result = await reparseDocumentStructure(trimmedDocName, parserMode);
      if (!result.success) {
        throw new Error(result.error || "Structure repair failed");
      }
      setLayoutStatus("loading");
      const nextLayout = await fetchTaskLayout(trimmedDocName);
      setLayout(nextLayout);
      setLayoutStatus("success");
      setRepairStatus("success");
      setCurrentRepairMode(resolveLayoutParserMode(nextLayout));
    } catch (error) {
      setRepairStatus("error");
      setLayoutStatus(layout ? "success" : "error");
      const message = error instanceof Error ? error.message : String(error);
      setRepairError(message);
    } finally {
      setActiveRepairMode(null);
    }
  }

  const unitCount =
    layout?.chapters?.reduce(
      (chapterTotal, chapter) =>
        chapterTotal +
        (chapter.sections || []).reduce(
          (sectionTotal, section) => sectionTotal + (section.task_units || []).length,
          0,
        ),
      0,
    ) || 0;
  const sectionCount =
    layout?.chapters?.reduce(
      (chapterTotal, chapter) => chapterTotal + (chapter.sections || []).length,
      0,
    ) || 0;
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

  return (
    <Box className="app-shell">
      <Paper component="header" elevation={0} className="topbar">
        <DocumentSearch
          value={docName}
          options={documentOptions}
          loading={layoutStatus === "loading"}
          searching={documentSearchLoading}
          onChange={setDocName}
          onOpen={loadDocumentOptions}
          onLoad={loadLayout}
        />
        <Box className="repair-controls">
          <Button
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
            id="repair-menu"
            anchorEl={repairMenuAnchor}
            open={repairMenuOpen}
            onClose={closeRepairMenu}
            MenuListProps={{ "aria-labelledby": "repair-menu-button" }}
          >
            <MenuItem
              selected={currentRepairMode === "common"}
              onClick={() => selectRepairMode("common")}
            >
              <ListItemIcon>
                {currentRepairMode === "common" ? <CheckIcon fontSize="small" /> : null}
              </ListItemIcon>
              <ListItemText>Common</ListItemText>
            </MenuItem>
            <MenuItem
              selected={currentRepairMode === "llm_enhanced"}
              onClick={() => selectRepairMode("llm_enhanced")}
            >
              <ListItemIcon>
                {currentRepairMode === "llm_enhanced" ? <CheckIcon fontSize="small" /> : null}
              </ListItemIcon>
              <ListItemText>LLM</ListItemText>
            </MenuItem>
          </Menu>
        </Box>
        <Typography aria-live="polite" className="status-region" color="text.secondary">
          {repairStatus === "loading" ? "Repairing structure" : ""}
          {repairStatus === "error" ? `Repair failed: ${repairError}` : ""}
          {repairStatus !== "loading" && repairStatus !== "error" && layoutStatus === "success"
            ? `${sectionCount} sections / ${unitCount} internal units`
            : ""}
          {layoutStatus === "loading" ? "Loading document" : ""}
          {layoutStatus === "error" ? "Document load failed" : ""}
        </Typography>
      </Paper>

      <Box component="main" className="reader-layout">
        <Paper
          component="aside"
          elevation={0}
          className="navigation-pane"
          aria-label="Document hierarchy"
          aria-busy={layoutStatus === "loading"}
        >
          <HierarchyNavigation
            layout={layout}
            selectedSectionId={selectedSection?.section.section_id || null}
            status={layoutStatus}
            error={layoutError}
            onSelectSection={selectSection}
          />
        </Paper>
        <Paper
          component="section"
          elevation={0}
          className="content-pane"
          aria-label="Reading content"
          aria-busy={contentStatus === "loading"}
        >
          <ReaderContent
            layoutStatus={layoutStatus}
            contentStatus={contentStatus}
            selectedSection={selectedSection}
            contentBlocks={contentBlocks}
            error={contentError}
          />
        </Paper>
      </Box>
    </Box>
  );
}
