import AddIcon from "@mui/icons-material/Add";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import PlaylistAddCheckIcon from "@mui/icons-material/PlaylistAddCheck";
import SaveOutlinedIcon from "@mui/icons-material/SaveOutlined";
import { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import type {
  DocumentTaskLayout,
  ManualStructureValidationIssue,
} from "../../types/api";
import { StateView } from "../../shared/components/StateView";
import { AppNotification } from "../../shared/components/AppNotification";
import { getSelectedRange } from "./model";
import { useTocEditorController } from "./controller";

interface TocEditorViewProps {
  docName: string;
  layout: DocumentTaskLayout | null;
  onBackToReader: () => void;
  onCommitSuccess: () => Promise<void>;
}

export function TocEditorView({
  docName,
  layout,
  onBackToReader,
  onCommitSuccess,
}: TocEditorViewProps) {
  if (!layout) {
    return (
      <StateView
        className="toc-editor-guard-state"
        title="Load document first"
        detail="TOC editing requires an already loaded task layout."
      />
    );
  }

  return (
    <LoadedTocEditor
      docName={docName}
      layout={layout}
      onBackToReader={onBackToReader}
      onCommitSuccess={onCommitSuccess}
    />
  );
}

interface LoadedTocEditorProps {
  docName: string;
  layout: DocumentTaskLayout;
  onBackToReader: () => void;
  onCommitSuccess: () => Promise<void>;
}

function LoadedTocEditor({
  docName,
  layout,
  onBackToReader,
  onCommitSuccess,
}: LoadedTocEditorProps) {
  const controller = useTocEditorController({ docName, layout, onCommitSuccess });
  const selectedRange = getSelectedRange(controller.chapters, controller.selectedItem);
  const [issueDialogOpen, setIssueDialogOpen] = useState(false);
  const allIssues = [
    ...controller.frontendIssues,
    ...(controller.backendValidation?.errors || []),
    ...(controller.backendValidation?.warnings || []),
  ];

  return (
    <Box className="toc-editor">
      <Stack
        direction="row"
        spacing={1}
        alignItems="center"
        justifyContent="space-between"
        className="toc-editor-header"
      >
        <Box>
          <Typography component="h1" variant="h1" className="toc-editor-title">
            Edit TOC
          </Typography>
          <Typography color="text.secondary" className="toc-editor-subtitle">
            {layout.title || docName}
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" onClick={onBackToReader}>
            Back to reader
          </Button>
          <Button
            variant="outlined"
            startIcon={<PlaylistAddCheckIcon />}
            onClick={controller.validateWithBackend}
            disabled={controller.validationStatus === "loading"}
          >
            Validate
          </Button>
          <Button
            variant="contained"
            color="warning"
            startIcon={<SaveOutlinedIcon />}
            onClick={controller.requestCommitConfirmation}
            disabled={controller.commitStatus === "loading"}
          >
            Commit reparse
          </Button>
        </Stack>
      </Stack>

      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1}
        alignItems={{ xs: "stretch", sm: "center" }}
        justifyContent="space-between"
        className="toc-editor-mode-bar"
      >
        <ToggleButtonGroup
          className="toc-editor-mode-toggle"
          value={controller.mode}
          exclusive
          size="small"
          onChange={(_event, nextMode) => {
            if (nextMode) {
              controller.setMode(nextMode);
            }
          }}
        >
          <ToggleButton value="from_scratch">From scratch</ToggleButton>
          <ToggleButton value="edit_existing">Edit existing</ToggleButton>
        </ToggleButtonGroup>
        <Typography color="text.secondary" className="toc-editor-mode-status">
          {controller.mode === "from_scratch"
            ? "Draft starts empty for missing or unrecognized TOC."
            : "Draft is seeded from the loaded layout."}
        </Typography>
      </Stack>

      <Alert severity="info" className="toc-editor-notice">
        Submitting TOC edits triggers hard reparse. Generated QA, summaries, quiz results,
        and derived artifacts will not be preserved.
      </Alert>

      <Box className="toc-editor-workspace">
        <Box className="toc-editor-tree">
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography component="h2" variant="h2">
              Table of contents
            </Typography>
            <Button size="small" startIcon={<AddIcon />} onClick={controller.addChapter}>
              Chapter
            </Button>
          </Stack>

          <Stack spacing={1.5} className="toc-editor-chapter-list">
            {controller.chapters.length === 0 ? (
              <StateView
                className="toc-editor-empty-draft-state"
                title="No TOC draft items"
                detail="Add a chapter to define a replacement structure."
              />
            ) : null}
            {controller.chapters.map((chapter, chapterIndex) => (
              <Box className="toc-editor-chapter" key={chapter.id}>
                <Stack direction="row" spacing={1} alignItems="center">
                  <TextField
                    className="toc-editor-chapter-title-input"
                    label={`Chapter ${chapterIndex + 1}`}
                    value={chapter.title}
                    onChange={(event) =>
                      controller.updateChapter(chapter.id, { title: event.target.value })
                    }
                    onFocus={() =>
                      controller.setSelectedItem({ kind: "chapter", chapterId: chapter.id })
                    }
                    size="small"
                    fullWidth
                  />
                  <MoveButtons
                    onUp={() => controller.moveChapter(chapter.id, -1)}
                    onDown={() => controller.moveChapter(chapter.id, 1)}
                  />
                  <Tooltip title="Delete chapter">
                    <IconButton onClick={() => controller.removeChapter(chapter.id)}>
                      <DeleteOutlineIcon />
                    </IconButton>
                  </Tooltip>
                </Stack>

                <Stack spacing={1} className="toc-editor-section-list">
                  {chapter.sections.map((section, sectionIndex) => (
                    <Stack
                      direction="row"
                      spacing={1}
                      alignItems="center"
                      className="toc-editor-section"
                      key={section.id}
                    >
                      <TextField
                        className="toc-editor-section-title-input"
                        label={`Section ${sectionIndex + 1}`}
                        value={section.title}
                        onChange={(event) =>
                          controller.updateSection(chapter.id, section.id, {
                            title: event.target.value,
                          })
                        }
                        onFocus={() =>
                          controller.setSelectedItem({
                            kind: "section",
                            chapterId: chapter.id,
                            sectionId: section.id,
                          })
                        }
                        size="small"
                        fullWidth
                      />
                      <MoveButtons
                        onUp={() => controller.moveSection(chapter.id, section.id, -1)}
                        onDown={() => controller.moveSection(chapter.id, section.id, 1)}
                      />
                      <Tooltip title="Delete section">
                        <IconButton
                          onClick={() => controller.removeSection(chapter.id, section.id)}
                        >
                          <DeleteOutlineIcon />
                        </IconButton>
                      </Tooltip>
                    </Stack>
                  ))}
                  <Button
                    size="small"
                    startIcon={<AddIcon />}
                    onClick={() => controller.addSection(chapter.id)}
                  >
                    Section
                  </Button>
                </Stack>
              </Box>
            ))}
          </Stack>
        </Box>

        <Box className="toc-editor-range-workspace">
          <Typography component="h2" variant="h2">
            Source range
          </Typography>
          <Typography color="text.secondary">
            Assign source anchors for the selected TOC item.
          </Typography>
          <Alert severity={controller.preferredAnchorType === "page_range" ? "success" : "info"}>
            {controller.preferredAnchorType === "page_range"
              ? "Page evidence is available. Page anchors are the default for this document."
              : "Page evidence is unavailable. Character anchors are the default fallback."}
          </Alert>
          {selectedRange ? (
            <Stack spacing={1.5}>
              <ToggleButtonGroup
                className="toc-editor-anchor-type-toggle"
                value={selectedRange.anchorType}
                exclusive
                size="small"
                onChange={(_event, nextAnchorType) => {
                  if (nextAnchorType) {
                    controller.updateSelectedRange({
                      ...selectedRange,
                      anchorType: nextAnchorType,
                    });
                  }
                }}
              >
                <ToggleButton value="page_range">Page</ToggleButton>
                <ToggleButton value="char_range">Char</ToggleButton>
              </ToggleButtonGroup>
              {selectedRange.anchorType === "page_range" ? (
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                  <TextField
                    label="Page start"
                    value={selectedRange.pageStart}
                    helperText="1-based page number"
                    onChange={(event) =>
                      controller.updateSelectedRange({
                        ...selectedRange,
                        pageStart: event.target.value,
                      })
                    }
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Page end"
                    value={selectedRange.pageEnd}
                    helperText="Leave blank for one page"
                    onChange={(event) =>
                      controller.updateSelectedRange({
                        ...selectedRange,
                        pageEnd: event.target.value,
                      })
                    }
                    size="small"
                    fullWidth
                  />
                </Stack>
              ) : (
                <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                  <TextField
                    label="char_start"
                    value={selectedRange.charStart}
                    onChange={(event) =>
                      controller.updateSelectedRange({
                        ...selectedRange,
                        charStart: event.target.value,
                      })
                    }
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="char_end"
                    value={selectedRange.charEnd}
                    onChange={(event) =>
                      controller.updateSelectedRange({
                        ...selectedRange,
                        charEnd: event.target.value,
                      })
                    }
                    size="small"
                    fullWidth
                  />
                </Stack>
              )}
            </Stack>
          ) : (
            <StateView
              className="toc-editor-range-empty-state"
              title="No TOC item selected"
              detail="Select a chapter or section to assign its source range."
            />
          )}
        </Box>
      </Box>

      <Dialog
        open={controller.confirmOpen}
        onClose={() => controller.setConfirmOpen(false)}
      >
        <DialogTitle>Confirm hard reparse</DialogTitle>
        <DialogContent>
          <Typography>
            This will replace the current document structure. Generated QA, summaries,
            quiz results, and derived artifacts will not be preserved.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => controller.setConfirmOpen(false)}>Cancel</Button>
          <Button color="warning" variant="contained" onClick={controller.commitHardReparse}>
            Hard reparse
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={issueDialogOpen} onClose={() => setIssueDialogOpen(false)}>
        <DialogTitle>Validation issues</DialogTitle>
        <DialogContent>
          <IssueList issues={allIssues} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIssueDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      <AppNotification
        notification={controller.notification}
        onClose={controller.closeNotification}
        onAction={allIssues.length > 0 ? () => setIssueDialogOpen(true) : undefined}
      />
    </Box>
  );
}

function MoveButtons({ onUp, onDown }: { onUp: () => void; onDown: () => void }) {
  return (
    <Stack direction="row" spacing={0.25}>
      <Tooltip title="Move up">
        <IconButton size="small" onClick={onUp}>
          <ArrowUpwardIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Tooltip title="Move down">
        <IconButton size="small" onClick={onDown}>
          <ArrowDownwardIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    </Stack>
  );
}

function IssueList({ issues }: { issues: ManualStructureValidationIssue[] }) {
  if (issues.length === 0) {
    return null;
  }
  return (
    <Stack spacing={1} className="toc-editor-issue-list">
      {issues.map((issue, index) => (
        <Alert
          severity={issue.severity === "error" ? "error" : "warning"}
          key={`${issue.code}:${issue.external_id || issue.entry_index || index}`}
        >
          {issue.message}
        </Alert>
      ))}
    </Stack>
  );
}
