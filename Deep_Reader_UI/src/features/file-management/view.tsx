import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { TocHintControl } from "../../shared/TocHintControl";
import type { TocHint } from "../../types/api";
import DriveFileRenameOutlineIcon from "@mui/icons-material/DriveFileRenameOutline";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FilterAltOutlinedIcon from "@mui/icons-material/FilterAltOutlined";
import FolderOpenOutlinedIcon from "@mui/icons-material/FolderOpenOutlined";
import RefreshOutlinedIcon from "@mui/icons-material/RefreshOutlined";
import BuildOutlinedIcon from "@mui/icons-material/BuildOutlined";
import CloseOutlinedIcon from "@mui/icons-material/CloseOutlined";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  MenuItem,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  IconButton,
  Tooltip,
} from "@mui/material";
import { PdfModeControl } from "./PdfModeControl";
import { StateView } from "../../shared/components/StateView";
import type { DocumentManagementItem, PdfExtractionMode, RequestStatus } from "../../types/api";
import {
  formatUploadFileSize,
  getDocumentPreparationStatusLabel,
  getDocumentManagementDetailLabel,
  getDocumentManagementTitle,
  getDocumentSourceStatusLabel,
  getDocumentSourceTypeLabel,
  getUploadStatusLabel,
  type DocumentDeleteState,
  type DocumentManagementFilterOptions,
  type DocumentManagementFilters,
  type DocumentRenameState,
  type FileUploadConflictState,
  type FileUploadDraft,
  type FileUploadResult,
  type FileUploadStatus,
  type RecentDocumentShortcut,
  type DocumentPrepareState,
} from "./model";

interface FileManagementViewProps {
  canOpenDocument: (docName: string) => boolean;
  openingDocName: string | null;
  openError: string;
  prepareState: DocumentPrepareState | null;
  onChangeUploadTocHint: (hint: TocHint) => void;
  onChangePrepareTocHint: (hint: TocHint) => void;
  onChangeUploadPdfMode: (mode: PdfExtractionMode) => void;
  onChangePreparePdfMode: (mode: PdfExtractionMode) => void;
  onRequestPrepare: (item: DocumentManagementItem) => void;
  onClosePrepare: () => void;
  onPrepareExisting: () => void;
  activeFilterCount: number;
  deleteState: DocumentDeleteState;
  error: string;
  filterOptions: DocumentManagementFilterOptions;
  filters: DocumentManagementFilters;
  items: DocumentManagementItem[];
  open: boolean;
  recentItems: RecentDocumentShortcut[];
  renameState: DocumentRenameState;
  status: RequestStatus;
  total: number;
  visibleTotal: number;
  uploadConflict?: FileUploadConflictState;
  uploadDraft: FileUploadDraft | null;
  uploadError: string;
  uploadResult: FileUploadResult | null;
  uploadStatus: FileUploadStatus;
  onCancelDelete: () => void;
  onCancelDuplicateUpload?: () => void;
  onCancelRename: () => void;
  onCancelUpload: () => void;
  onChangeFilter: (name: keyof DocumentManagementFilters, value: string) => void;
  onChangeRenameDraft: (nextName: string) => void;
  onConfirmDelete: () => void;
  onConfirmDuplicateUpload?: () => void;
  onConfirmRename: () => void;
  onClose: () => void;
  onOpenDocument: (docName: string) => void;
  onPrepareUploadedDocument: () => void;
  onCancelUploadedPreparation: () => void;
  onRefresh: () => void;
  onRequestDelete: (item: DocumentManagementItem) => void;
  onRequestRename: (item: DocumentManagementItem) => void;
  onResetFilters: () => void;
  onRetryUpload: () => void;
  onSelectUploadFile: (file: File | null) => void;
  onUpload: () => void;
}

export function FileManagementView({
  canOpenDocument,
  openingDocName,
  openError,
  prepareState,
  onChangeUploadTocHint,
  onChangePrepareTocHint,
  onChangeUploadPdfMode,
  onChangePreparePdfMode,
  onRequestPrepare,
  onClosePrepare,
  onPrepareExisting,
  activeFilterCount,
  deleteState,
  error,
  filterOptions,
  filters,
  items,
  open,
  recentItems,
  renameState,
  status,
  total,
  visibleTotal,
  uploadConflict,
  uploadDraft,
  uploadError,
  uploadResult,
  uploadStatus,
  onCancelDelete,
  onCancelDuplicateUpload,
  onCancelRename,
  onCancelUpload,
  onChangeFilter,
  onChangeRenameDraft,
  onConfirmDelete,
  onConfirmDuplicateUpload,
  onConfirmRename,
  onClose,
  onOpenDocument,
  onPrepareUploadedDocument,
  onCancelUploadedPreparation,
  onRefresh,
  onRequestDelete,
  onRequestRename,
  onResetFilters,
  onRetryUpload,
  onSelectUploadFile,
  onUpload,
}: FileManagementViewProps) {
  const loading = status === "loading";
  const deleting = deleteState.status === "deleting";
  const renaming = renameState.status === "renaming";
  const uploading = uploadStatus === "uploading";
  const safeUploadConflict = uploadConflict ?? {
    open: false,
    fileName: "",
    message: "",
  };
  const preparingUploadedDocument = uploadResult?.prepareStatus === "loading" || uploadResult?.prepareStatus === "queued";
  const preparingExisting = prepareState?.status === "loading";
  const operationPending = uploading || preparingUploadedDocument || preparingExisting;
  const deleteDialogOpen =
    deleteState.status === "confirming" ||
    deleteState.status === "deleting" ||
    deleteState.status === "error";
  const renameDialogOpen =
    renameState.status === "editing" ||
    renameState.status === "renaming" ||
    renameState.status === "error";
  const duplicateUploadDialogOpen =
    safeUploadConflict.open || uploadStatus === "conflict";
  const deleteItemTitle = deleteState.item
    ? getDocumentManagementTitle(deleteState.item)
    : "this document";
  const renameItemTitle = renameState.item
    ? getDocumentManagementTitle(renameState.item)
    : "this document";

  return (
    <>
      <Dialog
        className="file-management-dialog"
        fullWidth
        maxWidth="md"
        open={open}
        onClose={onClose}
      >
      <DialogTitle className="file-management-dialog-title">
        <Stack direction="row" alignItems="center" justifyContent="space-between" gap={2}>
          <Box minWidth={0}>
            <Typography component="span" variant="h6">
              Documents
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {visibleTotal} of {total} item{total === 1 ? "" : "s"}
            </Typography>
          </Box>
          <Button
            className="file-management-refresh-action"
            disabled={loading}
            startIcon={<RefreshOutlinedIcon />}
            variant="outlined"
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Stack>
      </DialogTitle>
      <DialogContent className="file-management-dialog-content" dividers>
        {openError && <Alert severity="warning" sx={{ mb: 1 }}>{openError}</Alert>}
        {openingDocName && <Typography role="status" variant="body2">Checking reader readiness.</Typography>}
        <Box className="file-management-upload-panel">
          <Stack
            className="file-management-upload-header"
            direction="row"
            alignItems="center"
            justifyContent="space-between"
            gap={2}
          >
            <Box minWidth={0}>
              <Typography className="file-management-upload-title" variant="subtitle2">
                Upload
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {uploadDraft ? uploadDraft.fileName : "No file selected"}
              </Typography>
            </Box>
            <Chip
              className={`file-management-upload-status file-management-upload-status-${uploadStatus}`}
              label={getUploadStatusLabel(uploadStatus)}
              size="small"
            />
          </Stack>
          {uploadDraft ? (
            <Typography className="file-management-upload-meta" variant="caption" color="text.secondary">
              {formatUploadFileSize(uploadDraft.fileSizeBytes)} · {uploadDraft.fileType}
            </Typography>
          ) : null}
          {uploadError ? (
            <Typography
              className="file-management-upload-error"
              variant="caption"
              color={uploadStatus === "uploaded" ? "success.main" : "error"}
            >
              {uploadError}
            </Typography>
          ) : null}
          {(uploadResult?.pdfMode || uploadDraft?.pdfMode) && (
            <PdfModeControl value={uploadResult?.pdfMode || uploadDraft?.pdfMode || "fast"}
              disabled={Boolean(operationPending || uploadStatus === "conflict" || uploadResult?.needsReconciliation)}
              onChange={onChangeUploadPdfMode} />
          )}
          {(uploadDraft || uploadResult) && <TocHintControl value={uploadResult?.tocHint || uploadDraft?.tocHint || "auto"}
            disabled={Boolean(operationPending) || uploadStatus === "conflict" || Boolean(uploadResult?.needsReconciliation)}
            onChange={onChangeUploadTocHint} />}
          <Stack className="file-management-upload-actions" direction="row" gap={1}>
            <Button
              className="file-management-select-upload-action"
              disabled={operationPending}
              component="label"
              size="small"
              startIcon={<UploadFileOutlinedIcon />}
              variant="outlined"
            >
              Select file
              <input
                disabled={operationPending}
                accept=".pdf,.txt,application/pdf,text/plain"
                hidden
                type="file"
                onChange={(event) => {
                  onSelectUploadFile(event.target.files?.[0] || null);
                  event.target.value = "";
                }}
              />
            </Button>
            <Button
              className="file-management-upload-action"
              disabled={
                !uploadDraft ||
                uploadStatus === "failed" ||
                uploadStatus === "uploaded" ||
                uploadStatus === "conflict" ||
                operationPending
              }
              size="small"
              variant="contained"
              onClick={onUpload}
            >
              {uploading ? "Uploading" : "Upload"}
            </Button>
            {uploadStatus !== "uploaded" && <Button
              className="file-management-cancel-upload-action"
              disabled={operationPending || (!uploadDraft && uploadStatus !== "failed")}
              size="small"
              onClick={onCancelUpload}
            >
              Cancel
            </Button>}
            {uploadStatus === "failed" ? (
              <Button
                className="file-management-retry-upload-action"
                disabled={uploading || !uploadDraft}
                size="small"
                startIcon={<RefreshOutlinedIcon />}
                onClick={onRetryUpload}
              >
                Retry upload
              </Button>
            ) : null}
          </Stack>
          {uploadStatus === "uploaded" && uploadResult ? (
            <Box className="file-management-post-upload-panel">
              <Typography
                className="file-management-post-upload-title"
                variant="subtitle2"
              >
                {uploadResult.docName}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {uploadResult.sourceFilename}
              </Typography>
              {uploadResult.prepareMessage ? (
                <Typography
                  className="file-management-post-upload-message"
                  variant="caption"
                  color={uploadResult.prepareStatus === "error" ? "error" :
                    uploadResult.prepareStatus === "success" ? "success.main" : "text.secondary"}
                >
                  {uploadResult.prepareMessage}
                </Typography>
              ) : null}
              <Stack className="file-management-post-upload-actions" direction="row" gap={1}>
                <Button
                  className="file-management-prepare-uploaded-action"
                  disabled={operationPending}
                  size="small"
                  startIcon={<BuildOutlinedIcon />}
                  variant="outlined"
                  onClick={onPrepareUploadedDocument}
                >
                  {uploadResult.prepareStatus === "queued" ? "Queued" : preparingUploadedDocument ? "Preparing" : uploadResult.needsReconciliation ? "Check readiness" : "Prepare"}
                </Button>
                {preparingUploadedDocument && <Button
                  className="file-management-cancel-preparation-action"
                  size="small" startIcon={<CloseOutlinedIcon />}
                  onClick={onCancelUploadedPreparation}>Cancel</Button>}
                <Button
                  className="file-management-open-uploaded-action"
                  disabled={Boolean(openingDocName) || !canOpenDocument(uploadResult.docName)}
                  size="small"
                  startIcon={<FolderOpenOutlinedIcon />}
                  variant="contained"
                  onClick={() => onOpenDocument(uploadResult.docName)}
                >
                  Open
                </Button>
                <Button
                  className="file-management-upload-another-action"
                  disabled={preparingUploadedDocument}
                  size="small"
                  onClick={onRetryUpload}
                >
                  Upload another
                </Button>
              </Stack>
            </Box>
          ) : null}
        </Box>
        <Divider className="file-management-section-divider" />
        <Accordion className="file-management-filter-panel" disableGutters>
          <AccordionSummary
            className="file-management-filter-summary"
            expandIcon={<ExpandMoreIcon />}
          >
            <Stack direction="row" alignItems="center" gap={1}>
              <FilterAltOutlinedIcon fontSize="small" />
              <Typography variant="subtitle2">Advanced filters</Typography>
              {activeFilterCount > 0 ? (
                <Chip label={activeFilterCount} size="small" />
              ) : null}
            </Stack>
          </AccordionSummary>
          <AccordionDetails className="file-management-filter-details">
            <Stack className="file-management-filter-grid" gap={1.5}>
              <TextField
                className="file-management-filter-query"
                fullWidth
                label="Search"
                size="small"
                value={filters.query}
                onChange={(event) => onChangeFilter("query", event.target.value)}
              />
              <TextField
                className="file-management-filter-language"
                disabled={filterOptions.languages.length === 0}
                label="Language"
                select
                size="small"
                value={filters.language}
                onChange={(event) => onChangeFilter("language", event.target.value)}
              >
                <MenuItem value="">All languages</MenuItem>
                {filterOptions.languages.map((language) => (
                  <MenuItem key={language} value={language}>
                    {language}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                className="file-management-filter-preparation"
                disabled={filterOptions.preparationStatuses.length === 0}
                label="Readiness"
                select
                size="small"
                value={filters.preparationStatus}
                onChange={(event) =>
                  onChangeFilter("preparationStatus", event.target.value)
                }
              >
                <MenuItem value="">All readiness states</MenuItem>
                {filterOptions.preparationStatuses.map((preparationStatus) => (
                  <MenuItem key={preparationStatus} value={preparationStatus}>
                    {getDocumentPreparationStatusLabel(
                      preparationStatus as DocumentManagementItem["preparationStatus"],
                    )}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                className="file-management-filter-profile"
                disabled={filterOptions.profileLabels.length === 0}
                label="Profile"
                select
                size="small"
                value={filters.profileLabel}
                onChange={(event) => onChangeFilter("profileLabel", event.target.value)}
              >
                <MenuItem value="">All profiles</MenuItem>
                {filterOptions.profileLabels.map((profileLabel) => (
                  <MenuItem key={profileLabel} value={profileLabel}>
                    {profileLabel}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                className="file-management-filter-source-type"
                disabled={filterOptions.sourceTypes.length === 0}
                label="Source type"
                select
                size="small"
                value={filters.sourceType}
                onChange={(event) => onChangeFilter("sourceType", event.target.value)}
              >
                <MenuItem value="">All source types</MenuItem>
                {filterOptions.sourceTypes.map((sourceType) => (
                  <MenuItem key={sourceType} value={sourceType}>
                    {getDocumentSourceTypeLabel(
                      sourceType as DocumentManagementItem["sourceType"],
                    )}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                className="file-management-filter-source-status"
                disabled={filterOptions.sourceStatuses.length === 0}
                label="Source status"
                select
                size="small"
                value={filters.sourceStatus}
                onChange={(event) => onChangeFilter("sourceStatus", event.target.value)}
              >
                <MenuItem value="">All source states</MenuItem>
                {filterOptions.sourceStatuses.map((sourceStatus) => (
                  <MenuItem key={sourceStatus} value={sourceStatus}>
                    {getDocumentSourceStatusLabel(
                      sourceStatus as DocumentManagementItem["sourceStatus"],
                    )}
                  </MenuItem>
                ))}
              </TextField>
              <Button
                className="file-management-reset-filters-action"
                disabled={activeFilterCount === 0}
                size="small"
                variant="outlined"
                onClick={onResetFilters}
              >
                Reset filters
              </Button>
            </Stack>
          </AccordionDetails>
        </Accordion>
        {deleteState.status === "success" && deleteState.message ? (
          <Alert
            className="file-management-delete-result"
            severity="success"
            sx={{ mb: 2 }}
          >
            {deleteState.message}
          </Alert>
        ) : null}
        {renameState.status === "success" && renameState.message ? (
          <Alert
            className="file-management-rename-result"
            severity="success"
            sx={{ mb: 2 }}
          >
            {renameState.message}
          </Alert>
        ) : null}
        {loading ? (
          <StateView
            className="file-management-state"
            loading
            title="Loading documents"
          />
        ) : status === "error" ? (
          <StateView
            className="file-management-state"
            severity="error"
            title="Document library failed"
            detail={error}
          />
        ) : status === "empty" ? (
          <StateView
            className="file-management-state"
            title="No documents"
            detail="No backend document candidates were returned."
          />
        ) : items.length === 0 ? (
          <StateView
            className="file-management-state"
            title="No matching documents"
            detail="Adjust or reset the current filters."
          />
        ) : (
          <>
            <TableContainer className="file-management-table-container">
              <Table className="file-management-table" size="small" aria-label="Document library">
                <TableHead>
                  <TableRow>
                    <TableCell>Document</TableCell>
                    <TableCell>Details</TableCell>
                    <TableCell align="right">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {items.map((item) => (
                    <TableRow className="file-management-row" key={item.docName} hover>
                      <TableCell className="file-management-document-cell">
                        <Typography className="file-management-document-title" variant="body2">
                          {getDocumentManagementTitle(item)}
                        </Typography>
                        <Typography
                          className="file-management-document-name"
                          variant="caption"
                          color="text.secondary"
                        >
                          {item.docName}
                        </Typography>
                      </TableCell>
                      <TableCell className="file-management-details-cell">
                        {item.reviewRequired && <Typography color="warning.main" variant="body2">OCR needs review</Typography>}
                        <Typography variant="caption" color="text.secondary">
                          {getDocumentManagementDetailLabel(item)}
                        </Typography>
                      </TableCell>
                      <TableCell className="file-management-actions-cell" align="right">
                        <Stack
                          className="file-management-row-actions"
                          direction="row"
                          justifyContent="flex-end"
                          gap={1}
                        >
                          {item.sourceStatus === "available" && (
                            <Tooltip title={item.sourceType === "pdf" ? "Prepare PDF" : "Prepare document"}>
                              <span><IconButton aria-label={`Prepare ${item.sourceType === "pdf" ? "PDF" : "document"} ${item.docName}`} size="small"
                                disabled={operationPending || renaming || deleting}
                                onClick={() => onRequestPrepare(item)}><BuildOutlinedIcon fontSize="small" /></IconButton></span>
                            </Tooltip>
                          )}
                          <Button
                            className="file-management-open-action"
                            disabled={Boolean(openingDocName) || !canOpenDocument(item.docName)}
                            size="small"
                            startIcon={<FolderOpenOutlinedIcon />}
                            variant="contained"
                            onClick={() => onOpenDocument(item.docName)}
                          >
                            Open
                          </Button>
                          <Button
                            className="file-management-rename-action"
                            disabled={renaming || operationPending}
                            size="small"
                            startIcon={<DriveFileRenameOutlineIcon />}
                            variant="outlined"
                            onClick={() => onRequestRename(item)}
                          >
                            Rename
                          </Button>
                          <Button
                            className="file-management-delete-action"
                            color="error"
                            disabled={deleting || renaming || operationPending}
                            size="small"
                            startIcon={<DeleteOutlineIcon />}
                            variant="outlined"
                            onClick={() => onRequestDelete(item)}
                          >
                            Delete
                          </Button>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            {recentItems.length > 0 ? (
              <Box className="file-management-recent-panel">
                <Typography className="file-management-recent-title" variant="subtitle2">
                  Recent reading
                </Typography>
                <Stack className="file-management-recent-list" direction="row" gap={1}>
                  {recentItems.map((item) => (
                    <Button
                      className="file-management-recent-action"
                      disabled={Boolean(openingDocName) || !canOpenDocument(item.docName)}
                      key={item.docName}
                      size="small"
                      startIcon={<FolderOpenOutlinedIcon />}
                      variant="outlined"
                      onClick={() => onOpenDocument(item.docName)}
                    >
                      <span className="file-management-recent-label">{item.title}</span>
                    </Button>
                  ))}
                </Stack>
              </Box>
            ) : null}
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
      </Dialog>
      <Dialog
        className="file-management-upload-conflict-dialog"
        maxWidth="xs"
        open={duplicateUploadDialogOpen}
        onClose={uploading ? undefined : onCancelDuplicateUpload}
      >
        <DialogTitle>Upload duplicate file?</DialogTitle>
        <DialogContent dividers>
          <Stack gap={1.5}>
            <Typography variant="body2">
              A source named "
              {safeUploadConflict.fileName || uploadDraft?.fileName || "this file"}"
              already exists. Uploading as a copy will keep the existing file and save
              this one with a unique name.
            </Typography>
            {safeUploadConflict.message ? (
              <Alert severity="info">{safeUploadConflict.message}</Alert>
            ) : null}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button disabled={uploading} onClick={onCancelDuplicateUpload}>
            Cancel
          </Button>
          <Button
            className="file-management-confirm-duplicate-upload-action"
            disabled={uploading || !uploadDraft}
            startIcon={<UploadFileOutlinedIcon />}
            variant="contained"
            onClick={onConfirmDuplicateUpload}
          >
            {uploading ? "Uploading" : "Upload as copy"}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        className="file-management-delete-confirmation-dialog"
        maxWidth="xs"
        open={deleteDialogOpen}
        onClose={deleting ? undefined : onCancelDelete}
      >
        <DialogTitle>Delete document?</DialogTitle>
        <DialogContent dividers>
          <Stack gap={1}>
            <Typography variant="body2">
              Delete "{deleteItemTitle}" and its backend file-backed artifacts.
            </Typography>
            {deleteState.item ? (
              <Typography variant="caption" color="text.secondary">
                {deleteState.item.docName}
              </Typography>
            ) : null}
            {deleteState.message ? (
              <Typography
                className="file-management-delete-message"
                variant="body2"
                color={deleteState.status === "error" ? "error" : "text.secondary"}
              >
                {deleteState.message}
              </Typography>
            ) : null}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button disabled={deleting} onClick={onCancelDelete}>
            Cancel
          </Button>
          <Button
            className="file-management-confirm-delete-action"
            color="error"
            disabled={deleting}
            startIcon={<DeleteOutlineIcon />}
            variant="contained"
            onClick={onConfirmDelete}
          >
            {deleting ? "Deleting" : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        className="file-management-rename-dialog"
        fullWidth
        maxWidth="xs"
        open={renameDialogOpen}
        onClose={renaming ? undefined : onCancelRename}
      >
        <DialogTitle>Rename document</DialogTitle>
        <DialogContent dividers>
          <Stack gap={2}>
            <Typography variant="body2">
              Rename "{renameItemTitle}" through the backend lifecycle route.
            </Typography>
            {renameState.item ? (
              <Typography variant="caption" color="text.secondary">
                Current: {renameState.item.docName}
              </Typography>
            ) : null}
            <TextField
              autoFocus
              className="file-management-rename-input"
              disabled={renaming}
              fullWidth
              label="New document name"
              size="small"
              value={renameState.draftName}
              onChange={(event) => onChangeRenameDraft(event.target.value)}
            />
            {renameState.message ? (
              <Typography
                className="file-management-rename-message"
                variant="body2"
                color={renameState.status === "error" ? "error" : "text.secondary"}
              >
                {renameState.message}
              </Typography>
            ) : null}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button disabled={renaming} onClick={onCancelRename}>
            Cancel
          </Button>
          <Button
            className="file-management-confirm-rename-action"
            disabled={renaming}
            startIcon={<DriveFileRenameOutlineIcon />}
            variant="contained"
            onClick={onConfirmRename}
          >
            {renaming ? "Renaming" : "Rename"}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={open && prepareState !== null} onClose={onClosePrepare} maxWidth="xs" fullWidth>
        <DialogTitle>{prepareState?.item.sourceType === "pdf" ? "Prepare PDF" : "Prepare document"}</DialogTitle>
        <DialogContent>
          <Typography sx={{ overflowWrap: "anywhere" }}>{prepareState?.item.docName}</Typography>
          <TocHintControl value={prepareState?.tocHint || "auto"} disabled={Boolean(operationPending)} onChange={onChangePrepareTocHint} />
          {prepareState?.pdfMode && <PdfModeControl value={prepareState.pdfMode} disabled={Boolean(operationPending)}
            onChange={onChangePreparePdfMode} />}
          {openError && <Alert severity="warning">{openError}</Alert>}
          {prepareState?.message && <Alert severity={prepareState.status === "error" ? "error" : "success"}>
            {prepareState.message}
          </Alert>}
        </DialogContent>
        <DialogActions>
          <Button disabled={preparingExisting} onClick={onClosePrepare}>Close</Button>
          {prepareState?.readerReady && !preparingExisting && <Button startIcon={<FolderOpenOutlinedIcon />}
            disabled={Boolean(openingDocName) || !canOpenDocument(prepareState.item.docName)}
            onClick={() => onOpenDocument(prepareState.item.docName)}>Open</Button>}
          <Button disabled={operationPending} startIcon={<BuildOutlinedIcon />} variant="contained"
            onClick={onPrepareExisting}>{preparingExisting ? "Preparing" : "Prepare"}</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
