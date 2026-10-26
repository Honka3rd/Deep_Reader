import { useRef, useState } from "react";
import { Box, Button, Collapse, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, Paper, Stack, TextField, Tooltip, Typography } from "@mui/material";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import SaveOutlinedIcon from "@mui/icons-material/SaveOutlined";
import CloseIcon from "@mui/icons-material/Close";

import type { CleanupHistory, CleanupRecord, ReadingState } from "../../services/ReadingContentService";
import { CLEANUP_PAGE_SIZE, editableCleanupRecord, restoreUnavailable } from "./cleanupHistoryModel";
import { useCleanupHistory, type CleanupAction, type CleanupCommand } from "./useCleanupHistory";
import type { ReviewNavigate } from "./useSourceReviewBlocks";
import { SourceReviewItemMenu, SourceReviewNavigateButton } from "./SourceReviewControls";
export type { CleanupHistory } from "../../services/ReadingContentService";

export function CleanupAudit({ history, busy, active = true, onPage, docName, snapshot, onMutate, onNavigate, onDirty, onError }: {
  active?: boolean;
  history?: CleanupHistory; busy: boolean; onPage: (offset: number) => void;
  docName: string; snapshot: ReadingState; onDirty: (dirty: boolean) => void; onError: (message: string) => void;
  onMutate: (action: CleanupAction, command: CleanupCommand) => Promise<boolean | undefined>;
  onNavigate: ReviewNavigate;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [offset, setOffset] = useState(0);
  const [discard, setDiscard] = useState<(() => void) | null>(null);
  const [deleting, setDeleting] = useState<CleanupRecord | null>(null);
  const deleteTrigger = useRef<HTMLElement | null>(null);
  const control = useCleanupHistory({ docName, snapshot, onMutate, onNavigate, onDirty, onError, active, blocked: busy });
  const disabled = !active || busy || control.busy;
  function leave(action: () => void) {
    if (control.dirty) setDiscard(() => action);
    else { control.setEditor(null); action(); }
  }
  return <Box component="section" className="cleanup-history" aria-label="OCR cleanup history" aria-busy={disabled} tabIndex={-1}>
    <Typography component="h3" variant="subtitle1">OCR cleanup history</Typography>
    {!history ? <Typography color="text.secondary">Cleanup history is unavailable.</Typography>
      : history.total === 0 ? <Typography color="text.secondary">No applied OCR cleanup recorded.</Typography>
      : <>
        {history.items.map(batch => <Box key={batch.revision} sx={{ borderBottom: "1px solid", borderColor: "divider", py: 1 }}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Tooltip title={expanded === batch.revision ? "Collapse cleanup details" : "Expand cleanup details"}>
              <IconButton aria-label={`${expanded === batch.revision ? "Collapse" : "Expand"} cleanup ${batch.revision}`}
                aria-expanded={expanded === batch.revision} aria-controls={`cleanup-details-${batch.revision}`}
                disabled={disabled} onClick={() => leave(() => { setExpanded(expanded === batch.revision ? null : batch.revision); setOffset(0); })}>
                {expanded === batch.revision ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Tooltip>
            <Box sx={{ minWidth: 0 }}>
              <Typography>{batch.applied_count} cleaned, {batch.skipped_count} skipped</Typography>
              <Typography variant="body2" color="text.secondary">
                {batch.deletions.length === 0 ? "History entries removed"
                  : batch.remaining_count === 0 ? "Undone" : batch.undoable ? "Undo available" : "Later changes prevent batch undo"}
              </Typography>
            </Box>
          </Stack>
          <Collapse in={expanded === batch.revision} unmountOnExit id={`cleanup-details-${batch.revision}`}>
            <Typography variant="caption" color="text.secondary">{new Date(batch.at).toLocaleString()}</Typography>
            {batch.deletions.length === 0 && <Typography className="cleanup-history-empty" color="text.secondary">No retained cleanup records.</Typography>}
            {batch.deletions.slice(offset, offset + CLEANUP_PAGE_SIZE).map((item, index) => {
              const editing = control.editor !== null && control.editor.id === item.record_id;
              const value = editing ? control.editor!.text : item.text ?? item.quote;
              const invalid = !value.trim() || Array.from(value).length > snapshot.max_chars;
              const unavailable = restoreUnavailable(item);
              return <Paper variant="outlined" key={item.record_id ?? index + offset} className="cleanup-history-record" data-record-id={item.record_id}
                sx={{ p: { xs: 1.5, sm: 2 }, my: 1.5, borderRadius: 1, minWidth: 0 }}>
              <Stack className="cleanup-history-record-header" direction="row" alignItems="center" spacing={1}>
              <Typography variant="body2" sx={{ minWidth: 0, flex: 1, overflowWrap: "anywhere" }}>{item.page_indices.length ? `Page ${item.page_indices.map(i => i + 1).join(", ")}` : "Page reference unavailable"}
                {" · "}{item.reason === "ocr_garbage" ? "Unreadable OCR text" : "Unreadable fragment"}</Typography>
              <SourceReviewItemMenu identity={item.record_id ?? `${batch.revision}:${index + offset}`}
                label={`Text actions for cleanup record ${item.record_id ?? index + offset}`} classPrefix="cleanup-history-record"
                disabled={disabled} capabilities={{ edit: editableCleanupRecord(item) && !editing,
                  delete: item.can_delete === true && Boolean(item.record_id), keep: !unavailable && !(editing && invalid) }}
                hints={{ keep: unavailable || "Restore saved text to reading content", delete: "Remove this history record, keeping reading content" }}
                onSelect={(action, trigger) => {
                  if (action === "edit") leave(() => control.setEditor({ id: item.record_id!, text: value, saved: value }));
                  else if (action === "delete") leave(() => { deleteTrigger.current = trigger; setDeleting(item); });
                  else if (control.editor && !editing) leave(() => void control.submit(item, "cleanup_restore"));
                  else void control.submit(item, "cleanup_restore");
                }} />
              </Stack>
              <Box component="blockquote" sx={{ m: 0, mt: .5, pl: 1, borderLeft: "2px solid", borderColor: "divider",
                whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: 180, overflow: "auto" }}>{item.quote}</Box>
              {item.edited_at && <Typography component="time" dateTime={item.edited_at} variant="caption">{new Date(item.edited_at).toLocaleString()}</Typography>}
              {item.text !== undefined && <Typography className="cleanup-history-draft-status" variant="caption" component="p" color="text.secondary">
                {item.status === "restored" ? "Previously restored · Saved draft" : "Saved restoration draft"}
              </Typography>}
              {editing && <TextField className="cleanup-history-editor" label="Restoration text" multiline minRows={3} maxRows={8}
                fullWidth value={value} disabled={disabled} error={invalid} helperText={`${Array.from(value).length} / ${snapshot.max_chars}`}
                onChange={event => control.setEditor(current => current && ({ ...current, text: event.target.value }))} />}
              {!editing && item.text !== undefined && item.text !== item.quote && <Typography className="cleanup-history-saved-text"
                sx={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: 180, overflow: "auto" }}>{item.text}</Typography>}
              <Stack direction="row" flexWrap="wrap" alignItems="center">
                {editing && <Tooltip title="Save restoration text"><span><IconButton
                  className="cleanup-history-save-action" aria-label="Save restoration text"
                  disabled={disabled || !editableCleanupRecord(item) || invalid || !control.dirty}
                  onClick={() => void control.submit(item, "cleanup_edit")}>
                  <SaveOutlinedIcon /></IconButton></span></Tooltip>}
                {editing && <Tooltip title="Cancel record edit"><IconButton aria-label="Cancel record edit" disabled={disabled}
                  onClick={() => leave(() => control.setEditor(null))}><CloseIcon /></IconButton></Tooltip>}
                {unavailable && <Typography className="cleanup-history-restore-status" variant="caption" color="text.secondary">{unavailable}</Typography>}
              </Stack>
              {item.links?.map(link => <Stack key={link.block_id} className="cleanup-history-block-link" direction="row" alignItems="center" spacing={1}>
                <Typography variant="caption" sx={{ overflowWrap: "anywhere", minWidth: 0, flex: 1 }}>{link.block_id} · {link.char_from}-{link.char_to}</Typography>
                <SourceReviewNavigateButton blockId={link.block_id} disabled={disabled || !link.can_navigate}
                  onClick={() => leave(() => void control.navigate(link))} />
              </Stack>)}
            </Paper>; })}
            {batch.deletions.length > CLEANUP_PAGE_SIZE && <Stack className="cleanup-history-record-pagination" direction="row" alignItems="center">
              <IconButton aria-label="Previous cleanup items" disabled={disabled || offset === 0}
                onClick={() => leave(() => setOffset(Math.max(0, offset - CLEANUP_PAGE_SIZE)))}><ArrowBackIcon /></IconButton>
              <Typography variant="caption">{offset + 1}-{Math.min(offset + CLEANUP_PAGE_SIZE, batch.deletions.length)} / {batch.deletions.length}</Typography>
              <IconButton aria-label="Next cleanup items" disabled={disabled || offset + CLEANUP_PAGE_SIZE >= batch.deletions.length}
                onClick={() => leave(() => setOffset(offset + CLEANUP_PAGE_SIZE))}><ArrowForwardIcon /></IconButton>
            </Stack>}
            {batch.skipped_count > 0 && <Typography variant="body2">Unverified proposals were kept unchanged.</Typography>}
          </Collapse>
        </Box>)}
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 1 }}>
          <Tooltip title="Previous cleanup records"><span><IconButton aria-label="Previous cleanup records"
            disabled={disabled || history.offset === 0} onClick={() => leave(() => { setExpanded(null); onPage(Math.max(0, history.offset - history.limit)); })}><ArrowBackIcon /></IconButton></span></Tooltip>
          <Typography variant="body2">{history.offset + (history.items.length ? 1 : 0)}-{history.offset + history.items.length} of {history.total}</Typography>
          <Tooltip title="Next cleanup records"><span><IconButton aria-label="Next cleanup records"
            disabled={disabled || history.offset + history.limit >= history.total}
            onClick={() => leave(() => { setExpanded(null); onPage(history.offset + history.limit); })}><ArrowForwardIcon /></IconButton></span></Tooltip>
        </Stack>
      </>}
    <Dialog open={Boolean(discard)} onClose={() => setDiscard(null)} aria-labelledby="cleanup-history-discard-title">
      <DialogTitle id="cleanup-history-discard-title">Discard unsaved restoration text?</DialogTitle>
      <DialogActions><Button onClick={() => setDiscard(null)}>Keep editing</Button><Button color="warning" onClick={() => {
        const action = discard; setDiscard(null); control.setEditor(null); onDirty(false); action?.();
      }}>Discard changes</Button></DialogActions>
    </Dialog>
    <Dialog open={Boolean(deleting)} disableRestoreFocus onClose={() => { if (!disabled) setDeleting(null); }}
      aria-labelledby="cleanup-history-delete-title" TransitionProps={{ onExited: () => {
        control.finishDeleteFocus();
        if (active && deleteTrigger.current?.isConnected) deleteTrigger.current.focus({ preventScroll: true });
        deleteTrigger.current = null;
      } }}>
      <DialogTitle id="cleanup-history-delete-title">Delete cleanup record?</DialogTitle>
      <DialogContent className="cleanup-history-delete-content">This removes the saved recovery text. Reading content stays unchanged. This cannot be undone.</DialogContent>
      <DialogActions><Button disabled={disabled} onClick={() => setDeleting(null)}>Cancel</Button>
        <Button color="error" disabled={disabled} onClick={async () => {
          if (!deleting) return;
          const batch = history?.items.find(item => item.revision === expanded);
          const success = await control.submit(deleting, "cleanup_delete");
          if (success === undefined) return;
          // Return to the review's existing notification surface on failure too.
          setDeleting(null);
          if (success) {
            if (batch) setOffset(current => Math.min(current, Math.max(0,
              Math.floor((batch.deletions.length - 2) / CLEANUP_PAGE_SIZE) * CLEANUP_PAGE_SIZE)));
          }
        }}>Delete record</Button></DialogActions>
    </Dialog>
  </Box>;
}
