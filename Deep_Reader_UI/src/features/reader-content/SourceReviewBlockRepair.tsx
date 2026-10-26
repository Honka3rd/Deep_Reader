import { useEffect, useRef, useState } from "react";
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, Typography } from "@mui/material";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import CheckIcon from "@mui/icons-material/Check";
import type { ReadingBlock, ReviewBlockAction, ReviewBlockCommand } from "../../services/ReadingContentService";
import { SourceReviewItemMenu, type SourceReviewAction } from "./SourceReviewControls";

export type ReviewBlockMode = SourceReviewAction;
export function reviewBlockReason(block: ReadingBlock) {
  if (!block.review_repair) return "Block repair is unavailable on this server.";
  switch (block.review_repair.blocked_reason) {
    case "block_manually_edited": return "This block was edited in the reader.";
    case "block_manually_deleted": return "This block was deleted in the reader.";
    case "review_block_kept": return "This block has already been confirmed.";
    case null: case undefined: return "";
    default: return "This block changed or is unavailable. Refresh related blocks.";
  }
}

export function SourceReviewBlockMenu({ block, disabled, onSelect }: {
  block: ReadingBlock; disabled: boolean; onSelect: (mode: ReviewBlockMode, trigger: HTMLElement) => void;
}) {
  const reason = reviewBlockReason(block);
  return <SourceReviewItemMenu identity={block.block_id} label={`Text actions for block ${block.block_id}`}
    classPrefix="source-review-block" disabled={disabled} onSelect={onSelect}
    capabilities={{ edit: block.review_repair?.can_edit === true, delete: block.review_repair?.can_delete === true,
      keep: block.review_repair?.can_keep === true }} hints={{ edit: reason, delete: reason, keep: reason }} />;
}

export function SourceReviewBlockEditor({ block, regionId, mode, busy, error, onClose, onDirty, onMutate }: {
  block: ReadingBlock; regionId: string; mode: ReviewBlockMode; busy: boolean; error: string;
  onClose: () => void; onDirty: (dirty: boolean) => void;
  onMutate: (action: ReviewBlockAction, command: ReviewBlockCommand) => Promise<boolean | undefined>;
}) {
  const [text, setText] = useState(block.content);
  const [confirmEdit, setConfirmEdit] = useState(false);
  const mounted = useRef(true);
  const locked = useRef(false);
  const dirty = mode === "edit" && text !== block.content;
  const count = Array.from(text).length;
  const valid = Boolean(text.trim()) && count <= 10000;
  useEffect(() => { onDirty(dirty); }, [dirty, onDirty]);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; onDirty(false); }; }, [onDirty]);
  function close() {
    if (busy || locked.current || (dirty && !window.confirm("Discard unsaved changes?"))) return;
    onClose();
  }
  async function submit(discard?: boolean) {
    if (busy || locked.current) return;
    locked.current = true;
    try {
      const success = await onMutate(`review_block_${mode}`, { region_id: regionId, block_id: block.block_id,
        ...(mode === "edit" ? { text, discard_interactions: discard } : {}) });
      if (success && mounted.current) { onDirty(false); onClose(); }
    } finally { if (mounted.current) locked.current = false; }
  }
  return <Dialog className="source-review-block-editor" open disableRestoreFocus fullWidth maxWidth="sm"
    onClose={close} aria-labelledby="source-review-block-editor-title">
    <DialogTitle id="source-review-block-editor-title">{mode === "edit" ? "Edit block" : mode === "delete" ? "Delete block?" : "Keep this block?"}</DialogTitle>
    <DialogContent dividers>
      {error && <Alert severity="error">{error}</Alert>}
      <Typography className="source-review-block-editor-target" variant="caption" sx={{ overflowWrap: "anywhere" }}>{block.block_id}</Typography>
      {mode === "edit" ? <TextField className="source-review-block-editor-text" label="Block text" autoFocus fullWidth multiline
        minRows={5} maxRows={14} value={text} disabled={busy || !block.review_repair?.can_edit}
        onChange={event => setText(event.target.value)} error={count > 10000} helperText={`${count} / 10000`} sx={{ mt: 2 }} />
        : <Typography className="source-review-block-confirmation" sx={{ mt: 2 }}>{mode === "keep"
          ? "Confirm this text is correct? Its entries will be removed from Source review and cannot be edited here again. This cannot be undone. Reading content and interactions are kept; the reader's Edit action remains available."
          : "Delete only this block from reading content, including its interactions and Notes? Other related blocks are unchanged. Undo restores text, not deleted interactions."}</Typography>}
      {confirmEdit && <Typography className="source-review-block-interaction-consent" sx={{ mt: 2 }}>
        Changed text always clears translations and affected summaries. Keep preserves other interactions and Notes; Clear deletes them too. Identical text changes nothing.
      </Typography>}
    </DialogContent>
    <DialogActions sx={{ flexWrap: "wrap" }}>
      <Button disabled={busy} onClick={close}>Cancel</Button>
      {mode === "edit" ? confirmEdit ? <>
        <Button disabled={busy || !valid} onClick={() => void submit(false)}>Keep and save</Button>
        <Button color="error" disabled={busy || !valid} onClick={() => void submit(true)}>Clear interactions and save</Button>
      </> : <Button startIcon={<EditOutlinedIcon />} disabled={busy || !valid || !block.review_repair?.can_edit}
        onClick={() => setConfirmEdit(true)}>Save</Button>
        : <Button color={mode === "delete" ? "error" : "primary"} disabled={busy || !block.review_repair?.[`can_${mode}`]}
          startIcon={mode === "keep" ? <CheckIcon /> : <DeleteOutlineIcon />} onClick={() => void submit()}>
          {mode === "keep" ? "Keep it" : "Delete block"}
        </Button>}
    </DialogActions>
  </Dialog>;
}
