import { Alert, Button, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from "@mui/material";
import { useRef } from "react";

export function IgnoreAllRegionsDialog({ open, count, busy, error, onCancel, onConfirm }: {
  open: boolean; count: number; busy: boolean; error: string;
  onCancel: () => void; onConfirm: () => void;
}) {
  const cancel = useRef<HTMLButtonElement | null>(null);
  return <Dialog className="source-review-ignore-all-dialog" open={open}
    TransitionProps={{ onEntered: () => cancel.current?.focus({ preventScroll: true }) }}
    onClose={() => { if (!busy) onCancel(); }} aria-labelledby="source-review-ignore-all-title"
    aria-describedby="source-review-ignore-all-description" fullWidth maxWidth="sm">
    <DialogTitle id="source-review-ignore-all-title">Ignore all {count} flagged areas?</DialogTitle>
    <DialogContent>
      <DialogContentText id="source-review-ignore-all-description">
        Carefully check the extracted text against the original pages before continuing.
        This permanently dismisses all {count} pending OCR flags in this document, including other pages.
        The text may still contain recognition errors or missing content. Body text and original images are kept.
        This action cannot be undone.
      </DialogContentText>
      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
    </DialogContent>
    <DialogActions>
      <Button ref={cancel} autoFocus disabled={busy} onClick={onCancel}>Cancel</Button>
      <Button color="error" disabled={busy || count === 0} onClick={onConfirm}>Ignore all permanently</Button>
    </DialogActions>
  </Dialog>;
}
