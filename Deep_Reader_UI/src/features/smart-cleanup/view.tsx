import CleaningServicesOutlinedIcon from "@mui/icons-material/CleaningServicesOutlined";
import RefreshIcon from "@mui/icons-material/Refresh";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Button } from "@mui/material";
import { useSmartCleanup, type SmartCleanupProps } from "./controller";

export function SmartCleanupDialog(props: SmartCleanupProps) {
  const state = useSmartCleanup(props);
  return <Dialog className="smart-cleanup-dialog" open={props.open} onClose={props.onClose}
    aria-labelledby="smart-cleanup-title" aria-describedby="smart-cleanup-status" maxWidth="sm" fullWidth>
    <DialogTitle id="smart-cleanup-title">Smart cleanup</DialogTitle>
    <DialogContent className="smart-cleanup-content" sx={{ overflowWrap: "anywhere" }}>
      {(state.phase === "checking" || state.phase === "running") && <CircularProgress size={24}
        className="smart-cleanup-progress" aria-label={state.phase === "checking" ? "Checking cleanup readiness" : "Cleanup in progress"} />}
      <DialogContentText id="smart-cleanup-status" role="status">
        {state.message}
      </DialogContentText>
      {state.phase === "confirm" && <DialogContentText className="smart-cleanup-consent" sx={{ mt: 2 }}>
        Remove safely identified OCR noise from the reading copy? Chapter structure, original text,
        saved notes and manual edits are preserved. Generated interactions are invalidated only if
        reading content changes. Cleanup can be undone.
      </DialogContentText>}
    </DialogContent>
    <DialogActions className="smart-cleanup-actions" sx={{ flexWrap: "wrap", gap: 1 }}>
      <Button className="smart-cleanup-close" autoFocus onClick={props.onClose}>
        {state.phase === "confirm" || state.phase === "checking" ? "Cancel" : "Close"}
      </Button>
      {(state.phase === "unknown" || state.phase === "reload") && <Button className="smart-cleanup-reconcile"
        startIcon={<RefreshIcon />} onClick={() => { void state.check(); }}>
        {state.phase === "reload" ? "Retry reading" : "Check stored state"}
      </Button>}
      {state.phase === "resume" && <Button className="smart-cleanup-resume" variant="contained"
        startIcon={<PlayArrowIcon />} onClick={() => { void state.resume(); }}>Resume reading</Button>}
      {(state.phase === "blocked" || state.phase === "done" || state.expired) && state.canCheck &&
        <Button className="smart-cleanup-fresh" startIcon={<RefreshIcon />} onClick={() => { void state.fresh(); }}>
          {state.result?.retry_action === "regenerate_cleanup" ? "Retry cleanup" : "Check again"}
        </Button>}
      {state.phase === "confirm" && <Button className="smart-cleanup-submit" startIcon={<CleaningServicesOutlinedIcon />}
        variant="contained" disabled={!state.canSubmit} onClick={() => { void state.submit(); }}>Clean text</Button>}
    </DialogActions>
  </Dialog>;
}
