import { Box, Button, Typography } from "@mui/material";
import { useReadingModeLabel, type createReadingModeQueue } from "./useReadingModeLabel";

export function ReadingModeLabel({ doc, active, read }: {
  doc: string; active: boolean; read: ReturnType<typeof createReadingModeQueue>;
}) {
  const { element, state, retry } = useReadingModeLabel(doc, active, read);
  const label = state.book?.modes.find(mode => mode.mode_id === state.book?.active_mode)?.label ?? "Unsupported saved mode";
  return <Box ref={element} className="file-management-mode-status" sx={{ minHeight: 20, maxWidth: 180 }}>
    <Typography className="file-management-mode-label" component="span" variant="caption" role="status" title="Current reading mode">
      {state.status === "loading" ? "Checking mode…" : state.status === "error" ? "Mode unavailable"
        : `${label}${state.book?.effective_mode ? "" : " · Unavailable"}`}
    </Typography>
    {state.status === "error" && <Button className="file-management-mode-retry" size="small"
      aria-label={`Retry reading mode for ${doc}`} onClick={retry}>Retry</Button>}
  </Box>;
}
