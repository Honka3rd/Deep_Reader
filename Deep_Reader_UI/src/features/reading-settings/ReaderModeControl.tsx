import { Box, Button, Typography } from "@mui/material";
import TuneIcon from "@mui/icons-material/Tune";
import type { BookPreferences } from "../../services/ReadingPreferencesService";
import type { ReadingSettingsController } from "./controller";
import { ReadingSettingsView } from "./view";

export function ReaderModeControl({ book, failed, disabled, onOpen, controller }: {
  book?: BookPreferences; failed: boolean; disabled: boolean; onOpen: () => void;
  controller: ReadingSettingsController;
}) {
  const label = book ? `${book.modes.find(mode => mode.mode_id === book.active_mode)?.label ?? "Unsupported saved mode"}${book.effective_mode ? "" : " · Unavailable"}`
    : failed ? "Could not load" : "Checking…";
  return <Box className="reader-mode-control" sx={{ maxWidth: "100%", minWidth: 0 }}>
    <Button className="reader-mode-switch" size="small" variant="outlined" startIcon={<TuneIcon />}
      aria-label={`Reading mode: ${label}. Change reading mode`} aria-haspopup="dialog"
      aria-expanded={!!controller.target} disabled={disabled} onClick={onOpen}
      sx={{ maxWidth: "100%", minHeight: 32, textTransform: "none", textAlign: "left" }}>
      <Typography className="reader-mode-label" component="span" variant="body2" sx={{ overflowWrap: "anywhere", minWidth: 0 }}>
        Mode: {label}
      </Typography>
    </Button>
    <ReadingSettingsView controller={controller} origin="reader" />
  </Box>;
}
