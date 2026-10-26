import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import { Box, IconButton, Paper, Tooltip } from "@mui/material";
import type { ReactNode } from "react";
import { useReaderFullscreen } from "./useReaderFullscreen";

export function ReaderContentPane({ children, busy, scopeKey }: {
  children: [ReactNode, ReactNode]; busy: boolean; scopeKey: string;
}) {
  const fullscreen = useReaderFullscreen(scopeKey);
  const label = fullscreen.active ? "Exit fullscreen reading" : "Fullscreen reading";
  return (
    <Paper component="section" id="reader-content-pane" elevation={0}
      className="content-pane reader-content-pane" aria-label="Reading content" aria-busy={busy}>
      <Box className="reader-content-toolbar">
        {children[0]}
        <Tooltip title={label}>
          <IconButton ref={fullscreen.buttonRef} className="reader-content-fullscreen-toggle"
            aria-label={label} aria-pressed={fullscreen.active} aria-controls="reader-content-pane"
            onClick={fullscreen.toggle}>
            {fullscreen.active ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </IconButton>
        </Tooltip>
      </Box>
      {children[1]}
    </Paper>
  );
}
