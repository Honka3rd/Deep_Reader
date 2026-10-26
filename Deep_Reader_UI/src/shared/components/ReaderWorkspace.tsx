import { Box } from "@mui/material";
import type { CSSProperties, ReactNode } from "react";
import { useReaderPaneResize } from "./useReaderPaneResize";

export function ReaderWorkspace({ children }: { children: [ReactNode, ReactNode] }) {
  const resize = useReaderPaneResize();
  return (
    <Box component="main" ref={resize.workspaceRef}
      className={`reader-layout reader-workspace${resize.dragging ? " reader-workspace-resizing" : ""}`}
      style={resize.width === null ? undefined : { "--reader-navigation-width": `${resize.width}px` } as CSSProperties}>
      {children[0]}
      <Box className="reader-pane-resizer" role="separator" tabIndex={0}
        aria-label="Resize document hierarchy" aria-orientation="vertical"
        aria-controls="document-hierarchy-pane" aria-valuemin={resize.minimumWidth}
        aria-valuemax={Math.round(resize.maximumWidth)} aria-valuenow={Math.round(resize.width ?? resize.minimumWidth)}
        title="Resize document hierarchy" {...resize.separatorProps} />
      {children[1]}
    </Box>
  );
}
