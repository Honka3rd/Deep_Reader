import { useEffect, type ReactNode } from "react";
import { Box, Button, LinearProgress, Stack, Typography } from "@mui/material";
import type { ReaderContentBlock } from "../reader-content/model";
import { useLearningPage, useReaderMode, useOrdinaryReading } from "./controller";
import { pageProgress, type LearningSessionMemory } from "./model";
import { ReaderModeControl, useReadingSettingsController } from "../reading-settings";

export interface ReaderModeScope { docName: string; suspended: boolean; revision: number; memory: LearningSessionMemory;
  onSettingsOpenChange?: (open: boolean) => void; onBeforeSettingsOpen?: () => boolean }
import { LearningContext, OrdinaryTranslationContext } from "./context";
interface ModeProps { scope?: ReaderModeScope; blocks: ReaderContentBlock[];
  onVisibleBlocks?: (blocks: ReaderContentBlock[]) => void;
  children: (presentation: { ordinary: boolean; status: ReactNode; modeControl: ReactNode; forwardBlocked: boolean; forwardBlockReason: string }) => ReactNode;
}
const labels = { queued: "Queued", reading: "Reading saved data", pending: "In progress", ready: "Saved", empty: "No learning content",
  partial: "Partially completed", failed: "Failed", unknown: "Outcome unknown", stale: "Source or settings changed", unavailable: "Unavailable", error: "Read failed" };
function LearningPageStatus({ c, progress }: { c: ReturnType<typeof useLearningPage>; progress: ReturnType<typeof pageProgress> }) {
  const summary = c.state.loading ? "Reading saved learning data…"
    : c.state.error ? "Learning progress could not be checked."
    : `${progress.completed} / ${progress.total} tasks complete${progress.active ? ` · ${progress.active} unfinished` : ""}${progress.attention ? ` · ${progress.attention} need attention` : ""}`;
  return <Box className="learning-page-status" component="section" aria-label="Language learning progress" sx={{ border: 1, borderColor: "divider", borderRadius: 1, p: 1.5 }}>
    <Typography variant="subtitle2">Language learning · current page</Typography>
    <Typography className="learning-progress-summary" role="status" variant="body2" sx={{ my: .5, minHeight: { xs: "3em", sm: "1.5em" } }}>{summary}</Typography>
    <LinearProgress className="learning-progress-bar" aria-label="Current-page learning completion"
      aria-valuetext={summary} variant={c.state.loading ? "indeterminate" : "determinate"}
      value={progress.percent} color={progress.attention || progress.checking || c.state.error ? "warning" : "primary"} sx={{ mb: 1 }} />
    <Typography className="learning-progress-navigation-hint" variant="caption" component="p" sx={{ minHeight: { xs: "3.5em", sm: "1.75em" } }}>
      {progress.checking ? "Check unfinished tasks below to unlock Next page."
        : progress.blocked ? "Preparing learning tasks. Next page is paused."
        : progress.attention || c.state.error ? "Some tasks need attention. You can continue reading."
        : "This page is ready. You can continue reading."}
    </Typography>
    {c.state.error && <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
      <Typography role="status" variant="body2">{c.state.error}</Typography>
      <Button className="learning-page-read-again" onClick={c.reload}>Read page again</Button>
    </Stack>}
    <Box component="details" className="learning-progress-overview">
      <Box component="summary" className="learning-progress-toggle" sx={{ cursor: "pointer", fontSize: 14 }}>Progress by paragraph</Box>
    <Box className="learning-progress-list" role="region" aria-label="Current-page learning components" tabIndex={0} sx={{ maxHeight: 180, overflowY: "auto" }}>
      {Object.entries(c.state.items).map(([key, item], index) => <Stack className="learning-progress-item" key={key} data-block-id={item.blockId} data-component={item.component}
        direction="row" alignItems="center" flexWrap="wrap" gap={1} sx={{ py: .5 }}>
        <Typography className="learning-progress-label" variant="body2">Paragraph {Math.floor(index / 2) + 1} · {item.component === "translation" ? "Translation" : "Vocabulary"}</Typography>
        <Typography className="learning-progress-state" variant="body2">{labels[item.phase]}{item.view.saved && item.phase !== "ready" ? " · saved result retained" : ""}</Typography>
        {item.message && <Typography className="learning-progress-message" variant="caption">{item.message}</Typography>}
        {!["queued", "ready", "empty"].includes(item.phase) && <Button className="learning-progress-check" size="small" onClick={() => c.check(key)}>Check progress</Button>}
      </Stack>)}
    </Box>
    </Box>
  </Box>;
}
// The stable provider keeps source DOM/focus/scroll mounted while mode/data reads finish.
export function ReaderModeBoundary({ scope, blocks, children, onVisibleBlocks }: ModeProps) {
  const settings = useReadingSettingsController(scope && !scope.suspended ? scope.docName : false, { closeOnBookSave: true });
  const settingsOpen = !!settings.target;
  const onSettingsOpenChange = scope?.onSettingsOpenChange;
  useEffect(() => { onSettingsOpenChange?.(settingsOpen); return () => onSettingsOpenChange?.(false); }, [settingsOpen, onSettingsOpenChange]);
  const mode = useReaderMode({ docName: scope?.docName ?? "", revision: scope?.revision ?? 0,
    suspended: !scope || scope.suspended || settingsOpen }, blocks);
  const book = !mode.hidden && mode.current.book?.effective_mode === "language_learning"
    && mode.current.book.modes.some(m => m.mode_id === "language_learning" && m.settings_version === "1") ? mode.current.book : null;
  const c = useLearningPage(book, mode.sources, scope?.memory);
  const progress = pageProgress(c.state, mode.sources.length * 2);
  const checkingMode = !!scope && !mode.hidden && !mode.current.book && !mode.current.error;
  const forwardBlocked = checkingMode || (!!book && progress.blocked);
  const forwardBlockReason = checkingMode ? "Checking this page’s reading mode…"
    : forwardBlocked ? "Current-page learning is unfinished. See Language learning progress below." : "";
  const ordinary = !scope || (!mode.hidden && mode.current.book?.effective_mode === "normal");
  useOrdinaryReading(ordinary, blocks, onVisibleBlocks);
  let status: ReactNode = null;
  if (!mode.hidden && mode.current.error) status = <Stack className="reader-mode-error" direction="row" alignItems="center" flexWrap="wrap" gap={1}>
    <Typography variant="body2" role="status">{mode.current.error}</Typography>
    <Button className="reader-mode-recheck" onClick={mode.retryRead}>Check reading mode</Button>
  </Stack>;
  else if (!mode.hidden && mode.current.book && !ordinary && !book) status = <Typography className="reader-mode-unavailable" role="status" variant="body2">
    Reading mode unavailable. Use the mode button above to review this book’s settings.
  </Typography>;
  else if (book) status = <LearningPageStatus c={c} progress={progress} />;
  const modeControl = scope ? <ReaderModeControl book={settingsOpen ? settings.book ?? undefined : mode.current.book}
    failed={!!mode.current.error} disabled={scope.suspended} controller={settings}
    onOpen={() => { if (scope.onBeforeSettingsOpen?.() !== false) settings.openBook(scope.docName, scope.docName); }} /> : null;
  return <OrdinaryTranslationContext.Provider value={ordinary}><LearningContext.Provider value={book ? { book, controller: c } : null}>{children({ ordinary, status, modeControl, forwardBlocked, forwardBlockReason })}</LearningContext.Provider></OrdinaryTranslationContext.Provider>;
}
