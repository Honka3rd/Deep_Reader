import { useEffect, useRef } from "react";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import CloseIcon from "@mui/icons-material/Close";
import { Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, List, ListItemButton, ListItemText, MenuItem, Stack, TextField, Typography, useMediaQuery, useTheme } from "@mui/material";
import { AppNotification } from "../../shared/components/AppNotification";
import type { ReadingSettingsController } from "./controller";
import { languageLabel } from "./model";
import { settingsFeatures } from "./features";

export function ReadingSettingsView({ controller: c, origin = "library" }: { controller: ReadingSettingsController; origin?: "library" | "reader" }) {
  const narrow = useMediaQuery(useTheme().breakpoints.down("sm"));
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (c.status !== "ready" && !c.confirm) return;
    const selector = c.confirm ? ".reading-settings-keep" : c.target?.kind === "workspace"
      ? "#reading-settings-native" : narrow && c.detail ? ".reading-mode-back" : '.reading-mode-option[aria-current="true"], .reading-mode-option';
    content.current?.querySelector<HTMLElement>(selector)?.focus();
  }, [c.status, c.target, c.detail, c.confirm, narrow]);
  if (!c.target) return null;
  const isBook = c.target.kind === "book", book = c.book;
  const candidate = book?.modes.find(m => m.mode_id === c.draft?.mode);
  const Panel = c.feature?.Panel;
  const disabled = c.busy || c.recovery;
  return <Dialog className="reading-settings-dialog" open fullWidth maxWidth="md" fullScreen={narrow}
    sx={{ "& .app-notification": { position: "absolute", bottom: { xs: 160, sm: 88 } } }}
    aria-labelledby="reading-settings-title" aria-describedby="reading-settings-purpose" onClose={c.close}>
    <DialogTitle className="reading-settings-title" id="reading-settings-title">
      <Stack direction="row" justifyContent="space-between" alignItems="start" gap={1}>
        <Box minWidth={0} sx={{ overflowWrap: "anywhere" }}>
          {isBook ? "Reading mode" : "Workspace preferences"}
          {c.target.kind === "book" && <Typography className="reading-settings-book-title" variant="body2" tabIndex={0} sx={{ maxHeight: 88, overflowY: "auto" }}>{c.target.title}</Typography>}
        </Box>
        <IconButton className="reading-settings-close" aria-label="Close reading settings" onClick={c.close}><CloseIcon /></IconButton>
      </Stack>
    </DialogTitle>
    <DialogContent ref={content} className="reading-settings-content" dividers sx={{ minHeight: 280, overflowY: "auto" }}>
      <Typography id="reading-settings-purpose" variant="body2" sx={{ mb: 2 }}>
        {isBook ? origin === "reader"
          ? "Choose a mode, then save and return to reading. Language learning prepares any missing translations and vocabulary for the current page."
          : "Preview a mode, then save and apply it to this book. Saving settings does not generate learning content."
          : "Your native language is shared by books in this workspace. Changing it can make a saved learning mode unavailable."}
      </Typography>
      {c.confirm && <Stack className="reading-settings-discard" role="region" aria-label="Unsaved settings" gap={2}>
        <Typography variant="h6">Leave these settings?</Typography>
        <Typography>{c.busy || c.recovery ? "A save may already have reached the server. Leaving stops this view; reopen settings to read the current result."
          : "Your unsaved changes will be discarded."}</Typography>
        <Stack direction="row" gap={1} flexWrap="wrap">
          <Button className="reading-settings-keep" autoFocus variant="contained" onClick={c.keepEditing}>Keep editing</Button>
          <Button className="reading-settings-discard-action" color="warning" onClick={c.discard}>Discard draft and continue</Button>
        </Stack>
      </Stack>}
      {/* Keep the active feature's local steps/draft mounted while confirming departure. */}
      <Box className="reading-settings-panel" hidden={c.confirm} sx={{ display: c.confirm ? "none" : "block" }}>
      {c.status !== "ready" ? <Stack className="reading-settings-state" gap={2} role="status">
        <Typography>{c.status === "loading" ? "Loading reading settings…" : "Reading settings could not be loaded."}</Typography>
        {c.status === "error" && <Button className="reading-settings-retry" onClick={() => { void c.reload(); }}>Try again</Button>}
      </Stack> : !isBook ? <Stack className="workspace-reading-settings" gap={2}>
        <TextField className="reading-settings-native" id="reading-settings-native" select label="Native language" fullWidth autoFocus
          value={c.native ?? ""} disabled={disabled} error={c.invalidNative}
          helperText={c.invalidNative ? "Choose a supported language." : "Used for sentence translations and vocabulary explanations."}
          onChange={e => c.changeNative(e.target.value || null)}>
          <MenuItem value="">Not set</MenuItem>
          {c.native && !c.options?.language_codes.includes(c.native) && <MenuItem value={c.native} disabled>Unsupported saved language</MenuItem>}
          {c.options?.language_codes.map(code => <MenuItem key={code} value={code}>{languageLabel(code)}</MenuItem>)}
        </TextField>
      </Stack> : book && c.draft && c.options ? <Stack className="reading-mode-layout" gap={2}>
        <Stack className="reading-mode-active" direction="row" flexWrap="wrap" gap={1} alignItems="center">
          <Typography variant="body2">Applied mode: {book.modes.find(m => m.mode_id === book.active_mode)?.label ?? "Unsupported saved mode"}</Typography>
          <Chip size="small" label={book.effective_mode ? "Active" : "Unavailable"} />
        </Stack>
        {!book.effective_mode && <Typography className="reading-mode-active-reason" variant="body2">
          {book.active_reasons.map(reason => reason.code === "book_deleted"
            ? "This book was re-imported. Review its settings, then Save and apply to activate a reading mode."
            : reason.message).join(" ")}
        </Typography>}
        <Box className="reading-mode-columns" sx={{ display: "grid", gridTemplateColumns: { xs: "minmax(0, 1fr)", sm: "minmax(180px, 1fr) minmax(0, 2fr)" }, gap: 2 }}>
          {(!narrow || !c.detail) && <List className="reading-mode-list" component="nav" aria-label="Reading modes" disablePadding>
            {book.modes.map(mode => {
              const supported = settingsFeatures[mode.mode_id]?.version === mode.settings_version;
              return <ListItemButton className="reading-mode-option" key={mode.mode_id} data-mode-id={mode.mode_id}
                selected={c.draft?.mode === mode.mode_id} aria-current={c.draft?.mode === mode.mode_id ? "true" : undefined}
                aria-controls="reading-mode-detail" disabled={c.busy} onClick={() => c.choose(mode.mode_id)} sx={{ alignItems: "start", borderRadius: 1 }}>
                <ListItemText primary={mode.label} secondary={<>
                  {mode.description}<br />
                  {mode.mode_id === book.active_mode ? "Applied · " : ""}{!supported ? "Unsupported" : mode.eligible ? "Available" : mode.reasons.some(reason => ["same_language", "book_deleted", "mode_unknown"].includes(reason.code)) ? "Unavailable" : "Needs settings"}
                </>} />
              </ListItemButton>;
            })}
          </List>}
          {(!narrow || c.detail) && <Stack className="reading-mode-detail" id="reading-mode-detail" component="section" aria-label="Mode settings" gap={2} sx={{ minWidth: 0 }}>
            {narrow && <Button className="reading-mode-back" startIcon={<ArrowBackIcon />} onClick={c.backToList} sx={{ alignSelf: "start" }}>Back to modes</Button>}
            <Typography className="reading-mode-candidate" variant="h6">{candidate?.label ?? "Unsupported saved mode"}</Typography>
            <Typography variant="caption">{c.draft.mode === book.active_mode ? "Applied mode settings" : "Preview · not applied"}</Typography>
            {Panel && <Panel key={c.draft.mode} draft={c.draft} book={book} options={c.options} disabled={disabled} onChange={c.changeDraft} />}
            {c.problems.length > 0 && <Box className="reading-mode-validation" role="status">
              {c.problems.map(problem => <Typography variant="body2" key={problem} sx={{ mb: 1 }}>{problem}</Typography>)}
            </Box>}
            <Button className="reading-mode-workspace" variant="outlined" disabled={c.busy} onClick={c.openWorkspace}>Workspace preferences</Button>
          </Stack>}
        </Box>
      </Stack> : null}
      </Box>
      {!c.confirm && c.recovery && <Typography className="reading-settings-recovery" role="status" sx={{ mt: 2 }}>Read current settings to resolve the previous save before submitting again.</Typography>}
      <AppNotification notification={c.notification} onClose={c.closeNotification} />
    </DialogContent>
    {!c.confirm && <DialogActions className="reading-settings-actions" sx={{ display: { xs: "grid", sm: "flex" }, gridTemplateColumns: "repeat(2, minmax(0, 1fr))", flexWrap: "wrap", gap: 1, p: 2, "& > :not(style) ~ :not(style)": { ml: 0 } }}>
      <Button className="reading-settings-cancel" onClick={c.close}>Back to {c.target.kind === "workspace" && c.target.back ? "book" : origin === "reader" ? "reader" : "Library"}</Button>
      {c.status === "ready" && <Button className="reading-settings-reload" disabled={c.busy} onClick={() => { void c.reload(); }}>Read current settings</Button>}
      {(!isBook || !narrow || c.detail) && <Button className="reading-settings-save" sx={{ gridColumn: "1 / -1" }} variant="contained" disabled={!c.canSave} onClick={() => { void c.save(); }}>
        {c.busy ? "Please wait…" : isBook ? "Save and apply" : "Save preferences"}
      </Button>}
    </DialogActions>}
  </Dialog>;
}
