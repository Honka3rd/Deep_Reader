import { useEffect, useRef, useState } from "react";
import { Autocomplete, Box, Button, Checkbox, Chip, CircularProgress, FormControlLabel, FormGroup, Dialog, DialogActions, DialogContent,
  DialogTitle, Drawer, IconButton, Stack, Tab, Tabs, TextField, Tooltip, Typography } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";
import TranslateIcon from "@mui/icons-material/Translate";
import RefreshIcon from "@mui/icons-material/Refresh";
import { AppNotification } from "../../shared/components/AppNotification";
import { languageLabel, savedSections } from "./model";
import type { BlockTranslationController } from "./controller";

export function BlockTranslationView({ controller: c }: { controller: BlockTranslationController }) {
  const [tab, setTab] = useState("translations");
  useEffect(() => { setTab("translations"); }, [c.selection?.doc_name, c.selection?.content_block_id, c.surface]);
  const visibleRecords = (c.data?.translations ?? []).filter(record => !c.inlineLanguages.includes(record.target_language));
  const inlineLanguageKey = c.inlineLanguages.join(",");
  const sections = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (c.surface !== "saved" || tab !== "translations" || !c.focusLanguage) return;
    const section = Array.from(sections.current?.querySelectorAll<HTMLElement>("[data-language-code]") ?? [])
      .find(item => item.dataset.languageCode === c.focusLanguage);
    if (section && sections.current) {
      sections.current.scrollTop = 0;
      section.focus({ preventScroll: true });
    }
  }, [c.surface, c.focusLanguage, c.status, tab, inlineLanguageKey]);
  const notification = <AppNotification notification={c.notification} onClose={c.dismissNotification}
    onAction={c.notificationAction} />;
  const loading = c.status === "loading";
  const state = loading ? <Stack className="translation-loading" role="status" direction="row" spacing={2} alignItems="center">
    <CircularProgress size={20} /><Typography>Loading translations...</Typography></Stack>
    : c.status === "failed" ? <Stack className="translation-failed" spacing={1}>
      <Typography>Translations unavailable.</Typography><Button startIcon={<RefreshIcon />} onClick={c.retry}>Retry</Button></Stack> : null;
  return <Box className="block-translation-feature">
    <Dialog className="translation-language-dialog" open={c.surface === "languages"} onClose={c.close}
      fullWidth maxWidth="sm" aria-labelledby="translation-language-title">
      <DialogTitle id="translation-language-title">Translate block</DialogTitle>
      <DialogContent sx={{ overflowY: "auto" }}>
        {state}
        <Stack spacing={3} sx={{ pt: 2 }}>
          {(["from", "to"] as const).map(direction => <Autocomplete key={direction}
            id={`translation-${direction}`} className={`translation-${direction}-control`}
            options={direction === "from" ? c.data?.language_codes ?? [] : c.targetLanguageCodes} value={direction === "from" ? c.from : c.to}
            disabled={c.status !== "ready"} freeSolo={false} autoHighlight
            getOptionLabel={languageLabel}
            onChange={(_event, value) => direction === "from" ? c.setFrom(value) : c.setTo(value)}
            renderInput={params => <TextField {...params} autoFocus={direction === "from"}
              label={`Translate ${direction}`} required />}
            renderOption={(props, code) => <li {...props} key={code} className={`${props.className ?? ""} translation-language-option`} data-language-code={code}>
              {languageLabel(code)}</li>} />)}
        </Stack>
      </DialogContent>
      <DialogActions><Button onClick={c.close}>Cancel</Button>
        <Button className="translation-submit" variant="contained" startIcon={<TranslateIcon />}
          disabled={!c.canSubmit} onClick={() => void c.submit()}>Translate</Button></DialogActions>
      {notification}
    </Dialog>
    <Drawer className="translation-drawer" anchor="right" open={c.surface === "saved"} onClose={c.close}
      PaperProps={{ role: "dialog", "aria-modal": true, "aria-labelledby": "translation-saved-title",
        sx: { width: { xs: "100%", sm: 560 }, maxWidth: "100vw", overflow: "hidden" } }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: "divider" }}>
        <Typography id="translation-saved-title" variant="h6" sx={{ flex: 1 }}>Translations</Typography>
        <Tooltip title="Reload translations"><span><IconButton aria-label="Reload translations" disabled={loading} onClick={c.retry}><RefreshIcon /></IconButton></span></Tooltip>
        <Tooltip title="Close translations"><IconButton autoFocus aria-label="Close translations" onClick={c.close}><CloseIcon /></IconButton></Tooltip>
      </Stack>
      <Tabs className="translation-drawer-tabs" value={tab} onChange={(_event, value) => setTab(value)}
        aria-label="Translation views" variant="fullWidth" sx={{ flexShrink: 0 }}>
        <Tab id="translation-saved-tab" aria-controls="translation-saved-panel" value="translations" label="Translations" />
        <Tab id="translation-display-tab" aria-controls="translation-display-panel" value="display" label="Reading display" />
      </Tabs>
      {state && <Box sx={{ p: 2 }}>{state}</Box>}
      <Box ref={sections} className="translation-saved-sections" id="translation-saved-panel"
        role="tabpanel" aria-labelledby="translation-saved-tab" hidden={tab !== "translations"}
        sx={{ flex: 1, minHeight: 0, overflowY: "auto", p: 2 }}>
        {c.status === "ready" && c.data?.translations.length === 0 && <Typography className="translation-empty">No saved translations.</Typography>}
        {c.status === "ready" && Boolean(c.data?.translations.length) && visibleRecords.length === 0
          && <Typography className="translation-all-inline">Your saved translations are shown below the original text. Change Reading display to show them here.</Typography>}
        {c.status === "ready" && savedSections(visibleRecords, c.focusLanguage).map(record => <Box component="section"
          className={`translation-saved-section${record.target_language === c.focusLanguage ? " translation-saved-section-active" : ""}`}
          key={record.target_language} data-language-code={record.target_language}
          aria-current={record.target_language === c.focusLanguage ? "true" : undefined}
          tabIndex={-1} aria-label={languageLabel(record.target_language)} sx={{ mb: 3, p: 1.5,
            borderLeft: 3, borderColor: record.target_language === c.focusLanguage ? "primary.main" : "transparent",
            bgcolor: record.target_language === c.focusLanguage ? "action.selected" : "transparent",
            overflowWrap: "anywhere" }}>
          <Stack direction="row" flexWrap="wrap" gap={1} alignItems="center">
            <Typography variant="subtitle1" component="h3">{languageLabel(record.target_language)}</Typography>
            <Chip size="small" label={record.status === "partial" ? "Partial" : "Completed"}
              color={record.status === "partial" ? "warning" : "default"} />
          </Stack>
          <Typography variant="caption" color="text.secondary" component="p">From {languageLabel(record.source_language)}</Typography>
          <Typography variant="caption" color="text.secondary" component="p">Updated <time dateTime={record.updated_at}>{new Date(record.updated_at).toLocaleString()}</time></Typography>
          <Typography className="translation-saved-text" dir="auto" sx={{ mt: 1, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{record.translated_text}</Typography>
        </Box>)}
      </Box>
      <Box className="translation-display-settings" id="translation-display-panel" role="tabpanel"
        aria-labelledby="translation-display-tab" hidden={tab !== "display"}
        sx={{ flex: 1, minHeight: 0, overflowY: "auto", p: 2 }}>
        <Typography component="h3" variant="subtitle1">Show translations while reading</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Choose languages to show below the original text, after Insight when shown.
          Your choices apply across books until you reload or close this page. Only saved translations are shown.
        </Typography>
        {c.status === "ready" && !c.data?.translations.length && <Typography>This block has no saved translations. Your language choices will apply when translations are available.</Typography>}
        <FormGroup aria-label="Languages shown during this session">
          {c.inlineLanguageOptions.map(code => <FormControlLabel key={code}
            className="translation-inline-language" data-language-code={code}
            label={languageLabel(code)} control={<Checkbox
              checked={c.inlineLanguages.includes(code)} disabled={c.status !== "ready"}
              onChange={(_event, checked) => c.setInlineLanguage(code, checked)} />} />)}
        </FormGroup>
      </Box>
      <Box sx={{ p: 2, borderTop: 1, borderColor: "divider" }}><Button startIcon={<AddIcon />} disabled={c.status !== "ready"}
        onClick={c.add}>Add translation</Button></Box>
      {notification}
    </Drawer>
    {!c.surface && notification}
  </Box>;
}
