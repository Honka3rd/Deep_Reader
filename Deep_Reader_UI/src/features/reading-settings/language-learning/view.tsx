import { Box, MenuItem, Stack, TextField, Typography } from "@mui/material";
import type { SettingsContext, SettingsDraft } from "../model";
import { languageLabel, schemeFor, selectedLevel, sourceLanguage, withLevel } from "../model";

export interface SettingsPanelProps extends SettingsContext {
  draft: SettingsDraft; disabled: boolean; onChange: (draft: SettingsDraft) => void;
}
export function SourceLanguageField({ draft, book, options, disabled, onChange }: SettingsPanelProps) {
  const value = draft.override ?? "profile";
  return <TextField className="reading-settings-source" id="reading-settings-source" select fullWidth label="Original language"
    value={value} disabled={disabled} onChange={e => onChange({ ...draft, override: e.target.value === "profile" ? null : e.target.value, learning: null })}
    helperText="A reading preference only; this does not change the book’s text or saved profile.">
    <MenuItem value="profile">Use saved profile{book.source_origin === "profile" && book.source_language ? ` — ${languageLabel(book.source_language)}` : ""}</MenuItem>
    {draft.override && !options.language_codes.includes(draft.override) && <MenuItem value={draft.override} disabled>Unsupported saved language</MenuItem>}
    {options.language_codes.map(code => <MenuItem value={code} key={code}>{languageLabel(code)}</MenuItem>)}
  </TextField>;
}
export function LanguageLearningSettings(props: SettingsPanelProps) {
  const { draft, book, disabled, onChange } = props, scheme = schemeFor(draft, props), level = selectedLevel(draft, props);
  const source = sourceLanguage(draft, book);
  return <Stack className="language-learning-settings" gap={2}>
    <Typography variant="body2">Learn the book’s original language with sentence translations into your native language and reading-level vocabulary guidance.</Typography>
    <SourceLanguageField {...props} />
    <Typography variant="body2">Native language: {book.workspace.native_language ? languageLabel(book.workspace.native_language) : "Not set"}</Typography>
    {source && source !== book.workspace.native_language && scheme && <Box className="reading-settings-proficiency">
      <TextField id="reading-settings-level" className="reading-settings-level" select fullWidth label="Reading level" value={level}
        disabled={disabled} onChange={e => onChange(withLevel(draft, props, e.target.value))}>
        <MenuItem value="" disabled>Choose your level</MenuItem>
        {scheme.levels.map(item => <MenuItem value={item.level_id} key={item.level_id}>{item.label}</MenuItem>)}
      </TextField>
      <Typography variant="caption" display="block" sx={{ mt: 1 }}>{scheme.label} — {scheme.scope_note}</Typography>
      {level && <Typography className="reading-settings-level-description" variant="body2" sx={{ mt: 1 }}>{scheme.levels.find(l => l.level_id === level)?.reading_description}</Typography>}
    </Box>}
  </Stack>;
}
