import { useEffect, useRef } from "react";
import { Box, Button, Chip, Stack, Typography } from "@mui/material";
import type { TranslationSavedRecord } from "../../services/TranslationService";
import type { InlineTranslationStatus } from "./useInlineTranslations";
import { languageLabel } from "./model";

export interface InlineTranslationViewState {
  records: TranslationSavedRecord[];
  status?: InlineTranslationStatus;
  onRetry?: () => void;
  focusLanguage?: string | null;
  canFocus?: boolean;
}
export function InlineTranslations({ records, status, onRetry, focusLanguage, canFocus }: InlineTranslationViewState) {
  const region = useRef<HTMLDivElement | null>(null);
  const focusedRecord = records.find(record => record.target_language === focusLanguage);
  useEffect(() => {
    if (!canFocus || !focusedRecord) return;
    const element = Array.from(region.current?.querySelectorAll<HTMLElement>("[data-language-code]") ?? [])
      .find(item => item.dataset.languageCode === focusedRecord.target_language);
    element?.focus({ preventScroll: true });
    element?.scrollIntoView({ block: "start", inline: "nearest" });
  }, [canFocus, focusedRecord]);
  if (!records.length && status !== "loading" && status !== "failed") return null;
  return <Stack ref={region} className="reader-content-block-inline-translations" spacing={1.5} sx={{ mt: 1.5 }}
    role="region" aria-label="Inline translations">
    {status === "loading" && <Typography role="status" variant="body2">Loading translations...</Typography>}
    {status === "failed" && <Stack direction="row" alignItems="center" spacing={1}>
      <Typography variant="body2">Translations could not be loaded.</Typography>
      <Button className="reader-inline-translation-retry" onClick={onRetry}>Retry translations</Button>
    </Stack>}
    {records.map(record => <Box component="section" className={`reader-inline-translation${record.target_language === focusLanguage ? " reader-inline-translation-active" : ""}`}
      tabIndex={-1} aria-current={record.target_language === focusLanguage ? "true" : undefined}
      key={record.target_language} data-language-code={record.target_language}
      aria-label={`${languageLabel(record.target_language)} translation`}
      sx={{ borderLeft: 3, borderColor: record.target_language === focusLanguage ? "primary.main" : "divider",
        bgcolor: record.target_language === focusLanguage ? "action.selected" : "transparent", pl: 2, minWidth: 0, overflowWrap: "anywhere" }}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography component="h3" variant="subtitle2">{languageLabel(record.target_language)}</Typography>
        {record.status === "partial" && <Chip size="small" color="warning" label="Partial" />}
      </Stack>
      <Typography className="reader-inline-translation-text" lang={record.target_language} dir="auto"
        variant="body1" sx={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{record.translated_text}</Typography>
    </Box>)}
  </Stack>;
}
