import type { ComponentType } from "react";
import { Stack, Typography } from "@mui/material";
import { LanguageLearningSettings, SourceLanguageField, type SettingsPanelProps } from "./language-learning/view";
import { learningProblems, type SettingsFeaturePolicy } from "./model";

export interface ReadingSettingsFeature extends SettingsFeaturePolicy { Panel: ComponentType<SettingsPanelProps> }
function NormalSettings(props: SettingsPanelProps) {
  return <Stack className="normal-reading-settings" gap={2}>
    <Typography variant="body2">Read the original text with the usual reading tools. No language-learning level is required.</Typography>
    <SourceLanguageField {...props} />
  </Stack>;
}
// Local typed components; the backend supplies the catalog, never executable UI.
export const settingsFeatures: Readonly<Record<string, ReadingSettingsFeature>> = {
  normal: { version: "1", Panel: NormalSettings, problems: () => [], selection: () => ({ mode_id: "normal", version: "1" }) },
  language_learning: { version: "1", Panel: LanguageLearningSettings, problems: learningProblems,
    selection: draft => { if (!draft.learning) throw new Error("Missing learning selection"); return draft.learning; } },
};
