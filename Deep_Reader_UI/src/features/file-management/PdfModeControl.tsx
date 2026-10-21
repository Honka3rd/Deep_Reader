import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import type { PdfExtractionMode } from "../../types/api";

export function PdfModeControl({ value, disabled, onChange }: {
  value: PdfExtractionMode;
  disabled: boolean;
  onChange: (mode: PdfExtractionMode) => void;
}) {
  return (
    <ToggleButtonGroup className="pdf-mode-control" aria-label="PDF extraction mode"
      exclusive size="small" value={value} disabled={disabled}
      onChange={(_, mode: PdfExtractionMode | null) => { if (mode) onChange(mode); }}
      sx={{ my: 1, maxWidth: "100%", "& button": { minWidth: 64 } }}>
      <ToggleButton value="fast">Fast</ToggleButton>
      <ToggleButton value="ocr">OCR</ToggleButton>
      <ToggleButton value="auto">Auto</ToggleButton>
    </ToggleButtonGroup>
  );
}
