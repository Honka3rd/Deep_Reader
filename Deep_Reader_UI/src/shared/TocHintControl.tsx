import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import { useId } from "react";
import type { TocHint } from "../types/api";

export function TocHintControl({ value, disabled, onChange }: {
  value: TocHint; disabled: boolean; onChange: (hint: TocHint) => void;
}) {
  const id = useId();
  return <FormControl size="small" fullWidth sx={{ my: 1 }}>
    <InputLabel id={id}>Printed table of contents</InputLabel>
    <Select labelId={id} label="Printed table of contents" value={value} disabled={disabled}
      onChange={(event) => onChange(event.target.value as TocHint)}>
      <MenuItem value="auto">Auto</MenuItem>
      <MenuItem value="present">Present</MenuItem>
      <MenuItem value="absent">Absent</MenuItem>
    </Select>
  </FormControl>;
}
