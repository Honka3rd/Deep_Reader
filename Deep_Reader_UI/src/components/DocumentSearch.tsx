import SearchIcon from "@mui/icons-material/Search";
import { Autocomplete, Box, Button, CircularProgress, TextField } from "@mui/material";
import { createFilterOptions } from "@mui/material/Autocomplete";
import type { FormEvent } from "react";

interface DocumentSearchProps {
  value: string;
  options: string[];
  loading: boolean;
  searching: boolean;
  onChange: (value: string) => void;
  onOpen: () => void;
  onLoad: () => void;
}

const filterOptions = createFilterOptions<string>({
  ignoreAccents: true,
  ignoreCase: true,
  matchFrom: "any",
  stringify: (option) => option,
  trim: true,
});

export function DocumentSearch({
  value,
  options,
  loading,
  searching,
  onChange,
  onOpen,
  onLoad,
}: DocumentSearchProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onLoad();
  }

  return (
    <Box component="form" className="document-search" onSubmit={handleSubmit}>
      <Autocomplete
        autoHighlight
        clearOnEscape
        options={options}
        filterOptions={filterOptions}
        value={options.includes(value) ? value : null}
        inputValue={value}
        onInputChange={(_, newValue) => onChange(newValue)}
        onChange={(_, newValue) => onChange(newValue || "")}
        onOpen={onOpen}
        disabled={loading}
        loading={searching}
        noOptionsText="No matching documents"
        renderInput={(params) => (
          <TextField
            {...params}
            label="Document"
            placeholder="Search documents"
            required
            size="small"
            helperText="Select one document returned by the API."
            inputProps={{
              ...params.inputProps,
              "aria-label": "Document name",
            }}
            InputProps={{
              ...params.InputProps,
              endAdornment: (
                <>
                  {searching ? <CircularProgress color="inherit" size={18} /> : null}
                  {params.InputProps.endAdornment}
                </>
              ),
            }}
          />
        )}
      />
      <Button
        type="submit"
        variant="contained"
        startIcon={<SearchIcon />}
        disabled={loading || searching || !options.includes(value)}
      >
        Load
      </Button>
    </Box>
  );
}
