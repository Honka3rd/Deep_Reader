import { Autocomplete, Box, CircularProgress, TextField } from "@mui/material";
import { createFilterOptions } from "@mui/material/Autocomplete";

interface BookSearchViewProps {
  value: string;
  inputValue: string;
  options: string[];
  loading: boolean;
  searching: boolean;
  onChange: (value: string) => void;
  onOpen: () => void;
  onSelect: (value: string) => void;
}

const filterOptions = createFilterOptions<string>({
  ignoreAccents: true,
  ignoreCase: true,
  matchFrom: "any",
  stringify: (option) => option,
  trim: true,
});

export function BookSearchView({
  value,
  inputValue,
  options,
  loading,
  searching,
  onChange,
  onOpen,
  onSelect,
}: BookSearchViewProps) {
  return (
    <Box className="document-search document-search-form">
      <Autocomplete
        className="document-search-combobox"
        autoHighlight
        clearOnEscape
        options={value && !options.includes(value) ? [value, ...options] : options}
        filterOptions={filterOptions}
        value={value || null}
        inputValue={inputValue}
        onInputChange={(_, newValue, reason) => {
          if (reason === "input" || reason === "clear") onChange(newValue);
        }}
        onChange={(_, newValue) => {
          const nextValue = newValue || "";
          onChange(nextValue);
          if (newValue) {
            onSelect(newValue);
          }
        }}
        onOpen={onOpen}
        disabled={loading}
        loading={searching}
        noOptionsText="No matching documents"
        renderInput={(params) => (
          <TextField
            {...params}
            className="document-search-input"
            label="Select one Document"
            placeholder="Search documents"
            required
            size="small"
            inputProps={{
              ...params.inputProps,
              "aria-label": "Document name",
            }}
            InputProps={{
              ...params.InputProps,
              endAdornment: (
                <>
                  {searching ? (
                    <CircularProgress
                      className="document-search-loading-indicator"
                      color="inherit"
                      size={18}
                    />
                  ) : null}
                  {params.InputProps.endAdornment}
                </>
              ),
            }}
          />
        )}
      />
    </Box>
  );
}
