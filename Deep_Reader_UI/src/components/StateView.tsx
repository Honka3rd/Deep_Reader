import { Alert, Box, CircularProgress, Typography } from "@mui/material";

interface StateViewProps {
  title: string;
  detail?: string;
  severity?: "info" | "error";
  loading?: boolean;
}

export function StateView({
  title,
  detail,
  severity = "info",
  loading = false,
}: StateViewProps) {
  if (severity === "error") {
    return (
      <Alert severity="error" role="alert">
        <Typography component="h2" variant="subtitle1" fontWeight={700}>
          {title}
        </Typography>
        {detail ? <Typography variant="body2">{detail}</Typography> : null}
      </Alert>
    );
  }

  return (
    <Box className="state-view">
      {loading ? <CircularProgress size={22} aria-label={title} /> : null}
      <Typography component="h2" variant="subtitle1" fontWeight={700}>
        {title}
      </Typography>
      {detail ? (
        <Typography variant="body2" color="text.secondary">
          {detail}
        </Typography>
      ) : null}
    </Box>
  );
}
