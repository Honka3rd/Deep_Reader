import { Alert, Box, CircularProgress, Typography } from "@mui/material";

interface StateViewProps {
  title: string;
  detail?: string;
  severity?: "info" | "error";
  loading?: boolean;
  className?: string;
}

export function StateView({
  title,
  detail,
  severity = "info",
  loading = false,
  className = "",
}: StateViewProps) {
  const stateClassName = ["state-view", className].filter(Boolean).join(" ");
  const alertClassName = ["state-alert", className].filter(Boolean).join(" ");

  if (severity === "error") {
    return (
      <Alert className={alertClassName} severity="error" role="alert">
        <Typography className="state-view-title" component="h2" variant="subtitle1" fontWeight={700}>
          {title}
        </Typography>
        {detail ? (
          <Typography className="state-view-detail" variant="body2">
            {detail}
          </Typography>
        ) : null}
      </Alert>
    );
  }

  return (
    <Box className={stateClassName}>
      {loading ? (
        <CircularProgress className="state-view-progress" size={22} aria-label={title} />
      ) : null}
      <Typography className="state-view-title" component="h2" variant="subtitle1" fontWeight={700}>
        {title}
      </Typography>
      {detail ? (
        <Typography className="state-view-detail" variant="body2" color="text.secondary">
          {detail}
        </Typography>
      ) : null}
    </Box>
  );
}
