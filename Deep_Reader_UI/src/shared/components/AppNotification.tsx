import CloseIcon from "@mui/icons-material/Close";
import {
  Alert,
  Button,
  IconButton,
  Snackbar,
  type AlertColor,
} from "@mui/material";

export interface AppNotificationState {
  open: boolean;
  message: string;
  severity: AlertColor;
  actionLabel?: string;
}

interface AppNotificationProps {
  notification: AppNotificationState;
  onClose: () => void;
  onAction?: () => void;
}

export function AppNotification({
  notification,
  onClose,
  onAction,
}: AppNotificationProps) {
  const autoHideDuration = notification.severity === "error" ? null : 5000;
  const action =
    notification.actionLabel && onAction ? (
      <Button color="inherit" size="small" onClick={onAction}>
        {notification.actionLabel}
      </Button>
    ) : null;

  return (
    <Snackbar
      className="app-notification"
      open={notification.open}
      autoHideDuration={autoHideDuration}
      anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      onClose={(_event, reason) => {
        if (reason !== "clickaway") {
          onClose();
        }
      }}
    >
      <Alert
        severity={notification.severity}
        variant="filled"
        action={
          <>
            {action}
            <IconButton
              aria-label="Close notification"
              color="inherit"
              size="small"
              onClick={onClose}
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </>
        }
      >
        {notification.message}
      </Alert>
    </Snackbar>
  );
}
