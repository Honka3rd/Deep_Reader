import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1f5f8b",
      dark: "#174766",
      light: "#d7ebf7",
    },
    secondary: {
      main: "#6a5b2f",
    },
    background: {
      default: "#f7f8f8",
      paper: "#ffffff",
    },
    text: {
      primary: "#172026",
      secondary: "#5a6872",
    },
  },
  shape: {
    borderRadius: 6,
  },
  typography: {
    fontFamily:
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h1: {
      fontSize: "1.55rem",
      lineHeight: 1.25,
      fontWeight: 700,
    },
    h2: {
      fontSize: "1.05rem",
      lineHeight: 1.3,
      fontWeight: 700,
    },
    h3: {
      fontSize: "0.92rem",
      lineHeight: 1.35,
      fontWeight: 700,
    },
    button: {
      textTransform: "none",
      fontWeight: 700,
    },
  },
});
