import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget = process.env.DEEP_READER_API_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: Number(process.env.DEEP_READER_UI_PORT || 5173),
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
