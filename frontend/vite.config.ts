import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    // dev proxy: forwards these paths to your FastAPI backend at localhost:8000
    proxy: {
      "/predict": "http://localhost:8000",
      "/register_user": "http://localhost:8000",
      "/users": "http://localhost:8000",
      "/history": "http://localhost:8000",
      // websocket proxy (for /ws/predict)
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
