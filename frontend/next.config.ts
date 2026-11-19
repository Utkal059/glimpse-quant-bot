import type { NextConfig } from "next";

const config: NextConfig = {
  // Self-contained server bundle for the Docker image.
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  // The UI talks to the engine directly from the browser (REST + WebSocket on
  // :8000); the Next server only serves static assets and the app shell.
};

export default config;
