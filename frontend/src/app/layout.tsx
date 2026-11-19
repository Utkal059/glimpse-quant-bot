import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Glimpse Quant Terminal",
  description: "Paper-first LS-LMSR trading terminal for glimpse-quant-bot.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#0d0e10",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
