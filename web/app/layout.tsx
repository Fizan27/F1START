// app/layout.tsx
//
// What it does: the frame around every page: the page title, the navigation
//   bar at the top and the footer at the bottom.
// Which files use it: Next.js wraps every page in app/ with it automatically.

import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1 Equalizer: every driver, the same car",
  description:
    "If every Formula 1 driver in history drove the same car, who would be fastest? A statistical model of driver skill since 1950, and races you can run yourself.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0d0d0d" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main>{children}</main>
        <footer>
          Races are simulated from a statistical model, not real results. Comparisons across eras are
          uncertain. Data: Jolpica F1 API and FastF1. Not affiliated with Formula 1.
        </footer>
      </body>
    </html>
  );
}
