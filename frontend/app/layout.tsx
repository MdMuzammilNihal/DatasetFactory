import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Dataset Factory", description: "Multi-domain data curation dashboard" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top"><a href="/">Dataset Factory</a><span>curation dashboard</span></header>
        <main>{children}</main>
      </body>
    </html>
  );
}
