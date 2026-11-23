import "@fontsource/ibm-plex-serif/400.css";
import "@fontsource/ibm-plex-serif/500.css";
import "@fontsource/ibm-plex-serif/600.css";
import "@fontsource/ibm-plex-serif/700.css";
import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Dataset Factory", description: "Multi-domain data curation dashboard" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="side">
            <a className="brand" href="/"><span className="logo">◆</span><span className="brandname">Dataset Factory</span></a>
            <div>
              <h6>Menu</h6>
              <nav>
                <a className="on" href="/">▦&nbsp; Dashboard</a>
                <a href="/#new">＋&nbsp; New job</a>
                <a href="/#jobs">☰&nbsp; Jobs</a>
              </nav>
            </div>
            <div className="foot"><b>Curation dashboard</b>Filter, score and export datasets.</div>
          </aside>
          <main className="page">{children}</main>
        </div>
      </body>
    </html>
  );
}
