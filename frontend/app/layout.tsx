import "@fontsource/ibm-plex-serif/400.css";
import "@fontsource/ibm-plex-serif/500.css";
import "@fontsource/ibm-plex-serif/600.css";
import "@fontsource/ibm-plex-serif/700.css";
import "./globals.css";
import type { ReactNode } from "react";
import { ThemeToggle } from "./theme-toggle";

// Applies a saved theme before first paint so the page does not flash the wrong palette.
const initTheme = `try{var t=localStorage.getItem("df-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export const metadata = { title: "Dataset Factory", description: "Multi-domain data curation dashboard" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: initTheme }} /></head>
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
            <div className="themewrap"><ThemeToggle /></div>
            <div className="foot"><b>Curation dashboard</b>Filter, score and export datasets.</div>
          </aside>
          <main className="page">{children}</main>
        </div>
      </body>
    </html>
  );
}
