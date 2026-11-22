"use client";
import { useEffect, useState } from "react";
import { api, ACTIVE, dur, n, pct, type JobState, type Meta } from "./lib";
import { Badge } from "./components";

type Target = "max_documents" | "max_tokens" | "max_candidates";

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [jobs, setJobs] = useState<JobState[]>([]);
  const [error, setError] = useState("");
  const [domain, setDomain] = useState("general");
  const [target, setTarget] = useState<Target>("max_documents");
  const [amount, setAmount] = useState("200");
  const [mode, setMode] = useState("curate");
  const [synthetic, setSynthetic] = useState(false);
  const [busy, setBusy] = useState(false);
  const [own, setOwn] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [textField, setTextField] = useState("text");
  const [txtSplit, setTxtSplit] = useState("blocks");
  const [license, setLicense] = useState("unknown");

  useEffect(() => { api<Meta>("/api/meta").then(setMeta).catch((e) => setError(String(e))); }, []);
  useEffect(() => {
    let live = true;
    const tick = () => api<JobState[]>("/api/jobs").then((j) => { if (live) { setJobs(j); setError(""); } })
      .catch((e) => live && setError(`Cannot reach the API: ${e}`));
    tick();
    const t = setInterval(tick, 2000);
    return () => { live = false; clearInterval(t); };
  }, []);

  async function start(e: React.FormEvent) {
    e.preventDefault(); setBusy(true);
    try {
      const body: Record<string, unknown> = { domain, mode, synthetic: own ? false : synthetic, [target]: Number(amount) };
      if (own) {
        if (!file) throw new Error("choose a file first");
        const form = new FormData();
        form.append("file", file); form.append("text_field", textField); form.append("txt_split", txtSplit);
        const up = await api<{ upload_id: string }>("/api/uploads", { method: "POST", body: form });
        body.upload_id = up.upload_id; body.input_license = license || "unknown";
        if (!Object.prototype.hasOwnProperty.call(body, target)) body[target] = Number(amount);
      }
      const s = await api<JobState>("/api/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      window.location.href = `/jobs/view?id=${encodeURIComponent(s.job_id)}`;
    } catch (err) { setError(String(err)); setBusy(false); }
  }

  return (
    <>
      <div className="card">
        <h2>New job</h2>
        <form className="new" onSubmit={start}>
          <label>Domain
            <select value={domain} onChange={(e) => setDomain(e.target.value)}>
              {(meta?.domains ?? ["general"]).map((d) => <option key={d}>{d}</option>)}
            </select></label>
          <label>Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {(meta?.modes ?? ["curate", "random"]).map((d) => <option key={d}>{d}</option>)}
            </select></label>
          <label>Target
            <select value={target} onChange={(e) => setTarget(e.target.value as Target)}>
              <option value="max_documents">documents</option>
              <option value="max_tokens">tokens</option>
              <option value="max_candidates">candidates read</option>
            </select></label>
          <label>Amount
            <input type="number" min={1} value={amount} onChange={(e) => setAmount(e.target.value)} required /></label>
          <label className="chk"><input type="checkbox" checked={own} onChange={(e) => setOwn(e.target.checked)} />
            your data (upload)</label>
          {own && (<>
            <label>File ({(meta?.upload?.formats ?? [".jsonl", ".txt", ".csv", ".parquet"]).join(" ")})
              <input type="file" accept={(meta?.upload?.formats ?? [".jsonl", ".txt", ".csv", ".parquet"]).join(",")}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)} required /></label>
            {file?.name.toLowerCase().endsWith(".txt") ? (
              <label>One document per
                <select value={txtSplit} onChange={(e) => setTxtSplit(e.target.value)}>
                  <option value="blocks">blank-line block</option><option value="file">whole file</option>
                </select></label>
            ) : (
              <label>Text field / column
                <input value={textField} onChange={(e) => setTextField(e.target.value)} /></label>
            )}
            <label>Licence of your data
              <input value={license} onChange={(e) => setLicense(e.target.value)} /></label>
          </>)}
          {!own && <label className="chk"><input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} />
            synthetic data (offline)</label>}
          <button className="primary" disabled={busy || !meta}>Start</button>
        </form>
        {own && <p className="mute">Your file is uploaded to this machine (max {meta?.upload?.max_mb ?? 200} MB), converted, and run through the same filters; the domain you pick decides which validator applies. Set the amount to at least your document count to read it all.</p>}
        {!own && !synthetic && <p className="mute">Real data is streamed from Hugging Face; sources take 20–110 s to open.</p>}
      </div>
      {error && <p className="err">{error}</p>}
      <div className="card">
        <h2>Jobs</h2>
        {jobs.length === 0 ? <span className="mute">no jobs yet</span> : (
          <table>
            <thead><tr><th>Job</th><th>Status</th><th>Mode</th><th className="num">Accepted</th><th className="num">Candidates</th>
              <th className="num">Yield</th><th className="num">Tokens</th><th className="num">Time</th></tr></thead>
            <tbody>{jobs.map((j) => (
              <tr key={j.job_id}>
                <td><a href={`/jobs/view?id=${encodeURIComponent(j.job_id)}`}>{j.job_id}</a></td>
                <td><Badge status={j.status} /></td><td>{j.mode}</td>
                <td className="num">{n(j.accepted)}</td><td className="num">{n(j.candidates)}</td>
                <td className="num">{ACTIVE.includes(j.status) && !j.candidates ? "–" : pct(j.yield_rate)}</td>
                <td className="num">{n(j.tokens)}</td><td className="num">{dur(j.elapsed_seconds)}</td>
              </tr>))}</tbody>
          </table>
        )}
      </div>
    </>
  );
}
