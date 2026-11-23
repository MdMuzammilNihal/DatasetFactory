"use client";
import { useEffect, useState } from "react";
import { api, ACTIVE, dur, n, pct, type JobState, type Meta } from "./lib";
import { Badge, JobsChart, YieldDonut } from "./components";

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

  async function removeJob(j: JobState) {
    if (!window.confirm(`Delete ${j.job_id} and all its data? This cannot be undone.`)) return;
    try {
      await api<void>(`/api/jobs/${encodeURIComponent(j.job_id)}`, { method: "DELETE" });
      setJobs((old) => old.filter((x) => x.job_id !== j.job_id)); setError("");
    } catch (err) { setError(String(err)); }
  }

  async function clearAll() {
    const finished = jobs.filter((j) => !ACTIVE.includes(j.status));
    if (!finished.length) return;
    const what = finished.length === 1 ? "the 1 finished job and its data" : `all ${finished.length} finished jobs and their data`;
    if (!window.confirm(`Delete ${what}? Running jobs are kept. This cannot be undone.`)) return;
    try {
      const out = await api<{ deleted: string[] }>("/api/jobs?confirm=true", { method: "DELETE" });
      const gone = new Set(out.deleted);
      setJobs((old) => old.filter((x) => !gone.has(x.job_id))); setError("");
    } catch (err) { setError(String(err)); }
  }

  const done = jobs.filter((j) => j.status === "completed");
  const accepted = jobs.reduce((a, j) => a + (j.accepted || 0), 0);
  const tokens = jobs.reduce((a, j) => a + (j.tokens || 0), 0);
  const cand = jobs.reduce((a, j) => a + (j.candidates || 0), 0);
  const running = jobs.filter((j) => ACTIVE.includes(j.status)).length;

  return (
    <>
      <div className="pagehead">
        <div><h1>Dashboard</h1><p>Overview of your curation jobs</p></div>
      </div>
      <div className="kpis">
        <div className="kpi t1"><span>Total jobs</span><b>{n(jobs.length)}</b><small>{running} running · {done.length} completed</small></div>
        <div className="kpi t2"><span>Documents accepted</span><b>{n(accepted)}</b><small>across all jobs</small></div>
        <div className="kpi t3"><span>Tokens</span><b>{n(tokens)}</b><small>in accepted data</small></div>
        <div className="kpi t4"><span>Overall yield</span><b>{cand ? pct(accepted / cand) : "–"}</b><small>{n(cand)} candidates read</small></div>
      </div>
      {error && <p className="err">{error}</p>}
      <div className="charts"><JobsChart jobs={jobs} /><YieldDonut jobs={jobs} /></div>
      <div className="cols">
      <div className="card" id="jobs">
        <div className="row spread">
          <h2>Jobs</h2>
          <button className="danger" onClick={clearAll} disabled={!jobs.some((j) => !ACTIVE.includes(j.status))}
            title="Delete every finished job and its data">Clear all</button>
        </div>
        {jobs.length === 0 ? <span className="mute">no jobs yet</span> : (
          <table className="jobs">
            <thead><tr><th>Job</th><th>Status</th><th>Mode</th><th className="num">Accepted</th><th className="num">Candidates</th>
              <th className="num">Yield</th><th className="num">Tokens</th><th className="num">Time</th><th /></tr></thead>
            <tbody>{jobs.map((j) => (
              <tr key={j.job_id}>
                <td className="c-job"><a href={`/jobs/view?id=${encodeURIComponent(j.job_id)}`}>{j.job_id}</a></td>
                <td className="c-status"><Badge status={j.status} /></td><td className="c-mode">{j.mode}</td>
                <td className="num c-acc" data-label="accepted">{n(j.accepted)}</td><td className="num c-cand">{n(j.candidates)}</td>
                <td className="num c-yield" data-label="yield">{ACTIVE.includes(j.status) && !j.candidates ? "–" : pct(j.yield_rate)}</td>
                <td className="num c-tok">{n(j.tokens)}</td><td className="num c-time" data-label="time">{dur(j.elapsed_seconds)}</td>
                <td className="c-del">
                  <button className="icon" onClick={() => removeJob(j)} disabled={ACTIVE.includes(j.status)}
                    aria-label={`Delete ${j.job_id}`} title={ACTIVE.includes(j.status) ? "Stop the job before deleting it" : "Delete this job"}>
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                      <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 11v6M14 11v6" />
                    </svg>
                  </button>
                </td>
              </tr>))}</tbody>
          </table>
        )}
      </div>
      <div className="card" id="new">
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
      </div>
    </>
  );
}
