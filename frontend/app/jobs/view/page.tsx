"use client";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { API, api, ACTIVE, dur, n, pct, type JobState } from "../../lib";
import { Badge, Browser, CountTable, Histogram, NestedTable, Stat } from "../../components";

function Detail() {
  const id = useSearchParams().get("id") ?? "";
  const [state, setState] = useState<JobState | null>(null);
  const [spec, setSpec] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"records" | "rejections">("records");
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() =>
    api<{ state: JobState; spec: Record<string, unknown> }>(`/api/jobs/${id}`)
      .then((r) => { setState(r.state); setSpec(r.spec); setError(""); }).catch((e) => setError(String(e))), [id]);

  useEffect(() => { if (id) refresh(); }, [id, refresh]);

  const active = state ? ACTIVE.includes(state.status) : false;
  // Live progress by SSE while the job is active; the stream ends by itself at a terminal status.
  useEffect(() => {
    if (!id || !active) return;
    const es = new EventSource(`${API}/api/jobs/${id}/events`);
    es.addEventListener("progress", (e) => { setState(JSON.parse((e as MessageEvent).data)); setTick((t) => t + 1); });
    es.addEventListener("end", () => { es.close(); refresh(); setTick((t) => t + 1); });
    es.onerror = () => { es.close(); setTimeout(refresh, 2000); };
    return () => es.close();
  }, [id, active, refresh]);

  async function act(what: "stop" | "resume") {
    try { await api(`/api/jobs/${id}/${what}`, { method: "POST" }); await refresh(); } catch (e) { setError(String(e)); }
  }

  if (!id) return <p className="err">No job id.</p>;
  if (!state) return <p className={error ? "err" : "mute"}>{error || "Loading…"}</p>;
  const target = spec && (["max_documents", "max_tokens", "max_candidates"] as const).find((k) => spec[k]);
  const have = target === "max_documents" ? state.accepted : target === "max_tokens" ? state.tokens : state.candidates;
  const progress = target ? Math.min(1, have / Number(spec![target])) : 0;
  const rejectedTotal = Object.values(state.rejected).reduce((a, b) => a + b, 0);

  return (
    <>
      <div className="row spread" style={{ marginBottom: 12 }}>
        <div className="row"><a href="/">← Jobs</a><h1>{state.job_id}</h1><Badge status={state.status} /></div>
        <div className="row">
          {active && <button onClick={() => act("stop")}>Stop</button>}
          {["stopped", "interrupted"].includes(state.status) && <button className="primary" onClick={() => act("resume")}>Resume</button>}
          {state.accepted > 0 && <a className="btn" href={`${API}/api/jobs/${id}/download`}>Download data.jsonl</a>}
        </div>
      </div>
      {error && <p className="err">{error}</p>}
      {state.message && <p className="mute">{state.message}</p>}
      {target && (
        <div className="card"><div className="row spread"><span>{target.replace("max_", "")} target: {n(have)} / {n(Number(spec![target]))}</span>
          <span className="mute">{pct(progress)}</span></div>
          <div className="bar"><i style={{ width: `${progress * 100}%` }} /></div></div>
      )}
      <div className="stats">
        <Stat label="candidates read" value={n(state.candidates)} />
        <Stat label="accepted" value={n(state.accepted)} />
        <Stat label="rejected" value={n(rejectedTotal)} />
        <Stat label="tokens (Pythia)" value={n(state.tokens)} />
        <Stat label="yield" value={state.candidates ? pct(state.yield_rate) : "–"} />
        <Stat label="docs/s (wall)" value={String(state.throughput_docs_per_s)} />
        <Stat label="docs/s (steady)" value={String(state.steady_throughput_docs_per_s)} />
        <Stat label="mean score" value={state.mean_accepted_score?.toFixed(3) ?? "–"} />
        <Stat label="elapsed" value={dur(state.elapsed_seconds)} />
      </div>
      {state.source_errors.length > 0 && (
        <div className="card"><h2 className="err">Source errors ({state.source_errors.length})</h2>
          <pre className="text">{state.source_errors.map((e) => JSON.stringify(e)).join("\n")}</pre></div>
      )}
      <div className="grid g2">
        <CountTable title="Rejected by label" data={state.rejected} />
        <Histogram bins={state.score_histogram} />
        <NestedTable title="By source" data={state.by_source} />
        <NestedTable title="By agent" data={state.by_agent} />
        <CountTable title="Licences" data={state.licenses} />
        <CountTable title="PII removed" data={state.pii_removed} empty="nothing removed" />
      </div>
      <h2 style={{ marginTop: 8 }}>Browse</h2>
      <div className="tabs">
        <button className={tab === "records" ? "on" : ""} onClick={() => setTab("records")}>Accepted records</button>
        <button className={tab === "rejections" ? "on" : ""} onClick={() => setTab("rejections")}>Rejections</button>
      </div>
      <Browser jobId={id} kind={tab} refreshKey={active ? 0 : tick} />
      {active && <p className="mute">Lists refresh when the job ends; use Prev/Next to reload while it runs.</p>}
    </>
  );
}

export default function JobPage() {
  return <Suspense fallback={<p className="mute">Loading…</p>}><Detail /></Suspense>;
}
