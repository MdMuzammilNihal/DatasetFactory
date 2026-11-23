"use client";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
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
  const [now, setNow] = useState(0);
  const [busy, setBusy] = useState<"" | "stop" | "resume">("");
  // Live clock: anchored to the last *new* server value, so a repeated (stale) value does not pull it back,
  // and never shown lower than before while the job runs.
  const anchor = useRef({ server: -1, at: 0 });
  const shown = useRef(0);

  const show = useCallback((s: JobState) => {
    if (s.elapsed_seconds !== anchor.current.server) anchor.current = { server: s.elapsed_seconds, at: Date.now() };
    setState(s);
  }, []);
  const refresh = useCallback(() =>
    api<{ state: JobState; spec: Record<string, unknown> }>(`/api/jobs/${id}`)
      .then((r) => { show(r.state); setSpec(r.spec); setError(""); }).catch((e) => setError(String(e))), [id, show]);

  useEffect(() => { if (id) refresh(); }, [id, refresh]);

  const status = state?.status ?? "";
  const active = ACTIVE.includes(status);
  // Live progress by SSE while the job is active. The browser reconnects a dropped stream by itself,
  // and a slow poll covers the case where it cannot, so the page never stays stuck on "running".
  useEffect(() => {
    if (!id || !active) return;
    const es = new EventSource(`${API}/api/jobs/${id}/events`);
    es.addEventListener("progress", (e) => show(JSON.parse((e as MessageEvent).data)));
    es.addEventListener("end", () => { es.close(); refresh(); });
    const poll = setInterval(refresh, 3000);
    return () => { es.close(); clearInterval(poll); };
  }, [id, active, refresh, show]);

  // When the job leaves the active states, load the final state and spec once and reload the record lists.
  const [wasActive, setWasActive] = useState(false);
  useEffect(() => {
    if (active) { setWasActive(true); return; }
    if (wasActive) { setWasActive(false); refresh(); setTick((t) => t + 1); }
  }, [active, wasActive, refresh]);

  // A clock that ticks every second while the job runs, between server updates.
  const ticking = status === "running" || status === "stopping";
  useEffect(() => {
    if (!ticking) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [ticking]);

  async function act(what: "stop" | "resume") {
    setBusy(what);
    try { await api(`/api/jobs/${id}/${what}`, { method: "POST" }); await refresh(); }
    catch (e) { setError(String(e)); await refresh(); }
    finally { setBusy(""); }
  }

  if (!id) return <p className="err">No job id.</p>;
  if (!state) return <p className={error ? "err" : "mute"}>{error || "Loading…"}</p>;
  const target = spec && (["max_documents", "max_tokens", "max_candidates"] as const).find((k) => spec[k]);
  const have = target === "max_documents" ? state.accepted : target === "max_tokens" ? state.tokens : state.candidates;
  const progress = target ? Math.min(1, have / Number(spec![target])) : 0;
  const rejectedTotal = Object.values(state.rejected).reduce((a, b) => a + b, 0);
  let elapsed = state.elapsed_seconds;
  if (ticking) {
    const { server, at } = anchor.current;
    elapsed = Math.max(shown.current, server + Math.max(0, now - at) / 1000);
    shown.current = elapsed;
  } else {
    shown.current = 0;                                   // finished: show the server's exact total
  }
  const stopping = status === "stopping" || busy === "stop";

  return (
    <>
      <div className="pagehead">
        <div>
          <a className="crumb" href="/">← Jobs</a>
          <div className="row"><h1>{state.job_id}</h1><Badge status={state.status} /></div>
          <p>{state.domain} · {state.mode}{state.started_at ? ` · started ${new Date(state.started_at).toLocaleString()}` : ""}</p>
        </div>
        <div className="row">
          {active && <button onClick={() => act("stop")} disabled={stopping}>{stopping ? "Stopping…" : "Stop"}</button>}
          {["stopped", "interrupted"].includes(state.status) && (
            <button className="primary" onClick={() => act("resume")} disabled={busy === "resume"}>{busy === "resume" ? "Resuming…" : "Resume"}</button>
          )}
          {state.accepted > 0 && <a className="btn" href={`${API}/api/jobs/${id}/download`}>Download data.jsonl</a>}
        </div>
      </div>
      {error && <p className="err">{error}</p>}
      {state.message && <p className="mute">{state.message}</p>}
      {status === "stopping" && <p className="mute">Stopping: the job finishes the chunk it is reading, saves a checkpoint and then stops (while sources are still opening this can take a minute).</p>}
      {target && (
        <div className="card progress"><div className="row spread"><span>{target.replace("max_", "")} target: {n(have)} / {n(Number(spec![target]))}</span>
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
        <Stat label="elapsed" value={dur(elapsed)} />
      </div>
      {state.source_errors.length > 0 && (
        <div className="card"><h2 className="err">Source errors ({state.source_errors.length})</h2>
          <pre className="text">{state.source_errors.map((e) => JSON.stringify(e)).join("\n")}</pre></div>
      )}
      <div className="grid detail">
        <Histogram bins={state.score_histogram} />
        <CountTable title="Rejected by label" data={state.rejected} />
        <NestedTable title="By source" data={state.by_source} />
        <NestedTable title="By agent" data={state.by_agent} />
        <div className="stack">
          <CountTable title="Licences" data={state.licenses} />
          <CountTable title="PII removed" data={state.pii_removed} empty="nothing removed" />
        </div>
      </div>
      <h2 className="browse">Browse</h2>
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
