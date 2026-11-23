"use client";
import { useCallback, useEffect, useState } from "react";
import { api, n, pct, type Counts, type JobState } from "./lib";

export function Badge({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

export function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat"><b>{value}</b><span>{label}</span></div>;
}

/** A label -> count table with bars, sorted by count. */
export function CountTable({ title, data, empty = "none yet" }: { title: string; data: Counts; empty?: string }) {
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = rows[0]?.[1] || 1;
  return (
    <div className="card">
      <h2>{title}</h2>
      {rows.length === 0 ? <span className="mute">{empty}</span> : (
        <table><tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td>{k}<div className="bar"><i style={{ width: `${(v / max) * 100}%` }} /></div></td>
              <td className="num">{n(v)}</td>
            </tr>
          ))}
        </tbody></table>
      )}
    </div>
  );
}

/** A name -> {counter: n} table (by_source, by_agent). */
export function NestedTable({ title, data }: { title: string; data: Record<string, Counts> }) {
  const cols = Array.from(new Set(Object.values(data).flatMap((o) => Object.keys(o))));
  const rows = Object.entries(data);
  return (
    <div className="card">
      <h2>{title}</h2>
      {rows.length === 0 ? <span className="mute">none yet</span> : (
        <table>
          <thead><tr><th />{cols.map((c) => <th key={c} className="num">{c}</th>)}</tr></thead>
          <tbody>{rows.map(([k, o]) => (
            <tr key={k}><td>{k}</td>{cols.map((c) => <td key={c} className="num">{n(o[c])}</td>)}</tr>
          ))}</tbody>
        </table>
      )}
    </div>
  );
}

export function Histogram({ bins }: { bins: number[] }) {
  const max = Math.max(1, ...bins);
  return (
    <div className="card wide">
      <h2>Quality score of accepted documents</h2>
      <div className="hist">
        {bins.map((v, i) => (
          <div key={i} style={{ height: `${(v / max) * 100}%` }}
               title={`${(i / bins.length).toFixed(2)}–${((i + 1) / bins.length).toFixed(2)}: ${v}`} />
        ))}
      </div>
      <div className="axis"><span>0</span><span>0.5</span><span>1.0</span></div>
    </div>
  );
}

type Row = Record<string, unknown>;

/** Paged browser of records or rejections. */
export function Browser({ jobId, kind, refreshKey }: { jobId: string; kind: "records" | "rejections"; refreshKey: number }) {
  const limit = kind === "records" ? 10 : 25;
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<Row[]>([]);
  const [more, setMore] = useState(false);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      const r = await api<Record<string, unknown>>(`/api/jobs/${jobId}/${kind}?offset=${offset}&limit=${limit}`);
      setRows(r[kind] as Row[]); setMore(Boolean(r.has_more)); setErr("");
    } catch (e) { setErr(String(e)); }
  }, [jobId, kind, offset, limit]);

  useEffect(() => { setOffset(0); }, [kind, jobId]);
  useEffect(() => { load(); }, [load, refreshKey]);

  return (
    <div>
      {err && <p className="err">{err}</p>}
      {rows.length === 0 && !err && <p className="mute">nothing yet</p>}
      {kind === "records" ? rows.map((r, i) => {
        const p = (r.provenance ?? {}) as Record<string, unknown>;
        return (
          <div className="card rec" key={String(r.id) + i}>
            <div className="row spread rechead">
              <span><b>#{offset + i + 1}</b> <span className="mute">{String(p.source)} · {String(p.agent)} · {String(p.license)}</span></span>
              <span className="meta"><span className="pill">score {String(r.quality_score)}</span><span className="pill">{n(Number(r.n_tokens))} tokens</span></span>
            </div>
            <pre className="text">{String(r.text)}</pre>
          </div>
        );
      }) : rows.length > 0 && (
        <div className="card"><table className="rej">
          <thead><tr><th>#</th><th>label</th><th>source</th><th>agent</th><th className="num">score</th></tr></thead>
          <tbody>{rows.map((r, i) => (
            <tr key={i}><td className="c-n">{offset + i + 1}</td><td className="c-label">{String(r.label)}</td><td className="c-src">{String(r.source)}</td>
              <td className="c-agent">{String(r.agent)}</td><td className="num c-score">{r.score == null ? "–" : String(r.score)}</td></tr>
          ))}</tbody>
        </table></div>
      )}
      <div className="row">
        <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>← Prev</button>
        <button disabled={!more} onClick={() => setOffset(offset + limit)}>Next →</button>
        <span className="mute">{rows.length ? `${offset + 1}–${offset + rows.length}` : ""}</span>
      </div>
    </div>
  );
}

/** Accepted vs rejected documents for the most recent jobs, as paired bars. */
export function JobsChart({ jobs }: { jobs: JobState[] }) {
  const rows = jobs.filter((j) => j.candidates > 0).slice(0, 8).reverse();
  const max = Math.max(1, ...rows.map((j) => j.candidates));
  return (
    <div className="card">
      <div className="row spread"><h2>Documents per job</h2>
        <span className="legend"><i className="dot a" />accepted<i className="dot r" />rejected</span></div>
      {rows.length === 0 ? <span className="mute">no data yet</span> : (
        <div className="bars">
          {rows.map((j) => (
            <a key={j.job_id} className="barcol" href={`/jobs/view?id=${encodeURIComponent(j.job_id)}`}
               title={`${j.job_id}: ${j.accepted} accepted of ${j.candidates} read`}>
              <div className="stack" style={{ height: `${(j.candidates / max) * 100}%` }}>
                <div className="r" style={{ flex: j.candidates - j.accepted }} />
                <div className="a" style={{ flex: j.accepted }} />
              </div>
              <span>{j.domain}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

/** Donut of the overall yield (accepted / candidates read) with the jobs-by-status counts below it. */
export function YieldDonut({ jobs }: { jobs: JobState[] }) {
  const read = jobs.reduce((a, j) => a + j.candidates, 0);
  const acc = jobs.reduce((a, j) => a + j.accepted, 0);
  const frac = read ? acc / read : 0;
  const R = 52, C = 2 * Math.PI * R;
  const status: Counts = {};
  jobs.forEach((j) => { status[j.status] = (status[j.status] ?? 0) + 1; });
  return (
    <div className="card donut">
      <h2>Overall yield</h2>
      <svg viewBox="0 0 140 140" role="img" aria-label={`overall yield ${pct(frac)}`}>
        <circle cx="70" cy="70" r={R} className="ring" />
        <circle cx="70" cy="70" r={R} className="arc" strokeDasharray={`${frac * C} ${C}`} transform="rotate(-90 70 70)" />
        <text x="70" y="68" textAnchor="middle" className="big">{read ? pct(frac) : "–"}</text>
        <text x="70" y="86" textAnchor="middle" className="sub">accepted</text>
      </svg>
      <div className="chips">
        {Object.entries(status).map(([k, v]) => <span key={k} className="chip"><Badge status={k} /> {v}</span>)}
      </div>
    </div>
  );
}
