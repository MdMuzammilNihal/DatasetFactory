"use client";
import { useCallback, useEffect, useState } from "react";
import { api, n, type Counts } from "./lib";

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
            <div className="row spread">
              <span><b>#{offset + i + 1}</b> <span className="mute">{String(p.source)} · {String(p.agent)} · {String(p.license)}</span></span>
              <span className="meta"><span className="pill">score {String(r.quality_score)}</span><span className="pill">{n(Number(r.n_tokens))} tokens</span></span>
            </div>
            <pre className="text">{String(r.text)}</pre>
          </div>
        );
      }) : rows.length > 0 && (
        <div className="card"><table>
          <thead><tr><th>#</th><th>label</th><th>source</th><th>agent</th><th className="num">score</th></tr></thead>
          <tbody>{rows.map((r, i) => (
            <tr key={i}><td>{offset + i + 1}</td><td>{String(r.label)}</td><td>{String(r.source)}</td>
              <td>{String(r.agent)}</td><td className="num">{r.score == null ? "–" : String(r.score)}</td></tr>
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
