export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Counts = Record<string, number>;
export interface JobState {
  job_id: string; name?: string; domain: string; mode: string; status: string; message: string;
  started_at: string; finished_at: string; elapsed_seconds: number;
  candidates: number; evaluated: number; accepted: number; tokens: number; words: number;
  rejected: Counts; by_source: Record<string, Counts>; by_agent: Record<string, Counts>;
  licenses: Counts; score_histogram: number[]; pii_removed: Counts;
  source_errors: { source?: string; error?: string; [k: string]: unknown }[];
  checkpoints: number; yield_rate: number; throughput_docs_per_s: number;
  mean_accepted_score: number | null; steady_throughput_docs_per_s: number;
}
export interface Meta { domains: string[]; modes: string[]; runtime: Record<string, unknown>; config_fingerprint: string; max_running: number; upload?: { formats: string[]; max_mb: number } }

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;     // e.g. DELETE: no body
  return res.json() as Promise<T>;
}

export const n = (v: number | undefined | null) => (v ?? 0).toLocaleString("en-US");
export const pct = (v: number) => (v * 100).toFixed(1) + "%";
/** Whole seconds: "7s", "1m 15s", "1h 2m 3s". */
export const dur = (s: number) => {
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = t % 60;
  return h ? `${h}h ${m}m ${sec}s` : m ? `${m}m ${sec}s` : `${sec}s`;
};
/** What people see: the friendly name ("Legal 2"), or the id for a job without one. */
export const label = (j: { job_id: string; name?: string }) => j.name || j.job_id;
export const ACTIVE = ["pending", "running", "stopping"];
