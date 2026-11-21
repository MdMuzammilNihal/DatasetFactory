export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Counts = Record<string, number>;
export interface JobState {
  job_id: string; domain: string; mode: string; status: string; message: string;
  started_at: string; finished_at: string; elapsed_seconds: number;
  candidates: number; evaluated: number; accepted: number; tokens: number; words: number;
  rejected: Counts; by_source: Record<string, Counts>; by_agent: Record<string, Counts>;
  licenses: Counts; score_histogram: number[]; pii_removed: Counts;
  source_errors: { source?: string; error?: string; [k: string]: unknown }[];
  checkpoints: number; yield_rate: number; throughput_docs_per_s: number;
  mean_accepted_score: number | null; steady_throughput_docs_per_s: number;
}
export interface Meta { domains: string[]; modes: string[]; runtime: Record<string, unknown>; config_fingerprint: string; max_running: number }

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const n = (v: number | undefined | null) => (v ?? 0).toLocaleString("en-US");
export const pct = (v: number) => (v * 100).toFixed(1) + "%";
export const dur = (s: number) => {
  const m = Math.floor(s / 60);
  return m ? `${m}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
};
export const ACTIVE = ["pending", "running"];
