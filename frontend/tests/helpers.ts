import { vi } from "vitest";
import type { JobState } from "../app/lib";

export function job(over: Partial<JobState> = {}): JobState {
  return {
    job_id: "legal-1", domain: "legal", mode: "curate", status: "completed", message: "",
    started_at: "2026-01-01T00:00:00", finished_at: "2026-01-01T00:01:00", elapsed_seconds: 75,
    candidates: 1000, evaluated: 1000, accepted: 600, tokens: 123456, words: 90000,
    rejected: { "legal:citations": 300, "score:below_threshold": 100 },
    by_source: { caselaw: { read: 1000, accepted: 600 } }, by_agent: { factory: { read: 1000 } },
    licenses: { "cc0-1.0": 600 }, score_histogram: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    pii_removed: {}, source_errors: [], checkpoints: 1, yield_rate: 0.6, throughput_docs_per_s: 50,
    mean_accepted_score: 0.8123, steady_throughput_docs_per_s: 80,
    ...over,
  };
}

export const META = {
  domains: ["general", "legal"], modes: ["curate", "random"], runtime: {}, config_fingerprint: "abc", max_running: 2,
  upload: { formats: [".jsonl", ".txt"], max_mb: 200 },
};

export type Route = unknown | ((init?: RequestInit) => unknown);

/** Stub fetch: `routes` maps "METHOD /path" (query string ignored) to a body or a function; unmatched -> 404. */
export function mockApi(routes: Record<string, Route>) {
  const calls: { method: string; path: string; init?: RequestInit }[] = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, "").split("?")[0];
    const method = (init?.method ?? "GET").toUpperCase();
    calls.push({ method, path, init });
    const route = routes[`${method} ${path}`];
    if (route === undefined) {
      return new Response(JSON.stringify({ detail: `no route ${method} ${path}` }), { status: 404 });
    }
    const body = typeof route === "function" ? (route as (i?: RequestInit) => unknown)(init) : route;
    if (body instanceof Response) return body;
    return new Response(JSON.stringify(body), { status: 200 });
  });
  vi.stubGlobal("fetch", fn);
  return { fn, calls };
}
