import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { job, mockApi } from "./helpers";

let currentId = "legal-1";
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(currentId ? { id: currentId } : {}) }));

import JobPage from "../app/jobs/view/page";

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  listeners: Record<string, ((e: MessageEvent) => void)[]> = {};
  onerror: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { FakeEventSource.instances.push(this); }
  addEventListener(name: string, fn: (e: MessageEvent) => void) { (this.listeners[name] ??= []).push(fn); }
  close() { this.closed = true; }
  emit(name: string, data?: unknown) {
    for (const fn of this.listeners[name] ?? []) fn({ data: JSON.stringify(data) } as MessageEvent);
  }
}

/** the stat tile with this label (the label text also appears in table headers). */
const statTile = (label: string) =>
  Array.from(document.querySelectorAll(".stats .stat")).find((t) => t.querySelector("span")?.textContent === label) as HTMLElement;

const detail = (state = job(), spec: Record<string, unknown> = { max_documents: 1000 }) => ({ state, spec });
const lists = { "GET /api/jobs/legal-1/records": { records: [], has_more: false },
                "GET /api/jobs/legal-1/rejections": { rejections: [], has_more: false } };

beforeEach(() => {
  currentId = "legal-1";
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
});

describe("job detail page", () => {
  it("shows the stats, tables and a download link for a finished job", async () => {
    mockApi({ "GET /api/jobs/legal-1": detail(), ...lists });
    render(<JobPage />);
    expect(await screen.findByRole("heading", { name: "legal-1" })).toBeInTheDocument();
    const stat = statTile;
    expect(stat("accepted")).toHaveTextContent("600");
    expect(stat("candidates read")).toHaveTextContent("1,000");
    expect(stat("rejected")).toHaveTextContent("400");            // 300 + 100
    expect(stat("yield")).toHaveTextContent("60.0%");
    expect(stat("mean score")).toHaveTextContent("0.812");
    expect(screen.getByText("documents target: 600 / 1,000")).toBeInTheDocument();
    expect(screen.getByText("legal:citations")).toBeInTheDocument();
    expect(screen.getByText("cc0-1.0")).toBeInTheDocument();
    expect(screen.getByText("nothing removed")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Download data.jsonl" })).toHaveAttribute("href", expect.stringMatching(/\/api\/jobs\/legal-1\/download$/));
    expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
    expect(FakeEventSource.instances).toHaveLength(0);              // no live stream for a finished job
  });

  it("follows a running job over SSE until it ends", async () => {
    const running = job({ status: "running", accepted: 10, candidates: 20 });
    const { calls } = mockApi({
      "GET /api/jobs/legal-1": (() => { let n = 0; return () => detail(++n === 1 ? running : job()); })(),
      ...lists,
    });
    render(<JobPage />);
    expect(await screen.findByRole("button", { name: "Stop" })).toBeInTheDocument();
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    const es = FakeEventSource.instances[0];
    expect(es.url).toMatch(/\/api\/jobs\/legal-1\/events$/);

    act(() => es.emit("progress", job({ status: "running", accepted: 250, candidates: 400 })));
    expect(within(statTile("accepted")).getByText("250")).toBeInTheDocument();

    act(() => es.emit("end"));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument());
    expect(es.closed).toBe(true);
    // initial load + at least one refresh at the end, which brings the final state without a page reload
    expect(calls.filter((c) => c.path === "/api/jobs/legal-1").length).toBeGreaterThanOrEqual(2);
    expect(within(statTile("accepted")).getByText("600")).toBeInTheDocument();
  });

  it("stops a running job from the Stop button", async () => {
    const { calls } = mockApi({
      "GET /api/jobs/legal-1": detail(job({ status: "running" })),
      "POST /api/jobs/legal-1/stop": {}, ...lists,
    });
    const user = userEvent.setup();
    render(<JobPage />);
    await user.click(await screen.findByRole("button", { name: "Stop" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/jobs/legal-1/stop")).toBe(true));
  });

  it("shows a disabled Stopping… button while a stop is pending", async () => {
    mockApi({ "GET /api/jobs/legal-1": detail(job({ status: "stopping" })), ...lists });
    render(<JobPage />);
    const btn = await screen.findByRole("button", { name: "Stopping…" });
    expect(btn).toBeDisabled();
    expect(screen.getByText(/saves a checkpoint and then stops/)).toBeInTheDocument();
  });

  it("ticks the elapsed clock every second while the job runs", async () => {
    mockApi({ "GET /api/jobs/legal-1": detail(job({ status: "running", elapsed_seconds: 10 })), ...lists });
    render(<JobPage />);
    await waitFor(() => expect(within(statTile("elapsed")).getByText("10s")).toBeInTheDocument());
    await waitFor(() => expect(within(statTile("elapsed")).queryByText("10s")).not.toBeInTheDocument(), { timeout: 2500 });
    expect(within(statTile("elapsed")).getByText(/^1[12]s$/)).toBeInTheDocument();
  });

  it("does not move the clock back when the server repeats an old elapsed value", async () => {
    mockApi({ "GET /api/jobs/legal-1": detail(job({ status: "running", elapsed_seconds: 10 })), ...lists });
    render(<JobPage />);
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    await waitFor(() => expect(within(statTile("elapsed")).getByText(/^1[12]s$/)).toBeInTheDocument(), { timeout: 2500 });
    const es = FakeEventSource.instances[0];
    act(() => es.emit("progress", job({ status: "running", elapsed_seconds: 10, accepted: 5 })));   // stale value again
    expect(within(statTile("elapsed")).queryByText("10s")).not.toBeInTheDocument();
    act(() => es.emit("progress", job({ status: "running", elapsed_seconds: 9.5, accepted: 6 })));  // even older
    expect(within(statTile("elapsed")).getByText(/^1[12]s$/)).toBeInTheDocument();
  });

  it("offers Resume for a stopped job and posts it", async () => {
    const { calls } = mockApi({
      "GET /api/jobs/legal-1": detail(job({ status: "stopped" })),
      "POST /api/jobs/legal-1/resume": {}, ...lists,
    });
    const user = userEvent.setup();
    render(<JobPage />);
    await user.click(await screen.findByRole("button", { name: "Resume" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/jobs/legal-1/resume")).toBe(true));
  });

  it("lists source errors", async () => {
    mockApi({ "GET /api/jobs/legal-1": detail(job({ source_errors: [{ source: "pubmed", error: "timeout" }] })), ...lists });
    render(<JobPage />);
    expect(await screen.findByText("Source errors (1)")).toBeInTheDocument();
    expect(screen.getByText(/"error":"timeout"/)).toBeInTheDocument();
  });

  it("switches between records and rejections", async () => {
    const { calls } = mockApi({ "GET /api/jobs/legal-1": detail(), ...lists });
    const user = userEvent.setup();
    render(<JobPage />);
    await screen.findByRole("heading", { name: "legal-1" });
    await user.click(screen.getByRole("button", { name: "Rejections" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/api/jobs/legal-1/rejections")).toBe(true));
  });

  it("shows the error for an unknown job", async () => {
    mockApi({});
    render(<JobPage />);
    expect(await screen.findByText(/404/)).toHaveClass("err");
  });

  it("asks for a job id when the URL has none", () => {
    currentId = "";
    mockApi({});
    render(<JobPage />);
    expect(screen.getByText("No job id.")).toBeInTheDocument();
  });
});
