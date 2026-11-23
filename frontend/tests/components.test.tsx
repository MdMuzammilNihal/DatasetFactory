import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Badge, Browser, CountTable, Histogram, JobsChart, NestedTable, Stat, YieldDonut } from "../app/components";
import { job, mockApi } from "./helpers";

describe("small components", () => {
  it("Badge puts the status in its class", () => {
    render(<Badge status="running" />);
    expect(screen.getByText("running")).toHaveClass("badge", "running");
  });
  it("Stat shows value and label", () => {
    render(<Stat label="accepted" value="1,000" />);
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("accepted")).toBeInTheDocument();
  });
});

describe("CountTable", () => {
  it("sorts rows by count and scales the bars", () => {
    const { container } = render(<CountTable title="Rejected" data={{ a: 10, b: 40, c: 20 }} />);
    const labels = Array.from(container.querySelectorAll("tr")).map((r) => r.querySelector("td")!.textContent);
    expect(labels).toEqual(["b", "c", "a"]);
    const widths = Array.from(container.querySelectorAll(".bar i")).map((i) => (i as HTMLElement).style.width);
    expect(widths).toEqual(["100%", "50%", "25%"]);
  });
  it("shows the empty text", () => {
    render(<CountTable title="PII" data={{}} empty="nothing removed" />);
    expect(screen.getByText("nothing removed")).toBeInTheDocument();
  });
});

describe("NestedTable", () => {
  it("builds columns from every row and fills gaps with 0", () => {
    render(<NestedTable title="By source" data={{ wiki: { read: 5, accepted: 3 }, fineweb: { read: 7 } }} />);
    expect(screen.getByRole("columnheader", { name: "read" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "accepted" })).toBeInTheDocument();
    const fineweb = screen.getByText("fineweb").closest("tr")!;
    expect(fineweb).toHaveTextContent("7");
    expect(fineweb).toHaveTextContent("0");
  });
  it("shows 'none yet' when empty", () => {
    render(<NestedTable title="By agent" data={{}} />);
    expect(screen.getByText("none yet")).toBeInTheDocument();
  });
});

describe("Histogram", () => {
  it("draws one bar per bin, the tallest at 100%", () => {
    const { container } = render(<Histogram bins={[0, 5, 10]} />);
    const bars = Array.from(container.querySelectorAll(".hist > div")) as HTMLElement[];
    expect(bars.map((b) => b.style.height)).toEqual(["0%", "50%", "100%"]);
  });
  it("does not divide by zero on an empty histogram", () => {
    const { container } = render(<Histogram bins={[0, 0]} />);
    const bars = Array.from(container.querySelectorAll(".hist > div")) as HTMLElement[];
    expect(bars.map((b) => b.style.height)).toEqual(["0%", "0%"]);
  });
});

const record = (i: number) => ({
  id: `d${i}`, text: `document text ${i}`, quality_score: 0.9, n_tokens: 1234,
  provenance: { source: "caselaw", agent: "factory", license: "cc0-1.0" },
});

describe("Browser", () => {
  it("lists records and pages forward and back", async () => {
    const { calls } = mockApi({
      "GET /api/jobs/j1/records": (() => {
        let call = 0;
        return () => (++call === 1
          ? { records: [record(1), record(2)], has_more: true }
          : { records: [record(3)], has_more: false });
      })(),
    });
    const user = userEvent.setup();
    render(<Browser jobId="j1" kind="records" refreshKey={0} />);
    expect(await screen.findByText("document text 1")).toBeInTheDocument();
    expect(screen.getAllByText(/caselaw · factory · cc0-1.0/)).toHaveLength(2);   // one line per record
    expect(screen.getByRole("button", { name: /Prev/ })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("document text 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next/ })).toBeDisabled();
    expect(screen.getByText("11–11")).toBeInTheDocument();   // page size 10 for records
    expect(calls.map((c) => c.path)).toEqual(["/api/jobs/j1/records", "/api/jobs/j1/records"]);
  });

  it("requests the offset and page size for rejections", async () => {
    const { fn } = mockApi({
      "GET /api/jobs/j1/rejections": { rejections: [{ label: "global:too_short", source: "wiki", agent: "factory", score: null }], has_more: false },
    });
    render(<Browser jobId="j1" kind="rejections" refreshKey={0} />);
    expect(await screen.findByText("global:too_short")).toBeInTheDocument();
    expect(screen.getByText("–")).toBeInTheDocument();   // a null score
    expect(String(fn.mock.calls[0][0])).toContain("/api/jobs/j1/rejections?offset=0&limit=25");
  });

  it("says 'nothing yet' for an empty list", async () => {
    mockApi({ "GET /api/jobs/j1/records": { records: [], has_more: false } });
    render(<Browser jobId="j1" kind="records" refreshKey={0} />);
    expect(await screen.findByText("nothing yet")).toBeInTheDocument();
  });

  it("shows the error when the API fails", async () => {
    mockApi({});
    render(<Browser jobId="j1" kind="records" refreshKey={0} />);
    await waitFor(() => expect(screen.getByText(/404/)).toHaveClass("err"));
  });
});

describe("dashboard charts", () => {
  it("JobsChart sizes bars by candidates and links to the job", () => {
    const { container } = render(<JobsChart jobs={[job({ job_id: "a", candidates: 1000, accepted: 600 }), job({ job_id: "b", candidates: 500, accepted: 500 })]} />);
    const cols = Array.from(container.querySelectorAll(".barcol")) as HTMLElement[];
    expect(cols).toHaveLength(2);
    expect(cols.map((c) => (c.querySelector(".stack") as HTMLElement).style.height).sort()).toEqual(["100%", "50%"]);
    expect(container.querySelector('a[href="/jobs/view?id=a"]')).not.toBeNull();
  });
  it("JobsChart says so when nothing has been read yet", () => {
    render(<JobsChart jobs={[job({ candidates: 0, accepted: 0 })]} />);
    expect(screen.getByText("no data yet")).toBeInTheDocument();
  });
  it("YieldDonut shows the overall yield and counts jobs by status", () => {
    render(<YieldDonut jobs={[job(), job({ job_id: "x", status: "failed", candidates: 0, accepted: 0 })]} />);
    expect(screen.getByLabelText("overall yield 60.0%")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
  });
  it("YieldDonut shows a dash with no jobs", () => {
    render(<YieldDonut jobs={[]} />);
    expect(screen.getByLabelText("overall yield 0.0%")).toBeInTheDocument();
  });
});
