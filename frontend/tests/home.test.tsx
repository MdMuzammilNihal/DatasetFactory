import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import Home from "../app/page";
import { META, job, mockApi } from "./helpers";

const base = { "GET /api/meta": META };

describe("Home page", () => {
  it("fills the form from /api/meta and lists jobs with their yield", async () => {
    mockApi({
      ...base,
      "GET /api/jobs": [job(), job({ job_id: "new-1", status: "pending", candidates: 0, accepted: 0, yield_rate: 0 })],
    });
    render(<Home />);
    expect(await screen.findByRole("option", { name: "legal" })).toBeInTheDocument();
    expect(await screen.findByRole("link", { name: "legal-1" })).toHaveAttribute("href", "/jobs/view?id=legal-1");
    const done = screen.getByRole("link", { name: "legal-1" }).closest("tr")!;
    expect(done).toHaveTextContent("60.0%");
    expect(done).toHaveTextContent("123,456");
    expect(done).toHaveTextContent("1m 15s");
    // a pending job that has read nothing yet has no yield to show
    expect(screen.getByRole("link", { name: "new-1" }).closest("tr")).toHaveTextContent("–");
  });

  it("says so when there are no jobs", async () => {
    mockApi({ ...base, "GET /api/jobs": [] });
    render(<Home />);
    expect(await screen.findByText("no jobs yet")).toBeInTheDocument();
  });

  it("shows an error when the API cannot be reached", async () => {
    mockApi({});
    render(<Home />);
    expect(await screen.findAllByText(/404|Cannot reach the API/)).not.toHaveLength(0);
  });

  it("posts the chosen job spec", async () => {
    const { calls } = mockApi({
      ...base, "GET /api/jobs": [],
      "POST /api/jobs": () => job({ job_id: "legal-2", status: "pending" }),
    });
    const user = userEvent.setup();
    render(<Home />);
    await screen.findByRole("option", { name: "legal" });
    await user.selectOptions(screen.getByLabelText("Domain"), "legal");
    await user.selectOptions(screen.getByLabelText("Target"), "max_tokens");
    const amount = screen.getByLabelText("Amount");
    await user.clear(amount);
    await user.type(amount, "2000000");
    await user.click(screen.getByLabelText(/synthetic data/));
    await user.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.path === "/api/jobs")).toBe(true));
    const post = calls.find((c) => c.method === "POST" && c.path === "/api/jobs")!;
    expect(JSON.parse(String(post.init!.body))).toEqual({
      domain: "legal", mode: "curate", synthetic: true, max_tokens: 2000000,
    });
  });

  it("shows an API error and re-enables Start", async () => {
    mockApi({
      ...base, "GET /api/jobs": [],
      "POST /api/jobs": new Response(JSON.stringify({ detail: "too many running" }), { status: 409 }),
    });
    const user = userEvent.setup();
    render(<Home />);
    await screen.findByRole("option", { name: "legal" });
    await user.click(screen.getByRole("button", { name: "Start" }));
    expect(await screen.findByText(/409: too many running/)).toHaveClass("err");
    expect(screen.getByRole("button", { name: "Start" })).toBeEnabled();
  });

  describe("your data (upload)", () => {
    it("swaps the synthetic option for the file fields", async () => {
      mockApi({ ...base, "GET /api/jobs": [] });
      const user = userEvent.setup();
      render(<Home />);
      await screen.findByRole("option", { name: "legal" });
      expect(screen.getByLabelText(/synthetic data/)).toBeInTheDocument();
      await user.click(screen.getByLabelText(/your data/));
      expect(screen.queryByLabelText(/synthetic data/)).not.toBeInTheDocument();
      expect(screen.getByLabelText(/^File/)).toBeInTheDocument();
      expect(screen.getByLabelText("Text field / column")).toHaveValue("text");
      expect(screen.getByLabelText("Licence of your data")).toHaveValue("unknown");
    });

    it("offers the split choice for a .txt file instead of the text field", async () => {
      mockApi({ ...base, "GET /api/jobs": [] });
      const user = userEvent.setup();
      render(<Home />);
      await screen.findByRole("option", { name: "legal" });
      await user.click(screen.getByLabelText(/your data/));
      await user.upload(screen.getByLabelText(/^File/), new File(["a\n\nb"], "notes.txt", { type: "text/plain" }));
      expect(screen.getByLabelText("One document per")).toBeInTheDocument();
      expect(screen.queryByLabelText("Text field / column")).not.toBeInTheDocument();
    });

    it("uploads the file, then starts a job with its upload_id and licence", async () => {
      const { calls } = mockApi({
        ...base, "GET /api/jobs": [],
        "POST /api/uploads": { upload_id: "up-0123456789abcdef", documents: 3 },
        "POST /api/jobs": () => job({ job_id: "general-up", status: "pending" }),
      });
      const user = userEvent.setup();
      render(<Home />);
      await screen.findByRole("option", { name: "legal" });
      await user.click(screen.getByLabelText(/your data/));
      await user.upload(screen.getByLabelText(/^File/), new File(['{"text":"hi"}'], "mine.jsonl"));
      const licence = screen.getByLabelText("Licence of your data");
      await user.clear(licence);
      await user.type(licence, "cc-by-4.0");
      // jsdom does not count a file attached by the test towards `required`, so it would block a click on
      // Start (a real browser does not); fire the form's submit event directly instead.
      fireEvent.submit(screen.getByRole("button", { name: "Start" }).closest("form")!);
      await waitFor(() => expect(calls.map((c) => `${c.method} ${c.path}`)).toContain("POST /api/jobs"));
      const upload = calls.find((c) => c.path === "/api/uploads")!;
      const form = upload.init!.body as FormData;
      expect((form.get("file") as File).name).toBe("mine.jsonl");
      expect(form.get("text_field")).toBe("text");
      const body = JSON.parse(String(calls.find((c) => c.path === "/api/jobs" && c.method === "POST")!.init!.body));
      expect(body).toMatchObject({ upload_id: "up-0123456789abcdef", input_license: "cc-by-4.0", synthetic: false, max_documents: 200 });
    });
  });
});
