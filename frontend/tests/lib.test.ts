import { describe, expect, it } from "vitest";
import { ACTIVE, api, dur, n, pct } from "../app/lib";
import { mockApi } from "./helpers";

describe("formatting", () => {
  it("n groups thousands and tolerates missing values", () => {
    expect(n(1234567)).toBe("1,234,567");
    expect(n(0)).toBe("0");
    expect(n(undefined)).toBe("0");
    expect(n(null)).toBe("0");
  });
  it("pct shows one decimal", () => {
    expect(pct(0.6)).toBe("60.0%");
    expect(pct(0.04567)).toBe("4.6%");
  });
  it("dur switches to minutes", () => {
    expect(dur(7.25)).toBe("7.3s");
    expect(dur(75)).toBe("1m 15s");
    expect(dur(600)).toBe("10m 0s");
  });
  it("ACTIVE lists the live statuses", () => {
    expect(ACTIVE).toEqual(["pending", "running"]);
  });
});

describe("api()", () => {
  it("returns parsed JSON", async () => {
    mockApi({ "GET /api/health": { status: "ok" } });
    await expect(api("/api/health")).resolves.toEqual({ status: "ok" });
  });
  it("turns a FastAPI error body into a readable message", async () => {
    mockApi({ "POST /api/jobs": new Response(JSON.stringify({ detail: "too many running" }), { status: 409 }) });
    await expect(api("/api/jobs", { method: "POST" })).rejects.toThrow("409: too many running");
  });
  it("falls back to the status text when the body is not JSON", async () => {
    mockApi({ "GET /api/x": new Response("boom", { status: 500, statusText: "Server Error" }) });
    await expect(api("/api/x")).rejects.toThrow("500: Server Error");
  });
  it("stringifies structured details (validation errors)", async () => {
    mockApi({ "POST /api/jobs": new Response(JSON.stringify({ detail: [{ msg: "bad" }] }), { status: 422 }) });
    await expect(api("/api/jobs", { method: "POST" })).rejects.toThrow('422: [{"msg":"bad"}]');
  });
});
