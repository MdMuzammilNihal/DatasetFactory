"""Long-run check: run a job, stop it, resume it, and report memory and resume time.

Usage (from the backend folder):
    python scripts/stability_check.py general --tokens 100M --stop-after 150
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import psutil

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dataset_factory.config import load_config  # noqa: E402
from dataset_factory.orchestrator import Job, JobSpec, load_resumable  # noqa: E402
from dataset_factory.cli import parse_count  # noqa: E402


def rss_mb() -> float:
    me = psutil.Process()
    return sum(p.memory_info().rss for p in [me, *me.children(recursive=True)]) / 1e6


def run_with_memory(job: Job, stop_after: float | None):
    peak, samples, t0 = 0.0, [], time.time()
    thread = threading.Thread(target=job.run, daemon=True)
    thread.start()
    first_progress = None
    while thread.is_alive():
        time.sleep(3)
        peak = max(peak, rss_mb())
        samples.append((round(time.time() - t0), job.state.snapshot()["candidates"], round(rss_mb())))
        if first_progress is None and job.state.snapshot()["candidates"] > 0:
            first_progress = time.time() - t0
        if stop_after and time.time() - t0 > stop_after:
            job.stop()
            stop_after = None
    return peak, samples, first_progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--tokens", default="100M")
    ap.add_argument("--stop-after", type=float, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume-for", type=float, default=None, help="stop the resumed job after this many seconds")
    args = ap.parse_args()
    config = load_config()
    spec = JobSpec(args.domain, max_tokens=parse_count(args.tokens), workers=args.workers)
    job = Job(spec, config)
    peak, samples, _ = run_with_memory(job, args.stop_after)
    snap = job.state.snapshot()
    print(f"phase 1: {snap['status']}, {snap['candidates']:,} candidates, {snap['tokens']:,} tokens, peak {peak:.0f} MB")
    print("samples (s, candidates, MB):", samples[::5])
    folder = job.dir
    resumed = load_resumable(folder, config)
    before = resumed.state.snapshot()["candidates"]
    t0 = time.time()
    peak2, samples2, _ = run_with_memory(resumed, args.resume_for)
    # time until the candidate counter moved past the checkpoint
    moved = next((s[0] for s in samples2 if s[1] > before), None)
    snap = resumed.state.snapshot()
    print(f"resume: from {before:,} candidates; first new candidates after ~{moved} s; "
          f"{snap['status']}, {snap['candidates']:,} candidates, {snap['tokens']:,} tokens, peak {peak2:.0f} MB, total {time.time()-t0:.0f} s")
    print("samples (s, candidates, MB):", samples2[::10])


if __name__ == "__main__":
    main()
