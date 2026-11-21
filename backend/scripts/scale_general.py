"""Scaling job: curate a large general dataset (default 300M tokens) with a RAM guard.

Run from `backend`:  .venv/Scripts/python scripts/scale_general.py [--tokens 300M] [--workers 3]
The job id is saved in `runs/scale-general.job`; running the script again resumes that job until it
finishes. If free RAM stays below --min-free-gb (default 4) for ~1 minute, the job is stopped at the
next chunk with a checkpoint (exit code 130) and can be resumed by running the script again.
Not used for training: it cannot change any experiment result.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_factory.config import load_config                      # noqa: E402
from dataset_factory.orchestrator import Job, JobSpec, load_resumable  # noqa: E402
from dataset_factory.cli import parse_count                        # noqa: E402

POINTER = Path("runs/scale-general.job")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="300M")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--check-every", type=float, default=15.0)
    a = ap.parse_args()

    def progress(snap):
        print(f"{snap['status']:9} read {snap['candidates']:>10,} accepted {snap['accepted']:>10,} "
              f"tokens {snap['tokens']:>13,}  free RAM {psutil.virtual_memory().available / 2**30:.1f} GB",
              flush=True)

    job = None
    if POINTER.exists():
        d = POINTER.read_text().strip()
        try:
            job = load_resumable(d, on_progress=progress, progress_every=60.0)
            if job.state.status in ("completed", "exhausted"):
                print(f"{d} already finished ({job.state.status}).")
                return 0
            print(f"resuming {d}", flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"cannot resume {d}: {e}", flush=True)
            return 1
    if job is None:
        spec = JobSpec(domain="general", max_tokens=parse_count(a.tokens), workers=a.workers)
        job = Job(spec, load_config(), output_dir="runs", on_progress=progress, progress_every=60.0)
        POINTER.parent.mkdir(exist_ok=True)
        POINTER.write_text(str(job.dir))
        print(f"new job {job.dir}", flush=True)

    done = threading.Event()

    def guard():
        low = 0
        while not done.wait(a.check_every):
            free = psutil.virtual_memory().available / 2**30
            low = low + 1 if free < a.min_free_gb else 0
            if low >= 4:
                print(f"RAM GUARD: free RAM {free:.1f} GB < {a.min_free_gb} GB, stopping at the next "
                      f"chunk; run this script again to resume", flush=True)
                job.stop()
                return

    threading.Thread(target=guard, daemon=True).start()
    try:
        job.run()
    finally:
        done.set()
    s = job.state.snapshot()
    print(f"{s['status']}: {s['accepted']:,} docs, {s['tokens']:,} tokens in {job.dir}")
    return 130 if s["status"] == "stopped" else (0 if s["status"] in ("completed", "exhausted") else 1)


if __name__ == "__main__":
    sys.exit(main())
