"""Throughput benchmark: documents per second for 1, 2, 4... worker processes.

This reproduces the paper's throughput-scaling test (Table 10). To measure the
pipeline and not the network, it reads a local JSON-lines file of real
documents (``fetch_benchmark_corpus`` streams one from FineWeb, the paper's
web corpus) and runs the real orchestrator on it with every stage switched on:
filters, validator, score, classifier, PII scrubbing, MinHash, token counting,
deduplication and writing.

Two throughputs are reported: *steady* excludes starting the worker processes
(each loads the language identifier, the classifier and the tokenizer);
*overall* is candidates divided by the whole wall time.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import Config, load_config
from .orchestrator import JobSpec, run_job
from .sources import BACKEND_DIR

DEFAULT_CORPUS = BACKEND_DIR / "data" / "bench" / "fineweb.jsonl"


def fetch_benchmark_corpus(n: int = 20_000, path: Path = DEFAULT_CORPUS, config: Config | None = None) -> Path:
    """Stream ``n`` FineWeb documents (corpus order) into a local JSON-lines file."""
    from .sources import open_source, source_specs
    from dataclasses import replace

    config = config or load_config()
    spec = replace(source_specs(config)["fineweb"], shuffle_buffer=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        stream = open_source(spec, domain="general", agent="generator", seed=0)
        try:
            for doc in itertools.islice(stream, n):
                fh.write(json.dumps({"id": doc.source_id, "url": doc.url, "text": doc.text},
                                    ensure_ascii=False) + "\n")
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
    os.replace(tmp, path)
    return path


def run_benchmark(
    domains: list[str],
    workers: list[int],
    corpus: Path = DEFAULT_CORPUS,
    *,
    config: Config | None = None,
    chunk_size: int | None = None,
    repeats: int = 1,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every (domain, workers) pair over the whole corpus; return one row each."""
    config = config or load_config()
    if not corpus.is_file():
        raise FileNotFoundError(f"{corpus} not found; fetch it first (--fetch N)")
    rows: list[dict[str, Any]] = []
    out = Path(tempfile.mkdtemp(prefix="df-bench-"))
    try:
        for domain in domains:
            single = None
            for w in workers:
                best = None
                for _ in range(repeats):
                    job = run_job(
                        JobSpec(domain, max_candidates=10**12, input_file=str(corpus),
                                input_license="odc-by-1.0", workers=w, chunk_size=chunk_size,
                                log_rejections=True),
                        config, output_dir=out,
                    )
                    s = job.state.snapshot()
                    shutil.rmtree(job.dir, ignore_errors=True)
                    if best is None or s["steady_throughput_docs_per_s"] > best["steady_throughput_docs_per_s"]:
                        best = s
                assert best is not None
                steady = best["steady_throughput_docs_per_s"]
                if single is None:
                    single = steady / max(1, w)       # per-worker reference (w is 1 normally)
                row = {
                    "domain": domain,
                    "workers": w,
                    "documents": best["candidates"],
                    "accepted": best["accepted"],
                    "startup_s": best["startup_seconds"],
                    "elapsed_s": best["elapsed_seconds"],
                    "steady_docs_per_s": steady,
                    "overall_docs_per_s": best["throughput_docs_per_s"],
                    "efficiency": round(steady / (max(1, w) * single), 3) if w else None,
                }
                rows.append(row)
                if report:
                    report(row)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return rows
