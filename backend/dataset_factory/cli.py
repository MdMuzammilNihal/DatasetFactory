"""Command line: ``python -m dataset_factory <command>`` (run from the backend folder).

    curate medical --tokens 2M                    curate real data up to 2 million tokens
    curate medical legal --docs 500 --synthetic   offline, built-in generator, two jobs
    curate general --docs 1000 --mode random      the same-source random baseline
    curate code --docs 100 --input my.jsonl       curate a local JSON-lines file
    resume runs/<job_id>                          continue a stopped or interrupted job
    status runs/<job_id>                          summary of a job folder
    benchmark --fetch 20000                       cache 20,000 FineWeb documents locally
    benchmark --workers 1 2 4 --domains general   throughput scaling (the paper's Table 10)

Ctrl+C stops a job at the next chunk with a checkpoint; ``resume`` continues it.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigError, load_config
from .orchestrator import MODES, Job, JobSpec, load_resumable
from .validators import DOMAINS

_SUFFIXES = {"k": 10**3, "m": 10**6, "b": 10**9}


def parse_count(text: str) -> int:
    """'2M' -> 2000000, '500k' -> 500000, '1500' -> 1500."""
    text = text.strip().lower().replace("_", "").replace(",", "")
    factor = _SUFFIXES.get(text[-1:], 1)
    number = text[:-1] if factor != 1 else text
    try:
        value = int(float(number) * factor)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a count: {text!r} (use e.g. 5000, 500k, 2M)")
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _printer(quiet: bool):
    def show(s: dict[str, Any]) -> None:
        if quiet:
            return
        print(
            f"\r[{s['job_id']}] {s['status']:<9} read {s['candidates']:>9,}  "
            f"accepted {s['accepted']:>8,}  tokens {s['tokens']:>12,}  "
            f"yield {s['yield_rate']:6.1%}  {s['throughput_docs_per_s']:>7,.0f} docs/s",
            end="", flush=True,
        )
    return show


def _run_with_ctrl_c(job: Job) -> None:
    """Run the job; the first Ctrl+C stops it cleanly with a checkpoint."""
    if threading.current_thread() is not threading.main_thread():
        job.run()
        return
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        print("\nstopping at the next chunk (Ctrl+C again to abort)...", flush=True)
        job.stop()
        signal.signal(signal.SIGINT, previous)

    signal.signal(signal.SIGINT, handler)
    try:
        job.run()
    finally:
        signal.signal(signal.SIGINT, previous)


def _summary(state: dict[str, Any], job_dir: Path) -> str:
    rejected = sorted(state["rejected"].items(), key=lambda kv: -kv[1])[:8]
    lines = [
        "",
        f"job      {state['job_id']}  ({state['mode']}, {state['domain']})",
        f"status   {state['status']}" + (f": {state['message']}" if state.get("message") else ""),
        f"folder   {job_dir}",
        f"read     {state['candidates']:,} candidates, accepted {state['accepted']:,} "
        f"({state['yield_rate']:.1%}), {state['tokens']:,} tokens, {state['words']:,} words",
        f"time     {state['elapsed_seconds']:.1f} s (start-up {state['startup_seconds']:.1f} s), "
        f"{state['throughput_docs_per_s']:,.0f} docs/s overall, "
        f"{state['steady_throughput_docs_per_s']:,.0f} after start-up",
    ]
    if state.get("mean_accepted_score") is not None:
        lines.append(f"score    mean accepted quality score {state['mean_accepted_score']:.3f}")
    if rejected:
        lines.append("rejected " + ", ".join(f"{k} {v:,}" for k, v in rejected))
    if state["licenses"]:
        lines.append("licences " + ", ".join(f"{k} {v:,}" for k, v in state["licenses"].items()))
    for source, counts in state["by_source"].items():
        share = counts["accepted"] / counts["read"] if counts["read"] else 0
        lines.append(f"source   {source}: {counts['accepted']:,} of {counts['read']:,} ({share:.0%})")
    for error in state["source_errors"][-3:]:
        lines.append(f"error    {error['source']} at {error['position']}: {error['error'][:120]}")
    return "\n".join(lines)


def cmd_curate(args: argparse.Namespace) -> int:
    config = load_config(args.config) if args.config else load_config()
    status = 0
    for domain in args.domains:
        spec = JobSpec(
            domain=domain, max_documents=args.docs, max_tokens=args.tokens,
            max_candidates=args.candidates, mode=args.mode, synthetic=args.synthetic,
            sources=args.sources.split(",") if args.sources else None, seed=args.seed,
            workers=args.workers, chunk_size=args.chunk_size,
            checkpoint_every=args.checkpoint_every, tokenizer=args.tokenizer,
            log_rejections=False if args.no_rejection_log else None,
            input_file=args.input, input_text_field=args.text_field,
            input_license=args.input_license,
        )
        job = Job(spec, config, output_dir=args.out, on_progress=_printer(args.quiet))
        _run_with_ctrl_c(job)
        print(_summary(job.state.snapshot(), job.dir))
        if job.state.status == "stopped":
            print(f"resume   python -m dataset_factory resume {job.dir}")
            return 130
        if job.state.status == "exhausted":
            print("note     the sources ran out before the target was reached")
            status = max(status, 3)
    return status


def cmd_resume(args: argparse.Namespace) -> int:
    job = load_resumable(args.job_dir, on_progress=_printer(args.quiet))
    if job.state.status in ("completed", "exhausted"):
        print(f"{args.job_dir} already finished ({job.state.status}).")
        return 0
    _run_with_ctrl_c(job)
    print(_summary(job.state.snapshot(), job.dir))
    return 130 if job.state.status == "stopped" else 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.job_dir) / "manifest.json"
    if not path.exists():
        print(f"no manifest.json in {args.job_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(_summary(manifest["state"], Path(args.job_dir)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    app = create_app(output_dir=args.out, max_running=args.max_running, upload_days=args.upload_days)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import DEFAULT_CORPUS, fetch_benchmark_corpus, run_benchmark

    corpus = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
    if args.fetch:
        print(f"streaming {args.fetch:,} FineWeb documents to {corpus} ...", flush=True)
        fetch_benchmark_corpus(args.fetch, corpus)
        print("done")
        if not args.run:
            return 0
    print(f"{'domain':<11}{'workers':>8}{'docs':>9}{'accepted':>10}{'start-up s':>12}"
          f"{'steady docs/s':>15}{'overall docs/s':>16}{'efficiency':>12}")

    def report(row):
        eff = "" if row["efficiency"] is None else f"{row['efficiency']:.1%}"
        print(f"{row['domain']:<11}{row['workers']:>8}{row['documents']:>9,}{row['accepted']:>10,}"
              f"{row['startup_s']:>12.1f}{row['steady_docs_per_s']:>15,.0f}"
              f"{row['overall_docs_per_s']:>16,.0f}{eff:>12}", flush=True)

    rows = run_benchmark(args.domains, args.workers, corpus, chunk_size=args.chunk_size,
                         repeats=args.repeats, report=report)
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dataset_factory", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("curate", help="build a dataset for one or more domains")
    p.add_argument("domains", nargs="+", choices=DOMAINS, metavar="domain",
                   help=f"one or more of: {', '.join(DOMAINS)}")
    p.add_argument("--docs", type=parse_count, help="stop after this many accepted documents")
    p.add_argument("--tokens", type=parse_count, help="stop once this many tokens are accepted (e.g. 2M)")
    p.add_argument("--candidates", type=parse_count, help="stop after reading this many candidates")
    p.add_argument("--mode", choices=MODES, default="curate",
                   help="'random' builds the unfiltered same-source baseline")
    p.add_argument("--synthetic", action="store_true", help="offline: read the built-in generator")
    p.add_argument("--sources", help="comma-separated source keys to read instead of the routing")
    p.add_argument("--input", help="curate a local JSON-lines file")
    p.add_argument("--text-field", default="text", help="text field of --input rows")
    p.add_argument("--input-license", default="unknown", help="licence of --input")
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--checkpoint-every", type=int)
    p.add_argument("--tokenizer", help='tokenizer model, or "none"')
    p.add_argument("--no-rejection-log", action="store_true")
    p.add_argument("--out", help="folder for job folders (default: [runtime] output_dir)")
    p.add_argument("--config", help="configuration file (default: config/default.toml)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_curate)

    p = sub.add_parser("resume", help="continue a stopped or interrupted job")
    p.add_argument("job_dir")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("status", help="summarise a job folder")
    p.add_argument("job_dir")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("serve", help="start the HTTP API for the dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", help="folder for job folders (default: [runtime] output_dir)")
    p.add_argument("--max-running", type=int, default=2, help="jobs that may run at once")
    p.add_argument("--upload-days", type=int, default=14,
                   help="delete uploads older than this many days at start-up (0 = keep forever)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("benchmark", help="throughput for several worker counts")
    p.add_argument("--fetch", type=parse_count, help="first stream this many FineWeb documents")
    p.add_argument("--run", action="store_true", help="with --fetch: also run the benchmark")
    p.add_argument("--corpus", help="JSON-lines file to read (default: data/bench/fineweb.jsonl)")
    p.add_argument("--domains", nargs="+", default=["general"], choices=DOMAINS)
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--repeats", type=int, default=1, help="runs per setting; the best is kept")
    p.add_argument("--save", help="write the rows to this JSON file")
    p.set_defaults(func=cmd_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "docs", None) is None and getattr(args, "tokens", None) is None \
            and getattr(args, "candidates", None) is None and args.command == "curate":
        print("give a target: --docs, --tokens or --candidates", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
