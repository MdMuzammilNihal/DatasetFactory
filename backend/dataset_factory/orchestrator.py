"""The orchestrator: runs one curation job from request to finished dataset.

A job asks for documents of one domain, up to a number of documents and/or a
token budget. The orchestrator

1. builds the Factory and Generator agents for the domain and reads candidate
   chunks from them in a background thread (streaming, never the whole corpus);
2. sends each chunk to a pool of worker processes, each holding its own Scorer
   agent (filters, validator, scorer, classifier, PII scrubber, tokenizer);
3. takes the results back **in chunk order**, removes duplicates, and writes
   accepted records to ``data.jsonl`` until the target is met.

Because results are merged in chunk order, the output does not depend on the
number of workers: one worker and four produce the same records.

A job folder (``runs/<job_id>/``) holds:

    config.toml        the exact configuration used (its fingerprint is in every record)
    data.jsonl         accepted records, one JSON object per line
    rejections.jsonl   one line per rejected candidate: source, id, stage and reason
    dedup.bin          content hash + MinHash signature of every accepted record
    checkpoint.json    where to resume (written every ``checkpoint_every`` candidates)
    manifest.json      the request, counts, rejection reasons, licences, timings

An interrupted job resumes from its last checkpoint with ``resume_job``: files
are cut back to their checkpointed sizes, the deduplication index is rebuilt
from ``dedup.bin`` and every source continues where the checkpoint says, so the
finished dataset is the same as an uninterrupted run would have produced.

``mode = "random"`` builds the comparison baseline instead: documents are taken
from the same sources in the same order with no filtering at all (only the
licence policy, which is a legal rule, not a quality one), up to the same
budget. Training on it versus the curated set isolates the effect of curation.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import queue
import secrets
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from . import __version__
from .agents import Chunk, ScorerAgent, build_agents, candidate_chunks
from .config import Config, ConfigError, load_config
from .dedup import DedupIndex, build_index, dedup_settings
from .licenses import build_policy, normalize_license
from .models import RawDocument
from .provenance import add_dedup, utc_now
from .sources import BACKEND_DIR
from .tokens import load_token_counter
from .validators import DOMAINS

MODES = ("curate", "random")
STATUSES = ("pending", "running", "completed", "exhausted", "stopped", "failed")
SCORE_BINS = 20                     # score histogram: 20 bins of width 0.05
HASH_BYTES = 16                     # content hashes are 16 hex characters

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_RUNTIME_DEFAULTS: dict[str, Any] = {
    "workers": 4,
    "chunk_size": 1000,
    "checkpoint_every": 10_000,
    "tokenizer": "EleutherAI/pythia-160m",
    "log_rejections": True,
    "output_dir": "runs",
}


def runtime_settings(config: Config) -> dict[str, Any]:
    raw = dict(config.runtime)
    unknown = sorted(set(raw) - set(_RUNTIME_DEFAULTS))
    if unknown:
        raise ConfigError(f"[runtime] unknown setting(s): {', '.join(unknown)}")
    settings = {**_RUNTIME_DEFAULTS, **raw}
    for key in ("workers", "chunk_size", "checkpoint_every"):
        value = settings[key]
        minimum = 0 if key == "workers" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"[runtime] {key} must be a whole number of at least {minimum}")
    for key in ("tokenizer", "output_dir"):
        if not isinstance(settings[key], str):
            raise ConfigError(f"[runtime] {key} must be text")
    if not isinstance(settings["log_rejections"], bool):
        raise ConfigError("[runtime] log_rejections must be true or false")
    return settings


@dataclass
class JobSpec:
    """What a job is asked to produce. None means "use the [runtime] setting"."""

    domain: str
    max_documents: int | None = None      # stop after this many accepted documents
    max_tokens: int | None = None         # stop once this many tokens are accepted
    max_candidates: int | None = None     # stop after reading this many candidates
    mode: str = "curate"                  # "curate" or "random" (the baseline)
    synthetic: bool = False               # read the built-in generator instead of real corpora
    sources: list[str] | None = None      # read only these sources (overrides the routing)
    seed: int | None = None               # defaults to [meta] seed
    workers: int | None = None
    chunk_size: int | None = None
    checkpoint_every: int | None = None
    tokenizer: str | None = None
    log_rejections: bool | None = None
    job_id: str | None = None
    input_file: str | None = None         # curate a local JSON-lines file instead
    input_text_field: str = "text"
    input_license: str = "unknown"

    def check(self) -> None:
        if self.domain not in DOMAINS:
            raise ConfigError(f"unknown domain '{self.domain}'. Domains: {', '.join(DOMAINS)}")
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of: {', '.join(MODES)}")
        for name in ("max_documents", "max_tokens", "max_candidates"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ConfigError(f"{name} must be positive")
        if self.max_documents is None and self.max_tokens is None and self.max_candidates is None:
            raise ConfigError("give a target: max_documents, max_tokens or max_candidates")


def resolve_spec(spec: JobSpec, config: Config) -> JobSpec:
    """The spec with every None replaced by its configured default."""
    spec.check()
    runtime = runtime_settings(config)
    values = {
        key: runtime[key] if getattr(spec, key) is None else getattr(spec, key)
        for key in ("workers", "chunk_size", "checkpoint_every", "tokenizer", "log_rejections")
    }
    resolved = replace(
        spec,
        seed=config.seed if spec.seed is None else spec.seed,
        job_id=spec.job_id or new_job_id(spec.domain, spec.mode),
        **values,
    )
    if resolved.max_tokens is not None and resolved.tokenizer.lower() in ("", "none"):
        raise ConfigError("a token budget needs a tokenizer; tokenizer is set to none")
    return resolved


INPUT_SOURCE = "input"


def with_input_source(config: Config, spec: JobSpec) -> Config:
    """The configuration plus a source reading ``spec.input_file``."""
    path = Path(spec.input_file).resolve()   # type: ignore[arg-type]
    if not path.is_file():
        raise ConfigError(f"input file not found: {path}")
    table = {
        "kind": "jsonl", "path": str(path), "text": spec.input_text_field,
        "license": spec.input_license, "id_field": "id", "url_field": "url",
    }
    return replace(config, sources={**config.sources, INPUT_SOURCE: table})


def new_job_id(domain: str, mode: str = "curate") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = domain if mode == "curate" else f"{domain}-{mode}"
    return f"{prefix}-{stamp}-{secrets.token_hex(2)}"


def default_output_dir(config: Config) -> Path:
    path = Path(runtime_settings(config)["output_dir"])
    return path if path.is_absolute() else BACKEND_DIR / path


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """What a worker sends back for one chunk (documents stay in the worker)."""

    index: int
    n_documents: int
    # Per document, in chunk order: (label, record or None, content hash, signature, score)
    outcomes: list[tuple[str, dict | None, str | None, np.ndarray | None, float | None]]
    seconds: float


class _Worker:
    """Per-process state: built once, then used for every chunk."""

    def __init__(self, domain: str, config: Config, mode: str, job_id: str, tokenizer: str) -> None:
        counter = load_token_counter(tokenizer)
        self.mode = mode
        self.counter = counter
        self.config = config
        self.domain = domain
        self.job_id = job_id
        if mode == "curate":
            self.scorer: ScorerAgent | None = ScorerAgent(
                domain, config, job_id=job_id, token_counter=counter
            )
        else:
            self.scorer = None
            self.policy = build_policy(config)

    def process(self, index: int, documents: list[RawDocument]) -> ChunkResult:
        start = time.perf_counter()
        outcomes = []
        for document in documents:
            if self.scorer is not None:
                ev = self.scorer.evaluate(document)
                outcomes.append((ev.label, ev.record, ev.content_hash, ev.signature, ev.score))
            else:
                outcomes.append(self._random(document))
        return ChunkResult(index, len(documents), outcomes, time.perf_counter() - start)

    def _random(self, document: RawDocument):
        """Baseline mode: keep the document as it is, with its provenance."""
        result = self.policy.check(document.license)
        if not result.passed:
            return (result.label, None, None, None, None)
        digest = hashlib.blake2b(document.text.encode("utf-8"), digest_size=8).hexdigest()
        record: dict[str, Any] = {
            "id": f"{self.domain}-random-{digest}",
            "text": document.text,
            "domain": self.domain,
            "quality_score": None,
            "n_words": len(document.text.split()),
        }
        if self.counter is not None:
            record["n_tokens"] = self.counter(document.text)
        record["provenance"] = {
            "source": document.source,
            "source_id": document.source_id,
            "url": document.url,
            "agent": document.agent,
            "license": normalize_license(document.license),
            "synthetic": document.source.startswith("synthetic"),
            "content_hash": digest,
            "stages": {},
            "mode": "random",
            "accepted_at": utc_now(),
            "job_id": self.job_id,
            "pipeline_version": __version__,
            "config_fingerprint": self.config.fingerprint,
        }
        return ("accepted", record, digest, None, None)


_WORKER: _Worker | None = None


def _init_worker(domain: str, config: Config, mode: str, job_id: str, tokenizer: str) -> None:
    global _WORKER
    _WORKER = _Worker(domain, config, mode, job_id, tokenizer)


def _ping() -> bool:
    time.sleep(0.05)            # long enough that each ping lands on a new process
    return _WORKER is not None


def _process_chunk(index: int, documents: list[RawDocument]) -> ChunkResult:
    assert _WORKER is not None, "worker not initialised"
    return _WORKER.process(index, documents)


class _InlineFuture:
    """Stands in for a Future when the job runs without worker processes."""

    def __init__(self, value: ChunkResult) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> ChunkResult:
        return self._value

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    """Progress of a job. ``snapshot()`` returns it as plain JSON-ready data."""

    job_id: str
    domain: str
    mode: str
    status: str = "pending"
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    candidates: int = 0               # documents read, including those the prefilter dropped
    evaluated: int = 0                # documents a worker looked at
    accepted: int = 0                 # written to data.jsonl
    tokens: int = 0
    words: int = 0
    chunks: int = 0
    worker_seconds: float = 0.0       # time workers spent on chunks
    startup_seconds: float = 0.0      # starting workers, loading models and opening the job folder
    rejected: dict[str, int] = field(default_factory=dict)        # label -> count
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    by_agent: dict[str, dict[str, int]] = field(default_factory=dict)
    licenses: dict[str, int] = field(default_factory=dict)
    score_histogram: list[int] = field(default_factory=lambda: [0] * SCORE_BINS)
    accepted_score_sum: float = 0.0
    pii_removed: dict[str, int] = field(default_factory=dict)
    source_errors: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: int = 0
    resumed_from_checkpoint: int = 0

    @property
    def yield_rate(self) -> float:
        return self.accepted / self.candidates if self.candidates else 0.0

    @property
    def throughput(self) -> float:
        """Candidates per second of wall time (reading included)."""
        return self.candidates / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["yield_rate"] = round(self.yield_rate, 4)
        data["throughput_docs_per_s"] = round(self.throughput, 1)
        data["mean_accepted_score"] = (
            round(self.accepted_score_sum / self.accepted, 4) if self.accepted and self.mode == "curate"
            else None
        )
        working = self.elapsed_seconds - self.startup_seconds
        data["steady_throughput_docs_per_s"] = round(self.candidates / working, 1) if working > 0 else 0.0
        data["elapsed_seconds"] = round(self.elapsed_seconds, 2)
        data["startup_seconds"] = round(self.startup_seconds, 2)
        data["worker_seconds"] = round(self.worker_seconds, 2)
        data.pop("accepted_score_sum")
        return data

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> "JobState":
        names = {f for f in cls.__dataclass_fields__}
        state = cls(**{k: v for k, v in data.items() if k in names})
        if data.get("mean_accepted_score") is not None:
            state.accepted_score_sum = data["mean_accepted_score"] * state.accepted
        return state


def _bump(counter: dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = counter.get(key, 0) + n


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[dict[str, Any]], None]
_END = object()


class Job:
    """One curation job. ``run()`` blocks until it ends; ``stop()`` (from any
    thread) ends it at the next chunk boundary with a checkpoint, so it can be
    resumed. ``state.snapshot()`` is safe to read while it runs."""

    def __init__(
        self,
        spec: JobSpec,
        config: Config | None = None,
        *,
        output_dir: str | Path | None = None,
        on_progress: ProgressCallback | None = None,
        progress_every: float = 1.0,
        _resume: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.spec = resolve_spec(spec, self.config)
        if self.spec.input_file:
            self.config = with_input_source(self.config, self.spec)
            self.spec.sources = [INPUT_SOURCE]
        base = Path(output_dir) if output_dir is not None else default_output_dir(self.config)
        self.dir = base / self.spec.job_id if _resume is None else Path(output_dir)  # type: ignore[arg-type]
        self.on_progress = on_progress
        self.progress_every = progress_every
        self.state = JobState(self.spec.job_id, self.spec.domain, self.spec.mode)  # type: ignore[arg-type]
        self._resume = _resume
        self._stop = threading.Event()
        self._t0: float | None = None
        self._last_progress = 0.0
        self._last_chunk: Chunk | None = None
        # Each agent's position after its latest finished chunk: (positions, next_source).
        self._latest: dict[str, tuple[dict, str | None]] = {}
        self._dedup_enabled = dedup_settings(self.config)["enabled"] and self.spec.mode == "curate"
        self._dedup_threshold = float(dedup_settings(self.config)["threshold"])

    # -- public -------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def data_path(self) -> Path:
        return self.dir / "data.jsonl"

    def run(self) -> JobState:
        state = self.state
        state.status = "running"
        state.started_at = state.started_at or utc_now()
        run_start = time.perf_counter()
        t0 = run_start - state.elapsed_seconds        # elapsed time carries over a resume
        self._t0 = t0
        self._progress(force=True)                    # report "running" before sources open
        executor = None
        reader = None
        try:
            self._prepare_folder()
            index = self._load_index()
            factory, generator, route = build_agents(
                self.spec.domain, self.config, synthetic=self.spec.synthetic,
                sources=self.spec.sources, prefilter=self.spec.mode == "curate",
                chunk_size=self.spec.chunk_size, seed=self.spec.seed,
            )
            self._agents = [a for a in (factory, generator) if a is not None]
            chunks = candidate_chunks(
                factory, generator, route.factory_share, resume=(self._resume or {}).get("resume")
            )
            reader = _Reader(chunks, maxsize=max(2, 2 * max(1, self.spec.workers)) + 2)  # type: ignore[arg-type]
            init_args = (self.spec.domain, self.config, self.spec.mode, self.spec.job_id,
                         self.spec.tokenizer)
            if self.spec.workers:
                executor = ProcessPoolExecutor(
                    max_workers=self.spec.workers, initializer=_init_worker, initargs=init_args
                )
                # Start every worker before reading begins, so loading errors
                # (e.g. a missing tokenizer) surface at once and the start-up
                # time can be told apart from the processing time.
                for future in [executor.submit(_ping) for _ in range(self.spec.workers)]:
                    future.result()
                submit = lambda c: executor.submit(_process_chunk, c.index, c.documents)  # noqa: E731
                in_flight = 2 * self.spec.workers
            else:
                worker = _Worker(*init_args)
                state.startup_seconds += time.perf_counter() - run_start
                submit = lambda c: _InlineFuture(worker.process(c.index, c.documents))  # noqa: E731
                in_flight = 1
            if self.spec.workers:
                state.startup_seconds += time.perf_counter() - run_start
            outcome = self._loop(reader, submit, in_flight, index, t0)
            state.status = outcome
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest, then re-raised
            state.status = "failed"
            state.message = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            if reader is not None:
                # On a user stop, do not wait long for a reader blocked on a network read (e.g. a source
                # still opening): it is a daemon thread and closes its streams itself; the checkpoint is saved.
                reader.close(timeout=2.0 if self._stop.is_set() else 30.0)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            self._close_files()
            state.elapsed_seconds = time.perf_counter() - t0
            state.finished_at = utc_now()
            self._collect_source_errors()
            if self.dir.exists():
                self._write_manifest()
            self._progress(force=True)
        return state

    # -- main loop ----------------------------------------------------------

    def _loop(self, reader: "_Reader", submit, in_flight: int, index: DedupIndex | None, t0: float) -> str:
        spec, state = self.spec, self.state
        pending: collections.deque[tuple[Chunk, Any]] = collections.deque()
        since_checkpoint = 0
        sources_done = False
        while True:
            while not sources_done and len(pending) < in_flight and not self._stop.is_set():
                if spec.max_candidates is not None and state.candidates + sum(
                    len(c.documents) + sum(c.prefiltered.values()) for c, _ in pending
                ) >= spec.max_candidates:
                    break
                chunk = reader.get(timeout=0.2 if pending else 0.5)
                if chunk is None:                   # nothing ready yet
                    self._progress()                # keeps elapsed time live while waiting
                    if pending:
                        break                       # work on what is already in flight
                    continue
                if chunk is _END:
                    sources_done = True
                    break
                pending.append((chunk, submit(chunk)))
            if not pending:
                if self._stop.is_set():
                    self._checkpoint(index)
                    return "stopped"
                return "exhausted" if sources_done else "completed"
            chunk, future = pending.popleft()
            while True:                             # wait in short steps so elapsed time and speed stay live
                try:
                    result: ChunkResult = future.result(timeout=1.0)
                    break
                except FutureTimeout:
                    if future.done():               # the task itself raised a TimeoutError: a real failure
                        raise
                    self._progress()
            done = self._merge(chunk, result, index)
            state.elapsed_seconds = time.perf_counter() - t0
            since_checkpoint += len(chunk.documents) + sum(chunk.prefiltered.values())
            if done:
                for _, other in pending:
                    other.cancel()
                return "completed"
            if self._stop.is_set():
                for _, other in pending:
                    other.cancel()
                self._checkpoint(index)
                return "stopped"
            if since_checkpoint >= spec.checkpoint_every:
                self._checkpoint(index)
                since_checkpoint = 0
            self._progress()

    def _target_met(self) -> bool:
        spec, state = self.spec, self.state
        return (
            (spec.max_documents is not None and state.accepted >= spec.max_documents)
            or (spec.max_tokens is not None and state.tokens >= spec.max_tokens)
            or (spec.max_candidates is not None and state.candidates >= spec.max_candidates)
        )

    def _merge(self, chunk: Chunk, result: ChunkResult, index: DedupIndex | None) -> bool:
        """Fold one chunk's results into the job, in document order. Returns
        True when the target is met (the rest of the chunk is then not used)."""
        state = self.state
        state.chunks += 1
        state.worker_seconds += result.seconds
        for label, n in chunk.prefiltered.items():
            _bump(state.rejected, label, n)
            state.candidates += n
            _bump(state.by_agent.setdefault(chunk.agent, {}), "read", n)
        lines, rejection_lines, dedup_rows = [], [], []
        target_met = False
        for document, (label, record, digest, signature, score) in zip(chunk.documents, result.outcomes):
            state.candidates += 1
            state.evaluated += 1
            src = state.by_source.setdefault(document.source, {"read": 0, "accepted": 0})
            src["read"] += 1
            agent = state.by_agent.setdefault(chunk.agent, {})
            _bump(agent, "read")
            if score is not None:
                state.score_histogram[min(SCORE_BINS - 1, max(0, int(score * SCORE_BINS)))] += 1
            if record is not None and index is not None and signature is not None:
                found = index.check_and_add(signature, digest)
                if found.duplicate:
                    label, record = f"dedup:{found.kind}", None
                else:
                    add_dedup(record, found.similarity, self._dedup_threshold)
                    dedup_rows.append(digest.encode("ascii").ljust(HASH_BYTES)[:HASH_BYTES]
                                      + signature.astype("<u2").tobytes())
            if record is None:
                _bump(state.rejected, label)
                if self.spec.log_rejections:
                    rejection_lines.append({
                        "source": document.source, "source_id": document.source_id,
                        "agent": chunk.agent, "label": label,
                        "score": None if score is None else round(score, 4),
                    })
            else:
                state.accepted += 1
                src["accepted"] += 1
                _bump(agent, "accepted")
                state.tokens += int(record.get("n_tokens") or 0)
                state.words += int(record.get("n_words") or 0)
                if record.get("quality_score") is not None:
                    state.accepted_score_sum += record["quality_score"]
                prov = record["provenance"]
                _bump(state.licenses, prov["license"])
                for kind, n in prov.get("pii_removed", {}).items():
                    _bump(state.pii_removed, kind, n)
                lines.append(record)
            if self._target_met():
                target_met = True
                break
        self._write(lines, rejection_lines, dedup_rows)
        if not target_met:
            self._last_chunk = chunk
            self._latest[chunk.agent] = (chunk.positions, chunk.next_source)
        return target_met

    # -- files --------------------------------------------------------------

    def _prepare_folder(self) -> None:
        if self._resume is None:
            self.dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(self.config.source_path, self.dir / "config.toml")
            sizes = {"data": 0, "rejections": 0, "dedup": 0}
        else:
            sizes = self._resume["files"]
        self._files = {}
        for key, name in (("data", "data.jsonl"), ("rejections", "rejections.jsonl"),
                          ("dedup", "dedup.bin")):
            path = self.dir / name
            fh = open(path, "ab")
            fh.truncate(sizes[key])          # drop anything written after the checkpoint
            fh.seek(sizes[key])
            self._files[key] = fh
        self._write_manifest()

    def _write(self, lines: list[dict], rejection_lines: list[dict], dedup_rows: list[bytes]) -> None:
        if lines:
            self._files["data"].write(b"".join(
                json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in lines
            ))
        if rejection_lines:
            self._files["rejections"].write(b"".join(
                json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in rejection_lines
            ))
        if dedup_rows:
            self._files["dedup"].write(b"".join(dedup_rows))

    def _close_files(self) -> None:
        for fh in getattr(self, "_files", {}).values():
            fh.close()

    def _load_index(self) -> DedupIndex | None:
        if not self._dedup_enabled:
            return None
        index = build_index(self.config)
        if self._resume is not None:
            row = HASH_BYTES + 2 * index.num_perm
            raw = (self.dir / "dedup.bin").read_bytes()[: self._resume["files"]["dedup"]]
            for start in range(0, len(raw), row):
                digest = raw[start:start + HASH_BYTES].decode("ascii").strip()
                signature = np.frombuffer(raw[start + HASH_BYTES:start + row], dtype="<u2")
                index.add(signature.astype(np.uint16), digest)
        return index

    def _checkpoint(self, index: DedupIndex | None) -> None:
        chunk = self._last_chunk
        for fh in self._files.values():
            fh.flush()
            os.fsync(fh.fileno())
        resume = dict((self._resume or {}).get("resume") or {})
        if chunk is not None:
            positions = dict(resume.get("positions") or {})
            next_source = dict(resume.get("next_source") or {})
            positions[chunk.agent] = chunk.positions
            next_source[chunk.agent] = chunk.next_source
            # Positions of the other agent come from its own latest chunk.
            for role, (pos, nxt) in self._latest.items():
                if role != chunk.agent:
                    positions[role], next_source[role] = pos, nxt
            resume = {
                "positions": positions, "next_source": next_source,
                "taken": chunk.taken, "next_index": chunk.index + 1,
            }
        self.state.checkpoints += 1
        checkpoint = {
            "job_id": self.spec.job_id,
            "spec": asdict(self.spec),
            "resume": resume,
            "files": {key: fh.tell() for key, fh in self._files.items()},
            "state": self.state.snapshot(),
            "written_at": utc_now(),
            "config_fingerprint": self.config.fingerprint,
        }
        _write_json_atomic(self.dir / "checkpoint.json", checkpoint)
        self._resume = checkpoint
        self._write_manifest()

    def _collect_source_errors(self) -> None:
        errors = []
        for agent in getattr(self, "_agents", []):
            for error in agent.errors:
                errors.append({"agent": agent.role, **asdict(error)})
        self.state.source_errors = errors[-50:]

    def _write_manifest(self) -> None:
        files = {}
        for name in ("data.jsonl", "rejections.jsonl"):
            path = self.dir / name
            if path.exists():
                files[name] = {"bytes": path.stat().st_size}
        if self.state.status in ("completed", "exhausted") and (self.dir / "data.jsonl").exists():
            files["data.jsonl"]["sha256"] = _sha256(self.dir / "data.jsonl")
        manifest = {
            "job_id": self.spec.job_id,
            "status": self.state.status,
            "spec": asdict(self.spec),
            "state": self.state.snapshot(),
            "files": files,
            "config": {
                "version": self.config.version,
                "fingerprint": self.config.fingerprint,
                "file": "config.toml",
            },
            "environment": {
                "pipeline_version": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        _write_json_atomic(self.dir / "manifest.json", manifest)

    def _progress(self, force: bool = False) -> None:
        if self.on_progress is None:
            return
        now = time.monotonic()
        if force or now - self._last_progress >= self.progress_every:
            self._last_progress = now
            if self.state.status == "running" and self._t0 is not None:
                self.state.elapsed_seconds = time.perf_counter() - self._t0
            self.on_progress(self.state.snapshot())


class _Reader:
    """Reads chunks in a background thread so streaming overlaps with work."""

    def __init__(self, chunks: Iterator[Chunk], maxsize: int) -> None:
        self._chunks = chunks
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="chunk-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for chunk in self._chunks:
                while not self._closed.is_set():
                    try:
                        self._queue.put(chunk, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._closed.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - handed to the main thread
            self._error = exc
        finally:
            self._chunks.close()               # closes every source stream, in this thread
            self._put_end()

    def _put_end(self) -> None:
        while True:
            try:
                self._queue.put(_END, timeout=0.2)
                return
            except queue.Full:
                if self._closed.is_set():
                    return

    def get(self, timeout: float | None = None):
        """The next chunk, ``_END`` after the last one, or None on timeout."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _END and self._error is not None:
            raise self._error
        return item

    def close(self, timeout: float = 30.0) -> None:
        self._closed.set()
        while True:                            # unblock the reader if it waits on a full queue
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Helpers and entry points
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_job(
    spec: JobSpec,
    config: Config | None = None,
    *,
    output_dir: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
) -> Job:
    """Run a job to the end and return it (``job.state``, ``job.dir``)."""
    job = Job(spec, config, output_dir=output_dir, on_progress=on_progress)
    job.run()
    return job


def load_resumable(job_dir: str | Path, config: Config | None = None, **kwargs: Any) -> Job:
    """Rebuild a stopped or interrupted job from its folder (call ``run()`` on it)."""
    job_dir = Path(job_dir)
    path = job_dir / "checkpoint.json"
    if not path.exists():
        raise ConfigError(f"{job_dir} has no checkpoint.json; it cannot be resumed")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    config = config or load_config(job_dir / "config.toml")
    if config.fingerprint != checkpoint["config_fingerprint"]:
        raise ConfigError("the configuration differs from the one the job started with")
    spec = JobSpec(**checkpoint["spec"])
    job = Job(spec, config, output_dir=job_dir, _resume=checkpoint, **kwargs)
    job.state = JobState.from_snapshot(checkpoint["state"])
    job.state.resumed_from_checkpoint = job.state.checkpoints
    job._latest = {
        role: (pos, checkpoint["resume"].get("next_source", {}).get(role))
        for role, pos in (checkpoint["resume"].get("positions") or {}).items()
    }
    return job


def resume_job(job_dir: str | Path, config: Config | None = None, **kwargs: Any) -> Job:
    """Resume a job from its last checkpoint and run it to the end."""
    job = load_resumable(job_dir, config, **kwargs)
    job.run()
    return job


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


__all__ = [
    "Job", "JobSpec", "JobState", "load_resumable", "read_records", "resume_job", "run_job",
    "runtime_settings", "MODES", "STATUSES",
]
