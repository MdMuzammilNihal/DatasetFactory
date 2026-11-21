"""Runs jobs in background threads and keeps their live state for the API."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from ..config import Config, ConfigError, load_config
from ..orchestrator import Job, JobSpec, default_output_dir, load_resumable

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
TERMINAL = ("completed", "exhausted", "stopped", "failed", "interrupted")


class JobNotFound(KeyError):
    pass


class JobConflict(RuntimeError):
    pass


class _Entry:
    def __init__(self, job: Job) -> None:
        self.job = job
        self.snapshot: dict[str, Any] = job.state.snapshot()
        self.thread: threading.Thread | None = None
        self.error: str | None = None


class JobManager:
    def __init__(self, config: Config | None = None, output_dir: str | Path | None = None,
                 max_running: int = 2) -> None:
        self.config = config or load_config()
        self.output_dir = Path(output_dir) if output_dir else default_output_dir(self.config)
        self.max_running = max_running
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    # -- helpers ------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise JobNotFound(job_id)
        path = self.output_dir / job_id
        if not path.is_dir():
            raise JobNotFound(job_id)
        return path

    def _running(self) -> int:
        return sum(1 for e in self._entries.values() if e.thread and e.thread.is_alive())

    def _launch(self, job: Job, entry: _Entry) -> None:
        def progress(snapshot: dict[str, Any]) -> None:
            entry.snapshot = snapshot

        def target() -> None:
            try:
                job.run()
            except Exception as exc:  # noqa: BLE001 - shown through the API
                entry.error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                entry.snapshot = job.state.snapshot()

        job.on_progress = progress
        entry.thread = threading.Thread(target=target, name=f"job-{job.spec.job_id}", daemon=True)
        entry.thread.start()

    # -- commands -----------------------------------------------------------

    def start(self, spec: JobSpec) -> dict[str, Any]:
        with self._lock:
            if self._running() >= self.max_running:
                raise JobConflict(f"{self.max_running} job(s) already running")
            job = Job(spec, self.config, output_dir=self.output_dir, progress_every=0.5)
            entry = _Entry(job)
            self._entries[job.spec.job_id] = entry
            self._launch(job, entry)
            return self.get(job.spec.job_id)

    def stop(self, job_id: str) -> dict[str, Any]:
        entry = self._entries.get(job_id)
        if entry is None or not (entry.thread and entry.thread.is_alive()):
            if entry is None:
                self.job_dir(job_id)
            raise JobConflict("job is not running")
        entry.job.stop()
        return self.get(job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id)
        with self._lock:
            old = self._entries.get(job_id)
            if old and old.thread and old.thread.is_alive():
                raise JobConflict("job is still running")
            if self._running() >= self.max_running:
                raise JobConflict(f"{self.max_running} job(s) already running")
            try:
                job = load_resumable(path, progress_every=0.5)
            except ConfigError as exc:
                raise JobConflict(str(exc)) from exc
            if job.state.status in ("completed", "exhausted"):
                raise JobConflict(f"job already {job.state.status}")
            entry = _Entry(job)
            self._entries[job_id] = entry
            self._launch(job, entry)
            return self.get(job_id)

    # -- queries ------------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any]:
        entry = self._entries.get(job_id)
        if entry is not None:
            data = dict(entry.snapshot)
            if entry.error:
                data["message"] = entry.error
            return data
        path = self.job_dir(job_id)
        manifest = path / "manifest.json"
        if manifest.exists():
            state = json.loads(manifest.read_text(encoding="utf-8"))["state"]
            if state.get("status") == "running":      # process died without finishing
                state["status"] = "interrupted"
            return state
        raise JobNotFound(job_id)

    def spec(self, job_id: str) -> dict[str, Any] | None:
        entry = self._entries.get(job_id)
        if entry is not None:
            return dict(vars(entry.job.spec))
        manifest = self.job_dir(job_id) / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8")).get("spec")
        return None

    def list(self) -> list[dict[str, Any]]:
        ids = set(self._entries)
        if self.output_dir.is_dir():
            ids |= {p.name for p in self.output_dir.iterdir() if p.is_dir() and JOB_ID_RE.match(p.name)}
        rows = []
        for job_id in ids:
            try:
                rows.append(self.get(job_id))
            except (JobNotFound, OSError, ValueError, KeyError):
                continue
        rows.sort(key=lambda s: s.get("started_at") or "", reverse=True)
        return rows

    def shutdown(self) -> None:
        for entry in self._entries.values():
            if entry.thread and entry.thread.is_alive():
                entry.job.stop()
        for entry in self._entries.values():
            if entry.thread:
                entry.thread.join(timeout=30)
