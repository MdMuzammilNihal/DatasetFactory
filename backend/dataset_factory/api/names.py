"""Friendly, permanent job names ("Legal 1", "Code random 3") kept in ``<runs>/job-names.json``.

The job id (``legal-20261123-224402-38f6``) stays the real identifier; the name is only for people.
Numbers count up per domain and mode and are never reused, so deleting "Legal 2" does not turn "Legal 3" into
"Legal 2". Jobs that existed before this file did get numbers in the order they were started. Job folders are not
touched, only this one small file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def name_prefix(domain: str, mode: str) -> str:
    base = (domain or "job").capitalize()
    return f"{base} random" if mode == "random" else base


class JobNames:
    def __init__(self, output_dir: Path) -> None:
        self.path = Path(output_dir) / "job-names.json"
        self._lock = threading.Lock()
        self._names: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        self._load()

    # -- storage ------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._names = {str(k): str(v) for k, v in dict(data.get("names", {})).items()}
            self._counters = {str(k): int(v) for k, v in dict(data.get("counters", {})).items()}
        except (OSError, ValueError, TypeError, AttributeError):
            self._names, self._counters = {}, {}          # missing or unreadable: start empty

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"counters": self._counters, "names": self._names}, indent=1), encoding="utf-8")
            for attempt in range(8):                      # Windows: the target can be briefly locked by another program
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            os.replace(tmp, self.path)
        except OSError:
            pass                                          # names are a convenience; never break a job over them

    # -- use ----------------------------------------------------------------

    def get(self, job_id: str) -> str:
        return self._names.get(job_id, "")

    def assign(self, job_id: str, domain: str, mode: str) -> str:
        with self._lock:
            return self._assign(job_id, domain, mode)

    def _assign(self, job_id: str, domain: str, mode: str) -> str:
        if job_id in self._names:
            return self._names[job_id]
        prefix = name_prefix(domain, mode)
        number = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = number
        self._names[job_id] = f"{prefix} {number}"
        self._save()
        return self._names[job_id]

    def forget(self, job_id: str) -> None:
        with self._lock:
            if self._names.pop(job_id, None) is not None:
                self._save()

    def sync(self, rows: list[dict[str, Any]]) -> None:
        """Give every row a name (oldest unnamed job first) and put it in ``row["name"]``."""
        with self._lock:
            for row in sorted((r for r in rows if r["job_id"] not in self._names),
                              key=lambda r: r.get("started_at") or ""):
                self._assign(row["job_id"], row.get("domain", ""), row.get("mode", "curate"))
            for row in rows:
                row["name"] = self._names.get(row["job_id"], "")
