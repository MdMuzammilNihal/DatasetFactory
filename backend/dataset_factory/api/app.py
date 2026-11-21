"""The FastAPI application. Run: ``python -m dataset_factory serve``."""

from __future__ import annotations

import asyncio
import itertools
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Config, ConfigError, load_config
from ..orchestrator import MODES, STATUSES, JobSpec, runtime_settings
from ..validators import DOMAINS
from .manager import TERMINAL, JobConflict, JobManager, JobNotFound


class JobRequest(BaseModel):
    domain: str
    max_documents: int | None = Field(None, ge=1)
    max_tokens: int | None = Field(None, ge=1)
    max_candidates: int | None = Field(None, ge=1)
    mode: str = "curate"
    synthetic: bool = False
    sources: list[str] | None = None
    seed: int | None = None
    workers: int | None = Field(None, ge=0, le=32)


def _read_lines(path: Path, offset: int, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """``limit`` JSON lines from ``offset``; second value is whether more follow."""
    if not path.exists():
        return [], False
    with path.open(encoding="utf-8") as fh:
        lines = list(itertools.islice((l for l in fh if l.strip()), offset, offset + limit + 1))
    rows = [json.loads(l) for l in lines[:limit]]
    return rows, len(lines) > limit


def create_app(config: Config | None = None, output_dir: str | Path | None = None,
               max_running: int = 2) -> FastAPI:
    manager = JobManager(config, output_dir, max_running)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await asyncio.to_thread(manager.shutdown)   # stop running jobs at a checkpoint

    app = FastAPI(title="Dataset Factory", version=__version__, lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.exception_handler(JobNotFound)
    async def _nf(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": f"no such job: {exc.args[0]}"}, status_code=404)

    @app.exception_handler(JobConflict)
    async def _conflict(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/meta")
    def meta():
        cfg = manager.config
        return {
            "domains": list(DOMAINS), "modes": list(MODES), "statuses": list(STATUSES),
            "runtime": runtime_settings(cfg), "config_fingerprint": cfg.fingerprint,
            "max_running": manager.max_running,
        }

    @app.get("/api/jobs")
    def list_jobs():
        return manager.list()

    @app.post("/api/jobs", status_code=202)
    def start_job(req: JobRequest):
        try:
            return manager.start(JobSpec(**req.model_dump()))
        except ConfigError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return {"state": manager.get(job_id), "spec": manager.spec(job_id)}

    @app.post("/api/jobs/{job_id}/stop", status_code=202)
    def stop_job(job_id: str):
        return manager.stop(job_id)

    @app.post("/api/jobs/{job_id}/resume", status_code=202)
    def resume_job(job_id: str):
        return manager.resume(job_id)

    @app.get("/api/jobs/{job_id}/manifest")
    def manifest(job_id: str):
        path = manager.job_dir(job_id) / "manifest.json"
        if not path.exists():
            raise HTTPException(404, "manifest is written when the job ends or checkpoints")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/records")
    def records(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=200)):
        rows, more = _read_lines(manager.job_dir(job_id) / "data.jsonl", offset, limit)
        return {"offset": offset, "limit": limit, "has_more": more, "records": rows}

    @app.get("/api/jobs/{job_id}/rejections")
    def rejections(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
        rows, more = _read_lines(manager.job_dir(job_id) / "rejections.jsonl", offset, limit)
        return {"offset": offset, "limit": limit, "has_more": more, "rejections": rows}

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        path = manager.job_dir(job_id) / "data.jsonl"
        if not path.exists():
            raise HTTPException(404, "no data yet")
        return FileResponse(path, media_type="application/x-ndjson", filename=f"{job_id}.jsonl")

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str):
        manager.get(job_id)   # 404 early

        async def stream():
            last = None
            while True:
                state = manager.get(job_id)
                payload = json.dumps(state)
                if payload != last:
                    last = payload
                    yield f"event: progress\ndata: {payload}\n\n"
                if state.get("status") in TERMINAL:
                    yield "event: end\ndata: {}\n\n"
                    return
                await asyncio.sleep(1.0)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app
