"""Live job feedback for the dashboard: start-up report, live elapsed time, "stopping", newest job first."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dataset_factory.api import create_app
from dataset_factory.api.manager import JobManager
from dataset_factory.api.names import JobNames
from dataset_factory.tokens import TokenizerUnavailable, load_token_counter
from dataset_factory.config import load_config
from dataset_factory.orchestrator import Job, JobSpec


def _entry(status: str, started_at: str = "", stop: bool = False):
    return SimpleNamespace(snapshot={"job_id": f"j-{status}-{started_at}", "status": status, "started_at": started_at},
                           error=None, job=SimpleNamespace(stop_requested=stop), thread=None)


def test_stop_requested_shows_as_stopping(tmp_path):
    manager = JobManager(output_dir=tmp_path)
    manager._entries = {"a": _entry("running", "2026-01-01", stop=True), "b": _entry("running", "2026-01-01"),
                        "c": _entry("stopped", "2026-01-01", stop=True)}
    assert manager.get("a")["status"] == "stopping"
    assert manager.get("b")["status"] == "running"
    assert manager.get("c")["status"] == "stopped"     # a finished stop is reported as such


def test_new_job_without_start_time_is_listed_first(tmp_path):
    manager = JobManager(output_dir=tmp_path)
    manager._entries = {"old": _entry("completed", "2026-01-01T00:00:00Z"),
                        "new": _entry("pending", ""),
                        "mid": _entry("completed", "2026-06-01T00:00:00Z")}
    order = [row["status"] + row["started_at"] for row in manager.list()]
    assert order == ["pending", "completed2026-06-01T00:00:00Z", "completed2026-01-01T00:00:00Z"]


def test_job_reports_running_at_once_and_elapsed_grows(tmp_path):
    snaps = []
    spec = JobSpec("general", max_documents=30, synthetic=True, workers=0)
    job = Job(spec, load_config(), output_dir=tmp_path, on_progress=snaps.append, progress_every=0.0)
    job.run()
    first = snaps[0]
    assert first["status"] == "running" and first["started_at"]       # before any document is read
    assert first["candidates"] == 0
    elapsed = [s["elapsed_seconds"] for s in snaps]
    assert elapsed == sorted(elapsed) and snaps[-1]["status"] == "completed"


# -- deleting jobs ----------------------------------------------------------

try:
    load_token_counter("EleutherAI/pythia-160m")
    _HAVE_TOKENIZER = True
except TokenizerUnavailable:
    _HAVE_TOKENIZER = False


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(output_dir=tmp_path, max_running=2)) as c:
        yield c


def _run_job(client, **kw):
    body = {"domain": "general", "synthetic": True, "max_documents": 20, "workers": 0, **kw}
    job_id = client.post("/api/jobs", json=body).json()["job_id"]
    end = time.time() + 60
    while time.time() < end:
        if client.get(f"/api/jobs/{job_id}").json()["state"]["status"] not in ("pending", "running", "stopping"):
            return job_id
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@pytest.mark.skipif(not _HAVE_TOKENIZER, reason="Pythia tokenizer not cached")
def test_delete_one_job_removes_it_and_its_folder(client, tmp_path):
    keep, drop = _run_job(client), _run_job(client)
    assert client.delete(f"/api/jobs/{drop}").status_code == 204
    assert not (tmp_path / drop).exists() and (tmp_path / keep).exists()
    assert client.get(f"/api/jobs/{drop}").status_code == 404
    assert [j["job_id"] for j in client.get("/api/jobs").json()] == [keep]
    assert client.delete(f"/api/jobs/{drop}").status_code == 404
    assert client.delete("/api/jobs/..%2Fsecret").status_code in (404, 405)
    (tmp_path / "uploads").mkdir(exist_ok=True)               # shared folders that are not jobs are never deleted
    (tmp_path / "uploads" / "keep.txt").write_text("x")
    assert client.delete("/api/jobs/uploads").status_code == 404
    assert (tmp_path / "uploads" / "keep.txt").exists()


@pytest.mark.skipif(not _HAVE_TOKENIZER, reason="Pythia tokenizer not cached")
def test_clear_all_needs_confirm_keeps_uploads_and_other_folders(client, tmp_path):
    a, b = _run_job(client), _run_job(client)
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "uploads" / "x.txt").write_text("keep me")
    (tmp_path / "notes").mkdir()
    assert client.delete("/api/jobs").status_code == 400          # no confirm: nothing happens
    assert (tmp_path / a).exists()
    out = client.delete("/api/jobs?confirm=true").json()
    assert sorted(out["deleted"]) == sorted([a, b]) and out["skipped"] == []
    assert client.get("/api/jobs").json() == []
    assert (tmp_path / "uploads" / "x.txt").exists() and (tmp_path / "notes").exists()


@pytest.mark.skipif(not _HAVE_TOKENIZER, reason="Pythia tokenizer not cached")
def test_running_job_cannot_be_deleted_and_survives_clear_all(client):
    # a real long job, so it is still running when we try to delete it
    job_id = client.post("/api/jobs", json={"domain": "general", "synthetic": True,
                                            "max_documents": 2_000_000, "workers": 0}).json()["job_id"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    out = client.delete("/api/jobs?confirm=true").json()
    assert out["deleted"] == [] and out["skipped"] == [job_id]
    client.post(f"/api/jobs/{job_id}/stop")
    end = time.time() + 60
    while time.time() < end and client.get(f"/api/jobs/{job_id}").json()["state"]["status"] != "stopped":
        time.sleep(0.2)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204


# -- friendly names -----------------------------------------------------------


def test_names_count_per_domain_and_mode_and_are_never_reused(tmp_path):
    names = JobNames(tmp_path)
    assert names.assign("a", "legal", "curate") == "Legal 1"
    assert names.assign("b", "legal", "curate") == "Legal 2"
    assert names.assign("c", "legal", "random") == "Legal random 1"
    assert names.assign("d", "code", "curate") == "Code 1"
    assert names.assign("a", "legal", "curate") == "Legal 1"          # asking again keeps the name
    names.forget("b")
    assert names.assign("e", "legal", "curate") == "Legal 3"          # "Legal 2" is not handed out again
    assert JobNames(tmp_path).get("e") == "Legal 3"                    # survives a restart


def test_older_jobs_are_named_oldest_first_and_new_ones_continue(tmp_path):
    names = JobNames(tmp_path)
    rows = [{"job_id": "new", "domain": "legal", "mode": "curate", "started_at": "2026-03-01"},
            {"job_id": "old", "domain": "legal", "mode": "curate", "started_at": "2026-01-01"},
            {"job_id": "mid", "domain": "legal", "mode": "curate", "started_at": "2026-02-01"}]
    names.sync(rows)
    assert {r["job_id"]: r["name"] for r in rows} == {"old": "Legal 1", "mid": "Legal 2", "new": "Legal 3"}
    assert names.assign("fresh", "legal", "curate") == "Legal 4"


def test_unreadable_names_file_does_not_break_anything(tmp_path):
    (tmp_path / "job-names.json").write_text("{not json")
    assert JobNames(tmp_path).assign("x", "news", "curate") == "News 1"


@pytest.mark.skipif(not _HAVE_TOKENIZER, reason="Pythia tokenizer not cached")
def test_api_returns_names_and_delete_forgets_them(client, tmp_path):
    first, second = _run_job(client), _run_job(client)
    names = {j["job_id"]: j["name"] for j in client.get("/api/jobs").json()}
    assert sorted(names.values()) == ["General 1", "General 2"]
    assert client.get(f"/api/jobs/{first}").json()["state"]["name"] == names[first]
    client.delete(f"/api/jobs/{first}")
    third = _run_job(client)
    assert client.get(f"/api/jobs/{third}").json()["state"]["name"] == "General 3"
    assert client.get(f"/api/jobs/{second}").json()["state"]["name"] == names[second]    # unchanged
