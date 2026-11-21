"""Tests for Module 8: the FastAPI backend (offline, synthetic source)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from dataset_factory.api import create_app
from dataset_factory.tokens import TokenizerUnavailable, load_token_counter

try:
    load_token_counter("EleutherAI/pythia-160m")
except TokenizerUnavailable:
    pytest.skip("Pythia tokenizer not cached", allow_module_level=True)


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(output_dir=tmp_path, max_running=2)) as c:
        yield c


def _wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get(f"/api/jobs/{job_id}").json()["state"]
        if state["status"] not in ("pending", "running"):
            return state
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def _start(client, **kw):
    body = {"domain": "general", "synthetic": True, "max_documents": 40, "workers": 0, **kw}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert "medical" in meta["domains"] and "curate" in meta["modes"]


def test_job_lifecycle_and_data(client):
    job_id = _start(client)
    state = _wait(client, job_id)
    assert state["status"] == "completed" and state["accepted"] >= 40
    assert any(j["job_id"] == job_id for j in client.get("/api/jobs").json())
    page = client.get(f"/api/jobs/{job_id}/records?limit=10").json()
    assert len(page["records"]) == 10 and page["has_more"]
    assert client.get(f"/api/jobs/{job_id}/records?offset=1000").json()["records"] == []
    assert "state" in client.get(f"/api/jobs/{job_id}/manifest").json()
    assert client.get(f"/api/jobs/{job_id}/rejections").status_code == 200
    d = client.get(f"/api/jobs/{job_id}/download")
    assert d.status_code == 200 and d.text.count("\n") >= 40


def test_events_stream_ends(client):
    job_id = _start(client)
    with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        body = "".join(r.iter_text())
    assert "event: progress" in body and "event: end" in body


def test_validation_and_errors(client):
    assert client.post("/api/jobs", json={"domain": "nope", "max_documents": 5}).status_code == 400
    assert client.post("/api/jobs", json={"domain": "general"}).status_code == 400   # no target
    assert client.post("/api/jobs", json={"domain": "general", "max_documents": 0}).status_code == 422
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fx").status_code == 404


def test_stop_and_resume(client):
    job_id = _start(client, max_documents=None, max_candidates=2_000_000)
    time.sleep(1.5)
    assert client.post(f"/api/jobs/{job_id}/stop").status_code == 202
    state = _wait(client, job_id)
    assert state["status"] == "stopped"
    assert client.post(f"/api/jobs/{job_id}/stop").status_code == 409   # not running now
    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 202
    client.post(f"/api/jobs/{job_id}/stop")
    assert _wait(client, job_id)["status"] in ("stopped", "completed", "exhausted")


def test_max_running(tmp_path):
    with TestClient(create_app(output_dir=tmp_path, max_running=1)) as c:
        first = _start(c, max_documents=None, max_candidates=2_000_000)
        r = c.post("/api/jobs", json={"domain": "general", "synthetic": True, "max_documents": 5, "workers": 0})
        assert r.status_code == 409
        c.post(f"/api/jobs/{first}/stop")
        _wait(c, first)
