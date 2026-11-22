"""Tests for uploads: validation and a job from an uploaded file (offline)."""

from __future__ import annotations

import io
import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from dataset_factory.api import create_app
from dataset_factory import synthetic
from dataset_factory.api import uploads
from dataset_factory.tokens import TokenizerUnavailable, load_token_counter

try:
    load_token_counter("EleutherAI/pythia-160m")
except TokenizerUnavailable:
    pytest.skip("Pythia tokenizer not cached", allow_module_level=True)

PARA = ("The committee reviewed the annual budget and said that the proposal, which "
        "had been discussed for several months, would be put to a vote on Friday. ")


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(output_dir=tmp_path)) as c:
        yield c


def _post(client, name, data, **form):
    return client.post("/api/uploads", files={"file": (name, data)}, data=form)


def _wait(client, job_id, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        state = client.get(f"/api/jobs/{job_id}").json()["state"]
        if state["status"] not in ("pending", "running"):
            return state
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_bad_type_missing_field_and_empty(client, tmp_path):
    assert _post(client, "x.exe", b"abc").status_code == 400
    assert _post(client, "x.jsonl", b'{"body": "hi"}\n').status_code == 400       # no 'text' field
    assert _post(client, "x.jsonl", b"not json\n").status_code == 400
    assert _post(client, "x.txt", b"   \n\n  ").status_code == 400
    assert _post(client, "x.csv", b"a,b\n1,2\n").status_code == 400
    assert _post(client, "x.txt", b"\xff\xfe\x00bad").status_code == 400
    # failed uploads leave nothing behind
    assert not list((tmp_path / "uploads").iterdir())


def test_too_big(tmp_path):
    with pytest.raises(uploads.UploadError, match="larger"):
        uploads.save_upload(tmp_path, "x.txt", io.BytesIO(b"a" * 2048), max_bytes=1000)
    assert not list((tmp_path / "uploads").iterdir())


def test_unknown_and_malicious_upload_id(client):
    body = {"domain": "general", "max_documents": 5, "upload_id": "up-0123456789abcdef"}
    assert client.post("/api/jobs", json=body).status_code == 404
    body["upload_id"] = "../../etc/passwd"
    assert client.post("/api/jobs", json=body).status_code == 404


def test_formats_convert(client):
    meta = _post(client, "a.jsonl", b'{"id":"k1","body":"hello world"}\n{"id":"k2","body":""}\n',
                 text_field="body").json()
    assert meta["documents"] == 1
    assert _post(client, "a.csv", b"title,txt\nx,first doc\ny,second doc\n", text_field="txt").json()["documents"] == 2
    assert _post(client, "a.txt", b"one\n\ntwo\r\n\r\nthree").json()["documents"] == 3
    assert _post(client, "a.txt", b"one\n\ntwo", txt_split="file").json()["documents"] == 1
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    buf = io.BytesIO()
    pq.write_table(pa.table({"text": ["p1", "p2"]}), buf)
    assert _post(client, "a.parquet", buf.getvalue()).json()["documents"] == 2


def test_job_from_upload(client, tmp_path):
    rng = random.Random(1)
    rows = [{"id": f"d{i}", "text": synthetic.general(rng)} for i in range(60)]
    data = "\n".join(json.dumps(r) for r in rows).encode()
    meta = _post(client, "mine.jsonl", data).json()
    assert meta["documents"] == 60 and (tmp_path / "uploads" / meta["upload_id"] / "mine.jsonl").is_file()
    assert "upload" in client.get("/api/meta").json()
    r = client.post("/api/jobs", json={"domain": "general", "max_candidates": 60, "workers": 0,
                                       "upload_id": meta["upload_id"], "input_license": "cc-by-4.0"})
    assert r.status_code == 202, r.text
    state = _wait(client, r.json()["job_id"])
    assert state["status"] in ("completed", "exhausted")
    assert state["candidates"] == 60 and state["accepted"] >= 30
    assert set(state["by_source"]) == {"file:mine.jsonl"} and "cc-by-4.0" in state["licenses"]
    # the uploads folder is not listed as a job
    assert all(j["job_id"] != "uploads" for j in client.get("/api/jobs").json())
