"""Upload cleanup: list, delete, in-use protection, purge of old uploads (offline)."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from dataset_factory.api import create_app, uploads

DOC = b'{"id": "a", "text": "Some text that is long enough to be a document."}\n'


def _upload(client, name="mine.jsonl"):
    r = client.post("/api/uploads", files={"file": (name, io.BytesIO(DOC))})
    assert r.status_code == 201, r.text
    return r.json()["upload_id"]


def _fake_job(root, job_id, upload_id, status):
    folder = root / job_id
    folder.mkdir()
    manifest = {"spec": {"input_file": str(root / "uploads" / upload_id / "mine.jsonl")},
                "state": {"job_id": job_id, "status": status}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _age(root, upload_id, days):
    meta_path = root / "uploads" / upload_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(output_dir=tmp_path)) as c:
        yield c


def test_list_and_delete(client, tmp_path):
    up = _upload(client)
    listed = client.get("/api/uploads").json()
    assert [u["upload_id"] for u in listed] == [up] and listed[0]["in_use"] is False
    assert client.delete(f"/api/uploads/{up}").status_code == 204
    assert not (tmp_path / "uploads" / up).exists()
    assert client.get("/api/uploads").json() == []
    assert client.delete(f"/api/uploads/{up}").status_code == 404
    assert client.delete("/api/uploads/up-../../x").status_code in (404, 405)
    assert client.delete("/api/uploads/not-an-id").status_code == 404


def test_upload_used_by_unfinished_job_is_kept(client, tmp_path):
    up = _upload(client)
    _fake_job(tmp_path, "job-stopped", up, "stopped")
    assert client.get("/api/uploads").json()[0]["in_use"] is True
    assert client.delete(f"/api/uploads/{up}").status_code == 409
    assert (tmp_path / "uploads" / up).is_dir()
    # once that job is finished the upload may go
    _fake_job(tmp_path, "job-done", up, "completed")
    (tmp_path / "job-stopped" / "manifest.json").write_text(
        json.dumps({"spec": {}, "state": {"job_id": "job-stopped", "status": "completed"}}), encoding="utf-8")
    assert client.delete(f"/api/uploads/{up}").status_code == 204


def test_purge_old_uploads_keeps_recent_and_in_use(tmp_path):
    with TestClient(create_app(output_dir=tmp_path)) as c:
        old, recent, busy = _upload(c), _upload(c), _upload(c)
    for up, days in ((old, 30), (recent, 1), (busy, 30)):
        _age(tmp_path, up, days)
    _fake_job(tmp_path, "job-running", busy, "interrupted")
    with TestClient(create_app(output_dir=tmp_path, upload_days=14)) as c:
        ids = {u["upload_id"] for u in c.get("/api/uploads").json()}
    assert ids == {recent, busy}
    with TestClient(create_app(output_dir=tmp_path, upload_days=0)):     # 0 = keep forever
        pass
    assert {u["upload_id"] for u in uploads.list_uploads(tmp_path)} == {recent, busy}
