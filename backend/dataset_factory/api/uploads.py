"""User uploads: stored under ``<runs>/uploads/<id>/`` and converted to JSON lines.

Only the upload id ever reaches a job; the API never takes a path. Supported
formats: ``.jsonl`` (one object per line), ``.txt``, ``.csv``, ``.parquet``.
Every format is converted once, at upload time, to ``<name>.jsonl`` with rows
``{"id", "text"}``, which the existing ``jsonl`` source reads.
"""

from __future__ import annotations

import csv
import json
import re
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

UPLOAD_ID_RE = re.compile(r"^up-[0-9a-f]{16}$")
FORMATS = (".jsonl", ".txt", ".csv", ".parquet")
MAX_BYTES = 200 * 1024 * 1024
MAX_DOCUMENTS = 2_000_000


class UploadError(ValueError):
    """Bad upload (type, size, content); shown to the user as HTTP 400."""


class UploadNotFound(KeyError):
    pass


def upload_root(output_dir: Path) -> Path:
    return output_dir / "uploads"


def upload_path(output_dir: Path, upload_id: str) -> Path:
    """the converted JSON-lines file of an upload; UploadNotFound for any unknown or malformed id."""
    if not UPLOAD_ID_RE.match(upload_id):
        raise UploadNotFound(upload_id)
    folder = upload_root(output_dir) / upload_id
    try:
        name = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["data_file"]
    except (OSError, ValueError, KeyError):
        raise UploadNotFound(upload_id) from None
    path = folder / name
    if Path(name).name != name or not path.is_file():
        raise UploadNotFound(upload_id)
    return path


def _jsonl_rows(raw: Path, field: str) -> Iterator[Any]:
    with raw.open(encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UploadError(f"line {number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise UploadError(f"line {number} is not a JSON object")
            if field not in row:
                raise UploadError(f"line {number} has no field '{field}'")
            yield row[field], row.get("id")


def _csv_rows(raw: Path, field: str) -> Iterator[Any]:
    csv.field_size_limit(50 * 1024 * 1024)
    with raw.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or field not in reader.fieldnames:
            raise UploadError(f"CSV has no column '{field}' (columns: {', '.join(reader.fieldnames or [])})")
        for row in reader:
            yield row[field], row.get("id")


def _parquet_rows(raw: Path, field: str) -> Iterator[Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise UploadError("parquet needs pyarrow (pip install pyarrow)") from exc
    try:
        pf = pq.ParquetFile(raw)
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"not a readable parquet file: {exc}") from exc
    names = pf.schema_arrow.names
    if field not in names:
        raise UploadError(f"parquet has no column '{field}' (columns: {', '.join(names)})")
    for batch in pf.iter_batches(columns=[field]):
        for value in batch.column(0).to_pylist():
            yield value, None


def _txt_rows(raw: Path, split: str) -> Iterator[Any]:
    text = raw.read_text(encoding="utf-8-sig")
    if split == "file":
        yield text, None
        return
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        yield block, None


def convert(raw: Path, suffix: str, dest: Path, field: str = "text", txt_split: str = "blocks") -> int:
    """Write ``dest`` (JSON lines of ``{"id", "text"}``) from ``raw``; returns the document count."""
    if suffix == ".jsonl":
        rows = _jsonl_rows(raw, field)
    elif suffix == ".csv":
        rows = _csv_rows(raw, field)
    elif suffix == ".parquet":
        rows = _parquet_rows(raw, field)
    elif suffix == ".txt":
        if txt_split not in ("blocks", "file"):
            raise UploadError("txt_split must be 'blocks' or 'file'")
        rows = _txt_rows(raw, txt_split)
    else:
        raise UploadError(f"unsupported file type '{suffix}'. Use: {', '.join(FORMATS)}")
    count = 0
    with dest.open("w", encoding="utf-8", newline="\n") as out:
        for value, given_id in rows:
            if not isinstance(value, str) or not value.strip():
                continue
            count += 1
            if count > MAX_DOCUMENTS:
                raise UploadError(f"more than {MAX_DOCUMENTS:,} documents")
            doc_id = str(given_id) if given_id not in (None, "") else str(count)
            out.write(json.dumps({"id": doc_id, "text": value}, ensure_ascii=False) + "\n")
    if count == 0:
        raise UploadError(f"no non-empty text found (field '{field}')")
    return count


def save_upload(output_dir: Path, filename: str, stream: BinaryIO, field: str = "text",
                txt_split: str = "blocks", max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    """Store and convert one upload; raises UploadError (nothing is left behind)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in FORMATS:
        raise UploadError(f"unsupported file type '{suffix or filename}'. Use: {', '.join(FORMATS)}")
    field = (field or "text").strip() or "text"
    upload_id = f"up-{secrets.token_hex(8)}"
    folder = upload_root(output_dir) / upload_id
    folder.mkdir(parents=True)
    try:
        raw = folder / f"raw{suffix}"
        size = 0
        with raw.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(f"file is larger than {max_bytes // (1024 * 1024)} MB")
                out.write(chunk)
        try:
            # named after the upload so the dashboard shows it as source "file:<name>.jsonl"
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem).strip("._")[:60] or "upload"
            data_file = f"{stem}.jsonl"
            documents = convert(raw, suffix, folder / data_file, field, txt_split)
        except UnicodeDecodeError as exc:
            raise UploadError("file is not valid UTF-8 text") from exc
        raw.unlink()
        meta = {
            "upload_id": upload_id, "filename": Path(filename).name[:200], "format": suffix[1:],
            "text_field": field, "data_file": data_file, "bytes": size, "documents": documents,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def list_uploads(output_dir: Path) -> list[dict[str, Any]]:
    """meta of every stored upload, newest first (unreadable folders are skipped)."""
    root = upload_root(output_dir)
    rows = []
    for folder in root.iterdir() if root.is_dir() else []:
        if not UPLOAD_ID_RE.match(folder.name):
            continue
        try:
            rows.append(json.loads((folder / "meta.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    rows.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return rows


def delete_upload(output_dir: Path, upload_id: str) -> None:
    """remove an upload folder; UploadNotFound for an unknown or malformed id."""
    upload_path(output_dir, upload_id)          # validates the id and that the upload exists
    shutil.rmtree(upload_root(output_dir) / upload_id)


def purge_old_uploads(output_dir: Path, max_age_days: int, keep: set[str] = frozenset()) -> list[str]:
    """delete uploads older than ``max_age_days`` that are not in ``keep``; returns the removed ids."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    removed = []
    for meta in list_uploads(output_dir):
        upload_id = meta.get("upload_id", "")
        try:
            created = datetime.fromisoformat(meta["created_at"])
        except (KeyError, ValueError):
            continue
        if upload_id in keep or created >= cutoff:
            continue
        try:
            delete_upload(output_dir, upload_id)
            removed.append(upload_id)
        except (UploadNotFound, OSError):
            continue
    return removed
