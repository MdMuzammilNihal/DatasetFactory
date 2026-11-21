"""Download training data for the learned quality classifier.

Usage (from the backend folder):

    python scripts/fetch_classifier_data.py          # writes data/classifier/*.jsonl

One file per corpus, one JSON object per line: {"text": "..."}.
Documents are streamed with the Hugging Face ``datasets`` library, so nothing
is downloaded in full. The script is resumable: a corpus whose file already
holds enough documents is skipped.

Any document that also appears in data/samples.json (the calibration samples)
is left out, which keeps that file an independent check of the trained models.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_samples import java_with_doc, pubmedqa, text_field  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND / "data" / "classifier"


def code_with_doc(row):
    """CodeSearchNet keeps the documentation apart from some functions; put it back."""
    code = row["whole_func_string"]
    doc = row.get("func_documentation_string") or ""
    if doc and doc[:40] not in code:
        return "// " + doc.replace("\n", "\n// ") + "\n" + code
    return code


def pubmed_files(*numbers):
    return [f"chunk/pubmed23n{n:04d}.jsonl" for n in numbers]


# name: (how to build the text, count, [load_dataset keyword arguments, ...])
# With several argument sets the count is split evenly between them, which
# spreads the sample across eras or jurisdictions of a corpus.
SPECS = {
    "fineweb": (text_field("text"), 4000, [dict(path="HuggingFaceFW/fineweb", name="sample-10BT")]),
    "wikipedia": (text_field("text"), 2000, [dict(path="wikimedia/wikipedia", name="20231101.en")]),
    "news_cnn": (text_field("article"), 2000, [dict(path="abisee/cnn_dailymail", name="3.0.0")]),
    "news_ccnews": (text_field("text"), 1500, [dict(path="vblagoje/cc_news")]),
    "med_pubmed": (text_field("contents"), 2000, [
        dict(path="MedRAG/pubmed", data_files=pubmed_files(n)) for n in (300, 600, 900, 1100)
    ]),
    "med_pubmedqa": (pubmedqa, 2000, [dict(path="qiaojin/PubMedQA", name="pqa_unlabeled")]),
    "legal_cap": (text_field("text"), 1500, [
        dict(path="common-pile/caselaw_access_project", data_files=[f"cap_{n:05d}.jsonl.gz"])
        for n in (20, 80, 140)
    ]),
    "sci_arxiv_sum": (text_field("article"), 1000, [dict(path="ccdv/arxiv-summarization", name="document")]),
    "sci_arxiv_cp": (text_field("text"), 400, [
        dict(path="common-pile/arxiv_papers", data_files=["00005_arxiv-papers.jsonl.gz"])
    ]),
    "code_py": (text_field("whole_func_string"), 1500, [dict(path="code-search-net/code_search_net", name="python")]),
    "code_java": (java_with_doc, 1500, [dict(path="code-search-net/code_search_net", name="java")]),
    "code_js": (code_with_doc, 500, [dict(path="code-search-net/code_search_net", name="javascript")]),
    "code_go": (code_with_doc, 500, [dict(path="code-search-net/code_search_net", name="go")]),
}


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).digest()


def calibration_digests() -> set[bytes]:
    path = BACKEND / "data" / "samples.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {digest(text) for texts in data.values() for text in texts}


def open_stream(kwargs: dict):
    """Open a streaming dataset; fall back to the Hub's Parquet conversion for
    corpora that are still published as a loading script."""
    try:
        return load_dataset(split="train", streaming=True, **kwargs)
    except Exception as first_error:
        try:
            return load_dataset(
                split="train", streaming=True, revision="refs/convert/parquet", **kwargs
            )
        except Exception:
            raise first_error


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded = calibration_digests()
    for name, (build, n, sources) in SPECS.items():
        path = OUT_DIR / f"{name}.jsonl"
        have = count_lines(path)
        if have >= n * 0.95:
            print(f"{name:<14} already has {have} documents, skipping", flush=True)
            continue
        written = 0
        per_source = -(-n // len(sources))   # ceiling division
        with path.open("w", encoding="utf-8") as fh:
            for kwargs in sources:
                taken = 0
                try:
                    for row in open_stream(kwargs):
                        try:
                            text = build(row)
                        except (KeyError, TypeError):
                            continue
                        if not isinstance(text, str) or not text.strip():
                            continue
                        if digest(text) in excluded:
                            continue
                        fh.write(json.dumps({"text": text}) + "\n")
                        written += 1
                        taken += 1
                        if taken >= per_source or written >= n:
                            break
                except Exception as exc:
                    print(f"  warning: {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        print(f"{name:<14} {written:>5} documents", flush=True)
    print(f"saved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
