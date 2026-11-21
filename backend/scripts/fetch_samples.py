"""Download a few hundred real documents per domain for calibration.

Usage (from the backend folder):

    python scripts/fetch_samples.py            # writes data/samples.json

Uses the public Hugging Face dataset viewer API, so it needs internet but no
extra libraries and no account. The output feeds scripts/calibrate.py.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/"
OUT = Path(__file__).resolve().parent.parent / "data" / "samples.json"


def get(path: str, **query) -> dict:
    url = API + path + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    error: Exception | None = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # network hiccup: wait and retry
            error = exc
            time.sleep(3)
    raise RuntimeError(f"could not fetch {url}: {error}")


def rows(dataset, config, split, n, step=50, start=0, stride=None):
    """Yield up to n raw rows, reading `step` rows every `stride` rows."""
    got, offset = 0, start
    while got < n:
        try:
            batch = get("rows", dataset=dataset, config=config, split=split,
                        offset=offset, length=step)["rows"]
        except RuntimeError as exc:
            print(f"  warning: {exc}", file=sys.stderr)
            return
        if not batch:
            return
        for item in batch:
            yield item["row"]
            got += 1
            if got >= n:
                return
        offset += stride or step


def text_field(field):
    return lambda row: row[field]


def pubmedqa(row):
    context = row["context"]
    return "\n".join(f"{label}: {text}" for label, text in zip(context["labels"], context["contexts"]))


def java_with_doc(row):
    doc = row["func_documentation_string"].replace("\n", "\n * ")
    return f"/**\n * {doc}\n */\n{row['func_code_string']}"


# name: (dataset, config, split, how to build the text, count, step, start, stride)
SPECS = {
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "train", text_field("text"), 300, 100, 0, None),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train", text_field("text"), 200, 100, 0, 20000),
    "news_cnn": ("abisee/cnn_dailymail", "3.0.0", "train", text_field("article"), 300, 100, 0, None),
    "med_pubmed": ("MedRAG/pubmed", "default", "train", text_field("contents"), 300, 50, 300000, 300000),
    "med_pubmedqa": ("qiaojin/PubMedQA", "pqa_unlabeled", "train", pubmedqa, 150, 50, 0, 20000),
    "legal_cap": ("common-pile/caselaw_access_project", "default", "train", text_field("text"), 200, 50, 0, 5000),
    "sci_arxiv_sum": ("ccdv/arxiv-summarization", "document", "train", text_field("article"), 150, 25, 0, 1000),
    "sci_arxiv_cp": ("common-pile/arxiv_papers", "default", "train", text_field("text"), 60, 10, 0, 500),
    "code_py": ("code-search-net/code_search_net", "python", "train", text_field("whole_func_string"), 200, 100, 0, 5000),
    "code_java": ("code-search-net/code_search_net", "java", "train", java_with_doc, 200, 100, 0, 5000),
}


def main() -> None:
    samples: dict[str, list[str]] = {}
    for name, (dataset, config, split, build, n, step, start, stride) in SPECS.items():
        texts = []
        for row in rows(dataset, config, split, n, step, start, stride):
            try:
                text = build(row)
            except (KeyError, TypeError):
                continue
            if isinstance(text, str) and text:
                texts.append(text)
        samples[name] = texts
        mean = sum(map(len, texts)) // max(1, len(texts))
        print(f"{name:<14} {len(texts):>4} documents, {mean:>6} characters on average")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(samples), encoding="utf-8")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
