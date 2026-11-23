"""Build the evaluation sets once, so every model is scored on identical text.

- ``wikitext.jsonl``: WikiText-103 test split, one article per line
  (general-language perplexity).
- ``medical.jsonl``: PubMed abstracts from two MedRAG files that no curation
  source reads (one from the 1980s, one from the 2020s). Only a length rule is
  applied: filtering them with our own validators would favour the curated
  model.
- ``pubmedqa_test.jsonl`` / ``pubmedqa_train.jsonl``: PubMedQA ``pqa_labeled``
  (the only 1,000 expert-labelled questions) split 500 / 500, stratified by
  answer, with a fixed seed. Curation reads ``pqa_unlabeled``, which shares no
  question with ``pqa_labeled``; training corpora are additionally checked for
  13-gram overlap with all of these files.
- ``code.jsonl`` / ``legal.jsonl`` (``build_domains``): CodeSearchNet test split and four
  Caselaw Access Project files that the caselaw source excludes; dev versions from the
  CodeSearchNet validation split and four other excluded CAP files.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

MEDICAL_FILES = ["chunk/pubmed23n0100.jsonl", "chunk/pubmed23n1100.jsonl"]
MEDICAL_PER_FILE = 1000
MEDICAL_MIN_WORDS = 100
MEDICAL_POOL = 6000           # read this many rows per file, then sample
PUBMEDQA_TEST = 500

FILES = ("wikitext", "medical", "pubmedqa_test", "pubmedqa_train")
DEV_MEDICAL_FILES = ["chunk/pubmed23n0060.jsonl", "chunk/pubmed23n1080.jsonl"]   # read by no curation source

# Code and legal (2026-11-19). Curation reads the CodeSearchNet *train* split, so the test set
# comes from its test split and the dev set from its validation split. The caselaw source reads
# every CAP file except these eight (see [sources.caselaw] data_files in backend/config/default.toml).
# The files are grouped by reporter, so four spread-out files per split.
DOMAIN_SETS = ("code", "legal")
CODE_DATASET = "code-search-net/code_search_net"
CODE_LANGS = ("python", "java", "javascript", "go")
CODE_PER_LANG = 500
CODE_MIN_CHARS = 100          # the code domain's global min_chars
CODE_MAX_CHARS = 20000
CODE_SPLITS = {"test": "test", "dev": "validation"}
LEGAL_DATASET = "common-pile/caselaw_access_project"
LEGAL_FILES = {"test": ["cap_00030.jsonl.gz", "cap_00075.jsonl.gz", "cap_00120.jsonl.gz", "cap_00165.jsonl.gz"],
               "dev": ["cap_00015.jsonl.gz", "cap_00060.jsonl.gz", "cap_00105.jsonl.gz", "cap_00150.jsonl.gz"]}
LEGAL_PER_FILE = 150
LEGAL_POOL = 1500             # read this many rows per file, then sample
LEGAL_MIN_WORDS = 100
LEGAL_MAX_CHARS = 40000       # bounds evaluation time; drops the longest opinions


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_HEADING = re.compile(r"^ = [^=].* = $")


def split_wikitext(lines: Iterable[str]) -> list[dict[str, str]]:
    """WikiText rows are lines; an article starts at a level-1 heading ' = Title = '."""
    articles: list[dict[str, str]] = []
    title, buf = None, []
    for line in lines:
        if _HEADING.match(line.rstrip("\n")):
            if title is not None:
                articles.append({"title": title, "text": "".join(buf).strip()})
            title, buf = line.strip(" =\n"), [line]
        elif title is not None:
            buf.append(line)
    if title is not None:
        articles.append({"title": title, "text": "".join(buf).strip()})
    return articles


def pubmedqa_context(row: dict[str, Any]) -> str:
    """Same layout as the curation source builder ("LABEL: text" per line)."""
    ctx = row["context"]
    return "\n".join(f"{lab}: {txt}" for lab, txt in zip(ctx["labels"], ctx["contexts"]))


def split_pubmedqa(rows: list[dict[str, Any]], n_test: int, seed: int):
    """Stratified by answer: test keeps the labelled set's yes/no/maybe mix."""
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for r in sorted(rows, key=lambda r: int(r["pubid"])):
        by_label.setdefault(r["label"], []).append(r)
    test, train = [], []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        k = round(len(group) * n_test / len(rows))
        test += group[:k]
        train += group[k:]
    # rounding can leave test one short or long; move the difference
    while len(test) > n_test:
        train.append(test.pop())
    while len(test) < n_test:
        test.append(train.pop())
    rng.shuffle(test)
    rng.shuffle(train)
    return test, train


def code_with_doc(row: dict[str, Any]) -> str:
    """Same text as the backend's ``code_with_doc`` builder (sources.py); copied because this
    package does not import the backend."""
    code = row["whole_func_string"]
    doc = row.get("func_documentation_string") or ""
    if doc and doc[:40] not in code:
        if row.get("language") in ("java", "javascript", "go", "php"):
            return "/**\n * " + doc.replace("\n", "\n * ") + "\n */\n" + code
        return "# " + doc.replace("\n", "\n# ") + "\n" + code
    return code


def keep_code(text: str) -> bool:
    return CODE_MIN_CHARS <= len(text) <= CODE_MAX_CHARS


def keep_legal(text: str) -> bool:
    return len(text) <= LEGAL_MAX_CHARS and len(text.split()) >= LEGAL_MIN_WORDS


def _sample_code(split: str, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset
    out: list[dict[str, Any]] = []
    for lang in CODE_LANGS:
        ds = load_dataset(CODE_DATASET, lang, split=CODE_SPLITS[split], streaming=True)
        pool = []
        for r in ds:
            t = code_with_doc(r)
            if keep_code(t):
                pool.append({"language": lang, "repo": r["repository_name"], "url": r["func_code_url"],
                             "text": t})
        out += rng.sample(pool, min(CODE_PER_LANG, len(pool)))
    return out


def _sample_legal(split: str, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset
    out: list[dict[str, Any]] = []
    for fname in LEGAL_FILES[split]:
        ds = load_dataset(LEGAL_DATASET, data_files=[fname], split="train", streaming=True)
        pool = []
        for i, r in enumerate(ds):
            if i >= LEGAL_POOL:
                break
            if keep_legal(r["text"]):
                pool.append({"id": r["id"], "file": fname, "text": r["text"]})
        out += rng.sample(pool, min(LEGAL_PER_FILE, len(pool)))
    return out


def _update_manifest(out_dir: Path, files: dict[str, Any], **extra) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}
    man["files"].update(files)
    man.update(extra)
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def _train_texts(train_files: list[Path], log=print):
    for f in train_files:
        log(f"  checking against {f}")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)["text"]


def build_domains(test_dir: Path, dev_dir: Path, train_files: list[Path], domains=DOMAIN_SETS,
                  seed: int = 0, log=print) -> dict[str, Any]:
    """Held-out perplexity sets for code and legal: ``<domain>.jsonl`` in ``test_dir`` and in
    ``dev_dir``. Text no curation source reads, with a length rule only (never our validators).
    Dev documents overlapping ``train_files`` are dropped (as in ``build_dev``); test documents
    are kept and the overlap only counted: the corpus builder drops the training documents."""
    sampler = {"code": _sample_code, "legal": _sample_legal}
    src = {"code": lambda sp: f"{CODE_DATASET} {','.join(CODE_LANGS)} split {CODE_SPLITS[sp]}",
           "legal": lambda sp: f"{LEGAL_DATASET} " + ", ".join(LEGAL_FILES[sp])}
    rule = {"code": {"min_chars": CODE_MIN_CHARS, "max_chars": CODE_MAX_CHARS, "per_language": CODE_PER_LANG},
            "legal": {"min_words": LEGAL_MIN_WORDS, "max_chars": LEGAL_MAX_CHARS, "per_file": LEGAL_PER_FILE,
                      "pool_per_file": LEGAL_POOL}}
    sets: dict[tuple[str, str], list] = {}
    for d in domains:
        for k, split in enumerate(("test", "dev")):
            log(f"{d} {split} ...")
            sets[(d, split)] = sampler[d](split, random.Random(seed + 10 * (k + 1)))
    keys = list(sets)
    flat = [(key, i) for key in keys for i in range(len(sets[key]))]
    hit = contaminated([sets[key][i]["text"] for key, i in flat], _train_texts(train_files, log)) \
        if train_files else {}
    bad: dict[tuple[str, str], set[int]] = {}
    for j in hit:
        key, i = flat[j]
        bad.setdefault(key, set()).add(i)
    report: dict[str, Any] = {}
    for (d, split), rows in sets.items():
        out_dir = test_dir if split == "test" else dev_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        over = bad.get((d, split), set())
        keep = rows if split == "test" else [r for i, r in enumerate(rows) if i not in over]
        write_jsonl(out_dir / f"{d}.jsonl", keep)
        entry = {"source": src[d](split), "documents": len(keep), **rule[d], "seed": seed,
                 "checked_against": [str(f) for f in train_files],
                 ("overlap_with_jobs" if split == "test" else "dropped_overlap"): len(over),
                 "sha256": sha256(out_dir / f"{d}.jsonl")}
        _update_manifest(out_dir, {d: entry})
        report[f"{d}/{split}"] = {k: v for k, v in entry.items() if k not in ("sha256", "checked_against")}
    return report


def _sample_medrag(files: list[str], rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset
    med: list[dict[str, Any]] = []
    for fname in files:
        ds = load_dataset("MedRAG/pubmed", data_files=[fname], split="train", streaming=True)
        pool = []
        for i, r in enumerate(ds):
            if i >= MEDICAL_POOL:
                break
            if len(r["contents"].split()) >= MEDICAL_MIN_WORDS:
                pool.append({"pmid": str(r["PMID"]), "file": fname, "text": r["contents"]})
        med += rng.sample(pool, min(MEDICAL_PER_FILE, len(pool)))
    return med


def build(out_dir: Path, seed: int = 0, log=print) -> dict[str, Any]:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"seed": seed, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "files": {}}

    log("wikitext-103 test ...")
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    articles = split_wikitext(wt["text"])
    n = write_jsonl(out_dir / "wikitext.jsonl", articles)
    manifest["files"]["wikitext"] = {"source": "Salesforce/wikitext wikitext-103-raw-v1 test",
                                     "documents": n}

    log("held-out PubMed abstracts ...")
    med = _sample_medrag(MEDICAL_FILES, random.Random(seed))
    n = write_jsonl(out_dir / "medical.jsonl", med)
    manifest["files"]["medical"] = {"source": "MedRAG/pubmed " + ", ".join(MEDICAL_FILES),
                                    "documents": n, "min_words": MEDICAL_MIN_WORDS}

    log("PubMedQA pqa_labeled ...")
    pq = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = [{"pubid": str(r["pubid"]), "question": r["question"], "context": pubmedqa_context(r),
             "long_answer": r["long_answer"], "label": r["final_decision"]} for r in pq]
    test, train = split_pubmedqa(rows, PUBMEDQA_TEST, seed)
    for name, part in (("pubmedqa_test", test), ("pubmedqa_train", train)):
        write_jsonl(out_dir / f"{name}.jsonl", part)
        counts = Counter(r["label"] for r in part)
        top, top_n = counts.most_common(1)[0]
        manifest["files"][name] = {"source": "qiaojin/PubMedQA pqa_labeled", "documents": len(part),
                                   "labels": dict(counts), "majority_label": top,
                                   "majority_accuracy": round(top_n / len(part), 4)}

    for name in FILES:
        manifest["files"][name]["sha256"] = sha256(out_dir / f"{name}.jsonl")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def contaminated(docs: list[str], train_texts: Iterable[str], n: int = 13,
                 min_shared: int = 8) -> dict[int, int]:
    """Indices of ``docs`` that share >= min(min_shared, half the n-grams of either text)
    distinct n-grams with any single training text -> the largest such count."""
    from .ngrams import ngram_hashes
    import numpy as np
    per = [np.unique(ngram_hashes(d, n)) for d in docs]
    H = np.concatenate(per) if per else np.empty(0, dtype=np.uint64)
    lab = np.concatenate([np.full(len(h), i) for i, h in enumerate(per)]) if per else np.empty(0, int)
    order = np.argsort(H, kind="stable")
    H, lab = H[order], lab[order]
    sizes = np.asarray([len(h) for h in per])
    out: dict[int, int] = {}
    for t in train_texts:
        h = np.unique(ngram_hashes(t, n))
        if not len(h):
            continue
        lo, hi = np.searchsorted(H, h, "left"), np.searchsorted(H, h, "right")
        m = hi > lo
        if not m.any():
            continue
        lo, hi = lo[m], hi[m]
        span = hi - lo                      # several dev docs can share one n-gram
        idx = np.repeat(lo, span) + np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span)
        ids, cnt = np.unique(lab[idx], return_counts=True)
        for i, c in zip(ids.tolist(), cnt.tolist()):
            need = max(1, min(min_shared, (len(h) + 1) // 2, (int(sizes[i]) + 1) // 2))
            if c >= need:
                out[i] = max(out.get(i, 0), c)
    return out


def build_dev(out_dir: Path, eval_dir: Path, train_files: list[Path], seed: int = 0, log=print) -> dict[str, Any]:
    """Development sets for choosing a curation recipe, so the test sets are used only once:
    ``wikitext.jsonl`` (WikiText-103 *validation*), ``medical.jsonl`` (two other MedRAG PubMed
    files), ``clinical.jsonl`` (the ``pubmedqa_train`` contexts). Any dev document overlapping a
    document of ``train_files`` (job ``data.jsonl`` files that training corpora draw from) is
    dropped from the dev set, since those corpora were not decontaminated against it."""
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log("wikitext-103 validation ...")
    sets = {"wikitext": split_wikitext(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                                    split="validation")["text"]),
            "medical": None, "clinical": None}
    log("dev PubMed abstracts ...")
    sets["medical"] = _sample_medrag(DEV_MEDICAL_FILES, random.Random(seed + 1))
    sets["clinical"] = [{"pubid": r["pubid"], "text": r["context"]}
                        for r in read_jsonl(eval_dir / "pubmedqa_train.jsonl")]
    names = list(sets)
    flat = [(k, i) for k in names for i in range(len(sets[k]))]
    docs = [sets[k][i]["text"] for k, i in flat]

    t0 = time.time()
    hit = contaminated(docs, _train_texts(train_files, log))
    log(f"overlap check {time.time() - t0:.0f}s: {len(hit)} dev documents overlap training data")
    files: dict[str, Any] = {}
    src = {"wikitext": "Salesforce/wikitext wikitext-103-raw-v1 validation",
           "medical": "MedRAG/pubmed " + ", ".join(DEV_MEDICAL_FILES),
           "clinical": "pubmedqa_train contexts (" + str(eval_dir) + ")"}
    base = 0
    for k in names:
        n = len(sets[k])
        keep = [r for i, r in enumerate(sets[k]) if base + i not in hit]
        dropped = n - len(keep)
        base += n
        write_jsonl(out_dir / f"{k}.jsonl", keep)
        files[k] = {"source": src[k], "documents": len(keep), "dropped_overlap": dropped,
                    "sha256": sha256(out_dir / f"{k}.jsonl")}
    # merged into an existing manifest, so code/legal entries (build_domains) are kept
    return _update_manifest(out_dir, files, seed=seed, dev=True, test_eval_dir=str(eval_dir),
                            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            checked_against=[str(f) for f in train_files])


def eval_texts(eval_dir: Path) -> dict[str, list[str]]:
    """Every text an evaluation can show a model, per set (for decontamination)."""
    out = {"wikitext": [r["text"] for r in read_jsonl(eval_dir / "wikitext.jsonl")],
           "medical": [r["text"] for r in read_jsonl(eval_dir / "medical.jsonl")]}
    for name in ("pubmedqa_test", "pubmedqa_train"):
        out[name] = [r["context"] + "\n" + r["question"] + "\n" + r["long_answer"]
                     for r in read_jsonl(eval_dir / f"{name}.jsonl")]
    for name in DOMAIN_SETS:                      # code / legal, once built
        if (eval_dir / f"{name}.jsonl").exists():
            out[name] = [r["text"] for r in read_jsonl(eval_dir / f"{name}.jsonl")]
    return out
