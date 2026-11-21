"""Pack curated (or baseline) jobs into one training corpus with an exact token budget.

Rules that make corpora comparable:
- Every corpus holds exactly ``budget`` training tokens, counted with the
  Pythia tokenizer **including** one end-of-text token after each document.
  The last document is cut at the budget.
- A corpus mixes parts (job folders or ``data.jsonl`` files) by token share;
  each part contributes documents in its file order until its share is full.
- Documents sharing a 13-word sequence with any evaluation set are dropped
  (counted per part and per set in ``meta.json``).
- About 1 in ``val_every`` documents (chosen by a hash of the document id, so
  independent of the budget) goes to ``val.bin`` instead, for monitoring.
- Documents are shuffled with a seed before packing, so parts interleave.

Output: ``train.bin`` / ``val.bin`` (uint16 token ids, readable with
``numpy.memmap``) and ``meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .evalsets import eval_texts, sha256
from .ngrams import OverlapIndex
from .tok import DEFAULT_TOKENIZER, encode_many, eos_id, load_tokenizer

DTYPE = np.uint16
BATCH = 512


class CorpusError(ValueError):
    pass


def parse_count(text: str) -> int:
    """'20M' -> 20_000_000, '1.5k' -> 1500, '200000' -> 200000."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmMbB]?)\s*", str(text))
    if not m:
        raise CorpusError(f"not a count: {text!r}")
    mult = {"": 1, "k": 10**3, "m": 10**6, "b": 10**9}[m.group(2).lower()]
    return int(round(float(m.group(1)) * mult))


@dataclass
class Part:
    path: Path          # job folder or .jsonl file
    share: float

    @property
    def data_file(self) -> Path:
        return self.path / "data.jsonl" if self.path.is_dir() else self.path

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"path": str(self.path), "share": self.share,
                                "data_bytes": self.data_file.stat().st_size}
        manifest = self.path / "manifest.json" if self.path.is_dir() else None
        if manifest and manifest.exists():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            spec, state = m.get("spec", {}), m.get("state", {})
            info.update(job_id=state.get("job_id") or self.path.name, domain=spec.get("domain"),
                        mode=spec.get("mode"), status=state.get("status"),
                        config_fingerprint=m.get("config_fingerprint"))
        return info


def parse_part(text: str) -> Part:
    """'runs/medical-x=0.5' or 'runs/medical-x' (share 1)."""
    path, _, share = text.rpartition("=") if "=" in text else (text, "", "1")
    try:
        value = float(share)
    except ValueError as exc:
        raise CorpusError(f"bad share in {text!r}") from exc
    p = Path(path)
    if value <= 0:
        raise CorpusError(f"share must be positive: {text!r}")
    if not (p / "data.jsonl").exists() and not (p.is_file() and p.suffix == ".jsonl"):
        raise CorpusError(f"no data.jsonl at {p}")
    return Part(p, value)


def _is_val(doc_id: str, every: int) -> bool:
    if every <= 0:
        return False
    return int(hashlib.blake2b(doc_id.encode(), digest_size=8).hexdigest(), 16) % every == 0


def _docs(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_corpus(name: str, parts: list[Part], budget: int, out_dir: Path,
                 eval_dir: Path | None, seed: int = 0, val_every: int = 100,
                 allow_short: bool = False, tokenizer: str = DEFAULT_TOKENIZER,
                 log=print) -> dict[str, Any]:
    if not parts:
        raise CorpusError("no parts given")
    if budget <= 0:
        raise CorpusError("budget must be positive")
    t0 = time.time()
    tok = load_tokenizer(tokenizer)
    eos = eos_id(tok)
    if tok.get_vocab_size() > np.iinfo(DTYPE).max:
        raise CorpusError("vocabulary does not fit uint16")

    index = OverlapIndex()
    if eval_dir is not None:
        for set_name, texts in eval_texts(eval_dir).items():
            index.add(set_name, texts)
        log(f"decontamination index: {len(index):,} distinct 13-grams")

    total_share = sum(p.share for p in parts)
    targets = [round(budget * p.share / total_share) for p in parts]
    targets[-1] += budget - sum(targets)

    train_docs: list[tuple[int, np.ndarray]] = []   # (part index, ids incl. EOS)
    val_docs: list[np.ndarray] = []
    part_stats = []
    for pi, (part, target) in enumerate(zip(parts, targets)):
        st = {**part.describe(), "target_tokens": target, "tokens": 0, "documents": 0,
              "read": 0, "val_documents": 0, "val_tokens": 0, "decontaminated": {},
              "by_source": {}, "by_domain": {}, "short_by": 0}
        batch: list[dict[str, Any]] = []

        def flush() -> bool:
            """Tokenize the batch; True once the part's target is reached."""
            ids = encode_many(tok, [d["text"] for d in batch])
            for d, doc_ids in zip(batch, ids):
                arr = np.asarray(doc_ids + [eos], dtype=DTYPE)
                if _is_val(str(d.get("id", "")), val_every):
                    val_docs.append(arr)
                    st["val_documents"] += 1
                    st["val_tokens"] += len(arr)
                    continue
                if st["tokens"] >= target:
                    continue
                train_docs.append((pi, arr))
                st["tokens"] += len(arr)
                st["documents"] += 1
                src = d.get("provenance", {}).get("source", "?")
                st["by_source"][src] = st["by_source"].get(src, 0) + 1
                dom = d.get("domain", "?")
                st["by_domain"][dom] = st["by_domain"].get(dom, 0) + 1
            batch.clear()
            return st["tokens"] >= target

        done = False
        for d in _docs(part.data_file):
            st["read"] += 1
            hits = index.hits(d["text"]) if len(index) else []
            if hits:
                for h in hits:
                    st["decontaminated"][h] = st["decontaminated"].get(h, 0) + 1
                continue
            batch.append(d)
            if len(batch) >= BATCH and flush():
                done = True
                break
        if not done and batch:
            flush()
        if st["tokens"] < target:
            st["short_by"] = target - st["tokens"]
            msg = (f"part {part.path} has only {st['tokens']:,} of its {target:,} tokens "
                   f"after decontamination (read {st['read']:,} documents)")
            if not allow_short:
                raise CorpusError(msg + "; curate more or pass --allow-short")
            log("WARNING: " + msg)
        log(f"part {pi}: {st['documents']:,} docs, {st['tokens']:,} tokens, "
            f"decontaminated {sum(st['decontaminated'].values())}")
        part_stats.append(st)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_docs))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_dir / "train.bin", "wb") as f:
        for i in order:
            arr = train_docs[i][1]
            if written + len(arr) > budget:
                arr = arr[: budget - written]
            arr.tofile(f)
            written += len(arr)
            if written >= budget:
                break
    val_order = rng.permutation(len(val_docs))
    with open(out_dir / "val.bin", "wb") as f:
        for i in val_order:
            val_docs[i].tofile(f)
    val_tokens = int(sum(len(v) for v in val_docs))

    meta = {
        "name": name, "budget": budget, "train_tokens": written, "val_tokens": val_tokens,
        "train_documents": len(train_docs), "val_documents": len(val_docs),
        "short": written < budget, "seed": seed, "val_every": val_every,
        "tokenizer": tokenizer, "eos_id": eos, "dtype": "uint16",
        "decontamination": {"n": index.n, "eval_dir": str(eval_dir) if eval_dir else None,
                            "eval_manifest": _eval_manifest(eval_dir)},
        "parts": part_stats,
        "sha256": {"train.bin": sha256(out_dir / "train.bin"), "val.bin": sha256(out_dir / "val.bin")},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _eval_manifest(eval_dir: Path | None) -> dict[str, str] | None:
    if eval_dir is None or not (eval_dir / "manifest.json").exists():
        return None
    m = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))
    return {k: v.get("sha256", "") for k, v in m.get("files", {}).items()}


def load_tokens(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=DTYPE, mode="r")
