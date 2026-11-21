"""Evaluate a model on the frozen evaluation sets (Module 10c).

- **Perplexity** on ``wikitext.jsonl``, ``medical.jsonl`` and ``clinical`` (the 500
  PubMedQA test contexts: clinical abstracts curation never read; they are in the
  decontamination index): every document is
  prefixed with EOS and scored token by token; documents longer than the window
  are scored with a sliding window (``stride`` new tokens per window, the rest
  as context). Perplexity = exp(total NLL / total scored tokens), pooled over
  the set. Per-document NLL and token counts are saved for paired comparisons.
- **PubMedQA** on ``pubmedqa_test.jsonl``: prompt "Context ... Question ...
  Answer:", prediction = the answer (" yes" / " no" / " maybe") with the highest
  log-likelihood. Also reported: a *calibrated* prediction that subtracts each
  answer's log-likelihood after a context-free prompt (small models prefer
  " yes" whatever the question), the prediction counts, macro-F1, the
  majority-class baseline and a bootstrap 95% CI. ``shots`` > 0 prepends
  labelled examples from ``pubmedqa_train.jsonl`` (fixed by the seed).
  Per-question log-likelihoods of the three answers are saved (``loglik``), and a
  **bias-free AUROC** of LL(yes) - LL(no) on the yes/no items: a constant preference
  for one word (the yes/no bias that dominates small models) does not change it,
  so it measures only whether the model ranks yes-questions above no-questions.
- ``compare`` gives paired-bootstrap CIs for the difference between two result
  files (same evaluation sets).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .evalsets import read_jsonl
from .tok import DEFAULT_TOKENIZER, eos_id, load_tokenizer

ANSWERS = ("yes", "no", "maybe")
PPL_TASKS = ("wikitext", "medical", "clinical", "code", "legal")
TASKS = ("wikitext", "medical", "clinical", "pubmedqa")   # default; code/legal on request


# ---------------------------------------------------------------- perplexity

def windows_for(n: int, max_len: int, stride: int) -> list[tuple[int, int, int]]:
    """(start, end, first_scored) windows over a sequence of ``n`` tokens.

    Tokens 1..n-1 are each scored exactly once (token 0 is the EOS prefix);
    every window after the first keeps ``max_len - stride`` tokens of context.
    """
    out = []
    scored_to = 1
    while scored_to < n:
        end = min(n, scored_to + (stride if out else max_len - 1))
        start = max(0, end - max_len)
        out.append((start, end, scored_to))
        scored_to = end
    return out


def _nll_batch(model, seqs: list[list[int]], targets_from: list[int], device, dtype):
    """Sum of NLL over positions >= targets_from[i] (as targets) for each sequence."""
    import torch
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    ids, att = ids.to(device), att.to(device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
        logits = model(input_ids=ids, attention_mask=att).logits
    # only the positions that predict scored tokens (saves memory for long prompts)
    lo = min(targets_from) - 1             # logits at position p predict token p+1
    # one row at a time: a full-batch fp32 log-softmax over 50k words (8 x 1024 windows) needs
    # several GB and spilled into shared memory on the 8 GB card (5x slower); same numbers
    tgt = ids[:, lo + 1:]
    out = []
    for i, s in enumerate(seqs):
        logp = torch.log_softmax(logits[i, lo:-1].float(), dim=-1)
        tok_lp = logp.gather(-1, tgt[i].unsqueeze(-1)).squeeze(-1)
        a = targets_from[i] - 1 - lo
        out.append((-float(tok_lp[a:len(s) - 1 - lo].sum()), len(s) - targets_from[i]))
    return out


def perplexity(model, texts: list[str], tok, device: str, dtype, max_len: int = 1024,
               stride: int = 512, batch: int = 8) -> dict[str, Any]:
    eos = eos_id(tok)
    jobs = []                                   # (doc index, window ids, first scored offset)
    for di, enc in enumerate(tok.encode_batch(texts, add_special_tokens=False)):
        ids = [eos] + enc.ids
        for start, end, first in windows_for(len(ids), max_len, stride):
            jobs.append((di, ids[start:end], first - start))
    jobs.sort(key=lambda j: len(j[1]))          # similar lengths per batch: less padding
    nll = np.zeros(len(texts))
    count = np.zeros(len(texts), dtype=np.int64)
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        for (di, _, _), (s, n) in zip(chunk, _nll_batch(model, [j[1] for j in chunk],
                                                         [j[2] for j in chunk], device, dtype)):
            nll[di] += s
            count[di] += n
    total = float(nll.sum()) / max(int(count.sum()), 1)
    return {"perplexity": math.exp(total), "nll_per_token": total, "tokens": int(count.sum()),
            "documents": len(texts), "doc_nll": nll.round(4).tolist(), "doc_tokens": count.tolist()}


def by_language(res: dict[str, Any], languages: list[str]) -> dict[str, Any]:
    """Pooled perplexity per language from the per-document NLL a result already holds
    (``languages[i]`` belongs to document i). Does not change the overall numbers."""
    nll, cnt = np.asarray(res["doc_nll"]), np.asarray(res["doc_tokens"])
    if len(languages) != len(nll):
        raise ValueError(f"{len(languages)} languages for {len(nll)} documents")
    out = {}
    for lang in sorted(set(languages)):
        m = np.asarray([x == lang for x in languages])
        out[lang] = {"perplexity": math.exp(float(nll[m].sum()) / max(int(cnt[m].sum()), 1)),
                     "documents": int(m.sum()), "tokens": int(cnt[m].sum())}
    return out


def add_code_languages(result_file: Path, eval_dir: Path) -> dict[str, Any] | None:
    """Add ``code.by_language`` to an existing result file (no model needed). Returns it, or None
    when the file has no code task."""
    r = json.loads(result_file.read_text(encoding="utf-8"))
    if "code" not in r:
        return None
    r["code"]["by_language"] = by_language(r["code"], [x["language"] for x in read_jsonl(eval_dir / "code.jsonl")])
    result_file.write_text(json.dumps(r), encoding="utf-8")
    return r["code"]["by_language"]


# ---------------------------------------------------------------- PubMedQA

def pubmedqa_prompt(row: dict[str, Any], with_answer: bool = False) -> str:
    p = f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer:"
    return p + (f" {row['label']}\n\n" if with_answer else "")


def few_shot_prefix(train: list[dict[str, Any]], shots: int, seed: int) -> str:
    """``shots`` examples, cycling through the answer classes so none dominates."""
    if shots <= 0:
        return ""
    rng = random.Random(seed)
    by = {a: [r for r in train if r["label"] == a] for a in ANSWERS}
    picks = []
    for i in range(shots):
        picks.append(rng.choice(by[ANSWERS[i % len(ANSWERS)]]))
    rng.shuffle(picks)
    return "".join(pubmedqa_prompt(r, with_answer=True) for r in picks)


def _answer_scores(model, tok, prompts: list[str], device, dtype, max_len: int, batch: int):
    """Log-likelihood of each answer after each prompt -> array [n_prompts, 3]."""
    ans_ids = [tok.encode(" " + a, add_special_tokens=False).ids for a in ANSWERS]
    eos = eos_id(tok)
    seqs, firsts = [], []
    for enc in tok.encode_batch(prompts, add_special_tokens=False):
        for a in ans_ids:
            room = max_len - len(a) - 1
            ids = [eos] + enc.ids[-room:]   # keep the end (question) if too long
            seqs.append(ids + a)
            firsts.append(len(ids))
    out = []
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res: dict[int, float] = {}
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        for i, (nll, _) in zip(idx, _nll_batch(model, [seqs[i] for i in idx], [firsts[i] for i in idx],
                                               device, dtype)):
            res[i] = -nll
    for i in range(len(seqs)):
        out.append(res[i])
    return np.asarray(out).reshape(len(prompts), len(ANSWERS))


def macro_f1(gold: list[str], pred: list[str]) -> float:
    f1s = []
    for a in ANSWERS:
        tp = sum(g == a and p == a for g, p in zip(gold, pred))
        fp = sum(g != a and p == a for g, p in zip(gold, pred))
        fn = sum(g == a and p != a for g, p in zip(gold, pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _auroc_matrix(gold: list[str], ll: np.ndarray):
    """Pairwise matrix [yes items, no items]: 1 if margin(yes) > margin(no), 0.5 on ties."""
    m = ll[:, 0] - ll[:, 1]
    y = np.asarray([g == "yes" for g in gold])
    n = np.asarray([g == "no" for g in gold])
    sy, sn = m[y][:, None], m[n][None, :]
    return (sy > sn) + 0.5 * (sy == sn)


def auroc_weights(n_yes: int, n_no: int, n: int, seed: int):
    """Stratified bootstrap multiplicities for the yes and the no items (shared for paired tests)."""
    rng = np.random.default_rng(seed)
    wy = np.stack([np.bincount(rng.integers(0, n_yes, n_yes), minlength=n_yes) for _ in range(n)])
    wn = np.stack([np.bincount(rng.integers(0, n_no, n_no), minlength=n_no) for _ in range(n)])
    return wy.astype(float), wn.astype(float)


def _boot_auroc(mat: np.ndarray, wy: np.ndarray, wn: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", wy, mat, wn) / (wy.sum(1) * wn.sum(1))


def yes_no_auroc(gold: list[str], ll: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, Any]:
    mat = _auroc_matrix(gold, ll)
    wy, wn = auroc_weights(*mat.shape, n, seed)
    b = _boot_auroc(mat, wy, wn)
    return {"auroc": float(mat.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n_yes": int(mat.shape[0]), "n_no": int(mat.shape[1])}


def _summary(gold: list[str], pred: list[str]) -> dict[str, Any]:
    correct = np.asarray([g == p for g, p in zip(gold, pred)], dtype=float)
    lo, hi = bootstrap_ci(correct)
    return {"accuracy": float(correct.mean()), "ci95": [lo, hi], "macro_f1": macro_f1(gold, pred),
            "predicted": {a: pred.count(a) for a in ANSWERS}, "correct": correct.astype(int).tolist()}


def pubmedqa(model, tok, test: list[dict[str, Any]], train: list[dict[str, Any]], device, dtype,
             shots: int = 0, seed: int = 0, max_len: int = 2048, batch: int = 6) -> dict[str, Any]:
    prefix = few_shot_prefix(train, shots, seed)
    scores = _answer_scores(model, tok, [prefix + pubmedqa_prompt(r) for r in test], device, dtype,
                            max_len, batch)
    # context-free prior of each answer (same few-shot prefix, empty context/question)
    prior = _answer_scores(model, tok, [prefix + "Context: N/A\nQuestion: N/A\nAnswer:"], device,
                           dtype, max_len, batch)[0]
    gold = [r["label"] for r in test]
    raw = [ANSWERS[int(i)] for i in scores.argmax(1)]
    cal = [ANSWERS[int(i)] for i in (scores - prior).argmax(1)]
    counts = {a: gold.count(a) for a in ANSWERS}
    top = max(counts, key=counts.get)
    return {"shots": shots, "n": len(test), "majority_label": top,
            "majority_accuracy": counts[top] / len(test),
            "raw": _summary(gold, raw), "calibrated": _summary(gold, cal),
            "yes_no_auroc": yes_no_auroc(gold, scores),
            "answer_prior": dict(zip(ANSWERS, prior.round(4).tolist())),
            "loglik": scores.round(4).tolist(), "gold": gold}


# ---------------------------------------------------------------- driver

def evaluate(model_path: str, eval_dir: Path, out: Path | None = None, tasks=TASKS,
             shots: tuple[int, ...] = (0, 3), seed: int = 0, model=None, device: str = "auto",
             precision: str = "auto", max_len: int = 1024, stride: int = 512,
             tokenizer: str = DEFAULT_TOKENIZER, limit: int = 0, quant: str = "none",
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """``model_path`` may also be a QLoRA run folder (has ``qlora_config.json``): the base is
    then loaded as in training (4-bit or not) with the adapter. ``quant="4bit"`` loads a plain
    model in 4-bit (to separate the quantization effect from the fine-tuning effect)."""
    import torch
    from .pretrain import _precision, load_model
    from .qlora import load_adapted, load_base, place

    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype, _ = _precision(precision, device)
    if model is None:
        if (Path(model_path) / "qlora_config.json").exists():
            model = load_adapted(model_path, device)
        elif quant != "none":
            model = load_base(model_path, quant, device)
        else:
            model = load_model(model_path)
    place(model, device).eval()
    tok = load_tokenizer(tokenizer)
    t0 = time.time()
    res: dict[str, Any] = {"model": model_path, "quant": quant, "eval_dir": str(eval_dir), "seed": seed,
                           "precision": str(dtype), "max_len": max_len, "stride": stride}
    man = eval_dir / "manifest.json"
    if man.exists():
        res["eval_sha256"] = {k: v.get("sha256") for k, v in
                              json.loads(man.read_text(encoding="utf-8"))["files"].items()}

    def rows(name):
        r = read_jsonl(eval_dir / f"{name}.jsonl")
        return r[:limit] if limit else r

    def texts(task):
        if task == "clinical":
            if (eval_dir / "clinical.jsonl").exists():          # dev sets
                return [r["text"] for r in rows("clinical")]
            return [r["context"] for r in rows("pubmedqa_test")]
        if not (eval_dir / f"{task}.jsonl").exists():
            raise FileNotFoundError(f"no {task}.jsonl in {eval_dir} (build it with 'dftrain domainsets')")
        return [r["text"] for r in rows(task)]

    for task in tasks:
        t = time.time()
        if task in PPL_TASKS:
            res[task] = perplexity(model, texts(task), tok, device, dtype, max_len, stride)
            if task == "code":
                res[task]["by_language"] = by_language(
                    res[task], [r["language"] for r in rows("code")])
            log(f"{task}: ppl {res[task]['perplexity']:.3f} ({res[task]['tokens']:,} tokens, "
                f"{time.time() - t:.0f}s)")
        elif task == "pubmedqa":
            test, train = rows("pubmedqa_test"), read_jsonl(eval_dir / "pubmedqa_train.jsonl")
            res["pubmedqa"] = {}
            for k in shots:
                r = pubmedqa(model, tok, test, train, device, dtype, shots=k, seed=seed)
                res["pubmedqa"][f"{k}shot"] = r
                log(f"pubmedqa {k}-shot: acc {r['raw']['accuracy']:.3f} (calibrated "
                    f"{r['calibrated']['accuracy']:.3f}; majority {r['majority_accuracy']:.3f}; "
                    f"yes/no AUROC {r['yes_no_auroc']['auroc']:.3f}) "
                    f"pred {r['raw']['predicted']} ({time.time() - t:.0f}s)")
        else:
            raise ValueError(f"unknown task {task}")
    res["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res), encoding="utf-8")
    return res


def compare(a: dict[str, Any], b: dict[str, Any], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paired bootstrap of B - A: perplexity ratio per set, accuracy difference per PubMedQA setting."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for task in PPL_TASKS:
        if task in a and task in b:
            na, nb = np.asarray(a[task]["doc_nll"]), np.asarray(b[task]["doc_nll"])
            c = np.asarray(a[task]["doc_tokens"])
            idx = rng.integers(0, len(c), size=(n, len(c)))
            ratio = np.exp((nb[idx].sum(1) - na[idx].sum(1)) / c[idx].sum(1))
            out[task] = {"ppl_a": a[task]["perplexity"], "ppl_b": b[task]["perplexity"],
                         "ratio_b_over_a": b[task]["perplexity"] / a[task]["perplexity"],
                         "ratio_ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]}
    for key in sorted(set(a.get("pubmedqa", {})) & set(b.get("pubmedqa", {}))):
        ra, rb = a["pubmedqa"][key], b["pubmedqa"][key]
        if "loglik" in ra and "loglik" in rb and ra["n"] == rb["n"]:
            gold = _gold_from(ra)
            ma = _auroc_matrix(gold, np.asarray(ra["loglik"]))
            mb = _auroc_matrix(gold, np.asarray(rb["loglik"]))
            wy, wn = auroc_weights(*ma.shape, n, seed)
            d = _boot_auroc(mb, wy, wn) - _boot_auroc(ma, wy, wn)
            out[f"pubmedqa_{key}_auroc"] = {"auroc_a": float(ma.mean()), "auroc_b": float(mb.mean()),
                                            "diff": float(mb.mean() - ma.mean()),
                                            "diff_ci95": [float(np.percentile(d, 2.5)),
                                                          float(np.percentile(d, 97.5))]}
        for mode in ("raw", "calibrated"):
            ca = np.asarray(a["pubmedqa"][key][mode]["correct"])
            cb = np.asarray(b["pubmedqa"][key][mode]["correct"])
            idx = rng.integers(0, len(ca), size=(n, len(ca)))
            d = cb[idx].mean(1) - ca[idx].mean(1)
            out[f"pubmedqa_{key}_{mode}"] = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                                             "diff": float(cb.mean() - ca.mean()),
                                             "diff_ci95": [float(np.percentile(d, 2.5)),
                                                           float(np.percentile(d, 97.5))]}
    return out


def _gold_from(r: dict[str, Any]) -> list[str]:
    """Gold labels of a saved PubMedQA result (stored with it; older files lack them)."""
    if "gold" not in r:
        raise KeyError("result has loglik but no gold labels")
    return r["gold"]
