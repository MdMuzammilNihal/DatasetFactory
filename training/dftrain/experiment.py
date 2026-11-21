"""Experiment runner (Module 10d): corpus -> pretrain -> evaluate (-> QLoRA) per arm and seed.

A plan is a TOML file::

    name = "medical-160m"
    model = "EleutherAI/pythia-160m"
    budget = "20M"                 # training tokens per corpus (incl. EOS)
    seeds = [0, 1, 2]              # training seeds (data order, LoRA init); evaluation seed is fixed
    reference = "random"           # arm every other arm is compared against
    tasks = ["wikitext", "medical", "clinical", "pubmedqa"]
    shots = [0, 3]
    qlora = true                   # also QLoRA-fine-tune each model on PubMedQA
    eval = "data/eval"
    out = "data/experiments"       # -> out/<name>/{corpora,runs,results,summary.json,summary.md}
    dev_eval = "data/eval-dev"     # optional: also score perplexity on dev sets (-> results-dev/, variant "dev")
    summary = "summary"            # optional: summary file name (several plans may share one folder)

    [pretrain]                     # any TrainConfig field (lr, global_batch, micro_batch, ...)
    [qlora_config]                 # any QLoRAConfig field (quant, epochs, ...)

    [[arms]]
    name = "base"                  # no parts: the released model, not pretrained further
    [[arms]]
    name = "curated"
    parts = ["../backend/runs/medical-...=1"]   # job folders or .jsonl, with optional =share

Each arm's corpus is built once (``corpus_seed``, default 0); each seed pretrains,
evaluates and fine-tunes. A finished step (its output file exists) is skipped and a
stopped pretraining run resumes from its checkpoint, so re-running the plan continues it.
The summary gives mean +- std over seeds per metric and, per seed, the paired-bootstrap
comparison (``evaluate.compare``) of each arm against the reference arm.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

import numpy as np

PLAN_KEYS = {"name", "model", "budget", "seeds", "reference", "tasks", "shots", "qlora", "eval",
             "out", "pretrain", "qlora_config", "arms", "corpus_seed", "allow_short", "device",
             "dev_eval", "summary"}
ARM_KEYS = {"name", "parts", "description", "seeds"}   # arm "seeds" overrides the plan's
EVAL_SEED = 0          # few-shot examples are the same for every model


class PlanError(ValueError):
    pass


def load_plan(path: Path) -> dict[str, Any]:
    plan = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = set(plan) - PLAN_KEYS
    if unknown:
        raise PlanError(f"unknown plan keys: {sorted(unknown)}")
    for k in ("name", "arms"):
        if k not in plan:
            raise PlanError(f"plan needs '{k}'")
    plan.setdefault("model", "EleutherAI/pythia-160m")
    plan.setdefault("budget", "20M")
    plan.setdefault("seeds", [0])
    plan.setdefault("tasks", ["wikitext", "medical", "clinical", "pubmedqa"])
    plan.setdefault("shots", [0, 3])
    plan.setdefault("qlora", False)
    plan.setdefault("eval", "data/eval")
    plan.setdefault("out", "data/experiments")
    plan.setdefault("pretrain", {})
    plan.setdefault("qlora_config", {})
    plan.setdefault("corpus_seed", 0)
    plan.setdefault("allow_short", False)
    plan.setdefault("device", "auto")
    plan.setdefault("dev_eval", "")
    plan.setdefault("summary", "summary")
    names = [a.get("name") for a in plan["arms"]]
    if len(set(names)) != len(names) or None in names:
        raise PlanError("every arm needs a unique name")
    for a in plan["arms"]:
        if set(a) - ARM_KEYS:
            raise PlanError(f"arm {a['name']}: unknown keys {sorted(set(a) - ARM_KEYS)}")
    plan.setdefault("reference", names[0])
    if plan["reference"] not in names:
        raise PlanError(f"reference arm {plan['reference']!r} is not in the plan")
    return plan


class Layout:
    def __init__(self, plan: dict[str, Any]):
        self.root = Path(plan["out"]) / plan["name"]

    def corpus(self, arm): return self.root / "corpora" / arm
    def run(self, arm, seed): return self.root / "runs" / f"{arm}-s{seed}"
    def qlora_run(self, arm, seed): return self.root / "runs" / f"{arm}-s{seed}-qlora"

    def result(self, arm, seed, qlora=False, folder="results"):
        if not qlora and seed is None:
            return self.root / folder / f"{arm}.json"
        return self.root / folder / f"{arm}-s{seed}{'-qlora' if qlora else ''}.json"


def steps(plan: dict[str, Any], only: list[str] | None = None) -> list[tuple[str, str, Any, Path]]:
    """(kind, arm, seed, output) in execution order; ``output`` existing means done."""
    lay, out = Layout(plan), []
    for arm in plan["arms"]:
        name = arm["name"]
        if only and name not in only:
            continue
        if arm.get("parts"):
            out.append(("corpus", name, None, lay.corpus(name) / "meta.json"))
        for seed in arm.get("seeds", plan["seeds"]):
            if arm.get("parts"):
                out.append(("pretrain", name, seed, lay.run(name, seed) / "summary.json"))
                out.append(("evaluate", name, seed, lay.result(name, seed)))
                if plan["dev_eval"]:
                    out.append(("evaluate_dev", name, seed, lay.result(name, seed, folder="results-dev")))
            elif seed == plan["seeds"][0]:
                out.append(("evaluate", name, None, lay.result(name, None)))   # deterministic
                if plan["dev_eval"]:
                    out.append(("evaluate_dev", name, None, lay.result(name, None, folder="results-dev")))
            if plan["qlora"]:
                out.append(("qlora", name, seed, lay.result(name, seed, qlora=True)))
    return out


def _model_path(plan, arm, seed) -> str:
    return str(Layout(plan).run(arm["name"], seed) / "model") if arm.get("parts") else plan["model"]


def run_step(plan: dict[str, Any], kind: str, arm: dict[str, Any], seed, log: Callable = print) -> None:
    from . import corpus as corpus_mod
    lay, name = Layout(plan), arm["name"]
    eval_dir = Path(plan["eval"])
    if kind == "corpus":
        corpus_mod.build_corpus(f"{plan['name']}-{name}", [corpus_mod.parse_part(p) for p in arm["parts"]],
                                corpus_mod.parse_count(plan["budget"]), lay.corpus(name), eval_dir,
                                seed=plan["corpus_seed"], allow_short=plan["allow_short"], log=log)
    elif kind == "pretrain":
        from .pretrain import TrainConfig, train
        run = lay.run(name, seed)
        cfg = TrainConfig(corpus=str(lay.corpus(name)), out=str(run), model=plan["model"], seed=seed,
                          device=plan["device"], **plan["pretrain"])
        train(cfg, log=log, resume=(run / "checkpoint.pt").exists())
    elif kind == "evaluate":
        from .evaluate import evaluate
        evaluate(_model_path(plan, arm, seed), eval_dir, lay.result(name, seed if arm.get("parts") else None),
                 tasks=plan["tasks"], shots=tuple(plan["shots"]), seed=EVAL_SEED, device=plan["device"],
                 log=log)
    elif kind == "evaluate_dev":
        from .evaluate import PPL_TASKS, evaluate
        evaluate(_model_path(plan, arm, seed), Path(plan["dev_eval"]),
                 lay.result(name, seed if arm.get("parts") else None, folder="results-dev"),
                 tasks=[t for t in plan["tasks"] if t in PPL_TASKS], seed=EVAL_SEED, device=plan["device"], log=log)
    elif kind == "qlora":
        from .evaluate import evaluate
        from .qlora import QLoRAConfig, train as qtrain
        run = lay.qlora_run(name, seed)
        if not (run / "summary.json").exists():
            qtrain(QLoRAConfig(model=_model_path(plan, arm, seed), out=str(run), seed=seed,
                               train_file=str(eval_dir / "pubmedqa_train.jsonl"), device=plan["device"],
                               **plan["qlora_config"]), log=log)
        evaluate(str(run), eval_dir, lay.result(name, seed, qlora=True), tasks=["pubmedqa"], shots=(0,),
                 seed=EVAL_SEED, device=plan["device"], log=log)
    else:
        raise ValueError(kind)


# ---------------------------------------------------------------- summary

def metrics(res: dict[str, Any]) -> dict[str, float]:
    m: dict[str, float] = {}
    for task in ("wikitext", "medical", "clinical", "code", "legal"):
        if task in res:
            m[f"{task}_ppl"] = res[task]["perplexity"]
    for key, r in res.get("pubmedqa", {}).items():
        m[f"pubmedqa_{key}_acc"] = r["raw"]["accuracy"]
        m[f"pubmedqa_{key}_f1"] = r["raw"]["macro_f1"]
        if "yes_no_auroc" in r:
            m[f"pubmedqa_{key}_auroc"] = r["yes_no_auroc"]["auroc"]
        for a, c in r["raw"]["predicted"].items():
            m[f"pubmedqa_{key}_pred_{a}"] = c
    return m


def _mean_std(values: list[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def summarize(plan: dict[str, Any]) -> dict[str, Any]:
    from .evaluate import compare
    lay = Layout(plan)

    def load(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    results: dict[str, dict[str, dict[int, Any]]] = {}     # variant -> arm -> seed -> result
    for arm in plan["arms"]:
        name = arm["name"]
        for seed in arm.get("seeds", plan["seeds"]):
            r = load(lay.result(name, seed if arm.get("parts") else None))
            if r:
                results.setdefault("pretrained", {}).setdefault(name, {})[seed] = r
            d = load(lay.result(name, seed if arm.get("parts") else None, folder="results-dev"))
            if d:
                results.setdefault("dev", {}).setdefault(name, {})[seed] = d
            q = load(lay.result(name, seed, qlora=True))
            if q:
                results.setdefault("qlora", {}).setdefault(name, {})[seed] = q
    out: dict[str, Any] = {"plan": {k: v for k, v in plan.items() if k != "arms"},
                           "arms": plan["arms"], "variants": {}}
    for variant, by_arm in results.items():
        v: dict[str, Any] = {"arms": {}, "vs_reference": {}}
        for name, by_seed in by_arm.items():
            per = [metrics(r) for r in by_seed.values()]
            keys = sorted(set().union(*per))
            v["arms"][name] = {"seeds": sorted(by_seed),
                               **{k: _mean_std([p[k] for p in per if k in p]) for k in keys}}
        ref = by_arm.get(plan["reference"], {})
        for name, by_seed in by_arm.items():
            if name == plan["reference"]:
                continue
            cmp = {}
            for seed, r in by_seed.items():
                base = ref.get(seed) or (next(iter(ref.values())) if len(ref) == 1 else None)
                if base:
                    cmp[seed] = compare(base, r, seed=seed)
            if cmp:
                v["vs_reference"][name] = cmp
        out["variants"][variant] = v
    return out


def _fmt(ms: dict[str, Any] | None, digits: int, pct: bool = False) -> str:
    if not ms:
        return "-"
    f = 100 if pct else 1
    s = f"{ms['mean'] * f:.{digits}f}"
    return s + (f" ± {ms['std'] * f:.{digits}f}" if ms["n"] > 1 else "")


def to_markdown(summary: dict[str, Any]) -> str:
    plan = summary["plan"]
    lines = [f"# Experiment {plan['name']}", "",
             f"Model {plan['model']}, budget {plan['budget']} tokens, seeds {plan['seeds']}, "
             f"reference arm `{plan['reference']}`. Mean ± std over seeds (std over training seeds only).", ""]
    for variant, v in summary["variants"].items():
        shots = sorted({k.split("_")[1] for a in v["arms"].values() for k in a if k.startswith("pubmedqa_")})
        ppl = [t for t in ("wikitext", "medical", "clinical", "code", "legal")
               if any(f"{t}_ppl" in a for a in v["arms"].values())]
        head = ["arm", "seeds"] + [("WikiText" if t == "wikitext" else t) + " ppl" for t in ppl]
        for s in shots:
            head += [f"PubMedQA {s} acc %", f"{s} macro-F1", f"{s} yes/no AUROC", f"{s} pred y/n/m"]
        lines += [f"## {variant}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for name, a in v["arms"].items():
            row = [name, str(len(a["seeds"]))] + [_fmt(a.get(f"{t}_ppl"), 2) for t in ppl]
            for s in shots:
                pred = "/".join(f"{a[f'pubmedqa_{s}_pred_{x}']['mean']:.0f}"
                                if f"pubmedqa_{s}_pred_{x}" in a else "-" for x in ("yes", "no", "maybe"))
                row += [_fmt(a.get(f"pubmedqa_{s}_acc"), 1, pct=True), _fmt(a.get(f"pubmedqa_{s}_f1"), 3),
                        _fmt(a.get(f"pubmedqa_{s}_auroc"), 3), pred]
            lines.append("| " + " | ".join(row) + " |")
        if v["vs_reference"]:
            lines += ["", f"Paired bootstrap vs `{plan['reference']}` (per seed: ppl ratio B/A or "
                          "accuracy difference, with 95% CI):", ""]
            for name, by_seed in v["vs_reference"].items():
                for seed, c in by_seed.items():
                    parts = []
                    for k, x in c.items():
                        if "ratio_b_over_a" in x:
                            parts.append(f"{k} {x['ratio_b_over_a']:.3f} [{x['ratio_ci95'][0]:.3f}, "
                                         f"{x['ratio_ci95'][1]:.3f}]")
                        elif k.endswith("_raw"):
                            parts.append(f"{k[9:-4]} acc {x['diff'] * 100:+.1f} pts "
                                         f"[{x['diff_ci95'][0] * 100:+.1f}, {x['diff_ci95'][1] * 100:+.1f}]")
                        elif k.endswith("_auroc"):
                            parts.append(f"{k[9:-6]} AUROC {x['diff']:+.3f} "
                                         f"[{x['diff_ci95'][0]:+.3f}, {x['diff_ci95'][1]:+.3f}]")
                    lines.append(f"- `{name}` seed {seed}: " + "; ".join(parts))
        lines.append("")
    return "\n".join(lines)


def write_summary(plan: dict[str, Any]) -> dict[str, Any]:
    s = summarize(plan)
    root = Layout(plan).root
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{plan['summary']}.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (root / f"{plan['summary']}.md").write_text(to_markdown(s), encoding="utf-8")
    return s


def run_plan(path: Path, only: list[str] | None = None, dry_run: bool = False,
             summary_only: bool = False, log: Callable = print) -> int:
    plan = load_plan(path)
    arms = {a["name"]: a for a in plan["arms"]}
    todo = steps(plan, only)
    if dry_run:
        for kind, arm, seed, out in todo:
            print(f"{'done' if out.exists() else 'todo'}  {kind:9s} {arm:20s} seed {seed}  -> {out}")
        return 0
    if not summary_only:
        for kind, arm, seed, out in todo:
            if out.exists():
                continue
            log(f"=== {kind} {arm} seed {seed}")
            run_step(plan, kind, arms[arm], seed, log=log)
            write_summary(plan)          # partial results are visible while the plan runs
    write_summary(plan)
    print(f"summary -> {Layout(plan).root / (plan['summary'] + '.md')}", file=sys.stderr)
    return 0
