"""Command line: ``python -m dftrain <command>`` (run from the ``training`` folder).

  evalsets [--out data/eval] [--seed 0]
  domainsets [--eval data/eval] [--dev data/eval-dev] [--domains code,legal] [--train-file JOB ...]
  devsets --train-file JOB_DIR_OR_JSONL [...] [--out data/eval-dev] [--eval data/eval]
  corpus NAME --budget 20M --part RUN_DIR[=SHARE] [--part ...] [--eval data/eval]
         [--out data/corpora] [--seed 0] [--val-every 100] [--allow-short]
  pretrain --corpus data/corpora/NAME --out data/runs/RUN [--model EleutherAI/pythia-160m]
         [--seed 0] [--global-batch 32] [--micro-batch 4] [--lr 6e-5] [--max-steps N] [--resume]
  evaluate MODEL --out results.json [--eval data/eval] [--tasks wikitext,medical,clinical,pubmedqa] [--shots 0,3]
  compare A.json B.json
  qlora MODEL --out data/runs/RUN [--quant 4bit|none] [--epochs 3] [--seed 0] [--eval data/eval|none]
  experiment PLAN.toml [--only arm1,arm2] [--dry-run] [--summary-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import corpus, evalsets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dftrain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evalsets", help="download and freeze the evaluation sets")
    e.add_argument("--out", default="data/eval")
    e.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("devsets", help="dev sets for choosing a recipe (test sets stay untouched)")
    d.add_argument("--out", default="data/eval-dev")
    d.add_argument("--eval", default="data/eval")
    d.add_argument("--train-file", action="append", required=True,
                   help="job folder or .jsonl whose documents training may use (dev docs overlapping them are dropped)")
    d.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("domainsets", help="code and legal perplexity sets (test -> --eval, dev -> --dev)")
    g.add_argument("--eval", default="data/eval")
    g.add_argument("--dev", default="data/eval-dev")
    g.add_argument("--domains", default="code,legal")
    g.add_argument("--train-file", action="append", default=[],
                   help="job folder or .jsonl training may use (overlapping dev docs are dropped)")
    g.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("corpus", help="pack job outputs into a training corpus")
    c.add_argument("name")
    c.add_argument("--budget", required=True, help="training tokens incl. EOS, e.g. 20M")
    c.add_argument("--part", action="append", required=True, metavar="DIR[=SHARE]")
    c.add_argument("--eval", default="data/eval", help="evaluation sets to decontaminate against "
                   "('none' to skip)")
    c.add_argument("--out", default="data/corpora")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--val-every", type=int, default=100)
    c.add_argument("--allow-short", action="store_true")

    t = sub.add_parser("pretrain", help="continued pretraining of a Pythia checkpoint")
    t.add_argument("--corpus", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="EleutherAI/pythia-160m")
    t.add_argument("--revision", default="main")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--seq-len", type=int, default=1024)
    t.add_argument("--global-batch", type=int, default=32)
    t.add_argument("--micro-batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=6e-5)
    t.add_argument("--warmup-frac", type=float, default=0.05)
    t.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=200)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.add_argument("--resume", action="store_true")

    v = sub.add_parser("evaluate", help="perplexity and PubMedQA on the frozen evaluation sets")
    v.add_argument("model", help="HF name (EleutherAI/pythia-160m) or a run's model folder")
    v.add_argument("--out", required=True)
    v.add_argument("--eval", default="data/eval")
    v.add_argument("--tasks", default="wikitext,medical,clinical,pubmedqa")
    v.add_argument("--shots", default="0,3")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    v.add_argument("--limit", type=int, default=0, help="first N items per set (quick checks)")
    v.add_argument("--quant", default="none", choices=["none", "4bit"],
                   help="load a plain model in 4-bit (QLoRA run folders load as trained)")

    q = sub.add_parser("qlora", help="QLoRA fine-tuning on PubMedQA, then PubMedQA evaluation")
    q.add_argument("model", help="HF name or a pretraining run's model folder")
    q.add_argument("--out", required=True)
    q.add_argument("--train", default="data/eval/pubmedqa_train.jsonl")
    q.add_argument("--quant", default="4bit", choices=["4bit", "none"])
    q.add_argument("--r", type=int, default=16)
    q.add_argument("--alpha", type=int, default=32)
    q.add_argument("--dropout", type=float, default=0.05)
    q.add_argument("--lr", type=float, default=2e-4)
    q.add_argument("--epochs", type=int, default=3)
    q.add_argument("--micro-batch", type=int, default=4)
    q.add_argument("--accum", type=int, default=4)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--gradient-checkpointing", action="store_true")
    q.add_argument("--eval", default="data/eval", help="'none' to skip the evaluation")
    q.add_argument("--shots", default="0")

    x = sub.add_parser("experiment", help="run an experiment plan (corpus -> pretrain -> evaluate "
                       "-> QLoRA) over seeds; resumes finished steps")
    x.add_argument("plan", help="experiment plan (.toml)")
    x.add_argument("--only", default="", help="comma-separated arm names")
    x.add_argument("--dry-run", action="store_true", help="print the steps and exit")
    x.add_argument("--summary-only", action="store_true", help="rebuild summary from results")

    k = sub.add_parser("compare", help="paired bootstrap of B vs A result files")
    k.add_argument("a")
    k.add_argument("b")

    lg = sub.add_parser("langppl", help="add per-language code perplexity to existing result files")
    lg.add_argument("files", nargs="+")
    lg.add_argument("--eval", default="data/eval", help="folder with code.jsonl (data/eval or data/eval-dev)")

    a = ap.parse_args(argv)
    if a.cmd == "langppl":
        from .evaluate import add_code_languages
        for f in a.files:
            r = add_code_languages(Path(f), Path(a.eval))
            if r:
                print(f, " ".join(f"{k} {v['perplexity']:.3f}" for k, v in r.items()))
        return 0
    if a.cmd == "evaluate":
        from .evaluate import evaluate
        r = evaluate(a.model, Path(a.eval), Path(a.out), tasks=a.tasks.split(","),
                     shots=tuple(int(x) for x in a.shots.split(",") if x != ""), seed=a.seed,
                     precision=a.precision, limit=a.limit, quant=a.quant)
        print(f"done in {r['seconds']} s -> {a.out}")
        return 0
    if a.cmd == "qlora":
        from .qlora import QLoRAConfig, train as qtrain
        cfg = QLoRAConfig(model=a.model, out=a.out, train_file=a.train, quant=a.quant, r=a.r,
                          alpha=a.alpha, dropout=a.dropout, lr=a.lr, epochs=a.epochs,
                          micro_batch=a.micro_batch, accum=a.accum, seed=a.seed,
                          gradient_checkpointing=a.gradient_checkpointing)
        print(json.dumps(qtrain(cfg), indent=2))
        if a.eval != "none":
            from .evaluate import evaluate
            evaluate(a.out, Path(a.eval), Path(a.out) / "results.json", tasks=["pubmedqa"],
                     shots=tuple(int(s) for s in a.shots.split(",")), seed=a.seed)
        return 0
    if a.cmd == "experiment":
        from .experiment import run_plan
        return run_plan(Path(a.plan), only=[s for s in a.only.split(",") if s],
                        dry_run=a.dry_run, summary_only=a.summary_only)
    if a.cmd == "compare":
        from .evaluate import compare
        ra, rb = (json.loads(Path(x).read_text(encoding="utf-8")) for x in (a.a, a.b))
        print(json.dumps(compare(ra, rb), indent=2))
        return 0
    if a.cmd == "pretrain":
        from .pretrain import TrainConfig, train
        cfg = TrainConfig(corpus=a.corpus, out=a.out, model=a.model, revision=a.revision, seed=a.seed,
                          seq_len=a.seq_len, global_batch=a.global_batch, micro_batch=a.micro_batch,
                          lr=a.lr, warmup_frac=a.warmup_frac, precision=a.precision,
                          eval_every=a.eval_every, save_every=a.save_every, max_steps=a.max_steps,
                          gradient_checkpointing=a.gradient_checkpointing)
        print(json.dumps(train(cfg, resume=a.resume), indent=2))
        return 0
    try:
        if a.cmd == "devsets":
            files = [Path(f) / "data.jsonl" if Path(f).is_dir() else Path(f) for f in a.train_file]
            m = evalsets.build_dev(Path(a.out), Path(a.eval), files, seed=a.seed)
            print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sha256"}
                              for k, v in m["files"].items()}, indent=2))
            return 0
        if a.cmd == "domainsets":
            files = [Path(f) / "data.jsonl" if Path(f).is_dir() else Path(f) for f in a.train_file]
            print(json.dumps(evalsets.build_domains(Path(a.eval), Path(a.dev), files,
                                                    domains=[x for x in a.domains.split(",") if x],
                                                    seed=a.seed), indent=2))
            return 0
        if a.cmd == "evalsets":
            m = evalsets.build(Path(a.out), seed=a.seed)
            print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sha256"}
                              for k, v in m["files"].items()}, indent=2))
        else:
            eval_dir = None if a.eval == "none" else Path(a.eval)
            if eval_dir is not None and not (eval_dir / "manifest.json").exists():
                print(f"no evaluation sets at {eval_dir}; run 'evalsets' first or pass --eval none",
                      file=sys.stderr)
                return 2
            meta = corpus.build_corpus(
                a.name, [corpus.parse_part(p) for p in a.part], corpus.parse_count(a.budget),
                Path(a.out) / a.name, eval_dir, seed=a.seed, val_every=a.val_every,
                allow_short=a.allow_short)
            print(json.dumps({k: meta[k] for k in ("name", "train_tokens", "val_tokens",
                              "train_documents", "short", "seconds")}, indent=2))
    except corpus.CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
