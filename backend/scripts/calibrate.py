"""Calibration reports on real sample documents.

Usage (from the backend folder; run scripts/fetch_samples.py first):

    python scripts/calibrate.py                 # validator acceptance matrix
    python scripts/calibrate.py medical         # ... plus failing checks for one domain
    python scripts/calibrate.py scores          # quality-score distribution per domain
    python scripts/calibrate.py classifier      # what the learned classifier adds
    python scripts/calibrate.py speed           # throughput of each stage

Add  --samples PATH  to use a different sample file. A sample file maps a
sample-set name to a list of texts, e.g. {"med_pubmed": ["...", "..."]}.

In the acceptance matrix each cell shows  accepted / documents that passed the
global filter.  A good validator accepts most documents from its own domain's
corpus and few from the others.
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dataset_factory.classifier import load_classifier  # noqa: E402
from dataset_factory.config import load_config  # noqa: E402
from dataset_factory.filters import GlobalFilter  # noqa: E402
from dataset_factory.scoring import build_scorer  # noqa: E402
from dataset_factory.textstats import TextStats  # noqa: E402
from dataset_factory.validators import DOMAINS, build_validator  # noqa: E402

# Which sample sets belong to which domain (used by the "scores" report).
OWN_SETS = {
    "general": ("wikipedia",),
    "news": ("news_cnn",),
    "medical": ("med_pubmedqa", "med_pubmed"),
    "legal": ("legal_cap",),
    "scientific": ("sci_arxiv_sum", "sci_arxiv_cp"),
    "code": ("code_py", "code_java"),
}


def percentiles(values: list[float]) -> str:
    if not values:
        return "(none)"
    values = sorted(values)
    pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]  # noqa: E731
    return "  ".join(f"{pick(p):.2f}" for p in (0.05, 0.25, 0.50, 0.75, 0.95))


def report_matrix(cfg, data, detail):
    sets = list(data)
    print(f"{'validator':<11}" + "".join(f"{s[:13]:>14}" for s in sets))
    for domain in DOMAINS:
        global_filter = GlobalFilter(cfg.global_filter_for(domain))
        validator = build_validator(domain, cfg)
        cells, failures = [], {}
        for s in sets:
            accepted = survived = 0
            counter: collections.Counter = collections.Counter()
            for text in data[s]:
                st = TextStats(text)
                if not global_filter.check(st).passed:
                    continue
                survived += 1
                result = validator.validate(st)
                accepted += result.passed
                counter.update(result.details["failed"])
            cells.append(f"{accepted}/{survived}")
            failures[s] = counter
        print(f"{domain:<11}" + "".join(f"{c:>14}" for c in cells))
        if detail == domain:
            for s in sets:
                print(f"    {s}: {dict(failures[s].most_common())}")


def report_scores(cfg, data):
    print("Quality score Q(x) of documents that passed both filters.")
    print("Percentiles shown: 5%  25%  50%  75%  95%\n")
    for domain in DOMAINS:
        global_filter = GlobalFilter(cfg.global_filter_for(domain))
        validator = build_validator(domain, cfg)
        scorer = build_scorer(domain, cfg)
        print(f"{domain}  (threshold {scorer.threshold})")
        for s in OWN_SETS[domain]:
            scores, sums = [], collections.Counter()
            for text in data.get(s, []):
                st = TextStats(text)
                if not global_filter.check(st).passed:
                    continue
                result = validator.validate(st)
                if not result.passed:
                    continue
                scored = scorer.score(st, result.details["signals"])
                scores.append(scored.score)
                sums.update(scored.features)
            kept = sum(1 for v in scores if v >= scorer.threshold)
            print(f"  {s:<14} n={len(scores):<4} {percentiles(scores)}   kept {kept}/{len(scores)}")
            if scores:
                means = "  ".join(f"{k}={v / len(scores):.2f}" for k, v in sums.items())
                print(f"      mean features: {means}")


def report_classifier(cfg, data):
    """What the learned classifier changes after the rule-based stages."""
    sets = list(data)
    print("Documents accepted by the rules (filter + validator + score), and how many")
    print("of those the learned classifier also accepts:  classifier / rules\n")
    print(f"{'domain':<11}" + "".join(f"{s[:13]:>14}" for s in sets))
    for domain in DOMAINS:
        classifier = load_classifier(domain, cfg)
        if classifier is None:
            raise SystemExit("the classifier is disabled in the configuration")
        global_filter = GlobalFilter(cfg.global_filter_for(domain))
        validator = build_validator(domain, cfg)
        scorer = build_scorer(domain, cfg)
        cells = []
        for s in sets:
            by_rules = by_both = 0
            for text in data[s]:
                st = TextStats(text)
                if not global_filter.check(st).passed:
                    continue
                result = validator.validate(st)
                if not result.passed or not scorer.score(st, result.details["signals"]).passed:
                    continue
                by_rules += 1
                by_both += classifier.check(st).passed
            cells.append(f"{by_both}/{by_rules}")
        print(f"{domain:<11}" + "".join(f"{c:>14}" for c in cells))

    docs = [TextStats(d) for s in sets for d in data[s]][:1500]
    for st in docs:
        st.words
    classifier = load_classifier("general", cfg)
    start = time.perf_counter()
    for st in docs:
        classifier.probability(st)
    print(f"\nclassifier speed: {len(docs) / (time.perf_counter() - start):.0f} docs/s (one process)")


def report_speed(cfg, data):
    docs = [d for s in data for d in data[s]][:1500]
    chars = sum(map(len, docs)) / len(docs)
    print(f"{len(docs)} documents, {chars:.0f} characters on average, one process\n")
    print(f"{'domain':<11}{'global':>10}{'validator':>11}{'scorer':>9}{'all three':>11}   docs/s")
    for domain in DOMAINS:
        global_filter = GlobalFilter(cfg.global_filter_for(domain))
        validator = build_validator(domain, cfg)
        scorer = build_scorer(domain, cfg)

        def timed(fn, items):
            start = time.perf_counter()
            out = [fn(x) for x in items]
            return out, len(items) / max(1e-9, time.perf_counter() - start)

        full = 0.0
        for _ in range(3):  # best of three, to smooth out machine noise
            start = time.perf_counter()
            for text in docs:  # the real pipeline: stop at the first stage that rejects
                st = TextStats(text)
                if global_filter.check(st).passed:
                    result = validator.validate(st)
                    if result.passed:
                        scorer.score(st, result.details["signals"])
            full = max(full, len(docs) / (time.perf_counter() - start))

        stats = [TextStats(text) for text in docs]
        _, g = timed(global_filter.check, stats)
        results, v = timed(validator.validate, stats)
        pairs = list(zip(stats, results))
        _, s = timed(lambda p: scorer.score(p[0], p[1].details["signals"]), pairs)
        print(f"{domain:<11}{g:>10.0f}{v:>11.0f}{s:>9.0f}{full:>11.0f}")


def main() -> None:
    args = sys.argv[1:]
    samples_path = BACKEND / "data" / "samples.json"
    if "--samples" in args:
        index = args.index("--samples")
        samples_path = Path(args[index + 1])
        del args[index:index + 2]
    mode = args[0] if args else None

    if not samples_path.exists():
        raise SystemExit(f"{samples_path} not found. Run: python scripts/fetch_samples.py")
    cfg = load_config()
    data: dict[str, list[str]] = json.loads(samples_path.read_text(encoding="utf-8"))

    if mode == "scores":
        report_scores(cfg, data)
    elif mode == "classifier":
        report_classifier(cfg, data)
    elif mode == "speed":
        report_speed(cfg, data)
    else:
        report_matrix(cfg, data, mode)


if __name__ == "__main__":
    main()
