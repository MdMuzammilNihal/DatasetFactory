"""Open every configured source, pull a few documents and curate them.

Usage (from the backend folder; needs an internet connection unless --synthetic):

    python scripts/check_sources.py                  # every domain, 60 documents per source
    python scripts/check_sources.py -n 200 legal     # more documents, one domain
    python scripts/check_sources.py --shuffle-buffer 0   # corpus order: much quicker to start
    python scripts/check_sources.py --synthetic      # offline: the built-in generator

For each domain and each source routed to it, this reads documents through the
Factory or Generator agent exactly as a run would, passes them to the Scorer
agent and removes duplicates. It prints how many were accepted, their licences
and the commonest reasons for rejection. A source that cannot be opened is
reported and the check carries on.

"yield" is accepted / read. Small samples: treat the percentages as rough.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import replace
import itertools
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dataset_factory.agents import FactoryAgent, GeneratorAgent, ScorerAgent  # noqa: E402
from dataset_factory.config import load_config  # noqa: E402
from dataset_factory.dedup import build_index  # noqa: E402
from dataset_factory.sources import domain_routes, source_specs  # noqa: E402
from dataset_factory.validators import DOMAINS  # noqa: E402


def check(agent, scorer, index, limit):
    """Read up to ``limit`` documents from the agent's sources and curate them."""
    labels: collections.Counter = collections.Counter()
    licenses: collections.Counter = collections.Counter()
    words = read = 0
    started = time.perf_counter()
    for item in itertools.islice(agent.documents(), limit):
        read += 1
        if isinstance(item, str):               # dropped by the Generator agent's prefilter
            labels[item] += 1
            continue
        evaluation = scorer.evaluate(item)
        if evaluation.accepted and index.check_and_add(
            evaluation.signature, evaluation.content_hash
        ).duplicate:
            labels["duplicate"] += 1
            continue
        labels[evaluation.label] += 1
        if evaluation.accepted:
            licenses[evaluation.record["provenance"]["license"]] += 1
            words += evaluation.record["n_words"]
    return read, labels, licenses, words, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("domains", nargs="*", help="domains to check (default: all)")
    parser.add_argument("-n", type=int, default=60, help="documents to read per source")
    parser.add_argument("--synthetic", action="store_true", help="use the built-in generator")
    parser.add_argument("--config", default=None, help="configuration file")
    parser.add_argument("--shuffle-buffer", type=int, default=None, metavar="N",
                        help="override every source's shuffle buffer (0 reads in corpus "
                             "order, which starts much sooner)")
    args = parser.parse_args()

    config = load_config(args.config)
    routes = domain_routes(config, DOMAINS)
    synthetic_key = next((k for k, s in source_specs(config).items() if s.kind == "synthetic"), None)
    domains = args.domains or list(routes)
    unknown = [d for d in domains if d not in routes]
    if unknown:
        parser.error(f"unknown domain(s): {', '.join(unknown)}")

    failed = 0
    print(f"config {config.fingerprint}, {args.n} documents per source\n")
    print(f"{'domain':<11}{'agent':<10}{'source':<17}{'read':>5}{'kept':>5}{'yield':>7}"
          f"{'words':>7}{'secs':>6}  licences / rejections")
    for domain in domains:
        route = routes[domain]
        scorer = ScorerAgent(domain, config)
        index = build_index(config)
        for cls, keys in ((FactoryAgent, route.factory), (GeneratorAgent, route.generator)):
            for key in ([synthetic_key] if args.synthetic else keys):
                agent = cls(domain, config, sources=[key], max_errors=1)
                if args.shuffle_buffer is not None:
                    agent.specs[key] = replace(agent.specs[key], shuffle_buffer=args.shuffle_buffer)
                read, labels, licenses, words, seconds = check(agent, scorer, index, args.n)
                kept = labels.pop("accepted", 0)
                share = f"{100 * kept / read:.0f}%" if read else "-"
                mean_words = f"{words // kept}" if kept else "-"
                notes = ", ".join(f"{name} x{n}" for name, n in licenses.most_common()) or "(none kept)"
                print(f"{domain:<11}{agent.role:<10}{key:<17}{read:>5}{kept:>5}{share:>7}"
                      f"{mean_words:>7}{seconds:>6.0f}  {notes}", flush=True)
                if labels:
                    print(" " * 68 + "rejected: "
                          + ", ".join(f"{name} x{n}" for name, n in labels.most_common(4)))
                for error in agent.errors:
                    failed += 1
                    print(" " * 68 + f"ERROR: {error.error}")
    if failed:
        print(f"\n{failed} source error(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
