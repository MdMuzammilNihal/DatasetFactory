"""Train the learned quality classifiers (one per domain).

Usage (from the backend folder; run scripts/fetch_classifier_data.py first):

    python scripts/train_classifier.py                 # all six domains
    python scripts/train_classifier.py medical legal   # only these

For each domain the positives are documents from trusted corpora of that
domain and the negatives are random web text plus the other domains' corpora.
One fifth of the data is held out for testing. The script prints, per domain:

  * accuracy and ROC AUC on the held-out fifth,
  * how often each corpus is accepted (probability >= 0.5),
  * the word features the model relies on most, so it can be inspected for
    shortcuts (for example keying on a corpus's formatting, not its content),
  * the same acceptance rates on data/samples.json, which the model never saw.

Models are written to models/quality_<domain>.npz and the numbers to
models/metrics.json. Training needs scikit-learn; using the models does not.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dataset_factory.classifier import (  # noqa: E402
    DEFAULT_DIM,
    DEFAULT_MAX_WORDS,
    FEATURE_VERSION,
    QualityClassifier,
    bucket_of,
    featurize,
    model_path,
    select_words,
)
from dataset_factory.config import derive_seed, load_config  # noqa: E402
from dataset_factory.filters import GlobalFilter  # noqa: E402
from dataset_factory.textstats import TextStats  # noqa: E402

DATA_DIR = BACKEND / "data" / "classifier"

POSITIVES = {
    "general": ("wikipedia",),
    "news": ("news_cnn", "news_ccnews"),
    "medical": ("med_pubmedqa", "med_pubmed"),
    "legal": ("legal_cap",),
    "scientific": ("sci_arxiv_sum", "sci_arxiv_cp"),
    "code": ("code_py", "code_java", "code_js", "code_go"),
}

# Corpora that are neither positive nor negative for a domain, because the
# right label is debatable. General knowledge legitimately includes news,
# science, medicine and law, so its negatives are web text and source code.
NEUTRAL = {
    "general": ("news_cnn", "news_ccnews", "med_pubmedqa", "med_pubmed", "legal_cap",
                "sci_arxiv_sum", "sci_arxiv_cp"),
    "scientific": ("med_pubmedqa", "med_pubmed"),
}

REGULARISATION_C = 4.0
MAX_PER_NEGATIVE_SET = 1500


def load_sets() -> dict[str, list[str]]:
    sets = {}
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            sets[path.stem] = [json.loads(line)["text"] for line in fh if line.strip()]
    if not sets:
        raise SystemExit(f"no data in {DATA_DIR}. Run: python scripts/fetch_classifier_data.py")
    return sets


def to_matrix(feature_lists: list[np.ndarray], dim: int) -> csr_matrix:
    """Rows are documents; each touched bucket has value 1/sqrt(buckets touched)."""
    indptr = np.zeros(len(feature_lists) + 1, dtype=np.int64)
    for i, buckets in enumerate(feature_lists):
        indptr[i + 1] = indptr[i] + buckets.size
    indices = np.concatenate(feature_lists) if feature_lists else np.empty(0, dtype=np.int64)
    values = np.concatenate(
        [np.full(b.size, 1.0 / np.sqrt(b.size), dtype=np.float32) for b in feature_lists]
    )
    return csr_matrix((values, indices, indptr), shape=(len(feature_lists), dim))


def top_tokens(weights: np.ndarray, vocabulary: dict[int, str], k: int = 12):
    known = np.array(sorted(vocabulary), dtype=np.int64)
    order = known[np.argsort(weights[known])]
    return [vocabulary[i] for i in order[::-1][:k]], [vocabulary[i] for i in order[:k]]


def train_domain(domain: str, sets: dict[str, list[str]], cfg) -> dict:
    global_filter = GlobalFilter(cfg.global_filter_for(domain))
    rng = np.random.default_rng(derive_seed(cfg.seed, "classifier", domain))
    positives = POSITIVES[domain]
    neutral = NEUTRAL.get(domain, ())

    features, labels, origin = [], [], []
    token_counts: Counter = Counter()
    for name, texts in sets.items():
        if name in neutral:
            continue
        label = 1 if name in positives else 0
        if label == 0 and len(texts) > MAX_PER_NEGATIVE_SET:
            picks = rng.choice(len(texts), MAX_PER_NEGATIVE_SET, replace=False)
            texts = [texts[i] for i in sorted(picks)]
        for text in texts:
            stats = TextStats(text)
            if not global_filter.check(stats).passed:
                continue  # the classifier only ever sees documents that passed stage 1
            buckets = featurize(stats.words, DEFAULT_MAX_WORDS, DEFAULT_DIM)
            if buckets.size == 0:
                continue
            features.append(buckets)
            labels.append(label)
            origin.append(name)
            words = select_words(stats.words, DEFAULT_MAX_WORDS)[:80]
            token_counts.update(set(words))
            token_counts.update({f"{a} {b}" for a, b in zip(words, words[1:])})

    # Label each bucket with the most common token that falls into it, so the
    # "strongest signs" below name the token that actually drives the weight.
    vocabulary: dict[int, str] = {}
    for token, count in token_counts.most_common():
        if count < 5:
            break
        vocabulary.setdefault(bucket_of(token), token)

    labels_arr = np.array(labels)
    origin_arr = np.array(origin)
    order = rng.permutation(len(labels))
    cut = int(len(order) * 0.8)
    train_idx, test_idx = order[:cut], order[cut:]

    matrix = to_matrix(features, DEFAULT_DIM)
    model = LogisticRegression(
        C=REGULARISATION_C, class_weight="balanced", solver="liblinear", max_iter=2000
    )
    model.fit(matrix[train_idx], labels_arr[train_idx])

    weights = model.coef_[0].astype(np.float32)
    bias = float(model.intercept_[0])
    meta = {
        "feature_version": FEATURE_VERSION,
        "max_words": DEFAULT_MAX_WORDS,
        "dim": DEFAULT_DIM,
        "trained_on": date.today().isoformat(),
        "positives": list(positives),
        "seed": cfg.seed,
        "config_fingerprint": cfg.fingerprint,
        "regularisation_c": REGULARISATION_C,
    }
    classifier = QualityClassifier(domain, weights, bias, meta)

    # Evaluate through the saved-and-reloaded model, i.e. exactly what the pipeline will run.
    path = model_path(domain, cfg)
    classifier.save(path)
    classifier = QualityClassifier.load(path)

    def probabilities(indices):
        out = np.empty(len(indices))
        for j, i in enumerate(indices):
            b = features[i]
            z = classifier.weights[b].sum() / np.sqrt(b.size) + classifier.bias
            out[j] = 1.0 / (1.0 + np.exp(-z))
        return out

    test_prob = probabilities(test_idx)
    test_labels = labels_arr[test_idx]
    accuracy = float(((test_prob >= 0.5) == test_labels).mean())
    auc = float(roc_auc_score(test_labels, test_prob))
    per_set = {}
    for name in sorted(set(origin)):
        mask = origin_arr[test_idx] == name
        if mask.any():
            per_set[name] = {
                "accepted": round(float((test_prob[mask] >= 0.5).mean()), 3),
                "n": int(mask.sum()),
                "positive": name in positives,
            }
    best, worst = top_tokens(weights, vocabulary)

    print(f"\n=== {domain} ===")
    print(f"train {len(train_idx)} / test {len(test_idx)} documents "
          f"({int(labels_arr.sum())} positive, {int((labels_arr == 0).sum())} negative)")
    print(f"held-out accuracy {accuracy:.3f}   ROC AUC {auc:.3f}")
    for name, row in per_set.items():
        kind = "own   " if row["positive"] else "other "
        print(f"  {kind}{name:<14} accepted {row['accepted']:.0%} of {row['n']}")
    print("  strongest signs FOR :", ", ".join(best))
    print("  strongest signs AGAINST:", ", ".join(worst))

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "train_documents": int(len(train_idx)),
        "test_documents": int(len(test_idx)),
        "per_corpus": per_set,
        "top_positive_tokens": best,
        "top_negative_tokens": worst,
        "model_file": path.name,
        "model_kb": round(path.stat().st_size / 1024),
    }


def independent_check(domains: list[str], cfg) -> dict:
    """Acceptance on data/samples.json, which shares no rows with the training data."""
    samples_path = BACKEND / "data" / "samples.json"
    if not samples_path.exists():
        return {}
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    names = list(data)
    print("\n=== independent check on data/samples.json (never used in training) ===")
    print("share of each sample set with probability >= 0.5")
    print(f"{'classifier':<11}" + "".join(f"{n[:12]:>13}" for n in names))
    out = {}
    for domain in domains:
        classifier = QualityClassifier.load(model_path(domain, cfg))
        row = {}
        for name in names:
            probs = [classifier.probability_text(t) for t in data[name]]
            row[name] = round(float(np.mean([p >= 0.5 for p in probs])), 3) if probs else None
        out[domain] = row
        print(f"{domain:<11}" + "".join(f"{row[n]:>13.0%}" for n in names))
    return out


def main() -> None:
    cfg = load_config()
    domains = sys.argv[1:] or list(POSITIVES)
    unknown = [d for d in domains if d not in POSITIVES]
    if unknown:
        raise SystemExit(f"unknown domain(s): {', '.join(unknown)}")
    sets = load_sets()
    print("training data:", ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    metrics_path = model_path(domains[0], cfg).parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    for domain in domains:
        missing = [name for name in POSITIVES[domain] if not sets.get(name)]
        if len(missing) == len(POSITIVES[domain]):
            raise SystemExit(f"no positive data for '{domain}' (missing {', '.join(missing)})")
        metrics[domain] = train_domain(domain, sets, cfg)
    check = independent_check(domains, cfg)
    for domain, row in check.items():
        metrics[domain]["independent_samples_accepted"] = row
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nmodels and metrics saved in {metrics_path.parent}")


if __name__ == "__main__":
    main()
