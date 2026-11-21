"""Learned quality classifier.

The rule-based validators check for explicit cues. This classifier adds a
learned judgement on top: for each domain, a small linear model is trained to
tell documents from trusted domain corpora apart from random web text and from
other domains. It outputs a probability between 0 and 1, and a document must
reach ``min_probability`` to be accepted (stage 4 of the pipeline, after the
quality score).

What it does and does not judge: it recognises whether a document reads like
the trusted text of its domain. That catches things the rules cannot express,
such as a maths paper full of braces passing for source code. It is not a
fine-grained ranking of quality among documents that are already in-domain.

The design follows what worked best in the DataComp-LM study of filtering
methods: a fastText-style linear classifier over word and word-pair features.
It needs no GPU, takes a few tens of microseconds per document, and runs
inside every worker process.

How a document becomes features:
  1. take up to ``max_words`` words (half from the start, half from the middle),
  2. hash every word and every adjacent word pair into one of ``dim`` buckets,
  3. the document is the set of buckets it touches, scaled to unit length.
The model is one weight per bucket plus a bias; the probability is the logistic
function of their weighted sum.

Training lives in scripts/train_classifier.py; this module only needs NumPy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import Config, ConfigError
from .hashing import word_hash as _word_hash
from .models import FilterResult
from .textstats import TextStats

STAGE = "classifier"
FEATURE_VERSION = 1
DEFAULT_DIM = 1 << 18           # 262,144 buckets
DEFAULT_MAX_WORDS = 400

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Mark-up left behind by specific corpora (for example "@xmath" where the arXiv
# set removed a formula). Ignored so the model learns the writing, not which
# corpus a document came from.
ARTIFACT_TOKENS = frozenset({"xmath", "xcite", "cnn"})

_MIX = np.uint64(0x9E3779B97F4A7C15)
_SHIFT = np.uint64(20)



def select_words(words: list[str], max_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    """The words the classifier looks at: start and middle of the document."""
    if len(words) > max_words:
        half = max_words // 2
        middle = len(words) // 2
        words = words[:half] + words[middle:middle + half]
    return [w for w in words if w not in ARTIFACT_TOKENS]


def featurize(words: list[str], max_words: int = DEFAULT_MAX_WORDS, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Return the sorted, distinct feature buckets of a document."""
    selected = select_words(words, max_words)
    if not selected:
        return np.empty(0, dtype=np.int64)
    hashes = np.fromiter(map(_word_hash, selected), dtype=np.uint64, count=len(selected))
    mask = np.uint64(dim - 1)
    unigrams = hashes & mask
    if len(selected) > 1:
        bigrams = ((hashes[:-1] * _MIX + hashes[1:]) >> _SHIFT) & mask
        buckets = np.concatenate((unigrams, bigrams))
    else:
        buckets = unigrams
    return np.unique(buckets).astype(np.int64)


def bucket_of(token: str, dim: int = DEFAULT_DIM) -> int:
    """Bucket of a single word or of a "word1 word2" pair (used to explain a model)."""
    parts = token.split(" ")
    if len(parts) == 1:
        return _word_hash(parts[0]) & (dim - 1)
    # The same arithmetic as featurize(), written with Python integers and an
    # explicit 64-bit wrap-around.
    mixed = (_word_hash(parts[0]) * int(_MIX) + _word_hash(parts[1])) & 0xFFFF_FFFF_FFFF_FFFF
    return (mixed >> int(_SHIFT)) & (dim - 1)


class QualityClassifier:
    """A trained per-domain classifier."""

    def __init__(self, domain: str, weights: np.ndarray, bias: float, meta: Mapping[str, Any]) -> None:
        dim = int(weights.shape[0])
        if dim & (dim - 1):
            raise ValueError("the number of buckets must be a power of two")
        self.domain = domain
        self.weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.bias = float(bias)
        self.dim = dim
        self.meta = dict(meta)
        self.max_words = int(self.meta.get("max_words", DEFAULT_MAX_WORDS))
        self.min_probability = 0.5

    def check(self, stats: TextStats) -> FilterResult:
        """Stage 4 of the pipeline: pass when the probability reaches the minimum."""
        probability = round(self.probability(stats), 4)
        passed = probability >= self.min_probability
        return FilterResult(
            passed=passed,
            stage=STAGE,
            reason=None if passed else "low_learned_quality",
            details={"probability": probability, "limit": self.min_probability},
        )

    def probability(self, stats: TextStats) -> float:
        """Probability (0-1) that the document belongs to this domain's trusted text."""
        buckets = featurize(stats.words, self.max_words, self.dim)
        if buckets.size == 0:
            return 0.0
        z = float(self.weights[buckets].sum()) / math.sqrt(buckets.size) + self.bias
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def probability_text(self, text: str) -> float:
        return self.probability(TextStats(text))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float16),   # halves the file; ample precision
            bias=np.float64(self.bias),
            meta=np.array(json.dumps({**self.meta, "domain": self.domain})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "QualityClassifier":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("feature_version") != FEATURE_VERSION:
                raise ConfigError(
                    f"{path} was trained with a different feature version; "
                    "retrain with scripts/train_classifier.py"
                )
            return cls(meta["domain"], data["weights"].astype(np.float32), float(data["bias"]), meta)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "enabled": True,
    "min_probability": 0.5,
    "min_probability_by_domain": {},   # per-domain exceptions to min_probability
    "model_dir": "models",
}


def _is_probability(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value <= 1


def classifier_settings(config: Config) -> dict[str, Any]:
    """Validated ``[classifier]`` settings with defaults filled in."""
    raw = dict(config.classifier)
    unknown = sorted(set(raw) - set(_DEFAULTS))
    if unknown:
        raise ConfigError(f"[classifier] unknown setting(s): {', '.join(unknown)}")
    settings = {**_DEFAULTS, **raw}
    if not isinstance(settings["enabled"], bool):
        raise ConfigError("[classifier] enabled must be true or false")
    if not _is_probability(settings["min_probability"]):
        raise ConfigError("[classifier] min_probability must be a number between 0 and 1")
    by_domain = settings["min_probability_by_domain"]
    if not isinstance(by_domain, Mapping) or not all(map(_is_probability, by_domain.values())):
        raise ConfigError(
            "[classifier.min_probability_by_domain] must map domains to numbers between 0 and 1"
        )
    if not isinstance(settings["model_dir"], str):
        raise ConfigError("[classifier] model_dir must be text")
    return settings


def model_path(domain: str, config: Config) -> Path:
    model_dir = Path(classifier_settings(config)["model_dir"])
    if not model_dir.is_absolute():
        model_dir = BACKEND_DIR / model_dir
    return model_dir / f"quality_{domain}.npz"


def load_classifier(domain: str, config: Config) -> QualityClassifier | None:
    """The domain's classifier, or None when the classifier is switched off."""
    settings = classifier_settings(config)
    if not settings["enabled"]:
        return None
    path = model_path(domain, config)
    if not path.exists():
        raise ConfigError(
            f"classifier model not found: {path}. Train it with "
            "'python scripts/train_classifier.py' or set enabled = false under [classifier]."
        )
    classifier = QualityClassifier.load(path)
    classifier.min_probability = float(
        settings["min_probability_by_domain"].get(domain, settings["min_probability"])
    )
    return classifier
