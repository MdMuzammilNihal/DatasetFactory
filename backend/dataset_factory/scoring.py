"""Stage 3: composite quality scoring.

For a document x that passed both filters, the paper defines

    Q(x) = sum_i  w_i * f_i(x)        and accepts x when   Q(x) >= tau_d

where the f_i are quality features, the w_i are per-domain weights and tau_d is
the domain's acceptance threshold. This module implements exactly that.

Every feature is a value between 0 and 1. Weights sum to 1, so Q(x) is also
between 0 and 1.

Length normalisation: every feature is either a rate (per 100 words, or a
ratio) or saturates at a target value, so a document cannot raise its score
simply by being longer or by repeating itself.

The features reuse the signals already measured by the domain validator, so
scoring adds very little work per document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import Config, ConfigError
from .textstats import TextStats
from .validators.base import numbers_per_100_words

FEATURES = (
    "terminology",          # density of domain vocabulary
    "citations",            # citations, attributions or comments per 100 words
    "structure",            # how many of the domain's structural cues are present
    "readability",          # sentence and word lengths in a normal range
    "factual_density",      # numbers and named entities per 100 words
    "lexical_diversity",    # variety of vocabulary
    "non_repetition",       # absence of repeated phrases (the repetition penalty)
    "source_authority",     # how trustworthy the source corpus is
    "syntactic_validity",   # for source code: well-formed blocks and delimiters
)

TARGETS = ("term_density", "citations_per_100w", "structure_cues", "facts_per_100w")


def _saturate(value: float) -> float:
    return 0.0 if value <= 0.0 else 1.0 if value >= 1.0 else value


def _plateau(value: float, zero_low: float, full_low: float, full_high: float, zero_high: float) -> float:
    """1 inside [full_low, full_high], falling linearly to 0 at the outer limits."""
    if value <= zero_low or value >= zero_high:
        return 0.0
    if value < full_low:
        return (value - zero_low) / (full_low - zero_low)
    if value > full_high:
        return (zero_high - value) / (zero_high - full_high)
    return 1.0


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    passed: bool
    threshold: float
    features: dict[str, float]


class QualityScorer:
    """Computes Q(x) for one domain."""

    def __init__(
        self,
        domain: str,
        settings: Mapping[str, Any],
        default_authority: float = 0.5,
    ) -> None:
        self.domain = domain
        where = f"scoring.{domain}"

        unknown = sorted(set(settings) - {"threshold", "weights", "targets", "disabled_features"})
        if unknown:
            raise ConfigError(f"[{where}] unknown setting(s): {', '.join(unknown)}")

        threshold = settings.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ConfigError(f"[{where}] threshold must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"[{where}] threshold must be between 0 and 1")
        self.threshold = float(threshold)

        weights = settings.get("weights")
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigError(f"[{where}] needs a [weights] table")
        bad = sorted(set(weights) - set(FEATURES))
        if bad:
            raise ConfigError(
                f"[{where}.weights] unknown feature(s): {', '.join(bad)}. "
                f"Valid features: {', '.join(FEATURES)}"
            )
        for name, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
                raise ConfigError(f"[{where}.weights] {name} must be a non-negative number")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"[{where}.weights] must sum to 1 (they sum to {total:.4f})")

        targets = settings.get("targets")
        if not isinstance(targets, Mapping) or set(targets) != set(TARGETS):
            raise ConfigError(f"[{where}.targets] must define exactly: {', '.join(TARGETS)}")
        for name, target in targets.items():
            if isinstance(target, bool) or not isinstance(target, (int, float)) or target <= 0:
                raise ConfigError(f"[{where}.targets] {name} must be a positive number")
        self.targets = {name: float(value) for name, value in targets.items()}

        disabled = settings.get("disabled_features", [])
        if not isinstance(disabled, (list, tuple)) or set(disabled) - set(FEATURES):
            raise ConfigError(f"[{where}] disabled_features must list valid feature names")
        self.disabled = frozenset(disabled)

        # Switching a feature off (for ablation) redistributes its weight over
        # the remaining features so that Q(x) stays on the same 0-1 scale.
        active = {n: float(w) for n, w in weights.items() if w > 0 and n not in self.disabled}
        active_total = sum(active.values())
        if active_total <= 0:
            raise ConfigError(f"[{where}] every feature is disabled or has zero weight")
        self.weights = {name: weight / active_total for name, weight in active.items()}

        self.default_authority = default_authority

    # -- features ---------------------------------------------------------

    def _feature(
        self, name: str, stats: TextStats, signals: Mapping[str, float], authority: float
    ) -> float:
        targets = self.targets
        if name == "terminology":
            return _saturate(signals.get("term_density", 0.0) / targets["term_density"])
        if name == "citations":
            return _saturate(
                signals.get("citations_per_100w", 0.0) / targets["citations_per_100w"]
            )
        if name == "structure":
            return _saturate(signals.get("structure_cues", 0.0) / targets["structure_cues"])
        if name == "readability":
            sentence_length = stats.n_words / stats.n_sentences
            return _plateau(sentence_length, 4, 12, 28, 60) * _plateau(
                stats.mean_word_length, 2.5, 4.0, 6.5, 9.0
            )
        if name == "factual_density":
            facts = signals.get("facts_per_100w")
            if facts is None:
                numbers = signals.get("numbers_per_100w")
                if numbers is None:
                    numbers = numbers_per_100_words(stats)
                facts = numbers + signals.get("proper_nouns_per_100w", 0.0)
            return _saturate(facts / targets["facts_per_100w"])
        if name == "lexical_diversity":
            # Typical prose has 0.45-0.65 distinct words in its first 500.
            return _saturate((stats.unique_word_ratio - 0.20) / 0.35)
        if name == "non_repetition":
            return 1.0 - _saturate(stats.repeated_ngram_ratio / 0.30)
        if name == "source_authority":
            return _saturate(authority)
        if name == "syntactic_validity":
            blocks = _saturate(signals.get("structural_line_ratio", 0.0) / 0.60)
            balance = 1.0 - _saturate(signals.get("delimiter_imbalance", 1.0) / 0.10)
            functions = _saturate(signals.get("function_definitions", 0.0))
            return 0.5 * blocks + 0.3 * balance + 0.2 * functions
        raise KeyError(name)  # pragma: no cover - guarded by config validation

    def score(
        self,
        stats: TextStats,
        signals: Mapping[str, float],
        authority: float | None = None,
    ) -> ScoreResult:
        """Score a document from its text statistics and validator signals.

        ``authority`` is the trust level (0-1) of the corpus the document came
        from; the configured default is used when it is not given.
        """
        if authority is None:
            authority = self.default_authority
        features = {
            name: round(self._feature(name, stats, signals, authority), 4)
            for name in self.weights
        }
        score = round(sum(self.weights[name] * value for name, value in features.items()), 4)
        return ScoreResult(score, score >= self.threshold, self.threshold, features)


def build_scorer(domain: str, config: Config) -> QualityScorer:
    domains = {k: v for k, v in config.scoring.items() if isinstance(v, Mapping)}
    if domain not in domains:
        raise ConfigError(
            f"no [scoring.{domain}] section. Configured domains: {', '.join(sorted(domains))}"
        )
    default_authority = config.scoring.get("default_authority", 0.5)
    if isinstance(default_authority, bool) or not isinstance(default_authority, (int, float)):
        raise ConfigError("[scoring] default_authority must be a number")
    return QualityScorer(domain, domains[domain], float(default_authority))


def check_scoring_config(config: Config, domains: tuple[str, ...]) -> None:
    """Raise ``ConfigError`` unless every domain has a valid scoring section."""
    extra = sorted(
        k for k, v in config.scoring.items() if isinstance(v, Mapping) and k not in domains
    )
    if extra:
        raise ConfigError(f"[scoring] unknown domain(s): {', '.join(extra)}")
    for domain in domains:
        build_scorer(domain, config)
