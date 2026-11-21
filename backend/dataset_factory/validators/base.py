"""Stage 2 framework: domain-specific validators, V_d(x) in the paper.

A validator measures a document once (``analyze``), turning it into a small
dictionary of numeric *signals*, then applies a list of named checks to those
signals. A document passes when every enabled check passes.

Keeping measurement and decision separate has two benefits: the scorer in the
next stage reuses the same signals without re-reading the text, and any check
can be switched off by name to reproduce the paper's leave-one-out ablation.

Speed matters here because this runs on every document that survives stage 1.
Single-word cues are therefore tested with dictionary lookups against the
document's word counts, multi-word cues with plain substring search, and
regular expressions are kept for patterns that genuinely need them.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, ClassVar, Iterable, Mapping

from ..config import ConfigError
from ..models import FilterResult
from ..textstats import TextStats

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
# A capitalised word that does not start a sentence: a cheap proxy for named
# entities (people, places, organisations).
_PROPER_NOUN_RE = re.compile(r"[a-z,;:] [A-Z][a-z]{2,}")
_FIRST_SECOND_PERSON = frozenset(
    "i me my mine myself we us our ours you your yours yourself".split()
)


def per_100_words(count: float, n_words: int) -> float:
    return count * 100.0 / n_words if n_words else 0.0


def count_terms(stats: TextStats, lexicon: frozenset[str]) -> int:
    """Total occurrences of vocabulary words in the document."""
    return sum(c for w, c in stats.word_counts.items() if w in lexicon)


def distinct_terms(stats: TextStats, lexicon: frozenset[str]) -> int:
    """How many different vocabulary words occur in the document."""
    return sum(1 for w in stats.word_counts if w in lexicon)


def term_density(stats: TextStats, lexicon: frozenset[str]) -> float:
    """Share of the document's words that belong to a domain vocabulary."""
    return count_terms(stats, lexicon) / stats.n_words if stats.n_words else 0.0


def phrases_present(lower_text: str, phrases: Iterable[str]) -> int:
    """How many of the given (lower-case) phrases occur at least once."""
    return sum(1 for phrase in phrases if phrase in lower_text)


def phrase_count(lower_text: str, phrases: Iterable[str]) -> int:
    """Total occurrences of the given (lower-case) phrases."""
    return sum(lower_text.count(phrase) for phrase in phrases)


def numbers_per_100_words(stats: TextStats) -> float:
    return per_100_words(len(_NUMBER_RE.findall(stats.text)), stats.n_words)


def proper_nouns_per_100_words(stats: TextStats) -> float:
    return per_100_words(len(_PROPER_NOUN_RE.findall(stats.text)), stats.n_words)


def person_ratio(stats: TextStats) -> float:
    """Share of first- and second-person pronouns (a sign of chatty or sales text)."""
    return term_density(stats, _FIRST_SECOND_PERSON)


def mean_sentence_length(stats: TextStats) -> float:
    return stats.n_words / stats.n_sentences


class DomainValidator:
    """Base class. Subclasses define ``domain``, ``CHECKS``, ``DEFAULTS``,
    ``analyze`` and ``run_checks``."""

    domain: ClassVar[str] = ""
    CHECKS: ClassVar[tuple[str, ...]] = ()
    DEFAULTS: ClassVar[dict[str, float]] = {}

    def __init__(self, settings: Mapping[str, Any] | None = None) -> None:
        values = dict(settings or {})
        where = f"validators.{self.domain}"

        disabled = values.pop("disabled_checks", [])
        if not isinstance(disabled, (list, tuple)) or not all(
            isinstance(name, str) for name in disabled
        ):
            raise ConfigError(f"[{where}] disabled_checks must be a list of check names")
        bad = sorted(set(disabled) - set(self.CHECKS))
        if bad:
            raise ConfigError(
                f"[{where}] unknown check name(s): {', '.join(bad)}. "
                f"Valid names: {', '.join(self.CHECKS)}"
            )

        unknown = sorted(set(values) - set(self.DEFAULTS))
        if unknown:
            raise ConfigError(f"[{where}] unknown setting(s): {', '.join(unknown)}")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"[{where}] {key} must be a number")

        self.disabled = frozenset(disabled)
        self.t = SimpleNamespace(**{**self.DEFAULTS, **values})  # thresholds

    def analyze(self, stats: TextStats) -> dict[str, float]:
        """Measure the document. Must include the three signals every domain
        shares with the scorer: ``term_density``, ``citations_per_100w`` and
        ``structure_cues``."""
        raise NotImplementedError

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        """Map each name in ``CHECKS`` to whether the document passes it."""
        raise NotImplementedError

    def validate(self, stats: TextStats) -> FilterResult:
        signals = self.analyze(stats)
        outcomes = self.run_checks(signals)
        failed = [
            name for name in self.CHECKS if name not in self.disabled and not outcomes[name]
        ]
        return FilterResult(
            passed=not failed,
            stage=self.domain,
            reason=failed[0] if failed else None,
            details={"signals": signals, "failed": failed},
        )

    def validate_text(self, text: str) -> FilterResult:
        return self.validate(TextStats(text))
