"""High-quality news validator: attribution, factual anchors, article structure."""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import (
    DomainValidator,
    count_terms,
    mean_sentence_length,
    numbers_per_100_words,
    per_100_words,
    person_ratio,
    phrase_count,
    phrases_present,
    proper_nouns_per_100_words,
)

_ATTRIBUTION_WORDS = frozenset(
    """
    said says told reported reports announced stated confirmed added explained
    warned argued noted spokesman spokeswoman spokesperson official officials
    sources authorities interview
    """.split()
)
_ATTRIBUTION_PHRASES = ("according to", "in a statement", "press conference", "news conference")
_QUOTE_RE = re.compile(r"[\"“][^\"”]{15,400}[\"”]")
_DATELINE_RE = re.compile(
    r"^\s*(?:[A-Z][A-Za-z .]+,\s+)?[A-Z]{3,}[A-Z .,]*\s*\((?:[A-Z][A-Za-z]+)\)\s*-{1,2}"  # LONDON (Reuters) --
    r"|^\s*\((?:CNN|Reuters|AP|AFP|BBC)\)\s*-{1,2}"
    r"|^\s*By\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
)
_DATE_WORDS = frozenset(
    """
    monday tuesday wednesday thursday friday saturday sunday january february
    april june july august september october november december yesterday
    """.split()
)
_CLICKBAIT_PHRASES = (
    "you won't believe", "you wont believe", "will blow your mind", "what happens next",
    "doctors hate", "one weird trick", "shocking truth", "click here", "things you need to know",
    "reasons why you", "!!",
)


class NewsValidator(DomainValidator):
    domain = "news"
    CHECKS = (
        "attribution",
        "factual_anchors",
        "article_structure",
        "no_clickbait",
        "reporting_voice",
    )
    DEFAULTS = {
        "min_attributions": 3,
        "min_attributions_per_100w": 1.2,
        "min_proper_nouns_per_100w": 2.5,
        "min_numbers_per_100w": 0.3,
        "min_sentences": 5,
        "min_sentence_length": 10,
        "max_sentence_length": 45,
        "max_clickbait_hits": 0,
        "max_person_ratio": 0.05,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        text = stats.text
        lower = stats.lower
        reporting = count_terms(stats, _ATTRIBUTION_WORDS) + phrase_count(
            lower, _ATTRIBUTION_PHRASES
        )
        attributions = reporting + len(_QUOTE_RE.findall(text))
        return {
            # For news, "terminology" is the density of reporting language.
            "term_density": reporting / stats.n_words if stats.n_words else 0.0,
            "citations_per_100w": per_100_words(attributions, stats.n_words),
            "structure_cues": int(bool(_DATELINE_RE.search(text[:200])))
            + int(count_terms(stats, _DATE_WORDS) > 0)
            + int(len(stats.lines) >= 3),
            "attributions": attributions,
            "proper_nouns_per_100w": proper_nouns_per_100_words(stats),
            "numbers_per_100w": numbers_per_100_words(stats),
            "sentences": stats.n_sentences,
            "sentence_length": mean_sentence_length(stats),
            "clickbait_hits": phrases_present(lower, _CLICKBAIT_PHRASES),
            "person_ratio": person_ratio(stats),
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "attribution": (
                s["attributions"] >= t.min_attributions
                and s["citations_per_100w"] >= t.min_attributions_per_100w
            ),
            "factual_anchors": (
                s["proper_nouns_per_100w"] >= t.min_proper_nouns_per_100w
                and s["numbers_per_100w"] >= t.min_numbers_per_100w
            ),
            "article_structure": (
                s["sentences"] >= t.min_sentences
                and t.min_sentence_length <= s["sentence_length"] <= t.max_sentence_length
            ),
            "no_clickbait": s["clickbait_hits"] <= t.max_clickbait_hits,
            "reporting_voice": s["person_ratio"] <= t.max_person_ratio,
        }
