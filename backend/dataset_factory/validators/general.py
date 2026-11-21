"""General-knowledge validator.

The most permissive of the six, as in the paper: it accepts well-formed,
factual, neutral prose of any topic (encyclopedia-style writing).
"""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import (
    DomainValidator,
    mean_sentence_length,
    numbers_per_100_words,
    per_100_words,
    person_ratio,
    phrase_count,
    phrases_present,
    proper_nouns_per_100_words,
)

_ENCYCLOPEDIC_PHRASES = (
    " is a ", " is an ", " was a ", " was an ", " are the ", "refers to", "known as",
    "also called", "defined as", "consists of", "born in", "born on", "died in",
    "died on", "founded in", "established in", "located in", "according to",
    "for example", "such as", "century", "population",
)
_YEAR_RE = re.compile(r"(?<!\d)(?:1\d|20)\d\d(?!\d)")
_PROMOTIONAL_PHRASES = (
    "buy now", "order now", "order today", "shop now", "best price", "lowest price",
    "free shipping", "limited time offer", "call us today", "contact us today",
    "our products", "our services", "our team", "our store", "we offer", "we provide",
    "sign up today", "money-back guarantee", "special offer", "discount code",
)


class GeneralValidator(DomainValidator):
    domain = "general"
    CHECKS = ("well_formed", "factual_density", "neutral_voice", "not_promotional")
    DEFAULTS = {
        "min_sentences": 3,
        "min_sentence_length": 8,
        "max_sentence_length": 50,
        "min_facts_per_100w": 3.0,
        "max_person_ratio": 0.03,
        "max_promotional_hits": 1,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        text = stats.text
        lower = stats.lower
        encyclopedic = phrase_count(lower, _ENCYCLOPEDIC_PHRASES)
        return {
            "term_density": encyclopedic / stats.n_words if stats.n_words else 0.0,
            "citations_per_100w": per_100_words(len(_YEAR_RE.findall(text)), stats.n_words),
            "structure_cues": phrases_present(lower, _ENCYCLOPEDIC_PHRASES),
            "facts_per_100w": numbers_per_100_words(stats) + proper_nouns_per_100_words(stats),
            "sentences": stats.n_sentences,
            "sentence_length": mean_sentence_length(stats),
            "person_ratio": person_ratio(stats),
            "promotional_hits": phrase_count(lower, _PROMOTIONAL_PHRASES),
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "well_formed": (
                s["sentences"] >= t.min_sentences
                and t.min_sentence_length <= s["sentence_length"] <= t.max_sentence_length
            ),
            "factual_density": s["facts_per_100w"] >= t.min_facts_per_100w,
            "neutral_voice": s["person_ratio"] <= t.max_person_ratio,
            "not_promotional": s["promotional_hits"] <= t.max_promotional_hits,
        }
