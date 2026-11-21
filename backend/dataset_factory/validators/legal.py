"""Legal case-law validator: citations, legal terminology and case structure."""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import DomainValidator, count_terms, per_100_words, phrase_count, term_density
from .lexicons import LEGAL

# Reporter citations such as "410 U.S. 113", "5 Ala. 123", "123 F.2d 456",
# "45 N.E.2d 678": volume, abbreviated reporter, page.
_REPORTER_RE = re.compile(
    r"\d{1,4}\s+(?:[A-Z][A-Za-z]{0,5}\.\s?){1,4}(?:[234](?:d|th)\.?\s)?\s?\d{1,5}\b"
)
# Case names: "Roe v. Wade", "State vs. Smith".
_CASE_NAME_RE = re.compile(r" vs?\.\s+[A-Z]")
# Statutes and code sections: "§ 1983", "Section 12", "Sec. 4", "Article IV".
_SECTION_RE = re.compile(r"sec(?:tion|\.)\s+\d")
_ARTICLE_RE = re.compile(r"art(?:icle|\.)\s+[ivx\d]")
_STATUTE_PHRASES = ("u.s.c.", "u. s. c.", "rev. stat", "gen. stat", "code §", "stat. ")

_PARTY_WORDS = frozenset(
    """
    plaintiff plaintiffs defendant defendants appellant appellants appellee
    appellees petitioner petitioners respondent respondents complainant
    complainants claimant claimants prosecution relator
    """.split()
)
_COURT_WORDS = frozenset(
    "court courts tribunal judge judges justice justices jury chancellor magistrate".split()
)
_DISPOSITION_WORDS = frozenset(
    """
    affirmed reversed remanded vacated dismissed overruled sustained denied
    granted decree verdict ordered adjudged
    """.split()
)
_DISPOSITION_PHRASES = (
    "judgment is", "judgment was", "judgment for", "judgment of", "judgment against",
    "it is ordered", "it is so ordered", "we hold", "held that", "the court held",
    "the court found", "the court ruled", "we conclude", "opinion of the court",
    "delivered the opinion",
)


class LegalValidator(DomainValidator):
    domain = "legal"
    CHECKS = ("citations", "legal_terminology", "case_structure")
    DEFAULTS = {
        "min_citations": 2,
        "min_term_density": 0.03,
        "min_party_terms": 1,
        "min_court_terms": 1,
        "min_disposition_terms": 1,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        text = stats.text
        lower = stats.lower
        citations = (
            len(_REPORTER_RE.findall(text))
            + len(_CASE_NAME_RE.findall(text))
            + text.count("§")
            + len(_SECTION_RE.findall(lower))
            + len(_ARTICLE_RE.findall(lower))
            + phrase_count(lower, _STATUTE_PHRASES)
        )
        parties = count_terms(stats, _PARTY_WORDS)
        courts = count_terms(stats, _COURT_WORDS)
        dispositions = count_terms(stats, _DISPOSITION_WORDS) + phrase_count(
            lower, _DISPOSITION_PHRASES
        )
        return {
            "term_density": term_density(stats, LEGAL),
            "citations": citations,
            "citations_per_100w": per_100_words(citations, stats.n_words),
            "party_terms": parties,
            "court_terms": courts,
            "disposition_terms": dispositions,
            "structure_cues": int(parties > 0) + int(courts > 0) + int(dispositions > 0),
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "citations": s["citations"] >= t.min_citations,
            "legal_terminology": s["term_density"] >= t.min_term_density,
            "case_structure": (
                s["party_terms"] >= t.min_party_terms
                and s["court_terms"] >= t.min_court_terms
                and s["disposition_terms"] >= t.min_disposition_terms
            ),
        }
