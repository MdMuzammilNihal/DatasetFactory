"""Scientific-paper validator: section organisation, citation density, terminology."""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import DomainValidator, distinct_terms, per_100_words, phrases_present, term_density
from .lexicons import SCIENTIFIC

_SECTION_WORDS = frozenset(
    """
    abstract introduction background preliminaries methodology methods
    experiments results discussion conclusion conclusions references
    bibliography acknowledgments acknowledgements acknowledgment appendix
    """.split()
)
_SECTION_PHRASES = (
    "related work", "materials and methods", "experimental setup", "in this paper",
    "in this work", "in this study", "in this article", "in this letter", "we propose",
    "we present", "we show", "we demonstrate", "we investigate", "we introduce",
    "we derive", "we consider", "we study", "we find", "the rest of this paper",
    "the rest of the paper", "the remainder of this paper", "@xmath", "\\begin{",
)
_FIGURE_RE = re.compile(r"fig(?:ure|\.)\s?\d")
_TABLE_RE = re.compile(r"table\s?\d")
_NUMBERED_SECTION_RE = re.compile(r"section\s?\d")
_EQUATION_RE = re.compile(r"eq(?:uation|\.)\s?\(?\d")

_NUMERIC_CITATION_RE = re.compile(r"\[\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*\]")     # [1], [2, 3]
_LABEL_CITATION_RE = re.compile(r"\[[A-Z][A-Za-z+]{1,7}\d{0,4}\]")            # [Kat], [BK98]
_AUTHOR_YEAR_RE = re.compile(r"\(\s*[A-Z][A-Za-z\-]+,?\s+(?:19|20)\d{2}[a-z]?\s*\)")
_CITATION_PHRASES = ("et al", "@xcite", "\\cite", "arxiv:", "doi:10.", "doi: 10.")


class ScientificValidator(DomainValidator):
    domain = "scientific"
    CHECKS = ("section_structure", "citation_density", "terminology", "substance")
    DEFAULTS = {
        "min_structure_cues": 3,
        "min_citations": 3,
        "min_citations_per_100w": 0.10,
        "min_term_density": 0.035,
        "min_words": 150,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        text = stats.text
        lower = stats.lower
        citations = (
            len(_NUMERIC_CITATION_RE.findall(text))
            + len(_LABEL_CITATION_RE.findall(text))
            + len(_AUTHOR_YEAR_RE.findall(text))
            + sum(lower.count(phrase) for phrase in _CITATION_PHRASES)
        )
        structure = (
            distinct_terms(stats, _SECTION_WORDS)
            + phrases_present(lower, _SECTION_PHRASES)
            + int(bool(_FIGURE_RE.search(lower)))
            + int(bool(_TABLE_RE.search(lower)))
            + int(bool(_NUMBERED_SECTION_RE.search(lower)))
            + int(bool(_EQUATION_RE.search(lower)))
        )
        return {
            "term_density": term_density(stats, SCIENTIFIC),
            "citations": citations,
            "citations_per_100w": per_100_words(citations, stats.n_words),
            "structure_cues": structure,
            "n_words": stats.n_words,
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "section_structure": s["structure_cues"] >= t.min_structure_cues,
            "citation_density": (
                s["citations"] >= t.min_citations
                and s["citations_per_100w"] >= t.min_citations_per_100w
            ),
            "terminology": s["term_density"] >= t.min_term_density,
            "substance": s["n_words"] >= t.min_words,
        }
