"""Medical validator.

Its checks are named so that each can be ablated on its own:
clinical structure, factual grounding, educational value, topic coherence and
verifiable claims, plus a misinformation screen.
"""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import (
    DomainValidator,
    distinct_terms,
    numbers_per_100_words,
    per_100_words,
    phrases_present,
)
from .lexicons import MEDICAL

# Clinical word endings (hepatitis, anemia, appendectomy, ...).
_SUFFIXES = (
    "itis", "osis", "emia", "aemia", "ectomy", "otomy", "ostomy", "opathy", "plasty",
    "oscopy", "algia", "uria", "plegia", "trophy", "carcinoma", "sarcoma", "lymphoma",
    "melanoma", "adenoma", "cyte", "cytes",
)
_SUFFIX_EXCEPTIONS = frozenset({"academia", "catastrophy"})

# Section labels of structured abstracts and clinical notes ("METHODS: ...").
_SECTION_LABELS = tuple(
    label + ":"
    for label in (
        "background", "objective", "objectives", "aim", "aims", "purpose", "introduction",
        "methods", "method", "materials and methods", "patients and methods", "design",
        "setting", "participants", "patients", "intervention", "interventions",
        "measurements", "main outcome measures", "results", "findings", "conclusion",
        "conclusions", "discussion", "case report", "case presentation",
        "history of present illness", "chief complaint", "physical examination",
        "assessment", "impression", "plan", "diagnosis", "follow-up",
    )
)
# Phrases that mark the same moves (aim, method, result, conclusion) in
# abstracts written as a single paragraph.
_STRUCTURE_PHRASES = (
    "we report", "we describe", "we present", "we studied", "we examined", "we evaluated",
    "we investigated", "we conducted", "we compared", "we analyzed", "we analysed",
    "we assessed", "we reviewed", "we found", "we conclude", "the aim of", "the purpose of",
    "the objective of", "this study", "the present study", "this report", "this case",
    "in conclusion", "it is concluded", "it is suggested", "presented with",
    "on examination", "results suggest", "results indicate", "results show",
    "findings suggest", "findings indicate", "data suggest", "data indicate",
    "these results", "these findings", "these data", "these observations",
    "suggests that", "suggest that", "indicates that", "indicate that",
)
# Verbs and phrases that report a finding or describe a method. Older and
# translated abstracts (a large part of PubMed) often have no section labels and
# none of the "we ..." phrases above, but they do state what was found.
_FINDING_WORDS = frozenset(
    """
    revealed showed demonstrated observed noted established resulted
    underwent presented permitted indicated confirmed
    """.split()
)
_FINDING_PHRASES = (
    "is based on", "are based on", "was based on", "led to", "is described",
    "are described", "is discussed", "are discussed", "is presented", "are presented",
    "the author", "the authors", "were found", "was found", "was revealed",
    "were revealed", "study of", "examination of", "investigation of",
)
_PASSIVE_RE = re.compile(
    r"(?:was|were|been) (?:studied|investigated|examined|determined|measured|found|observed|"
    r"shown|demonstrated|performed|compared|evaluated|assessed|analy[sz]ed|described|"
    r"reported|obtained|isolated|treated|administered|identified|detected|tested|"
    r"carried out|noted|admitted|diagnosed|randomi[sz]ed|enrolled|included|recorded)"
)

_EVIDENCE_WORDS = frozenset(
    """
    randomized randomised controlled placebo double-blind cohort prospective
    retrospective trial trials study studies patients subjects cases controls
    compared comparison significant significantly statistically correlation
    correlated associated association incidence prevalence mean median baseline
    follow-up outcome outcomes mortality survival sensitivity specificity
    increased decreased reduced elevated measured observed evaluated analysis
    effect effects response group groups level levels concentration rate ratio
    """.split()
)
_EVIDENCE_PHRASES = ("confidence interval", "95% ci", "odds ratio", "hazard ratio", "relative risk")
_PVALUE_RE = re.compile(r"p\s?[<=>]\s?0?\.\d+|n\s?=\s?\d+")

_BRACKET_CITATION_RE = re.compile(r"\[\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*\]")

_MISINFORMATION_PHRASES = (
    "miracle cure", "doctors hate", "big pharma doesn't want", "big pharma does not want",
    "cures all disease", "cures any disease", "cures every disease", "cures all cancer",
    "100% cure", "100% guaranteed", "100% effective", "they don't want you to know",
    "one weird trick", "detox your body", "vaccines cause autism", "vaccine causes autism",
    "no side effects whatsoever", "secret remedy",
)


class MedicalValidator(DomainValidator):
    domain = "medical"
    CHECKS = (
        "clinical_structure",
        "factual_grounding",
        "educational_value",
        "topic_coherence",
        "verifiable_claims",
        "misinformation",
    )
    DEFAULTS = {
        "min_structure_cues": 1,
        "min_evidence_terms": 2,
        "min_distinct_terms": 3,
        "min_sentences": 3,
        "min_term_density": 0.03,
        "min_numbers_per_100w": 0.5,
        "max_misinformation_hits": 0,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        lower = stats.lower
        counts = stats.word_counts

        term_hits = 0
        distinct = 0
        for word, count in counts.items():
            if word in MEDICAL or (
                len(word) > 6 and word.endswith(_SUFFIXES) and word not in _SUFFIX_EXCEPTIONS
            ):
                term_hits += count
                distinct += 1

        structure = (
            phrases_present(lower, _SECTION_LABELS)
            + phrases_present(lower, _STRUCTURE_PHRASES)
            + min(2, phrases_present(lower, _FINDING_PHRASES) + sum(1 for w in _FINDING_WORDS if w in counts))
            + len(set(_PASSIVE_RE.findall(lower)))
        )
        evidence = (
            distinct_terms(stats, _EVIDENCE_WORDS)
            + phrases_present(lower, _EVIDENCE_PHRASES)
            + len(set(_PVALUE_RE.findall(lower)))
        )
        citations = len(_BRACKET_CITATION_RE.findall(stats.text)) + lower.count("et al")

        return {
            "term_density": term_hits / stats.n_words if stats.n_words else 0.0,
            "distinct_terms": distinct,
            "structure_cues": structure,
            "evidence_terms": evidence,
            "numbers_per_100w": numbers_per_100_words(stats),
            "citations_per_100w": per_100_words(citations, stats.n_words),
            "citations": citations,
            "sentences": stats.n_sentences,
            "misinformation_hits": phrases_present(lower, _MISINFORMATION_PHRASES),
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "clinical_structure": s["structure_cues"] >= t.min_structure_cues,
            "factual_grounding": s["evidence_terms"] >= t.min_evidence_terms,
            "educational_value": (
                s["distinct_terms"] >= t.min_distinct_terms
                and s["sentences"] >= t.min_sentences
            ),
            "topic_coherence": s["term_density"] >= t.min_term_density,
            "verifiable_claims": (
                s["numbers_per_100w"] >= t.min_numbers_per_100w or s["citations"] >= 1
            ),
            "misinformation": s["misinformation_hits"] <= t.max_misinformation_hits,
        }
