"""Per-document text statistics.

Each statistic is computed at most once per document and then reused by the
global filter, the domain validators and the scorer, so no stage re-tokenises
the text.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import cached_property

_WORD_RE = re.compile(r"[a-z][a-z'\-]*")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")

STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be because been before being
    between both but by can could did do does during each for from had has have
    he her here his how i if in into is it its may more most must no not of on
    one only or other our out over she should so some such than that the their
    them then there these they this those through to under up was we were what
    when where which while who will with would you your
    """.split()
)


class TextStats:
    """Lazily computed statistics for one document."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n_chars = len(text)

    @cached_property
    def lower(self) -> str:
        return self.text.lower()

    @cached_property
    def words(self) -> list[str]:
        """Lower-cased alphabetic words."""
        return _WORD_RE.findall(self.lower)

    @cached_property
    def word_counts(self) -> Counter[str]:
        """How often each word occurs. Lets later stages test thousands of
        vocabulary terms with dictionary lookups instead of re-scanning the text."""
        return Counter(self.words)

    @cached_property
    def n_words(self) -> int:
        return len(self.words)

    @cached_property
    def alpha_chars(self) -> int:
        return sum(map(len, self.words))

    @cached_property
    def non_space_chars(self) -> int:
        t = self.text
        return self.n_chars - t.count(" ") - t.count("\n") - t.count("\t") - t.count("\r")

    @cached_property
    def alpha_ratio(self) -> float:
        """Share of non-space characters that belong to words."""
        return self.alpha_chars / self.non_space_chars if self.non_space_chars else 0.0

    @cached_property
    def mean_word_length(self) -> float:
        return self.alpha_chars / self.n_words if self.n_words else 0.0

    @cached_property
    def lines(self) -> list[str]:
        """Non-empty lines with surrounding whitespace removed."""
        return [s for line in self.text.splitlines() if (s := line.strip())]

    @cached_property
    def n_sentences(self) -> int:
        return max(1, len(_SENTENCE_END_RE.findall(self.text)))

    @cached_property
    def stopword_ratio(self) -> float:
        """Share of common function words among the first 1000 words."""
        head = self.words[:1000]
        if not head:
            return 0.0
        return sum(1 for w in head if w in STOPWORDS) / len(head)

    @cached_property
    def unique_word_ratio(self) -> float:
        """Distinct words / words, over the first 500 words.

        Measured on a fixed-size window because the ratio naturally falls as
        documents get longer.
        """
        head = self.words[:500]
        return len(set(head)) / len(head) if head else 0.0

    @cached_property
    def repeated_ngram_ratio(self) -> float:
        """Share of word 4-grams (first 1000 words) that repeat an earlier one."""
        head = self.words[:1000]
        total = len(head) - 3
        if total <= 0:
            return 0.0
        unique = len(set(zip(head, head[1:], head[2:], head[3:])))
        return 1.0 - unique / total

    @cached_property
    def duplicate_line_ratio(self) -> float:
        """Share of substantial lines (30+ characters) that duplicate another."""
        long_lines = [line for line in self.lines if len(line) >= 30]
        if len(long_lines) < 4:
            return 0.0
        return 1.0 - len(set(long_lines)) / len(long_lines)
