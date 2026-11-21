"""Word n-gram overlap, used to keep evaluation text out of training corpora.

A training document is dropped when it shares at least ``MIN_SHARED`` distinct
13-word sequences with one evaluation set (or half of its own 13-grams, for
short documents). GPT-3 dropped on a single shared 13-gram; on real PubMed text
that also drops documents that merely share a stock phrase ("sensitivity,
specificity, positive predictive value ..."), while real copies share dozens to
hundreds (a Wikipedia article in WikiText's test set shared 114). Words are lower-cased
``\\w+`` runs; each n-gram becomes one 64-bit rolling hash computed with NumPy,
so a 200M-token corpus is checked in minutes, not hours.
"""

from __future__ import annotations

import re
import zlib
from typing import Iterable

import numpy as np

N = 13
MIN_SHARED = 8
_WORD = re.compile(r"\w+")
_BASE = np.uint64(1_000_003)


def _word_hashes(text: str) -> np.ndarray:
    words = _WORD.findall(text.lower())
    # crc32 is stable across processes (Python's hash() is salted per run)
    return np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))


def ngram_hashes(text: str, n: int = N) -> np.ndarray:
    """Hashes of every n-word window of ``text`` (empty if it has fewer words)."""
    w = _word_hashes(text)
    if len(w) < n:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(len(w) - n + 1, dtype=np.uint64)
        for k in range(n):   # polynomial hash, wraps modulo 2**64
            h = h * _BASE + w[k: len(w) - n + 1 + k]
    return h


class OverlapIndex:
    """Sorted array of the evaluation n-gram hashes, tagged by evaluation set."""

    def __init__(self, n: int = N) -> None:
        self.n = n
        self._parts: dict[str, list[np.ndarray]] = {}
        self._sets: dict[str, np.ndarray] = {}

    def add(self, name: str, texts: Iterable[str]) -> None:
        self._parts.setdefault(name, []).extend(ngram_hashes(t, self.n) for t in texts)
        self._sets.pop(name, None)

    def _built(self) -> dict[str, np.ndarray]:
        for name, parts in self._parts.items():
            if name not in self._sets:
                arr = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
                self._sets[name] = np.unique(arr)
        return self._sets

    def __len__(self) -> int:
        return sum(len(a) for a in self._built().values())

    def shared(self, text: str) -> dict[str, int]:
        """Distinct n-grams of ``text`` found in each evaluation set (non-zero only)."""
        return self._shared(np.unique(ngram_hashes(text, self.n)))

    def _shared(self, h: np.ndarray) -> dict[str, int]:
        out: dict[str, int] = {}
        if not len(h):
            return out
        for name, arr in self._built().items():
            if len(arr):
                pos = np.searchsorted(arr, h)
                pos[pos == len(arr)] = 0
                k = int(np.count_nonzero(arr[pos] == h))
                if k:
                    out[name] = k
        return out

    def hits(self, text: str, min_shared: int = MIN_SHARED) -> list[str]:
        """Evaluation sets that ``text`` overlaps enough to count as contaminated."""
        h = np.unique(ngram_hashes(text, self.n))
        need = max(1, min(min_shared, (len(h) + 1) // 2))
        return [name for name, k in self._shared(h).items() if k >= need]
