"""Stage 5: deduplication.

The paper measures the similarity of two documents as the Jaccard similarity
of their shingle sets,

    Sim(x_i, x_j) = |Sh(x_i) & Sh(x_j)| / |Sh(x_i) | Sh(x_j)|

and drops a document when its similarity to an earlier accepted document
reaches the threshold (0.85). Comparing every pair exactly is far too slow, so
this module uses the standard MinHash technique:

* ``MinHasher`` turns a document into a short *signature* of ``num_perm``
  numbers. The share of positions where two signatures agree estimates the
  Jaccard similarity of the two documents. Signatures are computed inside the
  worker processes, because that is the expensive part.

* ``DedupIndex`` lives in the main process. It finds earlier documents likely
  to be similar using locality-sensitive hashing (signatures are cut into
  bands; documents sharing any band become candidates) and then confirms each
  candidate by comparing full signatures. Exact copies are caught first with a
  content hash.

A shingle is a run of ``shingle_size`` consecutive words (5 by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config, ConfigError, derive_seed
from .hashing import word_hash

_SHINGLE_BASE = np.uint64(0x100000001B3)
_MIX = np.uint64(0xFF51AFD7ED558CCD)
_BLOCK = 4096


class MinHasher:
    """Computes MinHash signatures. Every process must use the same seed."""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 1) -> None:
        if num_perm < 1 or shingle_size < 1:
            raise ValueError("num_perm and shingle_size must be positive")
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.default_rng(seed)
        # One random hash function per signature position: h(s) = a*s + b (mod 2^64).
        self._a = (rng.integers(0, 2**63, size=(num_perm, 1), dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
        self._b = rng.integers(0, 2**63, size=(num_perm, 1), dtype=np.uint64)

    def shingles(self, words: list[str]) -> np.ndarray:
        """Distinct 64-bit hashes of the document's word n-grams."""
        n, k = len(words), self.shingle_size
        if n == 0:
            return np.empty(0, dtype=np.uint64)
        hashes = np.fromiter(map(word_hash, words), dtype=np.uint64, count=n)
        if n < k:
            k = n                      # a very short text is one single shingle
        count = n - k + 1
        combined = np.zeros(count, dtype=np.uint64)
        for offset in range(k):        # polynomial rolling hash over k words
            combined = combined * _SHINGLE_BASE + hashes[offset:offset + count]
        combined ^= combined >> np.uint64(29)
        combined *= _MIX               # spread the bits before the min-hash step
        return np.unique(combined)

    def signature(self, words: list[str]) -> np.ndarray:
        """MinHash signature: ``num_perm`` 16-bit values."""
        shingles = self.shingles(words)
        if shingles.size == 0:
            return np.zeros(self.num_perm, dtype=np.uint16)
        minima = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, shingles.size, _BLOCK):   # blocks keep memory small
            block = shingles[start:start + _BLOCK][None, :]
            np.minimum(minima, (self._a * block + self._b).min(axis=1), out=minima)
        # Keep 16 well-mixed bits of each minimum: two different minima collide
        # only once in 65,536 times, and signatures take a quarter of the memory.
        return ((minima >> np.uint64(20)) & np.uint64(0xFFFF)).astype(np.uint16)


def estimated_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Estimated Jaccard similarity of two documents from their signatures."""
    return float(np.count_nonzero(a == b)) / a.size


def exact_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """True Jaccard similarity of two shingle sets (used to verify the estimate)."""
    if a.size == 0 and b.size == 0:
        return 1.0
    shared = np.intersect1d(a, b, assume_unique=True).size
    return shared / (a.size + b.size - shared)


@dataclass(frozen=True, slots=True)
class DedupResult:
    duplicate: bool
    kind: str | None = None         # "exact" or "near"
    match: int | None = None        # index of the earlier document it duplicates
    similarity: float = 0.0


class DedupIndex:
    """Remembers accepted documents and detects duplicates of them."""

    def __init__(
        self, threshold: float = 0.85, num_perm: int = 128, bands: int = 12, rows: int = 6
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if bands < 1 or rows < 1 or bands * rows > num_perm:
            raise ValueError("bands * rows must be between 1 and num_perm")
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self._signatures = np.empty((1024, num_perm), dtype=np.uint16)
        self._count = 0
        self._buckets: dict[int, int] = {}        # band key -> index of first document
        self._hashes: dict[str, int] = {}         # content hash -> index
        self.exact_duplicates = 0
        self.near_duplicates = 0

    def __len__(self) -> int:
        return self._count

    def _band_keys(self, signature: np.ndarray) -> list[int]:
        width = self.rows * 2                     # bytes per band (16-bit values)
        raw = signature[: self.bands * self.rows].tobytes()
        return [hash((band, raw[band * width:(band + 1) * width])) for band in range(self.bands)]

    def find(self, signature: np.ndarray, content_hash: str | None = None) -> DedupResult:
        """Look for an earlier document that this one duplicates."""
        if content_hash is not None:
            match = self._hashes.get(content_hash)
            if match is not None:
                return DedupResult(True, "exact", match, 1.0)
        buckets = self._buckets
        candidates = {buckets[key] for key in self._band_keys(signature) if key in buckets}
        best_match, best_similarity = None, 0.0
        for index in candidates:
            similarity = estimated_jaccard(signature, self._signatures[index])
            if similarity > best_similarity:
                best_match, best_similarity = index, similarity
        if best_match is not None and best_similarity >= self.threshold:
            return DedupResult(True, "near", best_match, best_similarity)
        return DedupResult(False, None, None, best_similarity)

    def add(self, signature: np.ndarray, content_hash: str | None = None) -> int:
        """Remember a document; returns its index."""
        index = self._count
        if index == self._signatures.shape[0]:
            grown = np.empty((index * 2, self.num_perm), dtype=np.uint16)
            grown[:index] = self._signatures
            self._signatures = grown
        self._signatures[index] = signature
        for key in self._band_keys(signature):
            self._buckets.setdefault(key, index)
        if content_hash is not None:
            self._hashes.setdefault(content_hash, index)
        self._count += 1
        return index

    def check_and_add(self, signature: np.ndarray, content_hash: str | None = None) -> DedupResult:
        """Return the duplicate found, or remember the document if it is new."""
        result = self.find(signature, content_hash)
        if result.duplicate:
            if result.kind == "exact":
                self.exact_duplicates += 1
            else:
                self.near_duplicates += 1
        else:
            self.add(signature, content_hash)
        return result

    def memory_bytes(self) -> int:
        """Rough memory used by the index (signatures plus lookup tables)."""
        per_entry = 100   # a Python dict entry with an integer key and value
        return int(self._signatures.nbytes + per_entry * (len(self._buckets) + len(self._hashes)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "threshold": 0.85,
    "num_perm": 128,
    "bands": 12,
    "rows": 6,
    "shingle_size": 5,
}


def dedup_settings(config: Config) -> dict[str, Any]:
    raw = dict(config.dedup)
    unknown = sorted(set(raw) - set(_DEFAULTS))
    if unknown:
        raise ConfigError(f"[dedup] unknown setting(s): {', '.join(unknown)}")
    settings = {**_DEFAULTS, **raw}
    if not isinstance(settings["enabled"], bool):
        raise ConfigError("[dedup] enabled must be true or false")
    threshold = settings["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ConfigError("[dedup] threshold must be a number above 0 and at most 1")
    for key in ("num_perm", "bands", "rows", "shingle_size"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"[dedup] {key} must be a positive whole number")
    if settings["bands"] * settings["rows"] > settings["num_perm"]:
        raise ConfigError("[dedup] bands x rows must not exceed num_perm")
    return settings


def build_minhasher(config: Config) -> MinHasher:
    settings = dedup_settings(config)
    return MinHasher(
        settings["num_perm"], settings["shingle_size"], seed=derive_seed(config.seed, "minhash")
    )


def build_index(config: Config) -> DedupIndex:
    settings = dedup_settings(config)
    return DedupIndex(
        float(settings["threshold"]), settings["num_perm"], settings["bands"], settings["rows"]
    )
