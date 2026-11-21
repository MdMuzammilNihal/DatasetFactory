"""Stable hashing helpers.

Python's built-in ``hash`` changes between processes, so it cannot be used for
anything that worker processes must agree on or that must be reproducible
between runs. These helpers give the same result everywhere.
"""

from __future__ import annotations

import hashlib
import zlib

_word_cache: dict[str, int] = {}


def word_hash(word: str) -> int:
    """Stable 32-bit hash of a word."""
    value = _word_cache.get(word)
    if value is None:
        if len(_word_cache) > 500_000:      # keep memory bounded on huge vocabularies
            _word_cache.clear()
        value = _word_cache[word] = zlib.crc32(word.encode("utf-8"))
    return value


def content_hash(text: str) -> str:
    """16-hex-digit fingerprint of a text, ignoring case and whitespace layout.

    Two documents with the same fingerprint are exact duplicates for our
    purposes (the same words in the same order).
    """
    normalised = " ".join(text.split()).lower()
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=8).hexdigest()
