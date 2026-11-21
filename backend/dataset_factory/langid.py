"""Language identification.

Uses the fastText ``lid.176`` model (the compact version that ships inside the
``fast-langdetect`` package, so it works offline). If that package is missing,
falls back to a simple English stop-word heuristic and says so through
``backend``, which is recorded in each run's manifest.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .textstats import STOPWORDS

_WORD_RE = re.compile(r"[a-z]+")
_WHITESPACE_RE = re.compile(r"\s+")


class LanguageIdentifier:
    """Identify the language of a text sample."""

    def __init__(self) -> None:
        try:
            from fast_langdetect import LangDetectConfig, LangDetector
        except ImportError:
            self._detect = None
            self.backend = "heuristic"
        else:
            # The library cuts its input to 80 characters unless told not to,
            # which makes the result depend on a document's first line alone.
            # The sample length is controlled by ``identify`` instead.
            detector = LangDetector(LangDetectConfig(model="lite", max_input_length=None))
            self._detect = detector.detect
            self.backend = "fasttext-lid.176"

    def identify(self, text: str, sample_chars: int = 2000) -> tuple[str, float]:
        """Return ``(language_code, confidence)``; ``("und", 0.0)`` if undetermined."""
        # fastText expects a single line of text.
        sample = _WHITESPACE_RE.sub(" ", text[:sample_chars]).strip()
        if not sample:
            return "und", 0.0
        if self._detect is None:
            return self._heuristic(sample)
        try:
            best = self._detect(sample, model="lite")[0]
        except Exception:
            return "und", 0.0
        return str(best["lang"]).lower(), float(best["score"])

    @staticmethod
    def _heuristic(sample: str) -> tuple[str, float]:
        words = _WORD_RE.findall(sample.lower())
        if not words:
            return "und", 0.0
        ratio = sum(1 for w in words if w in STOPWORDS) / len(words)
        if ratio >= 0.20:
            return "en", min(1.0, ratio / 0.40)
        return "und", 0.0


@lru_cache(maxsize=1)
def get_language_identifier() -> LanguageIdentifier:
    """Shared identifier (the model is loaded once per process)."""
    return LanguageIdentifier()
