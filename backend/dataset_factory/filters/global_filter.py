"""Stage 1: the global filter, G(x) in the paper.

Applied to every document regardless of domain. It removes documents that are
too short or long, badly encoded, full of markup, not in the target language,
not prose, low in information, repetitive, or mostly boilerplate.

Checks run cheapest-first and stop at the first failure, so obviously bad
documents cost almost nothing (the paper's "early rejection"). Any check can be
switched off through ``disabled_checks`` in the configuration.
"""

from __future__ import annotations

import re

from ..config import GlobalFilterConfig
from ..langid import LanguageIdentifier, get_language_identifier
from ..models import FilterResult
from ..textstats import TextStats

STAGE = "global"

_MARKUP_RE = re.compile(
    r"</?[a-zA-Z][a-zA-Z0-9]{0,15}(?:\s[^<>]{0,300})?/?>"   # HTML/XML tags
    r"|&(?:[a-z]{2,8}|#\d{1,6});"                           # HTML entities
)

# Byte sequences left behind when UTF-8 is decoded with the wrong encoding.
_MOJIBAKE_MARKERS = ("�", "Ã©", "Ã¨", "â€™", "â€œ", "â€\x9d", "â€“", "Â\xa0")

_BOILERPLATE_RE = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in (
            "all rights reserved",
            "privacy policy",
            "terms of service",
            "terms of use",
            "terms and conditions",
            "cookie policy",
            "accept all cookies",
            "accept cookies",
            "we use cookies",
            "this site uses cookies",
            "subscribe to our newsletter",
            "sign up for our newsletter",
            "enter your email",
            "click here",
            "read more",
            "skip to content",
            "skip to main content",
            "log in to your account",
            "sign in to continue",
            "create an account",
            "forgot your password",
            "add to cart",
            "free shipping",
            "share this article",
            "share on facebook",
            "follow us on",
            "leave a comment",
            "leave a reply",
            "you must be logged in",
            "javascript is disabled",
            "enable javascript",
            "page not found",
            "powered by wordpress",
        )
    )
)


class GlobalFilter:
    """Domain-independent quality filter."""

    def __init__(
        self,
        config: GlobalFilterConfig,
        language_identifier: LanguageIdentifier | None = None,
    ) -> None:
        self.cfg = config
        self._off = frozenset(config.disabled_checks)
        if "language" in self._off:
            self._langid = None
        else:
            self._langid = language_identifier or get_language_identifier()

    def check_text(self, text: str) -> FilterResult:
        return self.check(TextStats(text))

    def check(self, stats: TextStats) -> FilterResult:
        cfg = self.cfg
        off = self._off
        details: dict = {}

        def fail(reason: str, value: float, limit: float) -> FilterResult:
            details["value"] = round(value, 4)
            details["limit"] = limit
            return FilterResult(False, STAGE, reason, details)

        # 1. Length in characters: free, so it goes first.
        if "length" not in off:
            if stats.n_chars < cfg.min_chars:
                return fail("too_short", stats.n_chars, cfg.min_chars)
            if stats.n_chars > cfg.max_chars:
                return fail("too_long", stats.n_chars, cfg.max_chars)
        if stats.n_chars == 0:
            return fail("too_short", 0, cfg.min_chars)

        text = stats.text

        # 2. Encoding damage.
        if "encoding" not in off:
            damaged = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
            if damaged:
                ratio = damaged / stats.n_chars
                if ratio > cfg.max_replacement_char_ratio:
                    return fail("encoding_errors", ratio, cfg.max_replacement_char_ratio)

        # 3. Leftover markup. Checked before language because tag soup confuses
        #    the language model.
        if "markup" not in off and ("<" in text or "&" in text):
            per_1k = len(_MARKUP_RE.findall(text)) * 1000.0 / stats.n_chars
            if per_1k > cfg.max_markup_per_1k_chars:
                return fail("malformed_markup", per_1k, cfg.max_markup_per_1k_chars)

        # 4. Language.
        if self._langid is not None:
            lang, confidence = self._langid.identify(text, cfg.language_sample_chars)
            details["language"] = lang
            details["language_confidence"] = round(confidence, 3)
            if lang != cfg.language or confidence < cfg.min_language_confidence:
                return fail("wrong_language", confidence, cfg.min_language_confidence)

        # 5. Word-level checks (the text is tokenised here, once).
        if "length" not in off and stats.n_words < cfg.min_words:
            return fail("too_short", stats.n_words, cfg.min_words)

        if "alpha_ratio" not in off and stats.alpha_ratio < cfg.min_alpha_ratio:
            return fail("low_alpha_ratio", stats.alpha_ratio, cfg.min_alpha_ratio)

        if "word_length" not in off and stats.n_words:
            mean_len = stats.mean_word_length
            if mean_len < cfg.min_mean_word_length:
                return fail("abnormal_word_length", mean_len, cfg.min_mean_word_length)
            if mean_len > cfg.max_mean_word_length:
                return fail("abnormal_word_length", mean_len, cfg.max_mean_word_length)

        if "stopwords" not in off and stats.stopword_ratio < cfg.min_stopword_ratio:
            return fail("low_stopword_ratio", stats.stopword_ratio, cfg.min_stopword_ratio)

        # 6. Information density.
        if (
            "information_density" not in off
            and stats.unique_word_ratio < cfg.min_unique_word_ratio
        ):
            return fail(
                "low_information_density", stats.unique_word_ratio, cfg.min_unique_word_ratio
            )

        # 7. Repetition.
        if "repetition" not in off:
            if stats.duplicate_line_ratio > cfg.max_duplicate_line_ratio:
                return fail(
                    "repetitive_lines", stats.duplicate_line_ratio, cfg.max_duplicate_line_ratio
                )
            if stats.repeated_ngram_ratio > cfg.max_repeated_ngram_ratio:
                return fail(
                    "repetitive_ngrams", stats.repeated_ngram_ratio, cfg.max_repeated_ngram_ratio
                )

        # 8. Boilerplate.
        if "boilerplate" not in off:
            hits = _BOILERPLATE_RE.findall(stats.lower)
            if hits:
                per_100 = len(hits) * 100.0 / max(1, stats.n_words)
                if (
                    len(hits) >= cfg.min_boilerplate_hits
                    and per_100 > cfg.max_boilerplate_per_100_words
                ):
                    return fail("boilerplate", per_100, cfg.max_boilerplate_per_100_words)
                distinct = len(set(hits))
                if distinct >= cfg.max_boilerplate_distinct:
                    return fail("boilerplate", distinct, cfg.max_boilerplate_distinct)

        return FilterResult(True, STAGE, None, details)
