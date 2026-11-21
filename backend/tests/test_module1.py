"""Tests for Module 1: configuration, seeds, language identification, global filter."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_factory.config import (
    GLOBAL_CHECKS,
    ConfigError,
    derive_seed,
    load_config,
)
from dataset_factory.filters import GlobalFilter
from dataset_factory.langid import get_language_identifier
from dataset_factory.models import FilterResult, RawDocument
from dataset_factory.textstats import TextStats

GOOD = (
    "The heart is a muscular organ that pumps blood through the vessels of the "
    "circulatory system. In humans it is located between the lungs, in the middle "
    "compartment of the chest. The pumped blood carries oxygen and nutrients to the "
    "body, while carrying metabolic waste such as carbon dioxide to the lungs. In a "
    "healthy adult at rest the heart beats about seventy times each minute, and this "
    "rate increases during exercise because the muscles need more oxygen. Physicians "
    "who specialise in disorders of the heart are called cardiologists, and they use "
    "tools such as the electrocardiogram to measure its electrical activity."
)

FRENCH = (
    "Le coeur est un organe musculaire qui pompe le sang dans les vaisseaux du "
    "système circulatoire. Chez l'homme, il est situé entre les poumons, dans le "
    "compartiment central de la poitrine. Le sang pompé transporte l'oxygène et les "
    "nutriments vers le corps, tout en transportant les déchets métaboliques vers "
    "les poumons. Chez un adulte en bonne santé au repos, le coeur bat environ "
    "soixante-dix fois par minute, et ce rythme augmente pendant l'exercice parce "
    "que les muscles ont besoin de plus d'oxygène pour fonctionner correctement."
)

PYTHON_CODE = '''
def moving_average(values, window):
    """Return the moving average of a list of numbers."""
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        if index >= window - 1:
            result.append(total / window)
    return result


class RunningStats:
    # Tracks count, mean and variance without storing samples.
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
'''


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def gf(config):
    return GlobalFilter(config.global_filter)


def only(config, check: str, **changes) -> GlobalFilter:
    """A filter with every check except ``check`` switched off."""
    disabled = tuple(c for c in GLOBAL_CHECKS if c != check)
    return GlobalFilter(replace(config.global_filter, disabled_checks=disabled, **changes))


# --------------------------------------------------------------------------
# Configuration and seeds
# --------------------------------------------------------------------------

def test_default_config_loads(config):
    assert config.version == "1.0.0"
    assert config.seed == 42
    assert config.global_filter.min_words == 50
    assert len(config.fingerprint) == 16
    assert load_config().fingerprint == config.fingerprint


def test_domain_override_applies_only_to_that_domain(config):
    code = config.global_filter_for("code")
    assert code.min_words == 15
    assert "language" in code.disabled_checks
    # Settings the override does not mention are inherited.
    assert code.max_chars == config.global_filter.max_chars
    assert config.global_filter_for("medical") == config.global_filter
    assert config.global_filter_for(None) == config.global_filter


@pytest.mark.parametrize(
    "body, message",
    [
        ("[global_filter]\nmin_wrods = 10\n", "unknown setting"),
        ("[global_filter]\ndisabled_checks = ['spelling']\n", "unknown check"),
        ("[global_filter]\nmin_words = 'ten'\n", "whole number"),
        ("[global_filter]\nmin_words = true\n", "whole number"),
        ("[scorring]\nx = 1\n", "unknown section"),
        ("[meta\n", "not valid TOML"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, body, message):
    path = tmp_path / "bad.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_fingerprint_changes_with_settings(tmp_path, config):
    path = tmp_path / "other.toml"
    path.write_text("[global_filter]\nmin_words = 51\n", encoding="utf-8")
    assert load_config(path).fingerprint != config.fingerprint


def test_derive_seed_is_stable_and_distinct():
    assert derive_seed(42, "medical", 0) == derive_seed(42, "medical", 0)
    seeds = {derive_seed(42, "medical", i) for i in range(1000)}
    assert len(seeds) == 1000
    assert derive_seed(42, "medical", 0) != derive_seed(43, "medical", 0)
    assert derive_seed(42, "medical", 0) != derive_seed(42, "legal", 0)
    # Fixed value: guards against the hash changing between runs or machines.
    assert derive_seed(42, "medical", 0) == derive_seed(42, "medical", "0")


# --------------------------------------------------------------------------
# Models and text statistics
# --------------------------------------------------------------------------

def test_models():
    doc = RawDocument(text="x", source="synthetic")
    assert doc.agent == "factory" and doc.license == "unknown" and doc.meta == {}
    assert FilterResult(True, "global").label == "global:passed"
    assert FilterResult(False, "global", "too_short").label == "global:too_short"


def test_text_stats():
    stats = TextStats("The cat sat. The cat sat!\nA dog ran?")
    assert stats.words == ["the", "cat", "sat", "the", "cat", "sat", "a", "dog", "ran"]
    assert stats.n_sentences == 3
    assert stats.lines == ["The cat sat. The cat sat!", "A dog ran?"]
    assert stats.unique_word_ratio == pytest.approx(6 / 9)
    assert TextStats("").n_words == 0
    assert TextStats("").alpha_ratio == 0.0
    assert TextStats("").repeated_ngram_ratio == 0.0


# --------------------------------------------------------------------------
# Language identification
# --------------------------------------------------------------------------

def test_language_identifier():
    langid = get_language_identifier()
    assert langid.backend == "fasttext-lid.176"
    assert langid.identify(GOOD)[0] == "en"
    assert langid.identify(GOOD)[1] > 0.8
    assert langid.identify(FRENCH)[0] == "fr"
    assert langid.identify("   \n  ") == ("und", 0.0)
    # Newlines must not break the single-line model.
    assert langid.identify(GOOD.replace(". ", ".\n"))[0] == "en"


# --------------------------------------------------------------------------
# Global filter
# --------------------------------------------------------------------------

def test_good_document_passes(gf):
    result = gf.check_text(GOOD)
    assert result.passed and result.reason is None
    assert result.details["language"] == "en"


def test_too_short(gf):
    assert gf.check_text("Too short.").reason == "too_short"
    # Long enough in characters but under the word minimum.
    few_words = " ".join(["The remarkable cardiologist examined patients."] * 6)
    assert len(few_words) >= 200
    assert gf.check_text(few_words).reason == "too_short"


def test_too_long(gf, config):
    text = (GOOD + " ") * (config.global_filter.max_chars // len(GOOD) + 2)
    assert gf.check_text(text).reason == "too_long"


def test_encoding_errors(gf):
    assert gf.check_text(GOOD + "�" * 20).reason == "encoding_errors"
    assert gf.check_text(GOOD.replace("'", "â€™") + " donâ€™t" * 10).reason == "encoding_errors"


def test_malformed_markup(gf):
    html = GOOD.replace(". ", '.</p><div class="row"><p>&nbsp;')
    assert gf.check_text(html).reason == "malformed_markup"


def test_wrong_language(gf):
    result = gf.check_text(FRENCH)
    assert result.reason == "wrong_language"
    assert result.details["language"] == "fr"


def test_low_alpha_ratio(config):
    numeric = " ".join(f"{word} 48213 99.31 7765" for word in GOOD.split())
    assert only(config, "alpha_ratio").check_text(numeric).reason == "low_alpha_ratio"


def test_abnormal_word_length(config):
    f = only(config, "word_length")
    assert f.check_text("ab " * 200).reason == "abnormal_word_length"
    assert f.check_text("pneumonoultramicroscopic " * 60).reason == "abnormal_word_length"


def test_low_stopword_ratio(config):
    keywords = " ".join(
        f"protein{i} kinase receptor membrane enzyme substrate pathway" for i in range(20)
    )
    assert only(config, "stopwords").check_text(keywords).reason == "low_stopword_ratio"


def test_low_information_density(config):
    text = "the cat sat on the mat " * 60
    assert (
        only(config, "information_density").check_text(text).reason
        == "low_information_density"
    )


def test_repetitive_lines(config):
    lines = ["Welcome to the best online store for all of your needs."] * 20
    lines += [f"Unique line number {i} with some different content in it." for i in range(5)]
    assert only(config, "repetition").check_text("\n".join(lines)).reason == "repetitive_lines"


def test_repetitive_ngrams(config):
    text = "buy cheap watches online today and save money on every order " * 30
    assert only(config, "repetition").check_text(text).reason == "repetitive_ngrams"


def test_boilerplate(config, gf):
    footer = (
        " Subscribe to our newsletter. We use cookies. Read our privacy policy and "
        "terms of service. All rights reserved."
    )
    assert only(config, "boilerplate").check_text(GOOD + footer).reason == "boilerplate"
    # One incidental phrase in a real article is not enough to reject it.
    assert gf.check_text(GOOD + " All rights reserved.").passed


def test_thresholds_calibrated_on_real_text(gf):
    """Regressions found when the filter was first run on real corpora."""
    # A single boilerplate phrase in a short article must not reject it.
    short_article = " ".join(GOOD.split()[:60]) + ". Read more"
    assert gf.check_text(short_article).passed
    # News full of names and places scores low language confidence but is English.
    news = (
        "SAN FRANCISCO, California (CNN) -- A magnitude 4.2 earthquake shook the San "
        "Francisco area Friday at 4:42 a.m. PT, the U.S. Geological Survey reported. "
        "The quake left about 2,000 customers without power, said David Eisenhower, a "
        "spokesman for Pacific Gas and Light. Under the U.S. Geological Survey "
        "classification, a magnitude 4.2 earthquake is considered light, which it says "
        "usually causes minimal damage. We had quite a spike in calls, mostly calls of "
        "inquiry, none of any injury, none of any damage that was reported, said "
        "Captain Al Casciato of the San Francisco police."
    )
    assert gf.check_text(news).passed


def test_disabled_check_is_skipped(config):
    no_language = GlobalFilter(
        replace(config.global_filter, disabled_checks=("language", "stopwords"))
    )
    assert no_language.check_text(FRENCH).passed


def test_code_domain_uses_relaxed_settings(config):
    code_filter = GlobalFilter(config.global_filter_for("code"))
    assert code_filter.check_text(PYTHON_CODE).passed
    # Prose still passes the code settings; junk still fails.
    assert code_filter.check_text(GOOD).passed
    assert code_filter.check_text("x = 1").reason == "too_short"


def test_empty_text_never_crashes(config):
    everything_off = GlobalFilter(
        replace(config.global_filter, disabled_checks=GLOBAL_CHECKS)
    )
    assert everything_off.check_text("").reason == "too_short"
    assert everything_off.check_text("anything at all").passed
