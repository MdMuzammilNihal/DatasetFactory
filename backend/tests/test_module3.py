"""Tests for Module 3: composite quality scoring."""

from __future__ import annotations

import copy

import pytest

from dataset_factory.config import ConfigError, load_config
from dataset_factory.scoring import (
    FEATURES,
    TARGETS,
    QualityScorer,
    build_scorer,
    check_scoring_config,
)
from dataset_factory.textstats import TextStats
from dataset_factory.validators import DOMAINS, build_validator
from test_module2 import MEDICAL, SAMPLES


@pytest.fixture(scope="module")
def config():
    return load_config()


def score_text(config, domain, text, authority=None):
    stats = TextStats(text)
    signals = build_validator(domain, config).analyze(stats)
    return build_scorer(domain, config).score(stats, signals, authority)


def settings_for(config, domain):
    """A private, editable copy of one domain's scoring settings."""
    return copy.deepcopy(dict(config.scoring[domain]))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_every_domain_has_valid_scoring(config):
    check_scoring_config(config, DOMAINS)
    for domain in DOMAINS:
        scorer = build_scorer(domain, config)
        assert sum(scorer.weights.values()) == pytest.approx(1.0)
        assert set(scorer.weights) <= set(FEATURES)
        assert set(scorer.targets) == set(TARGETS)
        assert 0.0 < scorer.threshold < 1.0


def test_specialised_domains_have_stricter_thresholds(config):
    """As in the paper: legal and scientific are stricter than general and news."""
    threshold = {d: build_scorer(d, config).threshold for d in DOMAINS}
    assert min(threshold["legal"], threshold["scientific"]) >= max(
        threshold["general"], threshold["news"]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s["weights"].update(readability=0.9), "must sum to 1"),
        (lambda s: s["weights"].update(spelling=0.0), "unknown feature"),
        (lambda s: s["weights"].update(readability=-0.1), "non-negative"),
        (lambda s: s.update(threshold=1.5), "between 0 and 1"),
        (lambda s: s.update(threshold="high"), "must be a number"),
        (lambda s: s["targets"].pop("term_density"), "must define exactly"),
        (lambda s: s["targets"].update(term_density=0), "positive number"),
        (lambda s: s.update(disabled_features=["spelling"]), "valid feature names"),
        (lambda s: s.update(bonus=1), "unknown setting"),
        (lambda s: s.update(disabled_features=list(FEATURES)), "every feature is disabled"),
    ],
)
def test_invalid_scoring_settings(config, change, message):
    settings = settings_for(config, "medical")
    change(settings)
    with pytest.raises(ConfigError, match=message):
        QualityScorer("medical", settings)


def test_missing_and_unknown_domains(config, tmp_path):
    with pytest.raises(ConfigError, match="no \\[scoring.poetry\\]"):
        build_scorer("poetry", config)
    path = tmp_path / "extra.toml"
    path.write_text("[scoring.poetry]\nthreshold = 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown domain"):
        check_scoring_config(load_config(path), ())


# --------------------------------------------------------------------------
# The score itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("domain", sorted(SAMPLES))
def test_good_document_passes_its_domain_threshold(config, domain):
    result = score_text(config, domain, SAMPLES[domain])
    assert result.passed, (domain, result)
    assert result.threshold == build_scorer(domain, config).threshold


@pytest.mark.parametrize("domain", sorted(SAMPLES))
def test_score_is_the_weighted_sum_of_features(config, domain):
    scorer = build_scorer(domain, config)
    result = score_text(config, domain, SAMPLES[domain])
    assert set(result.features) == set(scorer.weights)
    assert all(0.0 <= value <= 1.0 for value in result.features.values())
    expected = sum(scorer.weights[name] * value for name, value in result.features.items())
    assert result.score == pytest.approx(expected, abs=1e-3)
    assert 0.0 <= result.score <= 1.0


def test_domain_text_scores_higher_in_its_own_domain(config):
    """A medical abstract is a better medical document than a legal opinion is."""
    for domain in ("medical", "legal", "scientific"):
        own = score_text(config, domain, SAMPLES[domain]).score
        for other in ("general", "news"):
            assert own > score_text(config, domain, SAMPLES[other]).score, (domain, other)


def test_longer_is_not_better(config):
    """Length normalisation: repeating a document must not raise its score."""
    once = score_text(config, "medical", MEDICAL)
    repeated = score_text(config, "medical", (MEDICAL + " ") * 10)
    assert repeated.score < once.score
    assert repeated.features["non_repetition"] < once.features["non_repetition"]


def test_source_authority_raises_the_score(config):
    low = score_text(config, "medical", MEDICAL, authority=0.1)
    high = score_text(config, "medical", MEDICAL, authority=0.9)
    assert high.score > low.score
    weight = build_scorer("medical", config).weights["source_authority"]
    assert high.score - low.score == pytest.approx(0.8 * weight, abs=1e-3)
    # Out-of-range values are clamped, not trusted.
    assert score_text(config, "medical", MEDICAL, authority=7).features["source_authority"] == 1.0


def test_threshold_decides_acceptance(config):
    settings = settings_for(config, "medical")
    stats = TextStats(MEDICAL)
    signals = build_validator("medical", config).analyze(stats)
    score = QualityScorer("medical", settings).score(stats, signals).score
    settings["threshold"] = min(1.0, score + 0.01)
    assert not QualityScorer("medical", settings).score(stats, signals).passed
    settings["threshold"] = score
    assert QualityScorer("medical", settings).score(stats, signals).passed


def test_disabled_feature_is_dropped_and_weights_renormalised(config):
    settings = settings_for(config, "medical")
    settings["disabled_features"] = ["terminology"]
    scorer = QualityScorer("medical", settings)
    assert "terminology" not in scorer.weights
    assert sum(scorer.weights.values()) == pytest.approx(1.0)
    stats = TextStats(MEDICAL)
    result = scorer.score(stats, build_validator("medical", config).analyze(stats))
    assert "terminology" not in result.features


def test_code_uses_syntactic_validity_not_readability(config):
    scorer = build_scorer("code", config)
    assert "syntactic_validity" in scorer.weights and "readability" not in scorer.weights
    result = score_text(config, "code", SAMPLES["code"])
    assert result.features["syntactic_validity"] > 0.7


def test_empty_document_scores_without_crashing(config):
    for domain in DOMAINS:
        result = build_scorer(domain, config).score(TextStats(""), {})
        assert 0.0 <= result.score <= 1.0 and not result.passed
