"""Tests for Module 4: the learned quality classifier."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dataset_factory.classifier import (
    ARTIFACT_TOKENS,
    DEFAULT_DIM,
    FEATURE_VERSION,
    QualityClassifier,
    bucket_of,
    classifier_settings,
    featurize,
    load_classifier,
    model_path,
    select_words,
)
from dataset_factory.config import ConfigError, load_config
from dataset_factory.textstats import TextStats
from dataset_factory.validators import DOMAINS
from test_module2 import SAMPLES

META = {"feature_version": FEATURE_VERSION, "max_words": 400}


@pytest.fixture(scope="module")
def config():
    return load_config()


def toy_classifier(positive=(), negative=(), bias=0.0) -> QualityClassifier:
    """A hand-made model: +8 for each positive token, -8 for each negative one."""
    weights = np.zeros(DEFAULT_DIM, dtype=np.float32)
    for token in positive:
        weights[bucket_of(token)] = 8.0
    for token in negative:
        weights[bucket_of(token)] = -8.0
    return QualityClassifier("toy", weights, bias, META)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def test_featurize_is_deterministic_sorted_and_in_range():
    words = "the patient was treated with aspirin and the patient recovered".split()
    a, b = featurize(words), featurize(words)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)                     # sorted, no duplicates
    assert a.min() >= 0 and a.max() < DEFAULT_DIM
    assert featurize([]).size == 0


def test_features_cover_words_and_word_pairs():
    buckets = set(featurize(["heart", "attack"]).tolist())
    assert buckets == {bucket_of("heart"), bucket_of("attack"), bucket_of("heart attack")}
    # Word order matters for pairs.
    assert bucket_of("heart attack") != bucket_of("attack heart")


def test_corpus_artifacts_are_ignored():
    assert "xmath" in ARTIFACT_TOKENS
    plain = featurize(["energy", "of", "the", "system"])
    marked = featurize(["energy", "xmath", "of", "xcite", "the", "system"])
    assert set(plain.tolist()) <= set(marked.tolist()) | {bucket_of("energy of")}
    assert bucket_of("xmath") not in set(marked.tolist())


def test_long_documents_are_sampled_from_start_and_middle():
    words = [f"w{i}" for i in range(2000)]
    picked = select_words(words, 400)
    assert len(picked) == 400
    assert picked[:200] == words[:200] and picked[200:] == words[1000:1200]
    # Text outside the two windows does not change the features.
    changed = words[:1500] + ["different"] * 500
    assert np.array_equal(featurize(words), featurize(changed))


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def test_probability_follows_the_weights():
    model = toy_classifier(positive=["patients"], negative=["touchdown"])
    assert model.probability_text("patients patients") > 0.9
    assert model.probability_text("touchdown touchdown") < 0.1
    assert model.probability_text("nothing relevant here") == pytest.approx(0.5)
    assert model.probability_text("") == 0.0


def test_probability_is_numerically_stable():
    weights = np.full(DEFAULT_DIM, 1e6, dtype=np.float32)
    assert QualityClassifier("toy", weights, 0.0, META).probability_text("any words") == 1.0
    assert QualityClassifier("toy", -weights, 0.0, META).probability_text("any words") == 0.0


def test_check_applies_the_minimum_probability():
    model = toy_classifier(positive=["patients"])
    model.min_probability = 0.6
    accepted = model.check(TextStats("patients patients"))
    assert accepted.passed and accepted.stage == "classifier" and accepted.reason is None
    rejected = model.check(TextStats("nothing relevant here"))
    assert not rejected.passed and rejected.reason == "low_learned_quality"
    assert rejected.details == {"probability": 0.5, "limit": 0.6}


def test_save_and_load_round_trip(tmp_path):
    model = toy_classifier(positive=["court", "the court"], negative=["return"], bias=-0.3)
    path = tmp_path / "quality_toy.npz"
    model.save(path)
    loaded = QualityClassifier.load(path)
    assert loaded.domain == "toy" and loaded.dim == DEFAULT_DIM
    for text in ("the court held", "return value", "unrelated words"):
        assert loaded.probability_text(text) == pytest.approx(model.probability_text(text), abs=1e-3)


def test_model_from_another_feature_version_is_refused(tmp_path):
    model = QualityClassifier("toy", np.zeros(DEFAULT_DIM, dtype=np.float32), 0.0,
                              {"feature_version": FEATURE_VERSION + 1})
    path = tmp_path / "old.npz"
    model.save(path)
    with pytest.raises(ConfigError, match="different feature version"):
        QualityClassifier.load(path)


def test_bucket_count_must_be_a_power_of_two():
    with pytest.raises(ValueError):
        QualityClassifier("toy", np.zeros(1000, dtype=np.float32), 0.0, META)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_default_settings(config):
    settings = classifier_settings(config)
    assert settings["enabled"] is True
    assert settings["min_probability"] == 0.5
    assert settings["min_probability_by_domain"] == {"general": 0.2}


@pytest.mark.parametrize(
    "table, message",
    [
        ({"weight": 0.3}, "unknown setting"),
        ({"enabled": "yes"}, "true or false"),
        ({"min_probability": 1.5}, "between 0 and 1"),
        ({"min_probability_by_domain": {"general": 2}}, "between 0 and 1"),
        ({"model_dir": 7}, "must be text"),
    ],
)
def test_invalid_settings(config, table, message):
    with pytest.raises(ConfigError, match=message):
        classifier_settings(replace(config, classifier=table))


def test_disabled_classifier_loads_nothing(config):
    assert load_classifier("medical", replace(config, classifier={"enabled": False})) is None


def test_missing_model_gives_a_helpful_error(config, tmp_path):
    missing = replace(config, classifier={"model_dir": str(tmp_path)})
    with pytest.raises(ConfigError, match="train_classifier.py"):
        load_classifier("medical", missing)


# --------------------------------------------------------------------------
# The shipped models
# --------------------------------------------------------------------------

def test_every_domain_ships_a_model(config):
    for domain in DOMAINS:
        assert model_path(domain, config).exists(), domain
        model = load_classifier(domain, config)
        assert model.domain == domain and model.meta["feature_version"] == FEATURE_VERSION
    assert load_classifier("general", config).min_probability == 0.2
    assert load_classifier("legal", config).min_probability == 0.5


@pytest.mark.parametrize("domain", ["news", "medical", "legal", "scientific", "code"])
def test_shipped_models_recognise_their_domain(config, domain):
    model = load_classifier(domain, config)
    own = model.probability_text(SAMPLES[domain])
    assert own >= 0.5, (domain, own)
    for other, text in SAMPLES.items():
        if other not in (domain, "general"):
            assert model.probability_text(text) < own, (domain, other)


def test_code_model_rejects_maths_prose_with_braces(config):
    """The weak spot of the rule-based code validator, which this model covers."""
    maths = (
        "# Bound states of nonlinear equations\n"
        "We consider the equation $-\\Delta u + V(x) u = \\lambda u$ in $\\mathbb{R}^{2}$ and "
        "prove that the ground state {u_n} is unique for every n. In this paper we show that "
        "the energy functional $E(u) = \\int |\\nabla u|^{2}$ attains its minimum, and that the "
        "sequence of eigenvalues {\\lambda_n} is increasing. The proof uses the results of "
        "Section 2 and the estimate (3.1) for the nonlinear term."
    )
    assert load_classifier("code", config).probability_text(maths) < 0.2
