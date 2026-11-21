"""Tests for Module 2: the six domain validators."""

from __future__ import annotations

import pytest

from dataset_factory.config import ConfigError, load_config
from dataset_factory.textstats import TextStats
from dataset_factory.validators import (
    DOMAINS,
    VALIDATORS,
    build_validator,
    check_validator_config,
)

GENERAL = (
    "The Nile is a major north-flowing river in northeastern Africa. It flows into the "
    "Mediterranean Sea and has historically been considered the longest river in the "
    "world, at about 6,650 kilometres. The river is the primary water source of Egypt, "
    "Sudan and South Sudan. Its drainage basin covers eleven countries, including "
    "Tanzania, Uganda, Rwanda, Ethiopia and Kenya. The northern section flows almost "
    "entirely through the Sudanese desert to Egypt, where Cairo is located on its large "
    "delta. Ancient Egyptian civilization depended on the river since about 3000 BC."
)

NEWS = (
    "LONDON, England (Reuters) -- The Bank of England raised interest rates by a quarter "
    "point on Thursday, the first increase in 14 months, officials said. Governor Andrew "
    "Bailey told reporters that inflation had reached 4.2 percent in October. \"We will "
    "act again if price pressures persist through the winter,\" Bailey said at a press "
    "conference in London. Analysts at Barclays said the move was widely expected. "
    "According to the Office for National Statistics, wages rose 3.1 percent over the "
    "same period. A spokesman for the Treasury said the government supported the "
    "decision. Markets in Frankfurt and Paris closed slightly lower on Thursday, traders "
    "reported."
)

MEDICAL = (
    "BACKGROUND: Hypertension is a major risk factor for cardiovascular disease and "
    "stroke. METHODS: We conducted a randomized controlled trial in 412 patients with "
    "stage 1 hypertension. Patients were randomized to receive amlodipine 5 mg daily or "
    "placebo for 24 weeks. Blood pressure was measured at baseline and at each "
    "follow-up visit. RESULTS: Mean systolic pressure decreased by 14.2 mmHg in the "
    "treatment group compared with 3.1 mmHg in the placebo group (p < 0.001). Adverse "
    "effects were observed in 8% of treated patients. CONCLUSIONS: These results "
    "suggest that amlodipine significantly reduced blood pressure in patients with "
    "early hypertension, with an acceptable rate of adverse outcomes."
)

LEGAL = (
    "Smith v. Jones, 410 U.S. 113 (1973). Appeal from the District Court. The plaintiff "
    "brought this action against the defendant for breach of contract and sought "
    "damages. The trial court entered judgment for the defendant, and the plaintiff "
    "appealed. The appellant contends that the court erred in excluding testimony "
    "under Section 12 of the Evidence Act. We have reviewed the record and the "
    "authorities cited, including Brown v. Board, 347 U.S. 483. The statute provides no "
    "remedy in equity where a remedy at law exists. We hold that the evidence was "
    "properly excluded and that the jury verdict is supported. The judgment of the "
    "District Court is affirmed."
)

SCIENTIFIC = (
    "Abstract. We propose a new method for estimating the parameters of nonlinear "
    "dynamical systems from noisy data. Introduction. Parameter estimation is a central "
    "problem in statistical physics [1, 2]. Previous approaches rely on linear "
    "approximation (Smith, 2015) or on numerical simulation [3]. In this paper we derive "
    "an estimator based on the variance of the observed distribution. Methods. The model "
    "is defined by a matrix of coupling coefficients, and the solution is obtained by "
    "optimization of the energy function, as shown in Fig. 1. Results. Experiments on "
    "three benchmark datasets demonstrate that the proposed algorithm reduces the "
    "estimation error compared with the method of Lee et al. [4]. The convergence "
    "analysis is given in Section 3, and the results are summarised in Table 2. "
    "Conclusion. The framework extends to higher dimensional systems. References. "
    "[1] A. Author, Journal of Physics. [2] B. Author, Physical Review. " * 2
)

CODE = '''
def binary_search(items, target):
    """Return the index of target in the sorted list items, or -1."""
    low = 0
    high = len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        # Narrow the window to the half that can contain the target.
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
'''

SAMPLES = {
    "general": GENERAL,
    "news": NEWS,
    "medical": MEDICAL,
    "legal": LEGAL,
    "scientific": SCIENTIFIC,
    "code": CODE,
}


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_six_domains_registered(config):
    assert set(DOMAINS) == {"general", "news", "medical", "legal", "scientific", "code"}
    check_validator_config(config)  # the shipped config is valid for every domain


def test_config_file_matches_code_defaults(config):
    """The TOML file and the in-code defaults must not drift apart."""
    for domain, cls in VALIDATORS.items():
        assert dict(config.validators[domain]) == cls.DEFAULTS, domain


@pytest.mark.parametrize("domain", sorted(SAMPLES))
def test_validator_accepts_its_own_domain(config, domain):
    result = build_validator(domain, config).validate_text(SAMPLES[domain])
    assert result.passed, (domain, result.details)
    assert result.stage == domain and result.reason is None


@pytest.mark.parametrize("domain", ["medical", "legal", "scientific", "code"])
def test_specialised_validators_reject_other_domains(config, domain):
    validator = build_validator(domain, config)
    for other, text in SAMPLES.items():
        if other != domain:
            assert not validator.validate_text(text).passed, (domain, other)


def test_news_rejects_non_news(config):
    validator = build_validator("news", config)
    for other in ("general", "medical", "scientific", "code"):
        assert not validator.validate_text(SAMPLES[other]).passed, other


def test_general_is_permissive_but_not_for_sales_or_chat(config):
    validator = build_validator("general", config)
    assert validator.validate_text(NEWS).passed  # factual news is general knowledge
    sales = (
        "Welcome to our store. We offer the best price on every item, and our products "
        "ship fast. Buy now and get free shipping on your first order. Our team is ready "
        "to help you choose, so contact us today. Order now while the special offer lasts."
    )
    assert {"neutral_voice", "not_promotional"} <= set(
        validator.validate_text(sales).details["failed"]
    )
    chat = (
        "I think you should really try this because I loved it and my friends loved it. "
        "We went there last week and I told my sister about it. You will not regret it, "
        "I promise you. My only complaint is that we had to wait, but you get used to it."
    )
    assert "neutral_voice" in validator.validate_text(chat).details["failed"]


def test_every_signal_set_has_the_shared_signals(config):
    for domain in DOMAINS:
        signals = build_validator(domain, config).analyze(TextStats(SAMPLES[domain]))
        for shared in ("term_density", "citations_per_100w", "structure_cues"):
            assert shared in signals, (domain, shared)
        assert all(isinstance(v, (int, float)) for v in signals.values()), domain


def test_result_lists_every_failed_check(config):
    result = build_validator("medical", config).validate_text(GENERAL)
    assert not result.passed
    assert result.reason == result.details["failed"][0]
    assert {"clinical_structure", "topic_coherence"} <= set(result.details["failed"])


def test_medical_misinformation_screen(config):
    validator = build_validator("medical", config)
    text = MEDICAL + " This miracle cure is one weird trick that doctors hate."
    assert validator.validate_text(text).details["failed"] == ["misinformation"]


def test_medical_checks_use_paper_names():
    assert set(VALIDATORS["medical"].CHECKS) >= {
        "clinical_structure",
        "factual_grounding",
        "educational_value",
        "topic_coherence",
        "verifiable_claims",
    }


def test_disabled_check_is_ignored():
    """Leave-one-out ablation: switching off the failing check lets the document through."""
    text = MEDICAL + " This miracle cure works."
    assert not VALIDATORS["medical"]().validate_text(text).passed
    ablated = VALIDATORS["medical"]({"disabled_checks": ["misinformation"]})
    assert ablated.validate_text(text).passed


def test_threshold_override():
    strict = VALIDATORS["legal"]({"min_citations": 50})
    assert strict.validate_text(LEGAL).reason == "citations"


def test_code_requires_comments_and_balanced_delimiters():
    validator = VALIDATORS["code"]()
    no_comments = "\n".join(
        line for line in CODE.splitlines() if "#" not in line and '"""' not in line
    )
    assert validator.validate_text(no_comments).details["failed"] == ["comment_structure"]
    unbalanced = CODE + "\nresult = compute((((((items, target\n"
    assert "balanced_delimiters" in validator.validate_text(unbalanced).details["failed"]


def test_code_accepts_other_languages():
    validator = VALIDATORS["code"]()
    java = """
/**
 * Returns the larger of two values.
 */
public static int max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}
"""
    javascript = """
// Debounce a function so it runs once after calls stop.
function debounce(fn, delay) {
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}
"""
    assert validator.validate_text(java).passed
    assert validator.validate_text(javascript).passed


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"min_citatons": 3}, "unknown setting"),
        ({"disabled_checks": ["spelling"]}, "unknown check"),
        ({"min_citations": "two"}, "must be a number"),
        ({"disabled_checks": "citations"}, "must be a list"),
    ],
)
def test_invalid_validator_settings(settings, message):
    with pytest.raises(ConfigError, match=message):
        VALIDATORS["legal"](settings)


def test_unknown_domain(config, tmp_path):
    with pytest.raises(ConfigError, match="unknown domain"):
        build_validator("poetry", config)
    path = tmp_path / "bad.toml"
    path.write_text("[validators.poetry]\nmin_rhymes = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown domain"):
        check_validator_config(load_config(path))


def test_empty_text_does_not_crash():
    for domain, cls in VALIDATORS.items():
        assert not cls().validate_text("").passed, domain
