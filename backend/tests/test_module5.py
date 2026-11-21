"""Tests for Module 5: deduplication, PII scrubbing, licences and provenance."""

from __future__ import annotations

import json
import random
import re
import zlib
from dataclasses import replace

import numpy as np
import pytest

from dataset_factory.classifier import load_classifier
from dataset_factory.config import ConfigError, load_config
from dataset_factory.dedup import (
    DedupIndex,
    MinHasher,
    build_index,
    build_minhasher,
    dedup_settings,
    estimated_jaccard,
    exact_jaccard,
)
from dataset_factory.filters import GlobalFilter
from dataset_factory.hashing import content_hash, word_hash
from dataset_factory.licenses import KNOWN, LicensePolicy, build_policy, normalize_license, summarize
from dataset_factory.models import RawDocument
from dataset_factory.pii import PiiScrubber, build_scrubber, pii_settings
from dataset_factory.provenance import add_dedup, build_record, record_id, validate_record
from dataset_factory.scoring import build_scorer
from dataset_factory.textstats import TextStats
from dataset_factory.validators import build_validator
from test_module2 import MEDICAL


@pytest.fixture(scope="module")
def config():
    return load_config()


def make_words(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f"word{rng.randrange(5000)}" for _ in range(n)]


def perturb(words: list[str], fraction: float, seed: int = 1) -> list[str]:
    """Replace a fraction of the words with new ones."""
    rng = random.Random(seed)
    out = list(words)
    for position in rng.sample(range(len(out)), max(1, int(len(out) * fraction))):
        out[position] = f"changed{rng.randrange(10**6)}"
    return out


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def test_hashes_are_stable():
    # A fixed, documented algorithm (CRC-32), not Python's per-process hash().
    assert word_hash("patient") == zlib.crc32(b"patient")
    assert content_hash("The  Cat\nsat.") == content_hash("the cat sat.")
    assert content_hash("the cat sat.") != content_hash("the cat sits.")
    assert re.fullmatch(r"[0-9a-f]{16}", content_hash("anything"))


# --------------------------------------------------------------------------
# MinHash
# --------------------------------------------------------------------------

def test_signature_shape_and_determinism():
    words = make_words(300, 1)
    hasher = MinHasher(num_perm=128, shingle_size=5, seed=7)
    signature = hasher.signature(words)
    assert signature.shape == (128,) and signature.dtype == np.uint16
    assert np.array_equal(signature, MinHasher(128, 5, seed=7).signature(words))
    assert not np.array_equal(signature, MinHasher(128, 5, seed=8).signature(words))


def test_identical_and_unrelated_documents():
    hasher = MinHasher(seed=1)
    a, b = make_words(400, 1), make_words(400, 2)
    assert estimated_jaccard(hasher.signature(a), hasher.signature(a)) == 1.0
    assert estimated_jaccard(hasher.signature(a), hasher.signature(b)) < 0.05


def test_estimate_tracks_exact_jaccard():
    hasher = MinHasher(seed=1)
    base = make_words(600, 3)
    for fraction in (0.01, 0.03, 0.08, 0.20):
        other = perturb(base, fraction)
        exact = exact_jaccard(hasher.shingles(base), hasher.shingles(other))
        estimate = estimated_jaccard(hasher.signature(base), hasher.signature(other))
        assert abs(exact - estimate) < 0.12, (fraction, exact, estimate)


def test_word_order_matters():
    hasher = MinHasher(seed=1)
    words = make_words(400, 4)
    shuffled = list(words)
    random.Random(0).shuffle(shuffled)
    assert estimated_jaccard(hasher.signature(words), hasher.signature(shuffled)) < 0.1


def test_short_and_empty_texts():
    hasher = MinHasher(seed=1)
    assert not hasher.signature([]).any()
    short = hasher.signature(["only", "three", "words"])
    assert np.array_equal(short, hasher.signature(["only", "three", "words"]))
    assert not np.array_equal(short, hasher.signature(["only", "three", "others"]))


def test_long_documents_use_block_processing():
    hasher = MinHasher(seed=1)
    words = make_words(30000, 5)            # far more shingles than one block
    signature = hasher.signature(words)
    assert estimated_jaccard(signature, hasher.signature(perturb(words, 0.005))) > 0.9


# --------------------------------------------------------------------------
# Dedup index
# --------------------------------------------------------------------------

def test_index_detects_exact_and_near_duplicates():
    hasher = MinHasher(seed=1)
    index = DedupIndex(threshold=0.85)
    original = make_words(500, 10)
    text = " ".join(original)

    first = index.check_and_add(hasher.signature(original), content_hash(text))
    assert not first.duplicate and len(index) == 1

    exact = index.check_and_add(hasher.signature(original), content_hash(text.upper() + "  "))
    assert exact.duplicate and exact.kind == "exact" and exact.match == 0

    near_words = perturb(original, 0.01)
    near = index.check_and_add(hasher.signature(near_words), content_hash(" ".join(near_words)))
    assert near.duplicate and near.kind == "near" and near.match == 0 and near.similarity >= 0.85

    different = make_words(500, 11)
    other = index.check_and_add(hasher.signature(different), content_hash(" ".join(different)))
    assert not other.duplicate and len(index) == 2
    assert (index.exact_duplicates, index.near_duplicates) == (1, 1)


def test_documents_below_the_threshold_are_kept():
    hasher = MinHasher(seed=1)
    index = DedupIndex(threshold=0.85)
    base = make_words(600, 12)
    half_changed = perturb(base, 0.20)      # exact Jaccard far below 0.85
    assert exact_jaccard(hasher.shingles(base), hasher.shingles(half_changed)) < 0.6
    index.add(hasher.signature(base))
    assert not index.check_and_add(hasher.signature(half_changed)).duplicate


def test_near_duplicates_are_found_reliably():
    """Recall at the threshold: lightly edited copies must almost always be caught."""
    hasher = MinHasher(seed=1)
    index = DedupIndex(threshold=0.85)
    originals = [make_words(400, 100 + i) for i in range(100)]
    for words in originals:
        index.add(hasher.signature(words))
    caught = sum(
        index.find(hasher.signature(perturb(words, 0.01, seed=i))).duplicate
        for i, words in enumerate(originals)
    )
    assert caught >= 98


def test_index_grows_past_its_initial_capacity():
    hasher = MinHasher(num_perm=64, seed=1)
    index = DedupIndex(threshold=0.85, num_perm=64, bands=10, rows=6)
    documents = [make_words(60, 1000 + i) for i in range(1500)]
    for words in documents:
        assert not index.check_and_add(hasher.signature(words)).duplicate
    assert len(index) == 1500
    again = index.find(hasher.signature(documents[1200]))
    assert again.duplicate and again.match == 1200
    assert index.memory_bytes() > 0


def test_dedup_settings(config):
    settings = dedup_settings(config)
    assert settings["threshold"] == 0.85 and settings["num_perm"] == 128
    assert build_minhasher(config).num_perm == 128
    assert build_index(config).threshold == 0.85
    # Every process must build the same hasher.
    words = make_words(200, 20)
    assert np.array_equal(build_minhasher(config).signature(words), build_minhasher(config).signature(words))


@pytest.mark.parametrize(
    "table, message",
    [
        ({"bands": 30, "rows": 6}, "must not exceed num_perm"),
        ({"threshold": 0}, "above 0"),
        ({"num_perm": 0}, "positive whole number"),
        ({"enabled": 1}, "true or false"),
        ({"shingles": 5}, "unknown setting"),
    ],
)
def test_invalid_dedup_settings(config, table, message):
    with pytest.raises(ConfigError, match=message):
        dedup_settings(replace(config, dedup=table))


def test_invalid_index_arguments():
    with pytest.raises(ValueError):
        DedupIndex(num_perm=64, bands=12, rows=6)
    with pytest.raises(ValueError):
        DedupIndex(threshold=0)


# --------------------------------------------------------------------------
# PII scrubbing
# --------------------------------------------------------------------------

def test_each_pii_type_is_replaced_and_counted():
    scrubber = PiiScrubber()
    text = (
        "Contact Jane at jane.doe@example.org or 555-123-4567. International: +44 20 7946 0958. "
        "Server 8.8.8.8 logged card 4111 1111 1111 1111 and SSN 123-45-6789."
    )
    out, counts = scrubber.scrub(text)
    assert counts == {"email": 1, "phone": 2, "ip": 1, "card": 1, "id_number": 1}
    for leaked in ("jane.doe", "555-123", "7946", "8.8.8.8", "4111", "123-45-6789"):
        assert leaked not in out
    for placeholder in ("[EMAIL]", "[PHONE]", "[IP_ADDRESS]", "[CARD_NUMBER]", "[ID_NUMBER]"):
        assert placeholder in out
    assert out.startswith("Contact Jane at [EMAIL] or [PHONE].")


@pytest.mark.parametrize(
    "text",
    [
        "The study ran from 2020-01-15 to 2021-03-30 with 1,234,567 participants.",
        "See Roe v. Wade, 410 U.S. 113 (1973), and No. 73-1815.",
        "Version 1.2.3 and build 10.4.2 were released in 1990-1995.",
        "The router at 192.168.1.1 and localhost 127.0.0.1 are private addresses.",
        "Order number 1234 5678 9012 3456 is not a valid card number.",
        "pi is 3.14159265358979 and e is 2.718281828459045.",
        # Found in real papers: both pass the card checksum but are not cards.
        "Fabrizio Illuminati 1122334411223344",
        "supported by the program of higher education ( no . 20100006110021 )",
        "ISBN 978-0-306-40615-7 and DOI 10.1000/182 identify publications.",
        "Mention @username on the forum, or use the @property decorator.",
    ],
)
def test_ordinary_numbers_are_left_alone(text):
    out, counts = PiiScrubber().scrub(text)
    assert out == text and counts == {}


def test_clean_text_is_returned_unchanged():
    out, counts = PiiScrubber().scrub(MEDICAL)
    assert out == MEDICAL and counts == {}


def test_only_selected_types_are_scrubbed():
    out, counts = PiiScrubber(("email",)).scrub("Mail a@b.org or call 555-123-4567.")
    assert out == "Mail [EMAIL] or call 555-123-4567." and counts == {"email": 1}
    with pytest.raises(ConfigError, match="unknown type"):
        PiiScrubber(("fingerprints",))


def test_code_domain_keeps_dotted_numbers_and_digit_groups(config):
    code = 'HOST = "8.8.4.4"  # contact admin@example.com, hotline 555-123-4567'
    out, counts = build_scrubber("code", config).scrub(code)
    assert "8.8.4.4" in out and "555-123-4567" in out
    assert counts == {"email": 1}
    general_out, general_counts = build_scrubber("general", config).scrub(code)
    assert general_counts == {"email": 1, "phone": 1, "ip": 1}


def test_pii_can_be_switched_off(config):
    assert build_scrubber("general", replace(config, pii={"enabled": False})) is None
    assert pii_settings(config)["enabled"] is True
    with pytest.raises(ConfigError, match="types must be a list"):
        pii_settings(replace(config, pii={"types": ["faces"]}))
    with pytest.raises(ConfigError, match="unknown setting"):
        pii_settings(replace(config, pii={"mode": "strict"}))


# --------------------------------------------------------------------------
# Licences
# --------------------------------------------------------------------------

def test_license_normalisation():
    assert normalize_license("CC-BY-SA") == "cc-by-sa-4.0"
    assert normalize_license("odc_by") == "odc-by-1.0"
    assert normalize_license("Apache 2.0") == "apache-2.0"
    assert normalize_license(None) == normalize_license("") == normalize_license("other") == "unknown"
    assert normalize_license("some-custom-terms") == "some-custom-terms"
    assert "apache-2.0" in KNOWN and "unknown" not in KNOWN


def test_license_policy():
    default = LicensePolicy()
    assert default.check("cc-by-4.0").passed
    assert default.check(None).passed                     # unknown is allowed by default
    strict = LicensePolicy(allow_unknown=False, blocked=("CC-BY-NC",))
    blocked = strict.check("cc-by-nc-4.0")
    assert not blocked.passed and blocked.reason == "blocked_license" and blocked.stage == "license"
    assert strict.check("mystery-terms").reason == "unknown_license"
    assert strict.check("mit").passed


def test_license_summary():
    summary = summarize({"cc-by-sa-4.0": 700, "public-domain": 250, "mystery": 50})
    assert list(summary["by_license"]) == ["cc-by-sa-4.0", "public-domain", "mystery"]
    assert summary["attribution_required"] and summary["share_alike_present"]
    assert not summary["non_commercial_present"]
    assert summary["documents_with_unrecognised_license"] == 50
    assert summarize({"cc-by-nc-4.0": 1})["non_commercial_present"]
    assert not summarize({"cc0-1.0": 5})["attribution_required"]


def test_license_config(config):
    assert build_policy(config).check("cc-by-nc-4.0").passed
    commercial = replace(config, licenses={"allow_unknown": False, "blocked": ["cc-by-nc-4.0"]})
    assert not build_policy(commercial).check("cc-by-nc-4.0").passed
    with pytest.raises(ConfigError, match="unknown setting"):
        build_policy(replace(config, licenses={"allowed": []}))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def curated_record(config, text=MEDICAL, **overrides):
    """Run one document through every stage and build its record."""
    domain = "medical"
    stats = TextStats(text)
    global_result = GlobalFilter(config.global_filter_for(domain)).check(stats)
    validator = build_validator(domain, config)
    validator_result = validator.validate(stats)
    score = build_scorer(domain, config).score(stats, validator_result.details["signals"], 0.9)
    classifier_result = load_classifier(domain, config).check(stats)
    scrubbed, pii_counts = build_scrubber(domain, config).scrub(text)
    document = RawDocument(
        text=text, source="hf:MedRAG/pubmed", agent="factory", source_id="row-17",
        license="PublicDomain",
    )
    arguments = dict(
        text=scrubbed, domain=domain, document=document, n_words=stats.n_words,
        content_hash=content_hash(scrubbed), score=score, global_result=global_result,
        validator_result=validator_result, validator_checks=validator.CHECKS,
        validator_disabled=validator.disabled, classifier_result=classifier_result,
        pii_counts=pii_counts, config_fingerprint=config.fingerprint, job_id="job-test",
    )
    arguments.update(overrides)
    return build_record(**arguments)


def test_record_is_complete_and_serialisable(config):
    record = curated_record(config, n_tokens=263)
    assert validate_record(record) == []
    assert json.loads(json.dumps(record)) == record       # plain JSON types only

    assert record["id"] == record_id("medical", record["provenance"]["content_hash"])
    assert record["domain"] == "medical" and record["n_tokens"] == 263
    assert record["quality_score"] == record["provenance"]["stages"]["score"]["value"]

    provenance = record["provenance"]
    assert provenance["source"] == "hf:MedRAG/pubmed" and provenance["agent"] == "factory"
    assert provenance["license"] == "public-domain" and provenance["synthetic"] is False
    assert provenance["config_fingerprint"] == config.fingerprint
    assert provenance["job_id"] == "job-test"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", provenance["accepted_at"])

    stages = provenance["stages"]
    assert stages["global"]["language"] == "en"
    assert "clinical_structure" in stages["validator"]["checks_passed"]
    assert stages["score"]["threshold"] == 0.70 and "terminology" in stages["score"]["features"]
    assert stages["classifier"]["probability"] >= 0.5


def test_record_keeps_scrubbed_text_and_pii_counts(config):
    text = MEDICAL + " Correspondence: dr.smith@hospital.org, telephone 555-123-4567."
    record = curated_record(config, text=text)
    assert "dr.smith" not in record["text"] and "[EMAIL]" in record["text"]
    assert record["provenance"]["pii_removed"] == {"email": 1, "phone": 1}


def test_record_variants(config):
    synthetic = curated_record(
        config, document=RawDocument(text=MEDICAL, source="synthetic", license="synthetic")
    )
    assert synthetic["provenance"]["synthetic"] is True
    assert synthetic["provenance"]["license"] == "synthetic"

    rules_only = curated_record(config, classifier_result=None)
    assert rules_only["provenance"]["stages"]["classifier"] is None
    assert validate_record(rules_only) == []
    assert "n_tokens" not in rules_only

    fixed_time = curated_record(config, accepted_at="2026-01-02T03:04:05Z")
    assert fixed_time["provenance"]["accepted_at"] == "2026-01-02T03:04:05Z"


def test_dedup_outcome_is_recorded(config):
    record = add_dedup(curated_record(config), max_similarity=0.123456, threshold=0.85)
    assert record["provenance"]["stages"]["dedup"] == {"max_similarity": 0.1235, "threshold": 0.85}


def test_validate_record_reports_problems(config):
    record = curated_record(config)
    del record["quality_score"]
    del record["provenance"]["license"]
    del record["provenance"]["stages"]["score"]
    record["text"] = "   "
    problems = validate_record(record)
    assert "missing field: quality_score" in problems
    assert "missing provenance field: license" in problems
    assert "missing stage: score" in problems
    assert "text is empty" in problems
    assert validate_record({"id": "x", "provenance": "nope"})
