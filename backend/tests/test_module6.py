"""Tests for Module 6: sources, the synthetic generator and the three agents.

Everything here runs offline, on the synthetic source.
"""

from __future__ import annotations

import collections
import itertools
import pickle
from dataclasses import replace

import pytest

from dataset_factory.agents import (
    Chunk,
    FactoryAgent,
    GeneratorAgent,
    RetrievingAgent,
    ScorerAgent,
    build_agents,
    candidate_chunks,
)
from dataset_factory.config import ConfigError, load_config
from dataset_factory.dedup import build_index
from dataset_factory.langid import get_language_identifier
from dataset_factory.models import RawDocument
from dataset_factory.provenance import validate_record
from dataset_factory.sources import (
    SourceSpec,
    domain_routes,
    get_field,
    open_source,
    row_to_document,
    source_specs,
)
from dataset_factory.synthetic import DOMAIN_BUILDERS, MIX, NOISE_BUILDERS, SyntheticGenerator
from dataset_factory.validators import DOMAINS


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def scorers(config):
    return {domain: ScorerAgent(domain, config, job_id="job-test") for domain in DOMAINS}


def synthetic(domain, role, n, seed=11):
    return list(itertools.islice(SyntheticGenerator(domain, role, seed).documents(), n))


# ---------------------------------------------------------------------------
# Language identification (regression: the library used to see 80 characters)
# ---------------------------------------------------------------------------

def test_language_is_judged_on_the_whole_sample_not_the_first_line():
    identifier = get_language_identifier()
    if identifier.backend == "heuristic":
        pytest.skip("fast-langdetect is not installed")
    english = "The committee met on Tuesday and approved the budget for the coming year. " * 12
    opening = "Compte rendu de la réunion du conseil municipal de la ville. "
    assert len(opening) < 80
    language, confidence = identifier.identify(opening + english)
    assert language == "en" and confidence > 0.5


# ---------------------------------------------------------------------------
# Sources and routing
# ---------------------------------------------------------------------------

def test_every_domain_is_routed_to_known_sources(config):
    specs = source_specs(config)
    routes = domain_routes(config, DOMAINS)
    assert set(routes) == set(DOMAINS)
    for route in routes.values():
        assert route.factory and set(route.factory + route.generator) <= set(specs)
    assert specs["synthetic"].kind == "synthetic" and specs["synthetic"].label == "synthetic"
    assert specs["wikipedia"].label == "hf:wikimedia/wikipedia/20231101.en"


def test_bad_source_and_route_settings_are_rejected(config):
    with pytest.raises(ConfigError, match="unknown setting"):
        source_specs(replace(config, sources={"x": {"path": "a/b", "txt": "text"}}))
    with pytest.raises(ConfigError, match="needs a path"):
        source_specs(replace(config, sources={"x": {}}))
    with pytest.raises(ConfigError, match="unknown text builder"):
        source_specs(replace(config, sources={"x": {"path": "a/b", "text": "builder:nope"}}))
    with pytest.raises(ConfigError, match="unknown source"):
        domain_routes(replace(config, domains={"news": {"factory": ["missing"]}}))
    with pytest.raises(ConfigError, match="not a known domain"):
        domain_routes(replace(config, domains={"poetry": {"factory": ["synthetic"]}}), DOMAINS)


def test_rows_become_documents():
    spec = SourceSpec(
        key="s", path="org/corpus", text="body", id_field="id", url_field="meta.url",
        license="unknown", license_field="meta.license", authority=0.8,
    )
    row = {"body": "Some text.", "id": 7,
           "meta": {"url": "https://example.org/7",
                    "license": "Creative Commons - https://creativecommons.org/licenses/by/4.0/"}}
    document = row_to_document(spec, row, "factory")
    assert document.source == "hf:org/corpus" and document.source_id == "7"
    assert document.url == "https://example.org/7" and document.license == "cc-by-4.0"
    assert document.authority == 0.8 and document.agent == "factory"
    assert row_to_document(spec, {"body": "   "}, "factory") is None
    assert row_to_document(spec, {"other": "x"}, "factory") is None
    assert get_field(row, "meta.missing") is None

    pubmedqa = SourceSpec(key="p", path="q/p", text="builder:pubmedqa")
    row = {"context": {"labels": ["BACKGROUND", "RESULTS"], "contexts": ["Why.", "What."]}}
    assert row_to_document(pubmedqa, row, "factory").text == "BACKGROUND: Why.\nRESULTS: What."
    assert row_to_document(pubmedqa, {"context": None}, "factory") is None


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

def test_synthetic_stream_is_reproducible_and_resumable():
    first = [d.text for d in synthetic("news", "generator", 60)]
    assert first == [d.text for d in synthetic("news", "generator", 60)]
    assert first != [d.text for d in synthetic("news", "generator", 60, seed=12)]
    resumed = itertools.islice(SyntheticGenerator("news", "generator", 11).documents(skip=40), 20)
    assert [d.text for d in resumed] == first[40:]


def test_synthetic_documents_are_labelled():
    documents = synthetic("legal", "generator", 400)
    assert {d.source for d in documents} == {"synthetic"}
    assert {d.license for d in documents} == {"synthetic"}
    kinds = collections.Counter(d.meta["kind"] for d in documents)
    assert set(kinds) == set(MIX["generator"])
    with pytest.raises(ValueError):
        SyntheticGenerator("poetry")


@pytest.mark.parametrize("domain", DOMAINS)
def test_synthetic_ground_truth_matches_the_pipeline(domain, config, scorers):
    """Clean documents pass every stage; every kind of noise is rejected."""
    index = build_index(config)
    outcome = collections.defaultdict(collections.Counter)
    for document in synthetic(domain, "generator", 700):
        evaluation = scorers[domain].evaluate(document)
        label = evaluation.label
        if evaluation.accepted and index.check_and_add(
            evaluation.signature, evaluation.content_hash
        ).duplicate:
            label = "duplicate"
        outcome[document.meta["kind"]][label] += 1

    clean = outcome["clean"]
    assert clean["accepted"] >= 0.97 * sum(clean.values())
    for kind in NOISE_BUILDERS:
        assert outcome[kind]["accepted"] == 0, (kind, outcome[kind])
    assert outcome["exact_duplicate"]["duplicate"] == sum(outcome["exact_duplicate"].values())
    near = outcome["near_duplicate"]
    assert near["duplicate"] >= 0.9 * sum(near.values())
    if domain != "general":     # general knowledge deliberately accepts good news and science
        off_topic = outcome["off_topic"]
        assert off_topic["accepted"] <= 0.02 * sum(off_topic.values())


# ---------------------------------------------------------------------------
# Scorer agent
# ---------------------------------------------------------------------------

def clean_document(domain, **changes):
    document = next(d for d in synthetic(domain, "factory", 50) if d.meta["kind"] == "clean")
    for name, value in changes.items():
        setattr(document, name, value)
    return document


def test_accepted_document_carries_a_complete_record(config, scorers):
    document = clean_document("medical")
    evaluation = scorers["medical"].evaluate(document)
    assert evaluation.accepted and evaluation.label == "accepted" and evaluation.stage == "accepted"
    record = evaluation.record
    assert validate_record(record) == []
    assert record["domain"] == "medical" and record["id"] == f"medical-{evaluation.content_hash}"
    provenance = record["provenance"]
    assert provenance["synthetic"] is True and provenance["license"] == "synthetic"
    assert provenance["agent"] == "factory" and provenance["job_id"] == "job-test"
    assert provenance["config_fingerprint"] == config.fingerprint
    assert set(provenance["stages"]) == {"global", "validator", "score", "classifier"}
    assert provenance["stages"]["global"]["language"] == "en"
    assert evaluation.signature.shape == (128,)
    assert evaluation.score == record["quality_score"] >= provenance["stages"]["score"]["threshold"]
    assert "n_tokens" not in record
    pickle.loads(pickle.dumps(evaluation))      # evaluations cross process boundaries


def test_each_stage_reports_its_own_rejection(config, scorers):
    scorer = scorers["medical"]
    assert scorer.evaluate(RawDocument("Too short.", "synthetic")).label == "global:too_short"
    off_topic = clean_document("legal")
    label = scorer.evaluate(off_topic).label
    assert label.startswith("medical:") and scorer.evaluate(off_topic).record is None

    blocked = replace(config, licenses={"blocked": ["cc-by-nc-4.0"]})
    evaluation = ScorerAgent("medical", blocked).evaluate(
        clean_document("medical", license="CC-BY-NC-4.0")
    )
    assert evaluation.label == "license:blocked_license"

    strict = replace(config, scoring={**config.scoring, "medical": {
        **config.scoring["medical"], "threshold": 0.999}})
    evaluation = ScorerAgent("medical", strict).evaluate(clean_document("medical"))
    assert evaluation.label == "score:below_threshold" and evaluation.score < 0.999


def test_classifier_is_a_separate_switchable_stage(config):
    news_text = clean_document("news")
    lenient = replace(config, validators={**config.validators, "medical": {
        "disabled_checks": ["clinical_structure", "factual_grounding", "educational_value",
                            "topic_coherence", "verifiable_claims", "misinformation"]}},
        scoring={**config.scoring, "medical": {**config.scoring["medical"], "threshold": 0.0}})
    assert ScorerAgent("medical", lenient).evaluate(news_text).label == "classifier:low_learned_quality"

    rules_only = replace(lenient, classifier={"enabled": False})
    evaluation = ScorerAgent("medical", rules_only).evaluate(news_text)
    assert evaluation.accepted and evaluation.record["provenance"]["stages"]["classifier"] is None


def test_pii_is_scrubbed_before_fingerprinting(scorers):
    plain = clean_document("general")
    with_email = clean_document("general")
    with_email.text += " Contact the archive at records@example.org for details."
    a, b = scorers["general"].evaluate(plain), scorers["general"].evaluate(with_email)
    assert b.accepted
    assert "records@example.org" not in b.record["text"] and "[EMAIL]" in b.record["text"]
    assert b.record["provenance"]["pii_removed"] == {"email": 1}
    assert a.record["provenance"]["pii_removed"] == {}
    from dataset_factory.hashing import content_hash
    assert b.content_hash == content_hash(b.record["text"]) != a.content_hash


def test_token_counter_and_internal_errors(config):
    scorer = ScorerAgent("general", config, token_counter=lambda text: len(text) // 4)
    document = clean_document("general")
    assert scorer.evaluate(document).record["n_tokens"] == len(document.text) // 4

    def broken(text):
        raise RuntimeError("tokenizer crashed")

    evaluation = ScorerAgent("general", config, token_counter=broken).evaluate(document)
    assert not evaluation.accepted and evaluation.label == "error:RuntimeError"
    assert "tokenizer crashed" in evaluation.details["error"]


# ---------------------------------------------------------------------------
# Retrieving agents
# ---------------------------------------------------------------------------

def numbered(key, n, fail_at=None, fail_times=1):
    """A fake source opener: documents "<key>-<i>", optionally failing part-way."""
    failures = {"left": fail_times}

    def stream(skip):
        for i in range(skip, n):
            if fail_at is not None and i == fail_at and failures["left"] > 0:
                failures["left"] -= 1
                raise ConnectionError("connection reset")
            yield RawDocument(f"{key}-{i}", f"fake:{key}", source_id=str(i))

    return stream


def fake_opener(streams):
    def opener(spec, *, domain, agent, seed, skip=0):
        return streams[spec.key](skip)
    return opener


def test_agents_read_their_routed_sources(config):
    assert list(FactoryAgent("code", config).specs) == [
        "code_python", "code_java", "code_javascript", "code_go"]
    assert list(GeneratorAgent("code", config).specs) == ["fineweb"]
    with pytest.raises(ConfigError, match="unknown source"):
        FactoryAgent("code", config, sources=["missing"])
    with pytest.raises(TypeError):
        RetrievingAgent("code", config)


def test_round_robin_chunks_and_positions(config):
    opener = fake_opener({"wikipedia": numbered("a", 5), "fineweb": numbered("b", 2)})
    agent = FactoryAgent("general", config, sources=["wikipedia", "fineweb"],
                         chunk_size=3, opener=opener)
    chunks = list(agent.chunks())
    assert all(isinstance(c, Chunk) and c.agent == "factory" and c.domain == "general" for c in chunks)
    assert [[d.text for d in c.documents] for c in chunks] == [
        ["a-0", "b-0", "a-1"], ["b-1", "a-2", "a-3"], ["a-4"]]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert chunks[0].positions == {"wikipedia": 2, "fineweb": 1}
    assert chunks[-1].positions == {"wikipedia": 5, "fineweb": 2}
    assert agent.exhausted == ["fineweb", "wikipedia"] and not agent.errors


def test_resuming_from_a_chunks_positions_continues_exactly(config):
    def make():
        opener = fake_opener({"wikipedia": numbered("a", 9), "fineweb": numbered("b", 9)})
        return FactoryAgent("general", config, sources=["wikipedia", "fineweb"],
                            chunk_size=4, opener=opener)

    full = list(make().chunks())
    resumed = list(make().chunks(skip=full[1].positions, start_index=2))
    assert [[d.text for d in c.documents] for c in resumed] == [
        [d.text for d in c.documents] for c in full[2:]]
    assert [c.index for c in resumed] == [c.index for c in full[2:]]


def test_a_failing_source_is_retried_then_dropped(config):
    opener = fake_opener({
        "wikipedia": numbered("a", 4, fail_at=2, fail_times=1),      # one glitch, then fine
        "fineweb": numbered("b", 6, fail_at=1, fail_times=99),       # never recovers
    })
    agent = FactoryAgent("general", config, sources=["wikipedia", "fineweb"],
                         chunk_size=100, opener=opener, max_errors=2)
    texts = [d.text for c in agent.chunks() for d in c.documents]
    assert texts == ["a-0", "b-0", "a-1", "a-2", "a-3"]      # nothing lost, nothing repeated
    assert agent.dropped == ["fineweb"] and agent.exhausted == ["wikipedia"]
    assert [(e.source, e.position, e.dropped) for e in agent.errors] == [
        ("fineweb", 1, False), ("wikipedia", 2, False), ("fineweb", 1, True)]
    assert "ConnectionError" in agent.errors[0].error


def test_a_source_that_cannot_be_opened_does_not_stop_the_others(config):
    def opener(spec, *, domain, agent, seed, skip=0):
        if spec.key == "fineweb":
            raise FileNotFoundError("dataset not found")
        return numbered("a", 3)(skip)

    agent = FactoryAgent("general", config, sources=["fineweb", "wikipedia"], opener=opener)
    assert [d.text for c in agent.chunks() for d in c.documents] == ["a-0", "a-1", "a-2"]
    assert agent.dropped == ["fineweb"] and len(agent.errors) == agent.max_errors


def test_generator_prefilter_drops_short_and_foreign_documents(config):
    agent = GeneratorAgent("news", config, sources=["synthetic"], chunk_size=50)
    chunks = list(itertools.islice(agent.chunks(), 6))
    reference = list(itertools.islice(
        open_source(agent.specs["synthetic"], domain="news", agent="generator", seed=agent.seed),
        chunks[-1].positions["synthetic"],
    ))
    kept = {d.source_id for c in chunks for d in c.documents}
    dropped = collections.Counter()
    for c in chunks:
        dropped.update(c.prefiltered)
    assert len(kept) + sum(dropped.values()) == len(reference)
    kinds = collections.Counter(d.meta["kind"] for d in reference if d.source_id not in kept)
    # Tag soup has no recognisable language, so some of it is dropped here too.
    assert set(kinds) <= {"fragment", "foreign_language", "off_topic", "markup"}
    assert dropped["prefilter:too_short"] >= kinds["fragment"] > 0
    assert dropped["prefilter:wrong_language"] == kinds["foreign_language"] + kinds["markup"]
    assert kinds["foreign_language"] > 0
    assert all(d.meta["kind"] not in ("fragment", "foreign_language")
               for c in chunks for d in c.documents)
    # The code domain switches the language check off, so only length is tested.
    assert GeneratorAgent("code", config, sources=["synthetic"])._langid is None


def test_prefilter_never_changes_the_outcome(config, scorers):
    """Whatever the Generator agent drops, the Scorer agent would reject too."""
    agent = GeneratorAgent("medical", config, sources=["synthetic"])
    for document in synthetic("medical", "generator", 400):
        reason = agent._reject(document)
        if reason is not None:
            label = scorers["medical"].evaluate(document).label
            # Rejected by the global filter either way; the reason can differ
            # because the global filter tests for markup before language.
            assert label.startswith("global:")


def test_candidates_are_mixed_by_factory_share(config):
    factory, generator, route = build_agents("legal", config, synthetic=True, chunk_size=20)
    assert route.factory_share == 0.85
    chunks = list(itertools.islice(candidate_chunks(factory, generator, route.factory_share), 60))
    assert [c.index for c in chunks] == list(range(60))
    read = collections.Counter()
    for c in chunks:
        read[c.agent] += len(c.documents) + sum(c.prefiltered.values())
    assert read["factory"] / sum(read.values()) == pytest.approx(0.85, abs=0.03)
    assert {d.agent for c in chunks for d in c.documents} == {"factory", "generator"}
    # The two agents draw different synthetic streams.
    first = {c.agent: c.documents[0].text for c in reversed(chunks)}
    assert first["factory"] != first["generator"]


def test_candidate_stream_survives_one_agent_running_out(config):
    opener = fake_opener({"wikipedia": numbered("a", 5), "fineweb": numbered("b" * 300, 40)})
    factory = FactoryAgent("general", config, chunk_size=5, opener=opener)
    generator = GeneratorAgent("general", config, chunk_size=5, opener=opener)
    generator._langid = None
    chunks = list(candidate_chunks(factory, generator, 0.5))
    assert sum(len(c.documents) for c in chunks) == 45
    assert list(candidate_chunks(None, None, 0.5)) == []


def test_end_to_end_on_synthetic_data(config, scorers):
    factory, generator, route = build_agents("scientific", config, synthetic=True, chunk_size=32)
    index = build_index(config)
    labels = collections.Counter()
    records = []
    for chunk in itertools.islice(candidate_chunks(factory, generator, route.factory_share), 12):
        labels.update(chunk.prefiltered)
        for evaluation in scorers["scientific"].evaluate_many(chunk.documents):
            if evaluation.accepted and index.check_and_add(
                evaluation.signature, evaluation.content_hash
            ).duplicate:
                labels["duplicate"] += 1
                continue
            labels[evaluation.label] += 1
            if evaluation.accepted:
                records.append(evaluation.record)
    assert len(records) == labels["accepted"] > 150
    assert labels["duplicate"] > 0 and len({r["id"] for r in records}) == len(records)
    assert all(validate_record(r) == [] and r["provenance"]["synthetic"] for r in records)
    assert {r["provenance"]["agent"] for r in records} == {"factory", "generator"}
    assert set(DOMAIN_BUILDERS) == set(DOMAINS)
