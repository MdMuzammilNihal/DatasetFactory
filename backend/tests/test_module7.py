"""Tests for Module 7: orchestrator, workers, checkpoints, CLI, benchmark.

Everything runs offline on the synthetic source. Token counting uses the
Pythia tokenizer when it is cached; the tests that need it skip otherwise.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import pytest

from dataset_factory.agents import build_agents, candidate_chunks
from dataset_factory.cli import main, parse_count
from dataset_factory.config import ConfigError, load_config
from dataset_factory.orchestrator import (
    Job,
    JobSpec,
    read_records,
    resume_job,
    run_job,
    runtime_settings,
)
from dataset_factory.provenance import validate_record
from dataset_factory.tokens import TokenizerUnavailable, load_token_counter


@pytest.fixture(scope="module")
def config():
    return load_config()


def _tokenizer_available() -> bool:
    try:
        load_token_counter("EleutherAI/pythia-160m")
        return True
    except TokenizerUnavailable:
        return False


needs_tokenizer = pytest.mark.skipif(not _tokenizer_available(), reason="Pythia tokenizer not cached")


def spec(domain="medical", **kwargs):
    defaults = dict(synthetic=True, workers=0, chunk_size=40, checkpoint_every=120, tokenizer="none")
    return JobSpec(domain, **{**defaults, **kwargs})


def ids(job):
    return [r["id"] for r in read_records(job.data_path)]


# ---------------------------------------------------------------------------
# Settings and requests
# ---------------------------------------------------------------------------

def test_runtime_settings_match_the_paper_defaults(config):
    settings = runtime_settings(config)
    assert settings["chunk_size"] == 1000 and settings["checkpoint_every"] == 10_000
    assert 1 <= settings["workers"] <= 4


def test_runtime_settings_reject_unknown_keys_and_bad_values(config):
    with pytest.raises(ConfigError):
        runtime_settings(replace(config, runtime={"wrokers": 2}))
    with pytest.raises(ConfigError):
        runtime_settings(replace(config, runtime={"chunk_size": 0}))


def test_a_request_needs_a_known_domain_and_a_target(config, tmp_path):
    with pytest.raises(ConfigError):
        Job(spec("cooking", max_documents=5), config, output_dir=tmp_path)
    with pytest.raises(ConfigError):
        Job(spec(), config, output_dir=tmp_path)
    with pytest.raises(ConfigError):   # a token budget needs a tokenizer
        Job(spec(max_tokens=1000), config, output_dir=tmp_path)


def test_parse_count():
    assert parse_count("2M") == 2_000_000
    assert parse_count("500k") == 500_000
    assert parse_count("1,500") == 1500


# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------

def test_job_stops_at_exactly_the_requested_number_of_documents(config, tmp_path):
    job = run_job(spec(max_documents=150), config, output_dir=tmp_path)
    assert job.state.status == "completed"
    assert job.state.accepted == 150 == len(ids(job))


def test_output_is_the_same_for_any_number_of_workers(config, tmp_path):
    inline = run_job(spec("legal", max_documents=120), config, output_dir=tmp_path)
    pooled = run_job(spec("legal", max_documents=120, workers=2), config, output_dir=tmp_path)
    assert ids(inline) == ids(pooled)
    assert inline.state.rejected == pooled.state.rejected


def test_records_are_valid_unique_and_stamped(config, tmp_path):
    job = run_job(spec("news", max_documents=120), config, output_dir=tmp_path)
    records = list(read_records(job.data_path))
    assert all(validate_record(r) == [] for r in records)
    hashes = [r["provenance"]["content_hash"] for r in records]
    assert len(set(hashes)) == len(hashes)
    assert {r["provenance"]["job_id"] for r in records} == {job.spec.job_id}
    assert {r["provenance"]["config_fingerprint"] for r in records} == {config.fingerprint}
    assert all("dedup" in r["provenance"]["stages"] for r in records)
    # The synthetic source repeats documents on purpose, so dedup must have fired.
    assert job.state.rejected.get("dedup:exact", 0) + job.state.rejected.get("dedup:near", 0) > 0


def test_every_candidate_is_accounted_for(config, tmp_path):
    job = run_job(spec("scientific", max_candidates=300), config, output_dir=tmp_path)
    s = job.state
    assert s.candidates == 300
    assert s.accepted + sum(s.rejected.values()) == s.candidates
    rejection_lines = (job.dir / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    prefiltered = sum(v for k, v in s.rejected.items() if k.startswith("prefilter:"))
    assert len(rejection_lines) == sum(s.rejected.values()) - prefiltered
    assert all("text" not in json.loads(line) for line in rejection_lines)


def test_job_folder_has_config_copy_and_manifest(config, tmp_path):
    job = run_job(spec(max_documents=50), config, output_dir=tmp_path)
    manifest = json.loads((job.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["config"]["fingerprint"] == config.fingerprint
    assert len(manifest["files"]["data.jsonl"]["sha256"]) == 64
    assert load_config(job.dir / "config.toml").fingerprint == config.fingerprint
    assert manifest["state"]["accepted"] == 50


def test_progress_is_reported(config, tmp_path):
    seen = []
    job = Job(spec(max_documents=100), config, output_dir=tmp_path, on_progress=seen.append,
              progress_every=0)
    job.run()
    assert seen and seen[-1]["status"] == "completed"
    assert [s["accepted"] for s in seen] == sorted(s["accepted"] for s in seen)


def test_random_mode_keeps_every_candidate_unfiltered(config, tmp_path):
    job = run_job(spec("general", max_documents=100, mode="random"), config, output_dir=tmp_path)
    records = list(read_records(job.data_path))
    assert job.state.accepted == job.state.candidates == 100
    assert job.state.rejected == {}
    assert all(r["provenance"]["mode"] == "random" and r["quality_score"] is None for r in records)


def test_licence_policy_applies_in_random_mode_too(config, tmp_path):
    blocked = replace(config, licenses={"allow_unknown": True, "blocked": ["synthetic"]})
    job = run_job(spec("general", max_candidates=50, mode="random"), blocked, output_dir=tmp_path)
    assert job.state.accepted == 0 and job.state.status == "completed"


def test_local_jsonl_input_is_curated(config, tmp_path):
    from dataset_factory.synthetic import SyntheticGenerator

    path = tmp_path / "in.jsonl"
    docs = itertools.islice(SyntheticGenerator("medical", "factory", 3).documents(), 80)
    path.write_text("".join(json.dumps({"id": f"d{i}", "body": d.text}) + "\n"
                            for i, d in enumerate(docs)), encoding="utf-8")
    job = run_job(spec(max_candidates=1000, synthetic=False, input_file=str(path),
                       input_text_field="body", input_license="cc-by-4.0"),
                  config, output_dir=tmp_path / "runs")
    assert job.state.status == "exhausted" and job.state.candidates == 80
    records = list(read_records(job.data_path))
    assert records and all(r["provenance"]["source"] == "file:in.jsonl" for r in records)
    assert {r["provenance"]["license"] for r in records} == {"cc-by-4.0"}


def test_missing_input_file_is_an_error(config, tmp_path):
    with pytest.raises(ConfigError):
        Job(spec(max_documents=5, input_file=str(tmp_path / "nope.jsonl")), config, output_dir=tmp_path)


@needs_tokenizer
def test_token_budget_is_met_with_pythia_counts(config, tmp_path):
    job = run_job(spec("news", max_tokens=20_000, tokenizer="EleutherAI/pythia-160m"),
                  config, output_dir=tmp_path)
    records = list(read_records(job.data_path))
    total = sum(r["n_tokens"] for r in records)
    assert total == job.state.tokens >= 20_000
    assert total - records[-1]["n_tokens"] < 20_000      # stopped at the first document over


@needs_tokenizer
def test_token_counter_matches_known_counts():
    counter = load_token_counter("EleutherAI/pythia-160m")
    assert counter("Hello world") == 2
    assert counter.count_many(["Hello world", ""]) == [2, 0]


# ---------------------------------------------------------------------------
# Checkpoints and resuming
# ---------------------------------------------------------------------------

def test_agents_resume_exactly_from_a_chunk(config):
    def make():
        factory, generator, route = build_agents("code", config, synthetic=True, chunk_size=7)
        return factory, generator, route.factory_share

    full = list(itertools.islice(candidate_chunks(*make()), 30))
    cut = full[12]
    # Positions and next source of each agent after chunk 12.
    latest = {}
    for c in full[:13]:
        latest[c.agent] = (c.positions, c.next_source)
    resume = {
        "positions": {role: pos for role, (pos, _) in latest.items()},
        "next_source": {role: nxt for role, (_, nxt) in latest.items()},
        "taken": cut.taken, "next_index": cut.index + 1,
    }
    rest = list(itertools.islice(candidate_chunks(*make(), resume=resume), 17))
    assert [[d.text for d in c.documents] for c in rest] == [[d.text for d in c.documents] for c in full[13:]]
    assert [c.index for c in rest] == [c.index for c in full[13:]]


def test_stopped_job_resumes_to_the_same_dataset(config, tmp_path):
    full = run_job(spec("news", max_documents=200), config, output_dir=tmp_path)

    job = Job(spec("news", max_documents=200), config, output_dir=tmp_path, progress_every=0)
    job.on_progress = lambda s: job.stop() if s["candidates"] >= 150 else None
    job.run()
    assert job.state.status == "stopped" and job.state.accepted < 200
    assert (job.dir / "checkpoint.json").exists()

    # A crash after the checkpoint leaves extra lines behind; resuming drops them.
    with job.data_path.open("ab") as fh:
        fh.write(b'{"partial": true}\n')
    resumed = resume_job(job.dir)
    assert resumed.state.status == "completed"
    assert ids(resumed) == ids(full)
    assert resumed.state.candidates == full.state.candidates
    assert resumed.state.rejected == full.state.rejected


def test_resume_refuses_a_changed_configuration(config, tmp_path):
    job = Job(spec(max_documents=500), config, output_dir=tmp_path, progress_every=0)
    job.on_progress = lambda s: job.stop() if s["candidates"] >= 80 else None
    job.run()
    toml = job.dir / "config.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace("seed = 42", "seed = 43"), encoding="utf-8")
    with pytest.raises(ConfigError):
        resume_job(job.dir)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_curate_status_and_resume(tmp_path, capsys):
    code = main(["curate", "legal", "--docs", "40", "--synthetic", "--workers", "0",
                 "--tokenizer", "none", "--chunk-size", "20", "--out", str(tmp_path), "-q"])
    assert code == 0
    job_dir = next(tmp_path.iterdir())
    assert len(list(read_records(job_dir / "data.jsonl"))) == 40
    assert main(["status", str(job_dir)]) == 0
    assert "completed" in capsys.readouterr().out


def test_cli_requires_a_target(capsys):
    assert main(["curate", "legal", "--synthetic"]) == 2
