"""Provenance: the record written for every accepted document.

Each record says where the document came from, which domain it was curated
for, its quality score, what every pipeline stage decided, and when it was
accepted. Together with the configuration fingerprint this makes a dataset
traceable and reproducible.

A record looks like this (one JSON object per line of the output file):

    {
      "id": "medical-9f2c41d07a3be615",
      "text": "...",
      "domain": "medical",
      "quality_score": 0.8312,
      "n_words": 187,
      "n_tokens": 263,
      "provenance": {
        "source": "hf:MedRAG/pubmed",
        "source_id": "pubmed23n0600_1734",
        "url": "",
        "agent": "factory",
        "license": "public-domain",
        "synthetic": false,
        "content_hash": "9f2c41d07a3be615",
        "stages": {
          "global": {"language": "en", "language_confidence": 0.93},
          "validator": {"checks_passed": ["clinical_structure", "..."], "checks_disabled": []},
          "score": {"value": 0.8312, "threshold": 0.7, "features": {"terminology": 0.9, "...": 0}},
          "classifier": {"probability": 0.99, "limit": 0.5},
          "dedup": {"max_similarity": 0.12, "threshold": 0.85}
        },
        "pii_removed": {"email": 1},
        "accepted_at": "2026-11-13T12:00:00Z",
        "job_id": "job-20261113-120000",
        "pipeline_version": "0.1.0",
        "config_fingerprint": "0f2f2ecf53d37665"
      }
    }
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from . import __version__
from .licenses import normalize_license
from .models import FilterResult, RawDocument
from .scoring import ScoreResult

REQUIRED_FIELDS = ("id", "text", "domain", "quality_score", "n_words", "provenance")
REQUIRED_PROVENANCE = (
    "source", "agent", "license", "synthetic", "content_hash", "stages",
    "accepted_at", "pipeline_version", "config_fingerprint",
)


def utc_now() -> str:
    """Current time as an ISO 8601 UTC timestamp, e.g. 2026-11-13T12:00:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_id(domain: str, content_hash: str) -> str:
    return f"{domain}-{content_hash}"


def build_record(
    *,
    text: str,
    domain: str,
    document: RawDocument,
    n_words: int,
    content_hash: str,
    score: ScoreResult,
    global_result: FilterResult,
    validator_result: FilterResult,
    validator_checks: tuple[str, ...],
    validator_disabled: frozenset[str] = frozenset(),
    classifier_result: FilterResult | None = None,
    pii_counts: Mapping[str, int] | None = None,
    config_fingerprint: str,
    job_id: str = "",
    n_tokens: int | None = None,
    accepted_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the output record for an accepted document.

    ``text`` is the final text (after PII scrubbing). The deduplication entry
    under ``stages`` is added by ``add_dedup`` once the main process has
    confirmed the document is not a duplicate.
    """
    global_details = {
        key: global_result.details[key]
        for key in ("language", "language_confidence")
        if key in global_result.details
    }
    stages: dict[str, Any] = {
        "global": global_details,
        "validator": {
            "checks_passed": [c for c in validator_checks if c not in validator_disabled],
            "checks_disabled": sorted(validator_disabled),
        },
        "score": {
            "value": score.score,
            "threshold": score.threshold,
            "features": dict(score.features),
        },
        "classifier": dict(classifier_result.details) if classifier_result is not None else None,
    }
    record: dict[str, Any] = {
        "id": record_id(domain, content_hash),
        "text": text,
        "domain": domain,
        "quality_score": score.score,
        "n_words": n_words,
    }
    if n_tokens is not None:
        record["n_tokens"] = n_tokens
    record["provenance"] = {
        "source": document.source,
        "source_id": document.source_id,
        "url": document.url,
        "agent": document.agent,
        "license": normalize_license(document.license),
        "synthetic": document.source.startswith("synthetic"),
        "content_hash": content_hash,
        "stages": stages,
        "pii_removed": dict(pii_counts or {}),
        "accepted_at": accepted_at or utc_now(),
        "job_id": job_id,
        "pipeline_version": __version__,
        "config_fingerprint": config_fingerprint,
    }
    return record


def add_dedup(record: dict[str, Any], max_similarity: float, threshold: float) -> dict[str, Any]:
    """Record the deduplication outcome on an accepted record."""
    record["provenance"]["stages"]["dedup"] = {
        "max_similarity": round(max_similarity, 4),
        "threshold": threshold,
    }
    return record


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return a list of problems with a record (empty when it is well-formed)."""
    problems = [f"missing field: {name}" for name in REQUIRED_FIELDS if name not in record]
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        problems += [
            f"missing provenance field: {name}"
            for name in REQUIRED_PROVENANCE
            if name not in provenance
        ]
        stages = provenance.get("stages")
        if isinstance(stages, Mapping):
            problems += [
                f"missing stage: {name}"
                for name in ("global", "validator", "score")
                if name not in stages
            ]
    elif "provenance" in record:
        problems.append("provenance must be an object")
    if "text" in record and not (isinstance(record["text"], str) and record["text"].strip()):
        problems.append("text is empty")
    return problems
