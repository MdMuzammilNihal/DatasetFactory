"""The three agents of the paper.

* ``FactoryAgent`` reads a domain's structured, trusted repositories.
* ``GeneratorAgent`` reads its large web corpora and drops the obviously
  unusable documents (wrong length, wrong language) before they travel further.
* ``ScorerAgent`` takes one candidate document through every per-document
  stage and either rejects it, saying where and why, or returns the finished
  record together with the fingerprints the deduplication step needs.

The two retrieving agents yield *chunks*: lists of documents that the
orchestrator hands to worker processes. Each chunk also says how far every
source has been read, which is what a checkpoint stores to resume a run.

Deduplication is not done here. It needs to see every accepted document, so it
runs in the main process on what the Scorer agents return.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .classifier import load_classifier
from .config import Config, ConfigError, derive_seed
from .dedup import build_minhasher
from .filters import GlobalFilter
from .hashing import content_hash
from .langid import get_language_identifier
from .licenses import build_policy
from .models import RawDocument
from .pii import build_scrubber
from .provenance import build_record
from .scoring import build_scorer
from .sources import DomainRoute, SourceSpec, domain_routes, open_source, source_specs
from .textstats import TextStats
from .validators import DOMAINS, build_validator

ROLES = ("factory", "generator")
DEFAULT_CHUNK_SIZE = 64
DEFAULT_MAX_ERRORS = 3          # failures tolerated per source before it is dropped

Opener = Callable[..., Iterator[RawDocument]]


# ---------------------------------------------------------------------------
# Retrieving agents
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Chunk:
    """A batch of candidate documents from one agent."""

    index: int
    domain: str
    agent: str
    documents: list[RawDocument]
    # Documents read from each source so far, counting this chunk. Reopening a
    # source with ``skip`` set to this number continues right after the chunk.
    positions: dict[str, int]
    # Documents the agent itself dropped while filling this chunk, by reason
    # (for example {"prefilter:too_short": 3}). Empty for the Factory agent.
    prefiltered: dict[str, int] = field(default_factory=dict)
    # The source the agent reads next (round-robin), needed to resume exactly.
    next_source: str | None = None
    # Candidates taken from each agent so far, counting this chunk (set by
    # ``candidate_chunks``; a checkpoint stores it to keep the interleaving).
    taken: dict[str, int] = field(default_factory=dict)


def _close(stream: Any) -> None:
    """Close a document stream if it can be closed (generators can)."""
    close = getattr(stream, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass


@dataclass(frozen=True, slots=True)
class SourceError:
    source: str                 # source key
    position: int               # documents read from the source when it failed
    error: str
    dropped: bool               # True when the agent gave up on the source


class RetrievingAgent:
    """Reads a domain's sources in turn, one document from each (round-robin)."""

    role = ""

    def __init__(
        self,
        domain: str,
        config: Config,
        *,
        sources: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        seed: int | None = None,
        max_errors: int = DEFAULT_MAX_ERRORS,
        opener: Opener = open_source,
    ) -> None:
        if self.role not in ROLES:
            raise TypeError("use FactoryAgent or GeneratorAgent")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        specs = source_specs(config)
        if sources is None:
            routes = domain_routes(config, DOMAINS)
            if domain not in routes:
                raise ConfigError(f"no [domains.{domain}] section")
            sources = getattr(routes[domain], self.role)
        missing = [key for key in sources if key not in specs]
        if missing:
            raise ConfigError(f"unknown source(s): {', '.join(missing)}")
        self.domain = domain
        self.config = config
        self.specs: dict[str, SourceSpec] = {key: specs[key] for key in sources}
        self.chunk_size = chunk_size
        self.seed = derive_seed(config.seed if seed is None else seed, domain)
        self.max_errors = max_errors
        self._opener = opener
        self.errors: list[SourceError] = []
        self.positions: dict[str, int] = {key: 0 for key in self.specs}
        self.exhausted: list[str] = []      # sources that ran out of documents
        self.dropped: list[str] = []        # sources abandoned after repeated errors
        self.next_source: str | None = None

    # Overridden by the Generator agent.
    def _reject(self, document: RawDocument) -> str | None:
        return None

    def _open(self, key: str) -> Iterator[RawDocument]:
        return iter(self._opener(
            self.specs[key], domain=self.domain, agent=self.role, seed=self.seed,
            skip=self.positions[key],
        ))

    def documents(
        self, skip: Mapping[str, int] | None = None, start_source: str | None = None
    ) -> Iterator[RawDocument | str]:
        """Documents in round-robin order. A string is yielded in place of a
        document the agent dropped itself (the string is the reason).

        ``skip`` and ``start_source`` resume a run: each source continues after
        ``skip[key]`` documents, and the round-robin starts at ``start_source``
        (a chunk's ``next_source``), so a resumed run reads exactly what an
        uninterrupted one would have. Streams are closed when this generator is.
        """
        self.positions = {key: int((skip or {}).get(key, 0)) for key in self.specs}
        streams: dict[str, Iterator[RawDocument] | None] = {key: None for key in self.specs}
        failures = {key: 0 for key in self.specs}
        active = list(self.specs)
        i = active.index(start_source) if start_source in active else 0
        self.next_source = active[i] if active else None
        try:
            while active:
                i %= len(active)
                key = active[i]
                try:
                    if streams[key] is None:
                        streams[key] = self._open(key)
                    document = next(streams[key])       # type: ignore[arg-type]
                except StopIteration:
                    active.pop(i)
                    self.exhausted.append(key)
                    _close(streams[key])
                    continue
                except Exception as exc:  # noqa: BLE001 - a bad source must not stop the run
                    failures[key] += 1
                    give_up = failures[key] >= self.max_errors
                    self.errors.append(SourceError(
                        key, self.positions[key], f"{type(exc).__name__}: {exc}"[:300], give_up
                    ))
                    _close(streams[key])
                    streams[key] = None                 # reopened at the same place next turn
                    if give_up:
                        active.pop(i)
                        self.dropped.append(key)
                    else:
                        i += 1
                    continue
                self.positions[key] += 1
                i += 1
                self.next_source = active[i % len(active)]
                reason = self._reject(document)
                yield reason if reason is not None else document
        finally:
            for stream in streams.values():
                _close(stream)

    def chunks(
        self,
        skip: Mapping[str, int] | None = None,
        start_index: int = 0,
        start_source: str | None = None,
    ) -> Iterator[Chunk]:
        """Chunks of ``chunk_size`` documents; the last one may be smaller."""
        index = start_index
        documents: list[RawDocument] = []
        prefiltered: dict[str, int] = {}

        def make() -> Chunk:
            return Chunk(index, self.domain, self.role, documents, dict(self.positions),
                         prefiltered, self.next_source)

        for item in self.documents(skip, start_source):
            if isinstance(item, str):
                prefiltered[item] = prefiltered.get(item, 0) + 1
                continue
            documents.append(item)
            if len(documents) == self.chunk_size:
                yield make()
                index += 1
                documents, prefiltered = [], {}
        if documents or prefiltered:
            yield make()


class FactoryAgent(RetrievingAgent):
    """Reads the structured, trusted repositories of a domain."""

    role = "factory"


class GeneratorAgent(RetrievingAgent):
    """Reads the large web corpora of a domain, with a cheap prefilter.

    Web text is mostly unusable for a specialised domain, so documents of the
    wrong length or language are dropped here, in the main process, rather
    than being sent to a worker. The Scorer agent applies the same two checks
    again as part of the global filter, so the prefilter changes cost, never
    whether a document is accepted.
    """

    role = "generator"

    def __init__(self, domain: str, config: Config, *, prefilter: bool = True, **kwargs: Any) -> None:
        super().__init__(domain, config, **kwargs)
        cfg = config.global_filter_for(domain)
        off = set(cfg.disabled_checks)
        self._cfg = cfg
        # prefilter=False reads web text untouched (the random-baseline mode).
        self._check_length = prefilter and "length" not in off
        self._langid = None if not prefilter or "language" in off else get_language_identifier()

    def _reject(self, document: RawDocument) -> str | None:
        cfg = self._cfg
        n_chars = len(document.text)
        if self._check_length:
            if n_chars < cfg.min_chars:
                return "prefilter:too_short"
            if n_chars > cfg.max_chars:
                return "prefilter:too_long"
        if self._langid is not None:
            language, confidence = self._langid.identify(document.text, cfg.language_sample_chars)
            if language != cfg.language or confidence < cfg.min_language_confidence:
                return "prefilter:wrong_language"
        return None


def candidate_chunks(
    factory: RetrievingAgent | None,
    generator: RetrievingAgent | None,
    factory_share: float,
    *,
    skip: Mapping[str, Mapping[str, int]] | None = None,
    resume: Mapping[str, Any] | None = None,
) -> Iterator[Chunk]:
    """Interleave the two agents' chunks so that about ``factory_share`` of the
    candidate documents come from the Factory agent. When one agent runs out,
    the other carries on alone. Chunks are numbered in the order they are
    yielded; the order is the same on every run.

    ``resume`` continues an interrupted run exactly. The orchestrator builds it
    from the last chunk it finished: ``{"positions": {role: {source: n}},
    "next_source": {role: key}, "taken": {role: n}, "next_index": n}``.
    ``skip`` is the positions-only form (the round-robin then restarts).
    """
    resume = dict(resume or {})
    positions = dict(resume.get("positions") or skip or {})
    next_source = resume.get("next_source") or {}
    streams = {
        agent.role: agent.chunks(positions.get(agent.role), 0, next_source.get(agent.role))
        for agent in (factory, generator) if agent is not None
    }
    taken = {"factory": 0, "generator": 0, **(resume.get("taken") or {})}
    index = int(resume.get("next_index", 0))
    try:
        yield from _interleave(streams, taken, index, factory_share)
    finally:
        for stream in streams.values():
            stream.close()


def _interleave(streams: dict, taken: dict, index: int, factory_share: float) -> Iterator[Chunk]:
    streams = dict(streams)
    while streams:
        total = taken["factory"] + taken["generator"]
        want_factory = total == 0 and factory_share > 0 or (
            total > 0 and taken["factory"] / total < factory_share
        )
        role = "factory" if want_factory else "generator"
        if role not in streams:
            role = next(iter(streams))
        chunk = next(streams[role], None)
        if chunk is None:
            del streams[role]
            continue
        taken[role] += len(chunk.documents) + sum(chunk.prefiltered.values())
        chunk.index = index
        chunk.taken = dict(taken)
        index += 1
        yield chunk


def build_agents(
    domain: str,
    config: Config,
    *,
    synthetic: bool = False,
    sources: Sequence[str] | None = None,
    prefilter: bool = True,
    **kwargs: Any,
) -> tuple[FactoryAgent | None, GeneratorAgent | None, DomainRoute]:
    """The two retrieving agents for a domain, as routed in the configuration.

    ``synthetic=True`` replaces every source with the built-in generator (the
    offline fallback). ``sources`` replaces the routing: the Factory agent
    alone reads those sources (used for local files and single-corpus
    baselines). ``prefilter=False`` turns off the Generator agent's prefilter.
    An agent is None when its role has no sources.
    """
    routes = domain_routes(config, DOMAINS)
    if domain not in routes:
        raise ConfigError(f"no [domains.{domain}] section")
    route = routes[domain]
    if sources:
        route = replace(route, factory=tuple(sources), generator=(), factory_share=1.0)
        return FactoryAgent(domain, config, sources=list(sources), **kwargs), None, route
    if synthetic:
        key = next((k for k, s in source_specs(config).items() if s.kind == "synthetic"), None)
        if key is None:
            raise ConfigError("no synthetic source is configured under [sources]")
        return (
            FactoryAgent(domain, config, sources=[key], **kwargs),
            GeneratorAgent(domain, config, sources=[key], prefilter=prefilter, **kwargs),
            route,
        )
    factory = FactoryAgent(domain, config, **kwargs) if route.factory else None
    generator = (
        GeneratorAgent(domain, config, prefilter=prefilter, **kwargs) if route.generator else None
    )
    return factory, generator, route


# ---------------------------------------------------------------------------
# Scorer agent
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Evaluation:
    """What the Scorer agent decided about one document."""

    accepted: bool
    label: str                              # e.g. "accepted", "global:too_short", "score:below_threshold"
    record: dict[str, Any] | None = None    # the output record, when accepted
    content_hash: str | None = None         # for exact-duplicate detection
    signature: np.ndarray | None = None     # MinHash signature, for near-duplicate detection
    score: float | None = None              # quality score, when the document got that far
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def stage(self) -> str:
        return self.label.split(":", 1)[0]


class ScorerAgent:
    """Runs every per-document stage for one domain.

        licence policy -> global filter -> domain validator -> quality score
          -> learned classifier -> PII scrub -> hash + MinHash signature -> record

    Building one loads the domain's classifier, so build it once per worker
    process and reuse it.
    """

    def __init__(
        self,
        domain: str,
        config: Config,
        *,
        job_id: str = "",
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.domain = domain
        self.config = config
        self.job_id = job_id
        self.token_counter = token_counter
        self.policy = build_policy(config)
        self.global_filter = GlobalFilter(config.global_filter_for(domain))
        self.validator = build_validator(domain, config)
        self.scorer = build_scorer(domain, config)
        self.classifier = load_classifier(domain, config)       # None when switched off
        self.scrubber = build_scrubber(domain, config)          # None when switched off
        self.minhasher = build_minhasher(config)

    def evaluate(self, document: RawDocument) -> Evaluation:
        try:
            return self._evaluate(document)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a worker
            return Evaluation(False, f"error:{type(exc).__name__}", details={"error": str(exc)[:300]})

    def _evaluate(self, document: RawDocument) -> Evaluation:
        license_result = self.policy.check(document.license)
        if not license_result.passed:
            return Evaluation(False, license_result.label, details=license_result.details)

        stats = TextStats(document.text)
        global_result = self.global_filter.check(stats)
        if not global_result.passed:
            return Evaluation(False, global_result.label, details=global_result.details)

        validator_result = self.validator.validate(stats)
        if not validator_result.passed:
            return Evaluation(
                False, validator_result.label, details={"failed": validator_result.details["failed"]}
            )

        score = self.scorer.score(stats, validator_result.details["signals"], document.authority)
        if not score.passed:
            return Evaluation(
                False, "score:below_threshold", score=score.score,
                details={"threshold": score.threshold, "features": score.features},
            )

        classifier_result = None
        if self.classifier is not None:
            classifier_result = self.classifier.check(stats)
            if not classifier_result.passed:
                return Evaluation(
                    False, classifier_result.label, score=score.score,
                    details=classifier_result.details,
                )

        text, pii_counts = document.text, {}
        if self.scrubber is not None:
            text, pii_counts = self.scrubber.scrub(text)
        # Fingerprints describe the text that is written, so they are taken
        # after scrubbing.
        final = stats if text == document.text else TextStats(text)
        digest = content_hash(text)
        signature = self.minhasher.signature(final.words)

        record = build_record(
            text=text,
            domain=self.domain,
            document=document,
            n_words=final.n_words,
            content_hash=digest,
            score=score,
            global_result=global_result,
            validator_result=validator_result,
            validator_checks=self.validator.CHECKS,
            validator_disabled=self.validator.disabled,
            classifier_result=classifier_result,
            pii_counts=pii_counts,
            config_fingerprint=self.config.fingerprint,
            job_id=self.job_id,
            n_tokens=self.token_counter(text) if self.token_counter is not None else None,
        )
        return Evaluation(
            True, "accepted", record=record, content_hash=digest, signature=signature,
            score=score.score,
        )

    def evaluate_many(self, documents: Sequence[RawDocument]) -> list[Evaluation]:
        return [self.evaluate(document) for document in documents]
