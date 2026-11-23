"""Data sources and domain routing.

A *source* is a corpus documents can be read from: a Hugging Face dataset that
is streamed (never downloaded in full), or the built-in synthetic generator.
Sources are declared under ``[sources.<key>]`` in the configuration.

A *route* says, for one domain, which sources the Factory agent reads
(structured, trusted repositories) and which the Generator agent reads (large
web corpora), and in what proportion. Routes live under ``[domains.<key>]``.
This is the source-routing step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .config import Config, ConfigError, derive_seed
from .licenses import normalize_license
from .models import RawDocument

KINDS = ("huggingface", "synthetic", "jsonl")
BACKEND_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Turning a dataset row into text
# ---------------------------------------------------------------------------

def _pubmedqa(row: Mapping[str, Any]) -> str:
    """PubMedQA stores a structured abstract as parallel lists of labels and texts."""
    context = row["context"]
    return "\n".join(
        f"{label}: {text}" for label, text in zip(context["labels"], context["contexts"])
    )


def _code_with_doc(row: Mapping[str, Any]) -> str:
    """CodeSearchNet keeps the documentation apart from some functions; put it back."""
    code = row["whole_func_string"]
    doc = row.get("func_documentation_string") or ""
    if doc and doc[:40] not in code:
        if row.get("language") in ("java", "javascript", "go", "php"):
            return "/**\n * " + doc.replace("\n", "\n * ") + "\n */\n" + code
        return "# " + doc.replace("\n", "\n# ") + "\n" + code
    return code


TEXT_BUILDERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "pubmedqa": _pubmedqa,
    "code_with_doc": _code_with_doc,
}


def get_field(row: Mapping[str, Any], path: str) -> Any:
    """Read a possibly nested field such as "metadata.license"; None if absent."""
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceSpec:
    key: str
    kind: str = "huggingface"
    path: str = ""                       # dataset repository, e.g. "MedRAG/pubmed"
    name: str = ""                       # dataset configuration, if it has several
    split: str = "train"
    data_files: tuple[str, ...] = ()     # optional subset of the repository's files
    text: str = "text"                   # a field name, or "builder:<name>"
    id_field: str = ""
    url_field: str = ""
    license: str = "unknown"             # licence of the corpus as a whole
    license_field: str = ""              # per-document licence field, if the corpus has one
    authority: float = 0.5               # trust level, 0-1 (the scorer's source_authority)
    shuffle_buffer: int = 1000           # 0 keeps the corpus's own order

    @property
    def label(self) -> str:
        """Name written to provenance, e.g. "hf:MedRAG/pubmed"."""
        if self.kind == "synthetic":
            return "synthetic"
        if self.kind == "jsonl":
            return f"file:{Path(self.path).name}"
        return f"hf:{self.path}" + (f"/{self.name}" if self.name else "")


_SPEC_TYPES = {
    "kind": str, "path": str, "name": str, "split": str, "data_files": list, "text": str,
    "id_field": str, "url_field": str, "license": str, "license_field": str,
    "authority": (int, float), "shuffle_buffer": int,
}


def source_specs(config: Config) -> dict[str, SourceSpec]:
    """Validated source definitions from ``[sources.*]``."""
    specs: dict[str, SourceSpec] = {}
    for key, table in config.sources.items():
        where = f"sources.{key}"
        if not isinstance(table, Mapping):
            raise ConfigError(f"[{where}] must be a table")
        unknown = sorted(set(table) - set(_SPEC_TYPES))
        if unknown:
            raise ConfigError(f"[{where}] unknown setting(s): {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for name, value in table.items():
            expected = _SPEC_TYPES[name]
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ConfigError(f"[{where}] {name} has the wrong type")
            values[name] = tuple(value) if name == "data_files" else value
        spec = SourceSpec(key=key, **values)
        if spec.kind not in KINDS:
            raise ConfigError(f"[{where}] kind must be one of: {', '.join(KINDS)}")
        if spec.kind in ("huggingface", "jsonl") and not spec.path:
            raise ConfigError(f"[{where}] a {spec.kind} source needs a path")
        if spec.text.startswith("builder:") and spec.text[8:] not in TEXT_BUILDERS:
            raise ConfigError(
                f"[{where}] unknown text builder '{spec.text[8:]}'. "
                f"Available: {', '.join(TEXT_BUILDERS)}"
            )
        if not 0.0 <= spec.authority <= 1.0:
            raise ConfigError(f"[{where}] authority must be between 0 and 1")
        if spec.shuffle_buffer < 0:
            raise ConfigError(f"[{where}] shuffle_buffer must not be negative")
        specs[key] = spec
    return specs


@dataclass(frozen=True, slots=True)
class DomainRoute:
    domain: str
    label: str
    description: str
    factory: tuple[str, ...]        # source keys read by the Factory agent
    generator: tuple[str, ...]      # source keys read by the Generator agent
    factory_share: float            # share of candidates taken from the Factory agent


def domain_routes(config: Config, valid_domains: tuple[str, ...] | None = None) -> dict[str, DomainRoute]:
    """Validated routing from ``[domains.*]``."""
    specs = source_specs(config)
    routes: dict[str, DomainRoute] = {}
    for domain, table in config.domains.items():
        where = f"domains.{domain}"
        if valid_domains is not None and domain not in valid_domains:
            raise ConfigError(f"[{where}] is not a known domain")
        if not isinstance(table, Mapping):
            raise ConfigError(f"[{where}] must be a table")
        unknown = sorted(
            set(table) - {"label", "description", "factory", "generator", "factory_share"}
        )
        if unknown:
            raise ConfigError(f"[{where}] unknown setting(s): {', '.join(unknown)}")
        lists = {}
        for role in ("factory", "generator"):
            keys = table.get(role, [])
            if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
                raise ConfigError(f"[{where}] {role} must be a list of source keys")
            missing = [k for k in keys if k not in specs]
            if missing:
                raise ConfigError(f"[{where}] {role} names unknown source(s): {', '.join(missing)}")
            lists[role] = tuple(keys)
        if not lists["factory"] and not lists["generator"]:
            raise ConfigError(f"[{where}] needs at least one factory or generator source")
        share = table.get("factory_share", 0.5)
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
            raise ConfigError(f"[{where}] factory_share must be a number between 0 and 1")
        routes[domain] = DomainRoute(
            domain=domain,
            label=str(table.get("label", domain.title())),
            description=str(table.get("description", "")),
            factory=lists["factory"],
            generator=lists["generator"],
            factory_share=float(share),
        )
    return routes


# ---------------------------------------------------------------------------
# Reading documents
# ---------------------------------------------------------------------------

def _open_stream(spec: SourceSpec):
    """Open a streaming dataset; fall back to the Hub's Parquet conversion for
    corpora that are still published as a loading script."""
    from datasets import load_dataset   # imported here so offline use needs no datasets

    kwargs: dict[str, Any] = {"path": spec.path, "split": spec.split, "streaming": True}
    if spec.name:
        kwargs["name"] = spec.name
    if spec.data_files:
        kwargs["data_files"] = list(spec.data_files)
    try:
        return load_dataset(**kwargs)
    except Exception as first_error:
        try:
            return load_dataset(revision="refs/convert/parquet", **kwargs)
        except Exception:
            raise first_error


def row_to_document(spec: SourceSpec, row: Mapping[str, Any], agent: str) -> RawDocument | None:
    """Convert one dataset row; None when the row has no usable text."""
    try:
        if spec.text.startswith("builder:"):
            text = TEXT_BUILDERS[spec.text[8:]](row)
        else:
            text = get_field(row, spec.text)
    except (KeyError, TypeError, AttributeError):
        return None
    if not isinstance(text, str) or not text.strip():
        return None

    license_id = spec.license
    if spec.license_field:
        per_document = get_field(row, spec.license_field)
        if isinstance(per_document, str) and per_document.strip():
            license_id = per_document
    source_id = get_field(row, spec.id_field) if spec.id_field else None
    url = get_field(row, spec.url_field) if spec.url_field else None
    return RawDocument(
        text=text,
        source=spec.label,
        agent=agent,
        source_id="" if source_id is None else str(source_id),
        url=url if isinstance(url, str) else "",
        license=normalize_license(license_id),
        authority=spec.authority,
    )


def open_source(
    spec: SourceSpec, *, domain: str, agent: str, seed: int, skip: int = 0
) -> Iterator[RawDocument]:
    """Stream documents from a source.

    ``skip`` resumes a stream: that many documents are passed over first. The
    order is fixed by ``seed``, so skipping lands on the same place every time.
    """
    if spec.kind == "synthetic":
        from .synthetic import SyntheticGenerator

        generator = SyntheticGenerator(domain, role=agent, seed=derive_seed(seed, "synthetic", agent))
        return generator.documents(skip=skip)
    if spec.kind == "jsonl":
        return _read_jsonl(spec, agent, skip)
    return _stream_huggingface(spec, agent, derive_seed(seed, "shuffle", spec.key), skip)


def _stream_huggingface(spec: SourceSpec, agent: str, seed: int, skip: int) -> Iterator[RawDocument]:
    stream = _open_stream(spec)
    if spec.shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=spec.shuffle_buffer)
    skipped = 0
    for row in stream:
        document = row_to_document(spec, row, agent)
        if document is None:
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield document


def _read_jsonl(spec: SourceSpec, agent: str, skip: int) -> Iterator[RawDocument]:
    """Read a local JSON-lines file (one object per line), in file order.

    A relative path is taken from the backend folder. Rows are converted like
    dataset rows, so ``text``, ``id_field`` and the other settings apply.
    """
    path = Path(spec.path)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            document = row_to_document(spec, json.loads(line), agent)
            if document is None:
                continue
            if skipped < skip:
                skipped += 1
                continue
            yield document
