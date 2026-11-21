"""Configuration loading, validation and seed handling.

All tunable behaviour lives in one TOML file (``config/default.toml``). The
loader rejects unknown keys and wrong types so a typo cannot silently change a
run, and it computes a fingerprint that is stored with every dataset produced.
"""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.toml"

# Names accepted in ``global_filter.disabled_checks``.
GLOBAL_CHECKS = (
    "length",
    "encoding",
    "alpha_ratio",
    "word_length",
    "markup",
    "language",
    "stopwords",
    "information_density",
    "repetition",
    "boilerplate",
)

_KNOWN_SECTIONS = {
    "meta", "global_filter", "validators", "scoring", "classifier", "dedup", "pii", "licenses",
    "sources", "domains", "runtime",
}


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""


@dataclass(frozen=True, slots=True)
class GlobalFilterConfig:
    language: str = "en"
    min_language_confidence: float = 0.25
    language_sample_chars: int = 2000

    min_chars: int = 200
    max_chars: int = 200_000
    min_words: int = 50

    max_replacement_char_ratio: float = 0.002
    min_alpha_ratio: float = 0.60
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0

    max_markup_per_1k_chars: float = 5.0
    min_stopword_ratio: float = 0.15
    min_unique_word_ratio: float = 0.25

    max_duplicate_line_ratio: float = 0.30
    max_repeated_ngram_ratio: float = 0.30

    min_boilerplate_hits: int = 3
    max_boilerplate_per_100_words: float = 1.0
    max_boilerplate_distinct: int = 4

    disabled_checks: tuple[str, ...] = ()


def _apply(base: GlobalFilterConfig, data: Mapping[str, Any], where: str) -> GlobalFilterConfig:
    """Return ``base`` with the values in ``data`` applied, validating each one."""
    defaults = {f.name: getattr(base, f.name) for f in fields(GlobalFilterConfig)}
    unknown = sorted(set(data) - set(defaults))
    if unknown:
        raise ConfigError(f"[{where}] unknown setting(s): {', '.join(unknown)}")

    values: dict[str, Any] = {}
    for key, value in data.items():
        default = defaults[key]
        if key == "disabled_checks":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ConfigError(f"[{where}] disabled_checks must be a list of check names")
            bad = sorted(set(value) - set(GLOBAL_CHECKS))
            if bad:
                raise ConfigError(
                    f"[{where}] unknown check name(s): {', '.join(bad)}. "
                    f"Valid names: {', '.join(GLOBAL_CHECKS)}"
                )
            values[key] = tuple(value)
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ConfigError(f"[{where}] {key} must be text")
            values[key] = value
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"[{where}] {key} must be a whole number")
            values[key] = value
        else:  # float
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"[{where}] {key} must be a number")
            values[key] = float(value)
    return replace(base, **values)


@dataclass(frozen=True)
class Config:
    version: str
    seed: int
    global_filter: GlobalFilterConfig
    global_filter_overrides: Mapping[str, GlobalFilterConfig]
    fingerprint: str
    source_path: str
    # Raw ``[validators.<domain>]`` tables. Each validator checks its own
    # settings when it is built (see ``validators.check_validator_config``).
    validators: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # Raw ``[scoring]`` table, checked by ``scoring.check_scoring_config``.
    scoring: Mapping[str, Any] = field(default_factory=dict)
    # Raw ``[classifier]`` table, checked by ``classifier.classifier_settings``.
    classifier: Mapping[str, Any] = field(default_factory=dict)
    # Raw ``[dedup]``, ``[pii]`` and ``[licenses]`` tables, each checked by its own module.
    dedup: Mapping[str, Any] = field(default_factory=dict)
    pii: Mapping[str, Any] = field(default_factory=dict)
    licenses: Mapping[str, Any] = field(default_factory=dict)
    # Raw ``[sources.*]`` and ``[domains.*]`` tables, checked by ``sources.py``.
    sources: Mapping[str, Any] = field(default_factory=dict)
    domains: Mapping[str, Any] = field(default_factory=dict)
    # Raw ``[runtime]`` table, checked by ``orchestrator.runtime_settings``.
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def global_filter_for(self, domain: str | None) -> GlobalFilterConfig:
        """Global-filter settings for a domain, with its overrides applied."""
        if domain is None:
            return self.global_filter
        return self.global_filter_overrides.get(domain, self.global_filter)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a configuration file (the default one if no path is given)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {cfg_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{cfg_path} is not valid TOML: {exc}") from exc

    unknown = sorted(set(raw) - _KNOWN_SECTIONS)
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")

    meta = raw.get("meta", {})
    version = meta.get("config_version", "0")
    seed = meta.get("seed", 0)
    if not isinstance(version, str):
        raise ConfigError("[meta] config_version must be text")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("[meta] seed must be a whole number")

    gf_raw = dict(raw.get("global_filter", {}))
    overrides_raw = gf_raw.pop("overrides", {})
    if not isinstance(overrides_raw, dict):
        raise ConfigError("[global_filter.overrides] must be a table of domains")

    base = _apply(GlobalFilterConfig(), gf_raw, "global_filter")
    overrides = {
        domain: _apply(base, data, f"global_filter.overrides.{domain}")
        for domain, data in overrides_raw.items()
    }

    validators_raw = raw.get("validators", {})
    if not isinstance(validators_raw, dict) or not all(
        isinstance(table, dict) for table in validators_raw.values()
    ):
        raise ConfigError("[validators] must contain one table per domain")

    scoring_raw = raw.get("scoring", {})
    if not isinstance(scoring_raw, dict):
        raise ConfigError("[scoring] must be a table")

    classifier_raw = raw.get("classifier", {})
    if not isinstance(classifier_raw, dict):
        raise ConfigError("[classifier] must be a table")

    simple_tables = {}
    for section in ("dedup", "pii", "licenses", "sources", "domains", "runtime"):
        table = raw.get(section, {})
        if not isinstance(table, dict):
            raise ConfigError(f"[{section}] must be a table")
        simple_tables[section] = table

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    return Config(
        version=version,
        seed=seed,
        global_filter=base,
        global_filter_overrides=overrides,
        fingerprint=fingerprint,
        source_path=str(cfg_path),
        validators=validators_raw,
        scoring=scoring_raw,
        classifier=classifier_raw,
        **simple_tables,
    )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable child seed, e.g. ``derive_seed(42, "medical", chunk_index)``.

    Uses a fixed hash (not Python's per-process ``hash``) so every worker
    process, on every run, derives the same seed from the same inputs.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") & 0x7FFF_FFFF


def seed_everything(seed: int) -> None:
    """Seed Python's and NumPy's random generators."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        return
    np.random.seed(seed)
