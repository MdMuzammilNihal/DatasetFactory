"""Stage 2: domain-specific validators.

``build_validator("medical", config)`` returns the validator for a domain with
the thresholds and switched-off checks from the configuration applied.
"""

from __future__ import annotations

from ..config import Config, ConfigError
from .base import DomainValidator
from .code import CodeValidator
from .general import GeneralValidator
from .legal import LegalValidator
from .medical import MedicalValidator
from .news import NewsValidator
from .scientific import ScientificValidator

VALIDATORS: dict[str, type[DomainValidator]] = {
    cls.domain: cls
    for cls in (
        GeneralValidator,
        NewsValidator,
        MedicalValidator,
        LegalValidator,
        ScientificValidator,
        CodeValidator,
    )
}

DOMAINS = tuple(VALIDATORS)


def build_validator(domain: str, config: Config | None = None) -> DomainValidator:
    try:
        cls = VALIDATORS[domain]
    except KeyError:
        raise ConfigError(
            f"unknown domain '{domain}'. Valid domains: {', '.join(DOMAINS)}"
        ) from None
    settings = config.validators.get(domain) if config is not None else None
    return cls(settings)


def check_validator_config(config: Config) -> None:
    """Raise ``ConfigError`` if any ``[validators.*]`` section is invalid."""
    unknown = sorted(set(config.validators) - set(VALIDATORS))
    if unknown:
        raise ConfigError(f"[validators] unknown domain(s): {', '.join(unknown)}")
    for domain in VALIDATORS:
        build_validator(domain, config)


__all__ = [
    "DOMAINS",
    "VALIDATORS",
    "DomainValidator",
    "build_validator",
    "check_validator_config",
]
