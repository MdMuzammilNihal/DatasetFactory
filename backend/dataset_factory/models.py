"""Core data types shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RawDocument:
    """A candidate document as it arrives from a source, before any filtering."""

    text: str
    source: str                 # e.g. "hf:HuggingFaceFW/fineweb" or "synthetic"
    agent: str = "factory"      # which agent retrieved it: "factory" or "generator"
    source_id: str = ""         # the document's id within its source, if any
    url: str = ""
    license: str = "unknown"    # licence of the source corpus
    authority: float | None = None   # trust level of the source (0-1), used by the scorer
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Outcome of one filtering stage for one document."""

    passed: bool
    stage: str                  # "global", or a domain name for stage 2
    reason: str | None = None   # why it was rejected; None when it passed
    details: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short label for statistics, e.g. ``global:too_short``."""
        return f"{self.stage}:{self.reason}" if self.reason else f"{self.stage}:passed"
