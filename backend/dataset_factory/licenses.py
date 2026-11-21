"""Licence tracking.

Every document carries the licence of the corpus it came from. This module
knows the obligations of the common licences, enforces a simple policy (which
licences may enter a dataset), and summarises what a finished dataset's mix of
licences requires of whoever uses it.

The obligations recorded here are a practical summary for bookkeeping, not
legal advice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .config import Config, ConfigError
from .models import FilterResult

STAGE = "license"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class License:
    id: str
    name: str
    commercial_use: bool        # may the text be used commercially?
    attribution: bool           # must the source be credited?
    share_alike: bool           # must derived works carry the same licence?
    url: str = ""


_ALL = [
    License("public-domain", "Public domain", True, False, False),
    License("cc0-1.0", "CC0 1.0", True, False, False,
            "https://creativecommons.org/publicdomain/zero/1.0/"),
    License("cc-by-4.0", "CC BY 4.0", True, True, False,
            "https://creativecommons.org/licenses/by/4.0/"),
    License("cc-by-3.0", "CC BY 3.0", True, True, False,
            "https://creativecommons.org/licenses/by/3.0/"),
    License("cc-by-sa-3.0", "CC BY-SA 3.0", True, True, True,
            "https://creativecommons.org/licenses/by-sa/3.0/"),
    License("cc-by-sa-4.0", "CC BY-SA 4.0", True, True, True,
            "https://creativecommons.org/licenses/by-sa/4.0/"),
    License("cc-by-nc-4.0", "CC BY-NC 4.0", False, True, False,
            "https://creativecommons.org/licenses/by-nc/4.0/"),
    License("cc-by-nc-sa-4.0", "CC BY-NC-SA 4.0", False, True, True,
            "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
    License("odc-by-1.0", "ODC Attribution 1.0", True, True, False,
            "https://opendatacommons.org/licenses/by/1-0/"),
    License("apache-2.0", "Apache 2.0", True, True, False,
            "https://www.apache.org/licenses/LICENSE-2.0"),
    License("mit", "MIT", True, True, False, "https://opensource.org/license/mit"),
    License("bsd-3-clause", "BSD 3-Clause", True, True, False,
            "https://opensource.org/license/bsd-3-clause"),
    # A corpus mixing several open-source licences (for example code from many
    # repositories): treat as attribution-required, terms vary per document.
    License("mixed-open-source", "Mixed open-source licences", True, True, False),
    # Text produced by this project's own synthetic generator.
    License("synthetic", "Synthetic (generated, no restrictions)", True, False, False),
]
KNOWN: dict[str, License] = {lic.id: lic for lic in _ALL}

_ALIASES = {
    "cc0": "cc0-1.0",
    "cc-by": "cc-by-4.0",
    "cc-by-sa": "cc-by-sa-4.0",
    "cc-by-nc": "cc-by-nc-4.0",
    "cc-by-nc-sa": "cc-by-nc-sa-4.0",
    "odc-by": "odc-by-1.0",
    "pddl": "public-domain",
    "publicdomain": "public-domain",
    "gfdl": "gfdl",
    "apache-2.0": "apache-2.0",
    "apache2.0": "apache-2.0",
    "": UNKNOWN,
    "other": UNKNOWN,
}


_CC_URL_RE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)")


def normalize_license(value: str | None) -> str:
    """Canonical licence id for a licence string; unrecognised strings are kept as given.

    Also understands the long forms some corpora store on each document, such
    as "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/".
    """
    if value is None:
        return UNKNOWN
    lowered = value.strip().lower()
    match = _CC_URL_RE.search(lowered)
    if match:
        return f"cc-{match.group(1)}-{match.group(2)}"
    if "creativecommons.org/publicdomain/zero" in lowered:
        return "cc0-1.0"
    key = lowered.replace("_", "-").replace(" ", "-")
    return _ALIASES.get(key, key)


class LicensePolicy:
    """Decides which source licences may enter a dataset."""

    def __init__(self, allow_unknown: bool = True, blocked: tuple[str, ...] = ()) -> None:
        self.allow_unknown = allow_unknown
        self.blocked = frozenset(normalize_license(b) for b in blocked)

    def check(self, license_id: str | None) -> FilterResult:
        normalised = normalize_license(license_id)
        details = {"license": normalised}
        if normalised in self.blocked:
            return FilterResult(False, STAGE, "blocked_license", details)
        if normalised not in KNOWN and not self.allow_unknown:
            return FilterResult(False, STAGE, "unknown_license", details)
        return FilterResult(True, STAGE, None, details)


def summarize(counts: Mapping[str, int]) -> dict[str, Any]:
    """What a dataset's mix of licences requires, for the run manifest.

    ``counts`` maps licence id to number of documents.
    """
    by_license = {}
    for license_id, n in sorted(counts.items(), key=lambda item: -item[1]):
        known = KNOWN.get(license_id)
        by_license[license_id] = {
            "documents": n,
            "name": known.name if known else license_id,
            "recognised": known is not None,
        }
    known_ids = [i for i in counts if i in KNOWN]
    unknown_docs = sum(n for i, n in counts.items() if i not in KNOWN)
    return {
        "by_license": by_license,
        "attribution_required": any(KNOWN[i].attribution for i in known_ids),
        "share_alike_present": any(KNOWN[i].share_alike for i in known_ids),
        "non_commercial_present": any(not KNOWN[i].commercial_use for i in known_ids),
        "documents_with_unrecognised_license": unknown_docs,
    }


def build_policy(config: Config) -> LicensePolicy:
    raw = dict(config.licenses)
    unknown = sorted(set(raw) - {"allow_unknown", "blocked"})
    if unknown:
        raise ConfigError(f"[licenses] unknown setting(s): {', '.join(unknown)}")
    allow_unknown = raw.get("allow_unknown", True)
    if not isinstance(allow_unknown, bool):
        raise ConfigError("[licenses] allow_unknown must be true or false")
    blocked = raw.get("blocked", [])
    if not isinstance(blocked, list) or not all(isinstance(b, str) for b in blocked):
        raise ConfigError("[licenses] blocked must be a list of licence ids")
    return LicensePolicy(allow_unknown, tuple(blocked))
