"""Personal-information scrubbing.

Accepted documents are scanned for e-mail addresses, phone numbers, public IP
addresses, payment-card numbers and ID numbers. Each match is replaced by a
placeholder such as ``[EMAIL]``, and the number of replacements per type is
recorded in the document's provenance.

The patterns are deliberately conservative: a missed match is preferable to
mangling ordinary numbers such as dates, citations or version numbers. Card
numbers must pass the Luhn checksum, and private or reserved IP ranges (which
identify nobody) are left alone.

This is pattern matching, not a guarantee. It removes the common, machine-
recognisable identifiers; it does not find names, addresses or health details.
"""

from __future__ import annotations

import re
from typing import Any

from .config import Config, ConfigError

PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ip": "[IP_ADDRESS]",
    "card": "[CARD_NUMBER]",
    "id_number": "[ID_NUMBER]",
}
PII_TYPES = tuple(PLACEHOLDERS)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

_PHONE_RE = re.compile(
    r"(?<![\w.+/-])"
    r"(?:"
    r"(?:\+?1[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"          # 555-123-4567, (555) 123 4567
    r"|\+\d{1,3}(?:[\s.-]\(?\d{1,4}\)?){2,5}"                     # +44 20 7946 0958
    r")"
    r"(?![\w-])"
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w.])")
_PRIVATE_IP_RE = re.compile(
    r"^(?:10\.|127\.|0\.|169\.254\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|22[4-9]\.|2[3-5]\d\.)"
)

_CARD_RE = re.compile(r"(?<![\w.-])\d(?:[ -]?\d){12,18}(?![\w.-])")

# US Social Security format. Other national ID formats are too close to
# ordinary numbers to match safely without context.
_ID_NUMBER_RE = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")


def _card_network_prefix(digits: str) -> bool:
    """Does the number start like a real payment card of that length?

    About one in ten random digit strings passes the Luhn checksum (grant
    numbers and index lists in papers did), so the prefix is checked as well.
    """
    n = len(digits)
    two, four = int(digits[:2]), int(digits[:4])
    if digits[0] == "4":                                   # Visa
        return n in (13, 16, 19)
    if 51 <= two <= 55 or 2221 <= four <= 2720:            # Mastercard
        return n == 16
    if two in (34, 37):                                    # American Express
        return n == 15
    if four == 6011 or two == 65 or 644 <= int(digits[:3]) <= 649:   # Discover
        return n in (16, 19)
    if two == 35:                                          # JCB
        return n in (16, 19)
    if two in (36, 38) or 300 <= int(digits[:3]) <= 305:   # Diners Club
        return n == 14
    if two == 62:                                          # UnionPay
        return 16 <= n <= 19
    return False


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


class PiiScrubber:
    """Replaces personal identifiers in text with placeholders."""

    def __init__(self, types: tuple[str, ...] = PII_TYPES) -> None:
        bad = sorted(set(types) - set(PII_TYPES))
        if bad:
            raise ConfigError(
                f"[pii] unknown type(s): {', '.join(bad)}. Valid types: {', '.join(PII_TYPES)}"
            )
        self.types = frozenset(types)

    def scrub(self, text: str) -> tuple[str, dict[str, int]]:
        """Return the scrubbed text and how many replacements were made per type."""
        counts: dict[str, int] = {}
        types = self.types
        has_digit = any(char.isdigit() for char in text[:20000]) or any(
            char.isdigit() for char in text[20000:]
        )

        if has_digit and "card" in types:
            def replace_card(match: re.Match[str]) -> str:
                digits = re.sub(r"\D", "", match.group())
                if 13 <= len(digits) <= 19 and _card_network_prefix(digits) and _luhn_valid(digits):
                    counts["card"] = counts.get("card", 0) + 1
                    return PLACEHOLDERS["card"]
                return match.group()

            text = _CARD_RE.sub(replace_card, text)

        if has_digit and "id_number" in types:
            text, n = _ID_NUMBER_RE.subn(PLACEHOLDERS["id_number"], text)
            if n:
                counts["id_number"] = n

        if has_digit and "phone" in types:
            def replace_phone(match: re.Match[str]) -> str:
                digits = sum(char.isdigit() for char in match.group())
                if 10 <= digits <= 15:
                    counts["phone"] = counts.get("phone", 0) + 1
                    return PLACEHOLDERS["phone"]
                return match.group()

            text = _PHONE_RE.sub(replace_phone, text)

        if has_digit and "ip" in types:
            def replace_ip(match: re.Match[str]) -> str:
                if _PRIVATE_IP_RE.match(match.group()):
                    return match.group()
                counts["ip"] = counts.get("ip", 0) + 1
                return PLACEHOLDERS["ip"]

            text = _IP_RE.sub(replace_ip, text)

        if "email" in types and "@" in text:
            text, n = _EMAIL_RE.subn(PLACEHOLDERS["email"], text)
            if n:
                counts["email"] = n

        return text, counts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def pii_settings(config: Config) -> dict[str, Any]:
    raw = dict(config.pii)
    unknown = sorted(set(raw) - {"enabled", "types", "skip_by_domain"})
    if unknown:
        raise ConfigError(f"[pii] unknown setting(s): {', '.join(unknown)}")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("[pii] enabled must be true or false")
    types = raw.get("types", list(PII_TYPES))
    if not isinstance(types, list) or set(types) - set(PII_TYPES):
        raise ConfigError(f"[pii] types must be a list drawn from: {', '.join(PII_TYPES)}")
    skip = raw.get("skip_by_domain", {})
    if not isinstance(skip, dict) or not all(
        isinstance(v, list) and not set(v) - set(PII_TYPES) for v in skip.values()
    ):
        raise ConfigError("[pii.skip_by_domain] must map domains to lists of PII types")
    return {"enabled": enabled, "types": types, "skip_by_domain": skip}


def build_scrubber(domain: str, config: Config) -> PiiScrubber | None:
    """The scrubber for a domain, or None when scrubbing is switched off."""
    settings = pii_settings(config)
    if not settings["enabled"]:
        return None
    skipped = set(settings["skip_by_domain"].get(domain, ()))
    types = tuple(t for t in settings["types"] if t not in skipped)
    return PiiScrubber(types) if types else None
