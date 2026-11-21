"""Source-code validator: syntax blocks, function definitions, comment structure."""

from __future__ import annotations

import re
from typing import Mapping

from ..textstats import TextStats
from .base import DomainValidator, per_100_words, term_density
from .lexicons import CODE_KEYWORDS

_FUNCTION_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+\w+\s*\("                                   # Python
    r"|\bfunction\s*\w*\s*\("                                           # JS, PHP
    r"|\bfunc\s+(?:\([^)]*\)\s*)?\w+\s*\("                              # Go
    r"|\bfn\s+\w+\s*[<(]"                                               # Rust
    r"|^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|virtual|"
    r"override|inline|extern|const)\s+)+[\w<>\[\],.?\s]*?\b\w+\s*\([^;{}]*\)\s*"
    r"(?:throws\s+[\w.,\s]+)?\{"                                        # Java, C#, C++
    r"|^\s*(?:[\w:<>*&]+\s+)+[\w:~]+\s*\([^;{}]*\)\s*(?:const\s*)?\{"   # C, C++
    r"|\)\s*=>\s*[{(\w]|\b\w+\s*=\s*(?:async\s+)?function\b"            # JS arrows
    r"|^\s*class\s+\w+[\s\w.,<>()]*[:{]"                                # classes
    r"|^\s*(?:sub|proc)\s+\w+",                                         # Perl, Tcl
    re.MULTILINE,
)

# Comment markers at the start of a line, docstring quotes, and trailing "//"
# comments. A trailing "//" only counts after ; { } or , so that Python's
# floor-division operator ("a // b") is not mistaken for a comment.
_COMMENT_RE = re.compile(
    r"^\s*(?:#(?!!)|//|/\*|\*\s|\*/|--\s|;;)|\"\"\"|'''|[;{},]\s*//\s|/\*\*", re.MULTILINE
)

# Lines that look like statements or block openers rather than prose.
_STRUCTURAL_LINE_RE = re.compile(
    r"[{};:]\s*$|^\s*[})\]]|^\s*(?:return|if|for|while|else|elif|import|from|try|except|"
    r"catch|switch|case|var|let|const|package|use|include|require)\b|[=!<>+\-*/]=|\w\(.*\)"
)


class CodeValidator(DomainValidator):
    domain = "code"
    CHECKS = ("syntax_blocks", "function_definition", "comment_structure", "balanced_delimiters")
    DEFAULTS = {
        # Docstrings and comment blocks are not structural lines, so well
        # documented code sits around 0.4-0.6; prose sits near 0.
        "min_structural_line_ratio": 0.30,
        "min_lines": 4,
        "min_function_definitions": 1,
        "min_comments": 1,
        "max_delimiter_imbalance": 0.10,
    }

    def analyze(self, stats: TextStats) -> dict[str, float]:
        text = stats.text
        lines = stats.lines
        n_lines = len(lines)
        structural = sum(1 for line in lines if _STRUCTURAL_LINE_RE.search(line))
        functions = len(_FUNCTION_RE.findall(text))
        comments = len(_COMMENT_RE.findall(text))

        opens = text.count("(") + text.count("{") + text.count("[")
        closes = text.count(")") + text.count("}") + text.count("]")
        delimiters = opens + closes
        imbalance = abs(opens - closes) / delimiters if delimiters else 1.0

        return {
            "term_density": term_density(stats, CODE_KEYWORDS),
            "citations_per_100w": per_100_words(comments, stats.n_words),  # comments play this role
            "structure_cues": functions,
            "n_lines": n_lines,
            "structural_line_ratio": structural / n_lines if n_lines else 0.0,
            "function_definitions": functions,
            "comments": comments,
            "delimiters": delimiters,
            "delimiter_imbalance": imbalance,
        }

    def run_checks(self, s: Mapping[str, float]) -> dict[str, bool]:
        t = self.t
        return {
            "syntax_blocks": (
                s["n_lines"] >= t.min_lines
                and s["structural_line_ratio"] >= t.min_structural_line_ratio
            ),
            "function_definition": s["function_definitions"] >= t.min_function_definitions,
            "comment_structure": s["comments"] >= t.min_comments,
            "balanced_delimiters": (
                s["delimiters"] >= 4 and s["delimiter_imbalance"] <= t.max_delimiter_imbalance
            ),
        }
