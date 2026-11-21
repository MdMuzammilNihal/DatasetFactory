"""Exact token counts with the Pythia (GPT-NeoX) tokenizer.

Token budgets are the unit the paper compares datasets by (200M tokens per
run), so accepted documents are counted with the tokenizer of the models that
will be trained on them. Only ``tokenizer.json`` is fetched from the Hugging
Face Hub, once; afterwards it is read from the local cache, also offline.

Counts exclude special tokens. Training adds one end-of-text token between
documents, so a packed dataset holds ``tokens + documents`` tokens.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Sequence

DEFAULT_TOKENIZER = "EleutherAI/pythia-160m"
NONE = ("", "none")

TokenCounter = Callable[[str], int]


class TokenizerUnavailable(RuntimeError):
    """The tokenizer could not be loaded (not cached and no connection)."""


@functools.lru_cache(maxsize=4)
def _load(name: str):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - declared dependencies
        raise TokenizerUnavailable(
            "token counting needs the 'tokenizers' and 'huggingface_hub' packages"
        ) from exc
    try:
        try:   # the cached copy first: no network request, works offline
            path = hf_hub_download(name, "tokenizer.json", local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            path = hf_hub_download(name, "tokenizer.json")
    except Exception as exc:  # noqa: BLE001 - offline and not cached, or a bad name
        raise TokenizerUnavailable(
            f"could not load the tokenizer of '{name}': {type(exc).__name__}: {exc}. "
            "Connect once to cache it, or set tokenizer = \"none\" to run without token counts."
        ) from exc
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


class TokenCounterImpl:
    """Callable token counter; ``count_many`` uses the tokenizer's batch mode."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tokenizer = _load(name)

    def __call__(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_many(self, texts: Sequence[str]) -> list[int]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(e.ids) for e in encodings]


def load_token_counter(name: str | None) -> TokenCounterImpl | None:
    """The counter for a tokenizer name, or None when counting is turned off."""
    if name is None or name.strip().lower() in NONE:
        return None
    return TokenCounterImpl(name)
