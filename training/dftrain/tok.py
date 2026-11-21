"""The Pythia (GPT-NeoX) tokenizer, read from the local Hugging Face cache first."""

from __future__ import annotations

import functools
import os

DEFAULT_TOKENIZER = "EleutherAI/pythia-160m"
EOS_TOKEN = "<|endoftext|>"


@functools.lru_cache(maxsize=2)
def load_tokenizer(name: str = DEFAULT_TOKENIZER):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    try:
        path = hf_hub_download(name, "tokenizer.json", local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        path = hf_hub_download(name, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    tok.no_padding()
    tok.no_truncation()
    return tok


def eos_id(tok) -> int:
    i = tok.token_to_id(EOS_TOKEN)
    if i is None:
        raise ValueError(f"tokenizer has no {EOS_TOKEN} token")
    return i


def encode_many(tok, texts: list[str]) -> list[list[int]]:
    """Token ids without special tokens (the curation counts use the same rule)."""
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
