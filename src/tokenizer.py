"""BPE Tokenization Utilities (`microsoft/harrier-oss-v1-0.6b` / `Qwen3` 151k BPE).

Provides fast subword token counting, token-bounded text slicing, and overlapping
passage chunking using the Rust `tokenizers` backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tokenizers import Tokenizer

from src.embedder.harrier import HARRIER_SNAPSHOT

_TOKENIZER_INSTANCE: Tokenizer | None = None
TOKENIZER_JSON_PATH: Path = HARRIER_SNAPSHOT / "tokenizer.json"


def get_bpe_tokenizer() -> Tokenizer:
    """Return the singleton Rust BPE tokenizer (`Qwen2/Qwen3/Harrier`, vocab_size=151,643)."""
    global _TOKENIZER_INSTANCE
    if _TOKENIZER_INSTANCE is None:
        _TOKENIZER_INSTANCE = Tokenizer.from_file(str(TOKENIZER_JSON_PATH))
    return _TOKENIZER_INSTANCE


def encode_bpe_ids(text: str) -> list[int]:
    """Encode `text` into BPE token IDs without special-token padding."""
    if not text:
        return []
    return get_bpe_tokenizer().encode(str(text), add_special_tokens=False).ids


def count_bpe_tokens(text: str) -> int:
    """Return the exact number of BPE subword tokens in `text`."""
    if not text:
        return 0
    return len(get_bpe_tokenizer().encode(str(text), add_special_tokens=False).ids)


def count_bpe_tokens_batch(texts: Sequence[str]) -> np.ndarray:
    """Return an `int32` NumPy array of exact BPE token counts for a batch of texts."""
    if not texts:
        return np.zeros(0, dtype=np.int32)
    tok = get_bpe_tokenizer()
    clean_texts = [str(t) if t is not None else "" for t in texts]
    encodings = tok.encode_batch_fast(clean_texts, add_special_tokens=False)
    return np.fromiter((len(e.ids) for e in encodings), dtype=np.int32, count=len(clean_texts))


def slice_by_tokens(
    text: str,
    max_tokens: int,
    from_end: bool = False,
    ellipsis: str = " ...",
) -> str:
    """Slice `text` to at most `max_tokens` BPE subword tokens (never cutting mid-token or by chars)."""
    if not text or max_tokens <= 0:
        return ""
    tok = get_bpe_tokenizer()
    ids = tok.encode(str(text), add_special_tokens=False).ids
    if len(ids) <= max_tokens:
        return str(text)
    if from_end:
        sliced_ids = ids[-max_tokens:]
        return f"... {tok.decode(sliced_ids, skip_special_tokens=True).lstrip()}"
    sliced_ids = ids[:max_tokens]
    return f"{tok.decode(sliced_ids, skip_special_tokens=True).rstrip()}{ellipsis}"


def split_bpe_tokens(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[str]:
    """Split `text` into overlapping passages of exact `chunk_size` BPE tokens with `overlap` BPE tokens."""
    if not text:
        return [""]
    tok = get_bpe_tokenizer()
    ids = tok.encode(str(text), add_special_tokens=False).ids
    if not ids:
        return [""]
    if len(ids) <= chunk_size:
        return [str(text).strip()]
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        piece_ids = ids[start : start + chunk_size]
        if not piece_ids:
            break
        chunks.append(tok.decode(piece_ids, skip_special_tokens=True).strip())
        if start + chunk_size >= len(ids):
            break
    return chunks
