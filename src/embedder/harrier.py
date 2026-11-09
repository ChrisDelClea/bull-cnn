"""Cached GPU/HTTP Embedding Provider for Microsoft Harrier (`microsoft/harrier-oss-v1-0.6b`).

Method & Literature References:
- Instruction-Tuned Decoder-Based Text Embeddings (Last-Token Pooling):
  [Improving Text Embeddings with Large Language Models (Wang et al., ACL 2024, arXiv:2401.00368)](https://arxiv.org/abs/2401.00368)
- Microsoft Harrier Open-Source Embedding Model:
  [microsoft/harrier-oss-v1-0.6b on Hugging Face](https://huggingface.co/microsoft/harrier-oss-v1-0.6b)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence
import urllib.request

import numpy as np

from src.embedder.base import BaseEmbedder

HARRIER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/"
    / "models--microsoft--harrier-oss-v1-0.6b/snapshots/"
    / "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
)
TORCH_PYTHON = Path(os.environ.get("TORCH_PYTHON", sys.executable))
QUERY_PREFIX = (
    "Instruct: Given a news search query, retrieve relevant news articles or passages "
    "that answer the query\nQuery: "
)


class CachedHarrierEmbedder(BaseEmbedder):
    """Computes and caches 1024-d L2-normalized embeddings using `microsoft/harrier-oss-v1-0.6b`."""

    def __init__(
        self,
        cache_dir: str | Path = "../data/embeddings_cache",
        http_url: str = "http://127.0.0.1:18082/embed",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http_url = http_url
        self.query_cache_path = self.cache_dir / "query_embeddings_cache.json"
        self.query_cache: dict[str, list[float]] = {}
        if self.query_cache_path.exists():
            self.query_cache = json.loads(self.query_cache_path.read_text(encoding="utf-8"))

    @property
    def dimension(self) -> int:
        return 1024

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query with instruction prefix and disk caching."""
        formatted = f"{QUERY_PREFIX}{query.strip()}"
        if formatted in self.query_cache:
            return np.asarray(self.query_cache[formatted], dtype=np.float32)

        payload = json.dumps({"inputs": formatted}).encode("utf-8")
        req = urllib.request.Request(
            self.http_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        vec = parsed[0] if (isinstance(parsed, list) and parsed and isinstance(parsed[0], list)) else parsed

        self.query_cache[formatted] = vec
        self.query_cache_path.write_text(
            json.dumps(self.query_cache), encoding="utf-8"
        )
        return np.asarray(vec, dtype=np.float32)

    def encode_passages_cached(
        self,
        passages: Sequence[str],
        cache_name: str,
        max_length: int = 320,
        batch_size: int = 12,
    ) -> np.ndarray:
        """Load cached `(M, 1024)` embeddings from disk or compute them on GPU (`cuda:0`)."""
        digest = hashlib.sha256(
            f"{cache_name}::{len(passages)}::{passages[0][:80] if passages else ''}".encode("utf-8")
        ).hexdigest()[:12]
        npy_path = self.cache_dir / f"{cache_name}_{len(passages)}_{digest}.npy"
        if npy_path.exists():
            return np.load(npy_path).astype(np.float32)

        tmp_in = self.cache_dir / f"_tmp_{cache_name}_in.json"
        tmp_in.write_text(json.dumps(list(passages), ensure_ascii=False), encoding="utf-8")

        worker_code = f"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, torch, numpy as np
from transformers import AutoModel, AutoTokenizer

snap = {str(HARRIER_SNAPSHOT)!r}
with open({str(tmp_in)!r}, "r", encoding="utf-8") as f:
    texts = json.load(f)

tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    snap, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
).eval()
if device == "cuda:0":
    for layer in model.layers:
        layer.to(device)
    model.norm.to(device)
    model.rotary_emb.to(device)

vecs = np.zeros((len(texts), 1024), dtype=np.float16)
bs = {int(batch_size)}
with torch.inference_mode():
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length={int(max_length)}, return_tensors="pt")
        inputs_embeds = model.embed_tokens(enc["input_ids"]).to(device)
        mask = enc["attention_mask"].to(device)
        hidden = model(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=False).last_hidden_state
        pos = torch.arange(mask.shape[1], device=device)
        last = (pos[None, :] * mask).amax(dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=device), last].float()
        normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs[i : i + len(batch)] = normed.cpu().numpy().astype(np.float16)

np.save({str(npy_path)!r}, vecs)
"""
        subprocess.run([str(TORCH_PYTHON), "-c", worker_code], check=True)
        tmp_in.unlink(missing_ok=True)
        return np.load(npy_path).astype(np.float32)
