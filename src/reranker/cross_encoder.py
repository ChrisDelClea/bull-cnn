"""Cross-Encoder Reranker using local `Qwen/Qwen3-Reranker-0.6B` service (port 18083).

Method & Literature References:
- Qwen3 Cross-Encoder Reranking (`P(yes | Instruction, Query, Document)`):
  [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (Zhang et al., June 2025, arXiv:2506.05176)](https://arxiv.org/abs/2506.05176)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import pandas as pd

from src.index.base import BaseSearchIndex
from src.index.dense import DensePassageIndex
from src.reranker.base import BaseReranker

QWEN_RERANKER_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/"
    / "models--Qwen--Qwen3-Reranker-0.6B/snapshots/"
    / "e61197ed45024b0ed8a2d74b80b4d909f1255473"
)
TORCH_PYTHON = Path(os.environ.get("TORCH_PYTHON", sys.executable))


class QwenCrossEncoderReranker(BaseReranker):
    """Scores `(query, document)` candidates via `Qwen/Qwen3-Reranker-0.6B` with GPU batching & disk caching."""

    def __init__(
        self,
        base_index: BaseSearchIndex,
        df: pd.DataFrame,
        rerank_depth: int = 25,
        passage_index: DensePassageIndex | None = None,
        endpoint_url: str = "http://127.0.0.1:18083/rerank",
        cache_path: str | Path = "../data/reranker_qwen_cache.json",
    ) -> None:
        super().__init__(base_index=base_index, rerank_depth=rerank_depth)
        self.df_by_id = df.set_index(df["Index"].astype(str))
        self.passage_index = passage_index
        self.endpoint_url = endpoint_url
        self.cache_path = Path(cache_path)
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def prewarm_queries_gpu(self, queries: list[str], depth: int | None = None) -> int:
        """Batch-score all uncached `(query, candidate)` pairs up to `depth` directly on GPU (`cuda:0`)."""
        max_k = depth or self.rerank_depth
        missing_items: list[dict[str, str]] = []
        seen_keys: set[str] = set()
        for q in queries:
            q_str = q.strip()
            stage1_hits = self.base_index.search_with_scores(q_str, top_k=max_k)
            for doc_id, _ in stage1_hits:
                d_str = str(doc_id)
                key = f"{q_str}::{d_str}"
                if key not in self.cache and key not in seen_keys and d_str in self.df_by_id.index:
                    seen_keys.add(key)
                    missing_items.append(
                        {"key": key, "query": q_str, "doc": self._build_doc_text(q_str, d_str)}
                    )

        if not missing_items:
            return 0

        tmp_in = self.cache_path.parent / "_tmp_qwen_rerank_in.json"
        tmp_out = self.cache_path.parent / "_tmp_qwen_rerank_out.json"
        tmp_in.write_text(json.dumps(missing_items, ensure_ascii=False), encoding="utf-8")

        worker = f"""
import json, torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

snap = {str(QWEN_RERANKER_SNAPSHOT)!r}
with open({str(tmp_in)!r}, "r", encoding="utf-8") as f:
    items = json.load(f)

tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, padding_side="left")
no_id = tok.encode("no", add_special_tokens=False)[0]
yes_id = tok.encode("yes", add_special_tokens=False)[0]
prefix = tok.encode(
    '<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n',
    add_special_tokens=False,
)
suffix = tok.encode("<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", add_special_tokens=False)
instr = "Given a web search query, retrieve relevant passages that answer the query"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    snap, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(device).eval()

prepared = []
for it in items:
    body = tok.encode(f"<Instruct>: {{instr}}\\n<Query>: {{it['query']}}\\n<Document>: {{it['doc']}}", add_special_tokens=False)[:1024]
    prepared.append(prefix + body + suffix)

out_scores = {{}}
bs = 32 if device == "cuda:0" else 8
with torch.inference_mode():
    for i in range(0, len(prepared), bs):
        batch_ids = prepared[i : i + bs]
        padded = tok.pad({{"input_ids": batch_ids}}, padding=True, return_tensors="pt").to(device)
        mask = padded["attention_mask"]
        pos = mask.cumsum(-1) - 1
        pos.masked_fill_(mask == 0, 0)
        logits = model(**padded, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        pair_logits = torch.stack((logits[:, no_id], logits[:, yes_id]), dim=1).float()
        probs = F.log_softmax(pair_logits, dim=1)[:, 1].exp().cpu().tolist()
        for j, p in enumerate(probs):
            out_scores[items[i + j]["key"]] = float(p)

with open({str(tmp_out)!r}, "w", encoding="utf-8") as f:
    json.dump(out_scores, f)
"""
        subprocess.run([str(TORCH_PYTHON), "-c", worker], check=True)
        scored = json.loads(tmp_out.read_text(encoding="utf-8"))
        self.cache.update(scored)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        return len(scored)

    def _build_doc_text(self, query: str, doc_id: str) -> str:
        row = self.df_by_id.loc[doc_id]
        words = str(row["article_clean"]).split()
        lead = " ".join(words[:240])
        best_chunk = (
            self.passage_index.get_best_passage_text(query, doc_id)
            if self.passage_index is not None
            else ""
        )
        passage_block = f"\nBest Matching Interior Passage:\n{best_chunk}" if best_chunk else ""
        return (
            f"Headline: {row['headline_clean']}\n"
            f"Date: {str(row['published'])[:10]} | Section: {row['Section']}\n"
            f"Description: {row['description_clean']}\n"
            f"Lead Snippet: {lead}{passage_block}"
        ).strip()

    def score_candidates(self, query: str, doc_ids: list[str]) -> list[float]:
        missing_ids = [d for d in doc_ids if f"{query.strip()}::{d}" not in self.cache]
        if missing_ids:
            texts = [self._build_doc_text(query, d) for d in missing_ids]
            payload = json.dumps({
                "query": query.strip(),
                "texts": texts,
                "return_text": False,
                "truncate": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                items = json.loads(resp.read().decode("utf-8"))
            for item in items:
                d_id = missing_ids[int(item["index"])]
                self.cache[f"{query.strip()}::{d_id}"] = float(item["score"])
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

        return [float(self.cache[f"{query.strip()}::{d}"]) for d in doc_ids]
