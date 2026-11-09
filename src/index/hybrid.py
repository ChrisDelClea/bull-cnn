"""Hybrid Sparse-Dense Retrieval Indexes (Weighted RRF & Dynamic Alpha Tuning).

Method & Literature References:
- Reciprocal Rank Fusion (RRF):
  [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods (Cormack, Clarke, & Buettcher, SIGIR 2009)](https://doi.org/10.1145/1571941.1572114)
- Dynamic Alpha Tuning (DAT) / Query-Adaptive Hybrid Fusion:
  [DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation (Hsu & Tzeng, March 2025, arXiv:2503.23013)](https://arxiv.org/abs/2503.23013)
  and [Adaptive Hybrid Retrieval for Portuguese Cultural Heritage (Barata, April 2026, arXiv:2604.06166)](https://arxiv.org/abs/2604.06166)
"""

from __future__ import annotations

import numpy as np

from src.index.base import BaseSearchIndex
from src.index.bm25 import SparseBM25Index, tokenize_text
from src.index.dense import DensePassageIndex


class HybridRRFIndex(BaseSearchIndex):
    """Weighted Reciprocal Rank Fusion (RRF, Cormack et al., SIGIR 2009)."""

    def __init__(
        self,
        lexical_index: BaseSearchIndex,
        dense_index: BaseSearchIndex,
        w_lexical: float = 0.25,
        w_dense: float = 0.75,
        rrf_k: int = 15,
        candidate_depth: int = 50,
    ) -> None:
        self.lexical_index = lexical_index
        self.dense_index = dense_index
        self.w_lexical = w_lexical
        self.w_dense = w_dense
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        lex_hits = self.lexical_index.search_with_scores(query, top_k=self.candidate_depth)
        dense_hits = self.dense_index.search_with_scores(query, top_k=self.candidate_depth)

        rrf_scores: dict[str, float] = {}
        for rank_idx, (doc_id, _) in enumerate(lex_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.w_lexical / (self.rrf_k + rank_idx)
        for rank_idx, (doc_id, _) in enumerate(dense_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.w_dense / (self.rrf_k + rank_idx)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class DynamicAlphaHybridIndex(BaseSearchIndex):
    """Query-Adaptive Hybrid Fusion (DAT, Hsu & Tzeng 2025; Barata 2026)."""

    def __init__(
        self,
        bm25_index: SparseBM25Index,
        dense_index: DensePassageIndex,
        candidate_depth: int = 100,
    ) -> None:
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.candidate_depth = candidate_depth
        self.max_idf = (
            float(np.percentile(self.bm25_index.idf, 95))
            if len(self.bm25_index.idf)
            else 8.0
        )

    def compute_dynamic_alpha(self, query: str) -> float:
        """Compute query-adaptive lexical weight `alpha in [0.10, 0.65]` from BM25 peak specificity & IDF."""
        tokens = tokenize_text(query)
        idfs = [
            float(self.bm25_index.idf[self.bm25_index.vocab[t]])
            for t in tokens
            if t in self.bm25_index.vocab
        ]
        if not idfs:
            return 0.10

        lex_hits = self.bm25_index.search_with_scores(query, top_k=20)
        if len(lex_hits) < 3:
            return 0.15
        scores = np.array([s for _, s in lex_hits], dtype=np.float32)
        top1 = float(scores[0])
        tail_mean = float(np.mean(scores[1:15]))
        tail_std = float(np.std(scores[1:15])) + 1e-4

        # Peak-to-tail Z-score and absolute BM25 confidence per query token
        z_lex = (top1 - tail_mean) / tail_std
        per_token_bm25 = top1 / max(len(tokens), 1)
        specificity = min(1.0, max(0.0, (0.12 * z_lex + 0.15 * per_token_bm25) - 0.35))
        return round(0.10 + 0.50 * specificity, 4)

    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        alpha = self.compute_dynamic_alpha(query)
        lex_hits = dict(self.bm25_index.search_with_scores(query, top_k=self.candidate_depth))
        dense_hits = dict(self.dense_index.search_with_scores(query, top_k=self.candidate_depth))

        union_ids = list(set(lex_hits.keys()) | set(dense_hits.keys()))
        if not union_ids:
            return []

        lex_vals = np.array([lex_hits.get(d, 0.0) for d in union_ids], dtype=np.float32)
        dense_min = min(dense_hits.values()) if dense_hits else 0.0
        dense_vals = np.array([dense_hits.get(d, dense_min) for d in union_ids], dtype=np.float32)

        if lex_vals.max() > lex_vals.min():
            lex_norm = (lex_vals - lex_vals.min()) / (lex_vals.max() - lex_vals.min())
        else:
            lex_norm = np.zeros_like(lex_vals)

        if dense_vals.max() > dense_vals.min():
            dense_norm = (dense_vals - dense_vals.min()) / (dense_vals.max() - dense_vals.min())
        else:
            dense_norm = np.zeros_like(dense_vals)

        combined = alpha * lex_norm + (1.0 - alpha) * dense_norm
        order = np.argsort(-combined)[:top_k]
        return [(union_ids[int(i)], float(combined[int(i)])) for i in order]
