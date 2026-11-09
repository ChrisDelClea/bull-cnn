"""Abstract base class for Stage-2 Rerankers (Decorator pattern over BaseSearchIndex).

Method & Literature References:
- Multi-Stage Retrieve-and-Rerank Architecture:
  [Passage Re-ranking with BERT (Nogueira & Cho, 2019, arXiv:1901.04085)](https://arxiv.org/abs/1901.04085)
- Reusable Document Prefixes for Pointwise LLM Reranking (DoPR):
  [DoPR: Reusable Compressed Document Prefixes for Efficient LLM Reranking (Sept 2026, arXiv:2609.03311)](https://arxiv.org/abs/2609.03311)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from src.index.base import BaseSearchIndex


class BaseReranker(BaseSearchIndex, ABC):
    """Wraps a Stage-1 `BaseSearchIndex` and re-scores the top `rerank_depth` candidates."""

    def __init__(self, base_index: BaseSearchIndex, rerank_depth: int = 15) -> None:
        self.base_index = base_index
        self.rerank_depth = rerank_depth

    @abstractmethod
    def score_candidates(self, query: str, doc_ids: list[str]) -> list[float]:
        """Return Stage-2 relevance scores for each candidate `doc_id`."""

    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        stage1 = self.base_index.search_with_scores(query, top_k=max(top_k, self.rerank_depth))
        if not stage1:
            return []

        head_ids = [doc_id for doc_id, _ in stage1[: self.rerank_depth]]
        tail = stage1[self.rerank_depth :]

        rerank_scores = self.score_candidates(query, head_ids)
        scored_head = [
            (doc_id, float(score) - 1e-4 * rank_i)
            for rank_i, (doc_id, score) in enumerate(zip(head_ids, rerank_scores))
        ]
        scored_head.sort(key=lambda x: x[1], reverse=True)

        min_head = scored_head[-1][1] if scored_head else 0.0
        adjusted_tail = [
            (doc_id, min_head - 1.0 - idx * 1e-3)
            for idx, (doc_id, _) in enumerate(tail)
        ]
        return (scored_head + adjusted_tail)[:top_k]
