"""Abstract base class for all search and retrieval indexes.

Method & Literature References:
- Modern Multi-Stage Information Retrieval Architecture:
  [Pretrained Transformers for Text Ranking: BERT and Beyond (Lin, Nogueira, & Yates, Synthesis Lectures on HLT 2021, arXiv:2010.06467)](https://arxiv.org/abs/2010.06467)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearchIndex(ABC):
    """Unified retrieval interface for lexical, dense, hybrid, and reranked indexes."""

    @abstractmethod
    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-K `(doc_id, score)` pairs ordered by descending score."""

    def search(self, query: str, top_k: int = 50) -> list[str]:
        """Template method returning top-K `doc_id` strings."""
        return [doc_id for doc_id, _ in self.search_with_scores(query, top_k=top_k)]
