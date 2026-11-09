"""Abstract base class for dense text and passage embedding models.

Method & Literature References:
- Bi-Encoder Dense Passage Retrieval (DPR):
  [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., EMNLP 2020, arXiv:2004.04906)](https://arxiv.org/abs/2004.04906)
- Instruction-Tuned Decoder-Based Embeddings (Last-Token Pooling):
  [Improving Text Embeddings with Large Language Models (Wang et al., ACL 2024, arXiv:2401.00368)](https://arxiv.org/abs/2401.00368)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for query and passage embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query into an L2-normalized 1D float32 vector."""

    @abstractmethod
    def encode_passages_cached(
        self,
        passages: Sequence[str],
        cache_name: str,
    ) -> np.ndarray:
        """Encode a sequence of passages into an `(M, D)` L2-normalized float32 matrix."""
