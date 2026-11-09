"""Abstract Base Class for Aspect- and Stance-Aware News Recommender Systems.

Method & Literature References:
- Aspect-Aware Candidate Generation (A2CG) & Multi-Perspective News Recommendation:
  [Democratizing News Recommenders: Modeling Multiple Perspectives for News Candidate Generation with VQ-VAE (Hardy, Padó, Wührl, & Ceron, Aug 2025, arXiv:2508.13978)](https://arxiv.org/abs/2508.13978)
- LLM-Based News Recommender Systems Taxonomy:
  [A Survey on LLM-based News Recommender Systems (Wang, Liesaputra, & Huang, Feb 2025, arXiv:2502.09797)](https://arxiv.org/abs/2502.09797)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseNewsRecommender(ABC):
    """Abstract interface for stance- and frame-aware news recommenders."""

    @abstractmethod
    def recommend_from_prompt(self, user_prompt: str, top_k: int = 5) -> pd.DataFrame:
        """Parse a natural-language user preference prompt and return top-K recommended articles."""
