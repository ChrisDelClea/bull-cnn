"""Stance-Aware & Aspect-Diversified News Recommender Package (`src.recommender`)."""

from src.recommender.base import BaseNewsRecommender
from src.recommender.stance import (
    ALL_POLICY_FRAMES,
    ArticleStanceProfile,
    PolicyFrame,
    StanceAwareNewsRecommender,
    StanceLabel,
    UserStanceIntent,
)

__all__ = [
    "ALL_POLICY_FRAMES",
    "ArticleStanceProfile",
    "BaseNewsRecommender",
    "PolicyFrame",
    "StanceAwareNewsRecommender",
    "StanceLabel",
    "UserStanceIntent",
]
