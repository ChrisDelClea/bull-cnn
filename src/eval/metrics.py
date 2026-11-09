"""Information Retrieval Evaluation Metrics, Statistical Significance Tests, & TREC Rubric Schema.

Method & Literature References:
- Normalized Discounted Cumulative Gain (nDCG):
  [Cumulated gain-based evaluation of IR techniques (Järvelin & Kekäläinen, ACM TOIS 2002)](https://doi.org/10.1145/582415.582418)
- Paired Permutation / Randomization Test & Bootstrap Confidence Intervals for IR:
  [A Comparison of Statistical Significance Tests for Information Retrieval Evaluation (Smucker, Allan, & Carterette, CIKM 2007)](https://doi.org/10.1145/1321440.1321528)
- Multi-Criteria LLM-as-a-Judge for TREC Relevance Assessment:
  [Large Language Models can Accurately Predict Searcher Preferences (Thomas et al., SIGIR 2024, arXiv:2309.10621)](https://arxiv.org/abs/2309.10621)
  and [LLMJudge: LLMs for Relevance Judgments (Rahmani et al., 2024, arXiv:2408.08896)](https://arxiv.org/abs/2408.08896)
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from pydantic import BaseModel, Field


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Reciprocal Rank (RR@K): `1 / rank` of the first relevant item in top-K."""
    for idx, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / float(idx + 1)
    return 0.0


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@K: fraction of relevant documents retrieved in top-K."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return float(hits) / float(len(relevant_ids))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Compute Precision@K: fraction of top-K slots occupied by relevant documents."""
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return float(hits) / float(k)


def average_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Average Precision (AP@K)."""
    if not relevant_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for idx, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += float(hits) / float(idx + 1)
    return sum_precisions / float(min(len(relevant_ids), k))


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    graded_qrels: dict[str, int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K, Järvelin & Kekäläinen 2002)."""
    if not graded_qrels:
        return 0.0

    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_ids[:k]):
        rel = graded_qrels.get(doc_id, 0)
        if rel > 0:
            dcg += float((2**rel) - 1) / math.log2(float(idx + 2))

    ideal_grades = sorted(graded_qrels.values(), reverse=True)[:k]
    idcg = sum(
        float((2**rel) - 1) / math.log2(float(idx + 2))
        for idx, rel in enumerate(ideal_grades)
        if rel > 0
    )
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def bootstrap_confidence_interval(
    scores: Sequence[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute non-parametric bootstrap `(mean, ci_low, ci_high)` for per-query IR metrics."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(arr.mean())
    if arr.size == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return mean_val, low, high


def paired_permutation_p_value(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Compute two-sided paired randomization/permutation p-value (Smucker et al., CIKM 2007)."""
    diffs = np.asarray(scores_b, dtype=np.float64) - np.asarray(scores_a, dtype=np.float64)
    if diffs.size == 0:
        return 1.0
    obs_diff = abs(float(diffs.mean()))
    if obs_diff == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diffs.size))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    p_val = float((np.sum(perm_means >= obs_diff) + 1.0) / (n_permutations + 1.0))
    return p_val


class MultiCriteriaRelevanceJudgment(BaseModel):
    """Multi-Aspect TREC Relevance Rubric (Thomas et al., SIGIR 2024, arXiv:2309.10621).

    `reasoning_short` precedes all numerical scores so constrained decoding reasons
    about query-article alignment before committing to the grades.
    """

    reasoning_short: str = Field(
        max_length=280,
        description="Concise 1-2 sentence analysis (max 250 chars) comparing query intent with article content.",
    )
    exactness: int = Field(
        ge=0,
        le=3,
        description="0-3: Does the article directly address the specific entities, events, or facts in the query?",
    )
    topicality: int = Field(
        ge=0,
        le=3,
        description="0-3: Is the central theme of the article aligned with the query topic?",
    )
    coverage: int = Field(
        ge=0,
        le=3,
        description="0-3: How thoroughly does the article cover the information need of the query?",
    )
    contextual_fit: int = Field(
        ge=0,
        le=3,
        description="0-3: Do the time period, geographic location, and stance match the query constraints?",
    )
    trec_grade: int = Field(
        ge=0,
        le=3,
        description=(
            "Final TREC relevance grade: 0 = Irrelevant, 1 = Marginally relevant, "
            "2 = Relevant, 3 = Highly relevant."
        ),
    )
