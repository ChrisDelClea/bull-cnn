"""Evaluation metrics, paired benchmark builder, cached TREC judge, and MLflow tracker."""

from src.eval.benchmark import (
    CachedTRECJudge,
    GeneratedQueryItem,
    PairedBenchmarkBuilder,
    QueryCategory,
    build_or_load_benchmark,
    select_stratified_anchors,
)
from src.eval.metrics import (
    MultiCriteriaRelevanceJudgment,
    average_precision_at_k,
    bootstrap_confidence_interval,
    ndcg_at_k,
    paired_permutation_p_value,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.eval.tracker import ExperimentTracker, evaluate_and_log_retriever

__all__ = [
    "CachedTRECJudge",
    "ExperimentTracker",
    "GeneratedQueryItem",
    "MultiCriteriaRelevanceJudgment",
    "PairedBenchmarkBuilder",
    "QueryCategory",
    "average_precision_at_k",
    "bootstrap_confidence_interval",
    "build_or_load_benchmark",
    "evaluate_and_log_retriever",
    "ndcg_at_k",
    "paired_permutation_p_value",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "select_stratified_anchors",
]
