"""Class-driven Experiment Tracker logging Strict, Fair Pooled TREC, Bootstrap CIs, and CQC metrics to MLflow.

Method & Literature References:
- TREC Depth-K Pooling & Fair Sibling Evaluation:
  [Test Collection Based Evaluation of Information Retrieval Systems (Sanderson, FnTIR 2010)](https://doi.org/10.1561/1500000009)
- Paired Permutation / Randomization Test & Bootstrap Confidence Intervals for IR:
  [A Comparison of Statistical Significance Tests for Information Retrieval Evaluation (Smucker, Allan, & Carterette, CIKM 2007)](https://doi.org/10.1145/1321440.1321528)
- Cross-Query Consistency (CQC) across Controlled Query Perturbations:
  [CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency (Sun et al., June 2026, arXiv:2606.13438)](https://arxiv.org/abs/2606.13438)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd

from src.eval.benchmark import CachedTRECJudge
from src.eval.metrics import (
    average_precision_at_k,
    bootstrap_confidence_interval,
    ndcg_at_k,
    paired_permutation_p_value,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.index.base import BaseSearchIndex


class ExperimentTracker:
    """Evaluates any `BaseSearchIndex` on a paired benchmark and logs to MLflow & TSV."""

    def __init__(
        self,
        experiment_name: str = "cnn_news_search_autoresearch",
        tracking_uri: str = "sqlite:///../mlflow.db",
        results_tsv_path: str | Path = "../data/autoresearch_results.tsv",
        judge: CachedTRECJudge | None = None,
        judge_top_k: int = 3,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.results_tsv_path = Path(results_tsv_path)
        self.judge = judge
        self.judge_top_k = judge_top_k

    def evaluate(
        self,
        run_name: str,
        index: BaseSearchIndex | Callable[[str, int], list[str]],
        benchmark_queries: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
        baseline_details_df: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """Run evaluation for `index`, update pooled qrels cache in mega-batches, and log to MLflow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        search_fn = index.search if isinstance(index, BaseSearchIndex) else index

        # 1. Execute retrieval for all queries first
        retrieved_per_query: list[tuple[dict[str, Any], list[str], float]] = []
        unjudged_pairs: list[tuple[dict[str, Any], str]] = []

        for q in benchmark_queries:
            q_text = q["query_text"]
            t0 = time.perf_counter()
            retrieved_ids = [str(doc_id) for doc_id in search_fn(q_text, 50)]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            retrieved_per_query.append((q, retrieved_ids, latency_ms))

            if self.judge is not None:
                for doc_id in retrieved_ids[: self.judge_top_k]:
                    unjudged_pairs.append((q, doc_id))

        # 2. Mega-batch judge any uncached `(query, doc_id)` pairs
        if self.judge is not None and unjudged_pairs:
            self.judge.judge_batch(unjudged_pairs, batch_size=30)

        # 3. Compute per-query IR metrics
        per_query_rows: list[dict[str, Any]] = []
        for q, retrieved_ids, latency_ms in retrieved_per_query:
            q_text = q["query_text"]
            source_doc_id = str(q["source_doc_id"])

            strict_set = {source_doc_id}
            strict_rr10 = reciprocal_rank_at_k(retrieved_ids, strict_set, k=10)
            strict_rec5 = recall_at_k(retrieved_ids, strict_set, k=5)
            strict_rec10 = recall_at_k(retrieved_ids, strict_set, k=10)
            strict_rec50 = recall_at_k(retrieved_ids, strict_set, k=50)

            graded_qrels: dict[str, int] = {source_doc_id: 3}
            if self.judge is not None:
                prefix = f"{q['query_id']}::"
                for k_cache, v_cache in self.judge.cache.items():
                    if k_cache.startswith(prefix) and self.judge._is_valid_cached_entry(k_cache):
                        d_id = k_cache.split("::", 1)[1]
                        graded_qrels[d_id] = int(v_cache["trec_grade"])
                for doc_id in retrieved_ids[: self.judge_top_k]:
                    graded_qrels[doc_id] = self.judge.judge_pair(q, doc_id)

            trec_rel_set = {d_id for d_id, grade in graded_qrels.items() if grade >= 2}
            n_sibling_matches = len(trec_rel_set - {source_doc_id})

            fair_rr10 = reciprocal_rank_at_k(retrieved_ids, trec_rel_set, k=10)
            fair_rec10 = recall_at_k(retrieved_ids, trec_rel_set, k=10)
            trec_ndcg5 = ndcg_at_k(retrieved_ids, graded_qrels, k=5)
            trec_ndcg10 = ndcg_at_k(retrieved_ids, graded_qrels, k=10)
            trec_map10 = average_precision_at_k(retrieved_ids, trec_rel_set, k=10)
            trec_p5 = precision_at_k(retrieved_ids, trec_rel_set, k=5)

            per_query_rows.append(
                {
                    "query_id": q["query_id"],
                    "anchor_id": q.get("anchor_id", q["query_id"]),
                    "query_category": q["query_category"],
                    "query_text": q_text,
                    "source_doc_id": source_doc_id,
                    "top1_doc_id": retrieved_ids[0] if retrieved_ids else None,
                    "n_sibling_matches": n_sibling_matches,
                    "strict_rr_at_10": strict_rr10,
                    "strict_recall_at_5": strict_rec5,
                    "strict_recall_at_10": strict_rec10,
                    "strict_recall_at_50": strict_rec50,
                    "fair_rr_at_10": fair_rr10,
                    "fair_recall_at_10": fair_rec10,
                    "trec_ndcg_at_5": trec_ndcg5,
                    "trec_ndcg_at_10": trec_ndcg10,
                    "trec_map_at_10": trec_map10,
                    "trec_p_at_5": trec_p5,
                    "latency_ms": latency_ms,
                }
            )

        details_df = pd.DataFrame(per_query_rows)

        anchor_grp = details_df.groupby("anchor_id")
        strict_cats_per_anchor = anchor_grp["strict_recall_at_10"].sum()
        fair_cats_per_anchor = anchor_grp["fair_recall_at_10"].sum()

        _, ndcg_ci_low, ndcg_ci_high = bootstrap_confidence_interval(details_df["trec_ndcg_at_10"].tolist())
        _, fair_mrr_ci_low, fair_mrr_ci_high = bootstrap_confidence_interval(details_df["fair_rr_at_10"].tolist())
        _, strict_mrr_ci_low, strict_mrr_ci_high = bootstrap_confidence_interval(details_df["strict_rr_at_10"].tolist())

        summary_metrics: dict[str, float] = {
            "strict_mrr_at_10": float(details_df["strict_rr_at_10"].mean()),
            "strict_mrr_ci95_low": strict_mrr_ci_low,
            "strict_mrr_ci95_high": strict_mrr_ci_high,
            "strict_recall_at_5": float(details_df["strict_recall_at_5"].mean()),
            "strict_recall_at_10": float(details_df["strict_recall_at_10"].mean()),
            "strict_recall_at_50": float(details_df["strict_recall_at_50"].mean()),
            "fair_mrr_at_10": float(details_df["fair_rr_at_10"].mean()),
            "fair_mrr_ci95_low": fair_mrr_ci_low,
            "fair_mrr_ci95_high": fair_mrr_ci_high,
            "fair_recall_at_10": float(details_df["fair_recall_at_10"].mean()),
            "trec_ndcg_at_5": float(details_df["trec_ndcg_at_5"].mean()),
            "trec_ndcg_at_10": float(details_df["trec_ndcg_at_10"].mean()),
            "trec_ndcg_ci95_low": ndcg_ci_low,
            "trec_ndcg_ci95_high": ndcg_ci_high,
            "trec_map_at_10": float(details_df["trec_map_at_10"].mean()),
            "trec_p_at_5": float(details_df["trec_p_at_5"].mean()),
            "strict_anchor_cats_hit_of_5": float(strict_cats_per_anchor.mean()),
            "fair_anchor_cats_hit_of_5": float(fair_cats_per_anchor.mean()),
            "fair_anchor_all5_consistency": float((fair_cats_per_anchor == 5.0).mean()),
            "latency_p50_ms": float(np.percentile(details_df["latency_ms"], 50)),
            "latency_p95_ms": float(np.percentile(details_df["latency_ms"], 95)),
        }

        if baseline_details_df is not None and len(baseline_details_df) == len(details_df):
            summary_metrics["p_value_vs_baseline_ndcg10"] = paired_permutation_p_value(
                baseline_details_df["trec_ndcg_at_10"].tolist(),
                details_df["trec_ndcg_at_10"].tolist(),
            )
            summary_metrics["p_value_vs_baseline_strict_mrr10"] = paired_permutation_p_value(
                baseline_details_df["strict_rr_at_10"].tolist(),
                details_df["strict_rr_at_10"].tolist(),
            )

        for cat, grp in details_df.groupby("query_category"):
            summary_metrics[f"ndcg10_{cat}"] = float(grp["trec_ndcg_at_10"].mean())
            summary_metrics[f"fair_mrr10_{cat}"] = float(grp["fair_rr_at_10"].mean())
            summary_metrics[f"strict_mrr10_{cat}"] = float(grp["strict_rr_at_10"].mean())

        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics({k: round(v, 4) for k, v in summary_metrics.items()})

        self.results_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        row_record = {"run_name": run_name, **(params or {}), **summary_metrics}
        row_df = pd.DataFrame([row_record])
        if self.results_tsv_path.exists():
            existing = pd.read_csv(self.results_tsv_path, sep="\t")
            existing = existing[existing["run_name"] != run_name]
            combined = pd.concat([existing, row_df], ignore_index=True)
            combined.to_csv(self.results_tsv_path, sep="\t", index=False)
        else:
            row_df.to_csv(self.results_tsv_path, sep="\t", index=False)

        return summary_metrics, details_df


def evaluate_and_log_retriever(
    experiment_name: str,
    run_name: str,
    search_fn: BaseSearchIndex | Callable[[str, int], list[str]],
    benchmark_queries: list[dict[str, Any]],
    judge: CachedTRECJudge | None = None,
    judge_top_k: int = 3,
    params: dict[str, Any] | None = None,
    tracking_uri: str = "sqlite:///../mlflow.db",
    results_tsv_path: str | Path = "../data/autoresearch_results.tsv",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Convenience wrapper around `ExperimentTracker.evaluate`."""
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        results_tsv_path=results_tsv_path,
        judge=judge,
        judge_top_k=judge_top_k,
    )
    return tracker.evaluate(
        run_name=run_name,
        index=search_fn,
        benchmark_queries=benchmark_queries,
        params=params,
    )
