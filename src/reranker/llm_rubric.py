"""Two-Stage 'Search + LLM' Pointwise Rubric Reranker using SGLang Structured Outputs.

Method & Literature References:
- Pointwise Multi-Criteria LLM Reranking & Reusable Document Prefixes:
  [DoPR: Reusable Compressed Document Prefixes for Efficient LLM Reranking (Sept 2026, arXiv:2609.03311)](https://arxiv.org/abs/2609.03311)
- Rubric-Calibrated LLM Relevance Judgments:
  [Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory (Sept 2026, arXiv:2609.35739)](https://arxiv.org/abs/2609.35739)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.eval.metrics import MultiCriteriaRelevanceJudgment
from src.index.base import BaseSearchIndex
from src.index.dense import DensePassageIndex
from src.llm.base import BaseStructuredLLMClient
from src.llm.sglang_client import SGLangStructuredClient
from src.reranker.base import BaseReranker


class LLMRubricReranker(BaseReranker):
    """Two-Stage 'Search + LLM' Retriever: Stage-1 Index + Pointwise LLM Rubric Reranker."""

    def __init__(
        self,
        base_index: BaseSearchIndex,
        df: pd.DataFrame,
        rerank_depth: int = 10,
        passage_index: DensePassageIndex | None = None,
        cache_path: str | Path = "../data/reranker_llm_cache.json",
        llm_client: BaseStructuredLLMClient | None = None,
    ) -> None:
        super().__init__(base_index=base_index, rerank_depth=rerank_depth)
        self.df_by_id = df.set_index(df["Index"].astype(str))
        self.passage_index = passage_index
        self.cache_path = Path(cache_path)
        self.llm_client = llm_client or SGLangStructuredClient()
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _score_single(self, query: str, doc_id: str) -> float:
        cache_key = f"{query.strip()}::{doc_id}"
        if cache_key not in self.cache:
            if doc_id not in self.df_by_id.index:
                return 0.0
            row = self.df_by_id.loc[doc_id]
            words = str(row["article_clean"]).split()
            lead = " ".join(words[:240])
            best_chunk = (
                self.passage_index.get_best_passage_text(query, doc_id)
                if self.passage_index is not None
                else ""
            )
            passage_block = f"\nBest Matching Interior Passage:\n{best_chunk}" if best_chunk else ""
            user_prompt = (
                f"Search Query: {query}\n\n"
                f"Candidate Article Headline: {row['headline_clean']}\n"
                f"Published Date: {str(row['published'])[:10]} | Section: {row['Section']}\n"
                f"Description: {row['description_clean']}\n"
                f"Lead Snippet:\n{lead}{passage_block}"
            )
            judgment: MultiCriteriaRelevanceJudgment = self.llm_client.generate_structured(
                system_prompt=(
                    "You are an expert search relevance reranker. Evaluate how well the candidate "
                    "news article satisfies the search query across exactness, topicality, coverage, "
                    "and contextual_fit (0-3), and assign trec_grade (0-3)."
                ),
                user_prompt=user_prompt,
                schema_cls=MultiCriteriaRelevanceJudgment,
            )  # type: ignore[assignment]
            self.cache[cache_key] = judgment.model_dump()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

        c = self.cache[cache_key]
        return float(
            2.0 * c["trec_grade"]
            + c["exactness"]
            + c["topicality"]
            + c["coverage"]
            + c["contextual_fit"]
        )

    def score_candidates(self, query: str, doc_ids: list[str]) -> list[float]:
        return [self._score_single(query, d_id) for d_id in doc_ids]
