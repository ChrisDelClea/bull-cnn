"""Controlled Multi-Query Perturbation Benchmark & Cached TREC LLM-as-a-Judge.

Method & Literature References:
- Controlled Multi-Query Perturbation & Cross-Query Consistency (CQC):
  [CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency (Sun et al., June 2026, arXiv:2606.13438)](https://arxiv.org/abs/2606.13438)
- Query Reformulation & Structural Query Variation Benchmarks:
  [Route Me If You Can: A Benchmark for Query Reformulation Selection (Sept 2026, arXiv:2609.14885)](https://arxiv.org/abs/2609.14885)
- Cross-Query Calibrated LLM Relevance Judgments:
  [Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory (Sept 2026, arXiv:2609.35739)](https://arxiv.org/abs/2609.35739)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.eval.metrics import MultiCriteriaRelevanceJudgment
from src.llm.base import BaseStructuredLLMClient
from src.llm.gemini_client import GeminiStructuredClient
from src.llm.sglang_client import SGLangStructuredClient

QueryCategory = Literal[
    "keyword_lookup",
    "semantic_paraphrase",
    "deep_fact_recall",
    "temporal_lookup",
    "stance_opinion",
]


class GeneratedQueryItem(BaseModel):
    """Structured output schema for generating a realistic user search query."""

    reasoning_short: str = Field(
        description="1-2 concise sentences (at most ~45 tokens) explaining how a real user would formulate this search query.",
    )
    user_query_style: Literal[
        "short_keywords_1_to_4_words",
        "medium_phrase_4_to_8_words",
        "natural_question_7_to_14_words",
    ] = Field(description="The realistic token length and style of the user query.")
    query_text: str = Field(
        description="The realistic user search query matching the requested category and length style."
    )
    target_entity: str = Field(
        description="Primary person, organization, or topic targeted by the query."
    )
    expected_stance: Literal["FAVOR", "AGAINST", "NEUTRAL"] = Field(
        description="Expected stance towards target_entity (use AGAINST/FAVOR for stance_opinion queries, else NEUTRAL)."
    )


class AnchorFiveCategoryItem(BaseModel):
    """All 5 perturbation queries generated for a single anchor article (`reasoning_short` first in each)."""

    anchor_index: int = Field(description="1-based anchor index matching the prompt input.")
    keyword_lookup: GeneratedQueryItem
    semantic_paraphrase: GeneratedQueryItem
    deep_fact_recall: GeneratedQueryItem
    temporal_lookup: GeneratedQueryItem
    stance_opinion: GeneratedQueryItem


class MultiAnchorBenchmarkBatch(BaseModel):
    """Mega-batch response containing 5-category query sets for multiple anchor articles in a single call."""

    reasoning_short: str = Field(
        description="Concise summary of how each anchor article's length stratum, section, and late facts were mapped into distinct queries."
    )
    anchors: list[AnchorFiveCategoryItem]


class SinglePairJudgmentItem(BaseModel):
    """A single `(pair_id, judgment)` entry inside a mega-batched TREC judge call."""

    pair_id: str = Field(description="Exact pair_id string from the prompt (`query_id::doc_id`).")
    reasoning_short: str = Field(
        description="1 concise sentence comparing query intent with candidate article content."
    )
    exactness: int = Field(ge=0, le=3)
    topicality: int = Field(ge=0, le=3)
    coverage: int = Field(ge=0, le=3)
    contextual_fit: int = Field(ge=0, le=3)
    trec_grade: int = Field(ge=0, le=3)


class BatchedRelevanceJudgmentList(BaseModel):
    """Mega-batch response of TREC relevance judgments for multiple `(query, doc)` pairs."""

    reasoning_short: str = Field(
        description="Brief overview of the batch relevance assessment criteria applied."
    )
    judgments: list[SinglePairJudgmentItem]


CATEGORY_INSTRUCTIONS: dict[QueryCategory, str] = {
    "keyword_lookup": (
        "Simulate a classic search-bar user typing a terse keyword query (ONLY 2 to 4 words, "
        "no punctuation) with specific proper nouns, names, or rare terms from the article "
        "(e.g., 'TuSimple autonomous truck' or 'Dr Oz Turkish citizenship')."
    ),
    "semantic_paraphrase": (
        "Simulate a user searching for the concept/event (4 to 8 words) using everyday synonyms "
        "and paraphrases because they do not remember the exact wording. "
        "STRICT RULE: Do NOT reuse any main nouns, verbs, or proper names from the headline!"
    ),
    "deep_fact_recall": (
        "Simulate a user asking a specific factual question (6 to 12 words) about a detail "
        "that ONLY appears in the provided late-article excerpt (from the final third of the article). "
        "Do NOT ask about the headline topic generally; target the specific fact/quote in the late excerpt."
    ),
    "temporal_lookup": (
        "Simulate a user searching for news from a specific time period and domain (3 to 8 words), "
        "explicitly including the year or month+year and topic (e.g., 'Senate race rankings February 2022')."
    ),
    "stance_opinion": (
        "Identify the main actor, organization, policy, or perspective in this article and the author's "
        "or subjects' critical (AGAINST) or supportive (FAVOR) stance towards it. Formulate a user request "
        "(5 to 12 words) explicitly asking for articles/commentary with that stance towards that target."
    ),
}


def select_stratified_anchors(
    df: pd.DataFrame,
    n_needed: int,
    exclude_doc_ids: set[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Select `n_needed` anchor articles stratified across length quartiles (Short/Median/Long) & sections.

    Avoids the 0.8% (`opinions` & `>=1200w`) selection bias by sampling proportionally across:
    - Stratum 1 (Short / Lower-Median: 350..600 words, 2-3 chunks)
    - Stratum 2 (Median / Typical: 600..1100 words, 4-6 chunks, centered on the 687w corpus median)
    - Stratum 3 (Long: >1100 words, 7+ chunks)
    and across diverse CNN editorial sections (`europe`, `politics`, `us`, `sport`, `business`, `world`, `health`, `opinions`, `uk`, `entertainment`).
    """
    rng = np.random.default_rng(seed)
    usable = df[
        (~df["flag_empty_article"])
        & (df["n_words_clean"] >= 350)
        & (~df["Index"].astype(str).isin(exclude_doc_ids))
    ].copy()

    # Assign length strata
    conditions = [
        usable["n_words_clean"] < 600,
        (usable["n_words_clean"] >= 600) & (usable["n_words_clean"] <= 1100),
        usable["n_words_clean"] > 1100,
    ]
    usable["length_stratum"] = np.select(conditions, ["short_350_600", "median_600_1100", "long_gt_1100"], default="median_600_1100")

    target_sections = [
        "politics",
        "us",
        "europe",
        "world",
        "business",
        "sport",
        "opinions",
        "uk",
        "health",
        "entertainment",
    ]
    strata_names = ["short_350_600", "median_600_1100", "long_gt_1100"]

    selected_indices: list[int] = []
    used_indices: set[int] = set()

    # Round-robin across (length_stratum, section) cells for balanced coverage
    while len(selected_indices) < n_needed:
        added_in_pass = 0
        for s_name in strata_names:
            for sec in target_sections:
                if len(selected_indices) >= n_needed:
                    break
                cell = usable[
                    (usable["length_stratum"] == s_name)
                    & (usable["Section"].astype(str).str.lower() == sec)
                    & (~usable.index.isin(used_indices))
                ]
                if len(cell) > 0:
                    chosen_idx = int(rng.choice(cell.index.to_numpy(), size=1)[0])
                    selected_indices.append(chosen_idx)
                    used_indices.add(chosen_idx)
                    added_in_pass += 1
        if added_in_pass == 0:
            # Fill any remaining from general usable pool
            rem = usable[~usable.index.isin(used_indices)]
            if len(rem) == 0:
                break
            take = min(n_needed - len(selected_indices), len(rem))
            extra = rng.choice(rem.index.to_numpy(), size=take, replace=False)
            for idx_val in extra:
                selected_indices.append(int(idx_val))
                used_indices.add(int(idx_val))
            break

    return df.loc[selected_indices].copy()


class PairedBenchmarkBuilder:
    """Builds or loads a Controlled Multi-Query Perturbation benchmark (`n_anchors x 5 categories`)."""

    def __init__(
        self,
        benchmark_path: str | Path = "../data/eval_benchmark_queries.json",
        n_anchors: int = 50,
        seed: int = 42,
        llm_client: BaseStructuredLLMClient | None = None,
        batch_size: int = 10,
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.n_anchors = n_anchors
        self.seed = seed
        self.llm_client = llm_client or GeminiStructuredClient(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        self.batch_size = batch_size

    def build_or_load(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return `n_anchors * 5` paired benchmark queries, expanding with stratified anchors if needed."""
        existing_queries: list[dict[str, Any]] = []
        if self.benchmark_path.exists():
            raw_existing = json.loads(self.benchmark_path.read_text(encoding="utf-8"))
            target_model = getattr(self.llm_client, "model", None)
            if target_model == "gemini-3.8-flash":
                existing_queries = [
                    q
                    for q in raw_existing
                    if q.get("generator_model") == "gemini-3.8-flash"
                    and q.get("thinking_level") == "HIGH"
                ]
            else:
                existing_queries = raw_existing
            if len(existing_queries) >= self.n_anchors * 5:
                return existing_queries[: self.n_anchors * 5]

        existing_anchors = sorted({str(q["anchor_id"]) for q in existing_queries})
        existing_doc_ids = {str(q["source_doc_id"]) for q in existing_queries}
        n_existing_anchors = len(existing_anchors)
        n_to_generate = self.n_anchors - n_existing_anchors

        if n_to_generate <= 0:
            return existing_queries

        sampled_df = select_stratified_anchors(
            df=df,
            n_needed=n_to_generate,
            exclude_doc_ids=existing_doc_ids,
            seed=self.seed,
        )

        categories: list[QueryCategory] = [
            "keyword_lookup",
            "semantic_paraphrase",
            "deep_fact_recall",
            "temporal_lookup",
            "stance_opinion",
        ]

        queries = list(existing_queries)
        rows_list = [row for _, row in sampled_df.iterrows()]

        # Generate in mega-batches of `self.batch_size` anchors per LLM call
        for batch_start in range(0, len(rows_list), self.batch_size):
            batch_rows = rows_list[batch_start : batch_start + self.batch_size]
            prompt_blocks: list[str] = []

            for local_i, row in enumerate(batch_rows, start=1):
                words = str(row["article_clean"]).split()
                lead_excerpt = " ".join(words[:220])
                late_start = max(int(len(words) * 0.65), min(220, len(words) - 120))
                late_excerpt = " ".join(words[late_start : late_start + 220])
                prompt_blocks.append(
                    f"=== ANCHOR #{local_i} ===\n"
                    f"Headline: {row['headline_clean']}\n"
                    f"Published Date: {str(row['published'])[:10]} (Year: {row['year']})\n"
                    f"Section: {row['Section']} | Total Words: {row['n_words_clean']}\n"
                    f"Description: {row['description_clean']}\n"
                    f"Lead Excerpt (First Paragraphs):\n{lead_excerpt}\n"
                    f"Late Excerpt (Final Third of Article — use ONLY this for deep_fact_recall):\n{late_excerpt}\n"
                )

            cat_rules = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_INSTRUCTIONS.items())
            user_prompt = (
                f"Generate all 5 search query categories for each of the {len(batch_rows)} CNN articles below.\n\n"
                f"Category Rules:\n{cat_rules}\n\n"
                + "\n".join(prompt_blocks)
            )

            batch_res: MultiAnchorBenchmarkBatch = self.llm_client.generate_structured(
                system_prompt=(
                    "You are a senior NIST TREC Information Retrieval benchmark architect. "
                    "For every anchor article, generate realistic, high-precision user queries across all 5 categories."
                ),
                user_prompt=user_prompt,
                schema_cls=MultiAnchorBenchmarkBatch,
            )  # type: ignore[assignment]

            for local_i, row in enumerate(batch_rows):
                anchor_num = n_existing_anchors + batch_start + local_i + 1
                doc_id = str(row["Index"])
                anchor_item = (
                    batch_res.anchors[local_i]
                    if local_i < len(batch_res.anchors)
                    else batch_res.anchors[-1]
                )
                for category in categories:
                    gen: GeneratedQueryItem = getattr(anchor_item, category)
                    q_id = f"anchor{anchor_num:02d}_{category}"
                    queries.append(
                        {
                            "query_id": q_id,
                            "anchor_id": f"anchor_{anchor_num:02d}",
                            "query_category": category,
                            "user_query_style": gen.user_query_style,
                            "query_text": gen.query_text,
                            "n_query_words": len(gen.query_text.split()),
                            "reasoning_short": gen.reasoning_short,
                            "target_entity": gen.target_entity,
                            "expected_stance": gen.expected_stance,
                            "source_doc_id": doc_id,
                            "source_headline": str(row["headline_clean"]),
                            "source_section": str(row["Section"]),
                            "source_year": int(row["year"]),
                            "source_n_words": int(row["n_words_clean"]),
                            "generator_model": getattr(self.llm_client, "model", "unknown"),
                            "thinking_level": getattr(self.llm_client, "thinking_level", "unknown"),
                            "known_qrels": {doc_id: 3},
                        }
                    )

            self.benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            self.benchmark_path.write_text(
                json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"[{getattr(self.llm_client, 'model', 'LLM')} | thinking={getattr(self.llm_client, 'thinking_level', 'N/A')}] "
                f"Saved {len(queries)}/{self.n_anchors * 5} queries to {self.benchmark_path.name}",
                flush=True,
            )

        return queries


def build_or_load_benchmark(
    df: pd.DataFrame,
    benchmark_path: str | Path = "../data/eval_benchmark_queries.json",
    n_per_category: int = 50,
    seed: int = 42,
    llm_client: BaseStructuredLLMClient | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper around `PairedBenchmarkBuilder`."""
    return PairedBenchmarkBuilder(
        benchmark_path=benchmark_path,
        n_anchors=n_per_category,
        seed=seed,
        llm_client=llm_client,
    ).build_or_load(df)


class CachedTRECJudge:
    """Evaluates `(query, article)` relevance using `MultiCriteriaRelevanceJudgment` with disk caching."""

    def __init__(
        self,
        df: pd.DataFrame,
        cache_path: str | Path = "../data/eval_qrels_cache.json",
        llm_client: BaseStructuredLLMClient | None = None,
    ) -> None:
        self.df_by_id = df.set_index(df["Index"].astype(str))
        self.cache_path = Path(cache_path)
        self.llm_client = llm_client or GeminiStructuredClient(
            model="gemini-3.8-flash", thinking_level="HIGH"
        )
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _is_valid_cached_entry(self, key: str) -> bool:
        if key not in self.cache:
            return False
        target_model = getattr(self.llm_client, "model", None)
        if target_model == "gemini-3.8-flash":
            entry = self.cache[key]
            return (
                entry.get("judge_model") == "gemini-3.8-flash"
                and entry.get("thinking_level") == "HIGH"
            )
        return True

    def judge_pair(
        self,
        query_obj: dict[str, Any],
        doc_id: str,
        force_recompute: bool = False,
    ) -> int:
        """Return TREC grade (0..3) for `(query_id, doc_id)`, querying LLM only on cache miss."""
        q_id = query_obj["query_id"]
        if str(doc_id) == str(query_obj["source_doc_id"]) and not force_recompute:
            return 3

        cache_key = f"{q_id}::{doc_id}"
        if not force_recompute and self._is_valid_cached_entry(cache_key):
            return int(self.cache[cache_key]["trec_grade"])

        if os.environ.get("BULL_CNN_JUDGE_CACHE_ONLY") == "1":
            if self._is_valid_cached_entry(cache_key):
                return int(self.cache[cache_key]["trec_grade"])
            return 3 if str(doc_id) == str(query_obj["source_doc_id"]) else 0

        if str(doc_id) not in self.df_by_id.index:
            return 0

        row = self.df_by_id.loc[str(doc_id)]
        words = str(row["article_clean"]).split()
        if len(words) > 500:
            snippet = " ".join(words[:300]) + " ... [middle omitted] ... " + " ".join(words[-200:])
        else:
            snippet = " ".join(words)

        user_prompt = (
            f"Search Query: {query_obj['query_text']}\n"
            f"Query Category: {query_obj['query_category']}\n"
            f"Expected Stance (if opinion query): {query_obj.get('expected_stance', 'NEUTRAL')} "
            f"towards {query_obj.get('target_entity', 'N/A')}\n\n"
            f"Candidate Article Headline: {row['headline_clean']}\n"
            f"Published Date: {str(row['published'])[:10]} | Section: {row['Section']}\n"
            f"Description: {row['description_clean']}\n"
            f"Article Text:\n{snippet}"
        )
        judgment: MultiCriteriaRelevanceJudgment = self.llm_client.generate_structured(
            system_prompt=(
                "You are an expert NIST TREC Information Retrieval assessor. "
                "Grade the relevance of the candidate news article to the user's search query "
                "across exactness, topicality, coverage, and contextual_fit (0-3), then assign "
                "the final trec_grade (0=Irrelevant, 1=Marginal, 2=Relevant, 3=Highly Relevant)."
            ),
            user_prompt=user_prompt,
            schema_cls=MultiCriteriaRelevanceJudgment,
        )  # type: ignore[assignment]

        dumped = judgment.model_dump()
        dumped["judge_model"] = getattr(self.llm_client, "model", "unknown")
        dumped["thinking_level"] = getattr(self.llm_client, "thinking_level", "unknown")
        self.cache[cache_key] = dumped
        self.save_cache()
        return int(judgment.trec_grade)

    def judge_batch(
        self,
        unjudged_pairs: list[tuple[dict[str, Any], str]],
        batch_size: int = 95,
    ) -> None:
        """Judge a list of `(query_obj, doc_id)` pairs in mega-batches to minimize API RPD usage."""
        if os.environ.get("BULL_CNN_JUDGE_CACHE_ONLY") == "1":
            return
        to_process: list[tuple[dict[str, Any], str]] = []
        seen_keys: set[str] = set()
        for q_obj, doc_id in unjudged_pairs:
            d_str = str(doc_id)
            if d_str == str(q_obj["source_doc_id"]):
                continue
            key = f"{q_obj['query_id']}::{d_str}"
            if key not in seen_keys and not self._is_valid_cached_entry(key) and d_str in self.df_by_id.index:
                seen_keys.add(key)
                to_process.append((q_obj, d_str))

        n_batches = (len(to_process) + batch_size - 1) // batch_size if to_process else 0
        for b_idx, start in enumerate(range(0, len(to_process), batch_size), start=1):
            chunk = to_process[start : start + batch_size]
            blocks: list[str] = []
            for q_obj, d_str in chunk:
                pair_id = f"{q_obj['query_id']}::{d_str}"
                row = self.df_by_id.loc[d_str]
                words = str(row["article_clean"]).split()
                snippet = (
                    " ".join(words[:120]) + " ... " + " ".join(words[-80:])
                    if len(words) > 220
                    else " ".join(words)
                )
                blocks.append(
                    f"[PAIR_ID: {pair_id}]\n"
                    f"Query ({q_obj['query_category']}): {q_obj['query_text']}\n"
                    f"Article ({str(row['published'])[:10]} | {row['Section']}): {row['headline_clean']} — {row['description_clean']}\n"
                    f"Excerpt: {snippet}\n"
                )

            prompt = (
                f"Assess the relevance (0..3) for each of the {len(chunk)} (query, article) pairs below.\n"
                "Keep reasoning_short very concise (at most 10 words per item). Return one judgment per PAIR_ID.\n\n"
                + "\n".join(blocks)
            )
            res: BatchedRelevanceJudgmentList = self.llm_client.generate_structured(
                system_prompt=(
                    "You are an expert NIST TREC Information Retrieval assessor. "
                    "Grade each (query, candidate article) pair across exactness, topicality, coverage, "
                    "contextual_fit (0-3), and trec_grade (0=Irrelevant, 1=Marginal, 2=Relevant, 3=Highly Relevant)."
                ),
                user_prompt=prompt,
                schema_cls=BatchedRelevanceJudgmentList,
            )  # type: ignore[assignment]

            by_id = {item.pair_id: item for item in res.judgments}
            for q_obj, d_str in chunk:
                pair_id = f"{q_obj['query_id']}::{d_str}"
                if pair_id in by_id:
                    it = by_id[pair_id]
                    self.cache[pair_id] = {
                        "reasoning_short": it.reasoning_short,
                        "exactness": it.exactness,
                        "topicality": it.topicality,
                        "coverage": it.coverage,
                        "contextual_fit": it.contextual_fit,
                        "trec_grade": it.trec_grade,
                        "judge_model": getattr(self.llm_client, "model", "unknown"),
                        "thinking_level": getattr(self.llm_client, "thinking_level", "unknown"),
                    }
                else:
                    self.cache[pair_id] = {
                        "reasoning_short": "Fallback grade 0 (missing from batch output).",
                        "exactness": 0,
                        "topicality": 0,
                        "coverage": 0,
                        "contextual_fit": 0,
                        "trec_grade": 0,
                        "judge_model": getattr(self.llm_client, "model", "unknown"),
                        "thinking_level": getattr(self.llm_client, "thinking_level", "unknown"),
                    }
            self.save_cache()
            print(
                f"[{getattr(self.llm_client, 'model', 'Judge')} | thinking={getattr(self.llm_client, 'thinking_level', 'N/A')}] "
                f"Judged batch {b_idx}/{n_batches} ({min(start + batch_size, len(to_process))}/{len(to_process)} pairs)",
                flush=True,
            )

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
