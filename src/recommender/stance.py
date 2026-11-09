"""Stance-Aware & Policy-Frame-Diversified News Recommender ('A2CG / Search + LLM + MMR').

Method & Literature References:
- Cross-Target & Cross-Domain LLM Stance Detection (`AGAINST`, `FAVOR`, `NEUTRAL_ANALYTICAL`):
  [StanceEval 2026: Cross-Target and Cross-Domain Stance Detection (Oct 2026, arXiv:2610.03215)](https://arxiv.org/abs/2610.03215)
  and [Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions (Pangtey et al., Jan 2026, arXiv:2505.08464)](https://arxiv.org/abs/2505.08464)
- Auditing LLM Media Framing & Policy Frames in News:
  [Who Judges the Frame? Auditing Multimodal LLM Judges for News Framing Across Event-Level Perspectives (Tommasel et al., Sept 2026, arXiv:2610.00071)](https://arxiv.org/abs/2610.00071),
  [Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation (Jan 2026, arXiv:2601.05835)](https://arxiv.org/abs/2601.05835),
  and [Frame In, Frame Out (FIFO): Measuring Framing Bias in LLM-Generated News Summaries (May 2025 / 2026, arXiv:2505.05406)](https://arxiv.org/abs/2505.05406)
- Aspect-Aware Candidate Generation & Perspective Diversity in News Recommenders:
  [Democratizing News Recommenders: Modeling Multiple Perspectives for News Candidate Generation with VQ-VAE (Hardy, Padó, Wührl, & Ceron, Aug 2025, arXiv:2508.13978)](https://arxiv.org/abs/2508.13978)
  and [A Survey on LLM-based News Recommender Systems (Wang, Liesaputra, & Huang, Feb 2025, arXiv:2502.09797)](https://arxiv.org/abs/2502.09797)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.embedder.base import BaseEmbedder
from src.index.base import BaseSearchIndex
from src.llm.base import BaseStructuredLLMClient
from src.llm.sglang_client import SGLangStructuredClient
from src.recommender.base import BaseNewsRecommender

StanceLabel = Literal["AGAINST", "FAVOR", "NEUTRAL_ANALYTICAL"]

PolicyFrame = Literal[
    "constitutional_democracy_rule_of_law",
    "crime_justice_law_enforcement",
    "economy_fiscal_inflation_trade",
    "foreign_policy_war_national_security",
    "immigration_border_asylum",
    "healthcare_science_climate_environment",
    "civil_rights_identity_social_values",
    "electoral_horserace_campaign_strategy",
    "partisan_governance_legislative_gridlock",
    "character_competence_ethics_corruption",
    "media_disinformation_public_discourse",
    "human_interest_lived_impact",
]

ALL_POLICY_FRAMES: tuple[PolicyFrame, ...] = (
    "constitutional_democracy_rule_of_law",
    "crime_justice_law_enforcement",
    "economy_fiscal_inflation_trade",
    "foreign_policy_war_national_security",
    "immigration_border_asylum",
    "healthcare_science_climate_environment",
    "civil_rights_identity_social_values",
    "electoral_horserace_campaign_strategy",
    "partisan_governance_legislative_gridlock",
    "character_competence_ethics_corruption",
    "media_disinformation_public_discourse",
    "human_interest_lived_impact",
)


class UserStanceIntent(BaseModel):
    """Structured extraction of user recommendation intent (Chain-of-Thought `reasoning_short` first)."""

    reasoning_short: str = Field(
        description="Brief reasoning identifying the target entity/topic and what stance the user wants to read."
    )
    target_entity: str = Field(
        description="Canonical name of the political figure, institution, or policy targeted by the user."
    )
    desired_stance: StanceLabel = Field(
        description="Whether the user wants articles critical/opposing (AGAINST), supportive/praising (FAVOR), or neutral (NEUTRAL_ANALYTICAL) toward target_entity."
    )
    retrieval_query: str = Field(
        description="Expanded search query combining the target entity with opinion/editorial themes for Stage-1 retrieval."
    )


class ArticleStanceProfile(BaseModel):
    """Structured target-specific stance and 12-dimension policy frame profile for a candidate article."""

    reasoning_short: str = Field(
        description="Concise 1-2 sentence analysis of how the author portrays the target entity and the core argument."
    )
    target_is_central: bool = Field(
        description="True if the target entity is a substantive subject of the article rather than a passing mention."
    )
    stance_toward_target: StanceLabel = Field(
        description="Author's stance toward the target entity: AGAINST (critical/opposing), FAVOR (supportive/defending), or NEUTRAL_ANALYTICAL."
    )
    stance_intensity: int = Field(
        ge=0,
        le=5,
        description="Intensity of the stance from 0 (purely neutral report) to 5 (sharply polemical/passionate critique or defense).",
    )
    policy_frame: PolicyFrame = Field(
        description="Primary news & policy frame (from the 12-frame taxonomy) through which the article evaluates the target entity."
    )
    argument_thesis: str = Field(
        description="One-sentence summary of the author's specific thesis regarding the target entity."
    )


class StanceAwareNewsRecommender(BaseNewsRecommender):
    """Recommends stance-consistent, frame-diverse opinion articles via Hybrid Search + LLM Stance Profiling + MMR."""

    def __init__(
        self,
        base_index: BaseSearchIndex,
        df: pd.DataFrame,
        embedder: BaseEmbedder,
        llm_client: BaseStructuredLLMClient | None = None,
        candidate_depth: int = 20,
        mmr_lambda: float = 0.65,
        frame_redundancy_penalty: float = 0.35,
        max_workers: int = 8,
        cache_path: str | Path = "../data/stance_recommender_cache.json",
    ) -> None:
        self.base_index = base_index
        self.df_by_id = df.set_index(df["Index"].astype(str))
        self.embedder = embedder
        self.llm_client = llm_client or SGLangStructuredClient()
        self.candidate_depth = candidate_depth
        self.mmr_lambda = mmr_lambda
        self.frame_redundancy_penalty = frame_redundancy_penalty
        self.max_workers = max_workers
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")

    def parse_user_intent(self, user_prompt: str) -> UserStanceIntent:
        """Extract target entity and desired stance from natural language via LLM Structured Outputs."""
        cache_key = f"intent::{user_prompt.strip()}"
        if cache_key not in self.cache:
            intent: UserStanceIntent = self.llm_client.generate_structured(
                system_prompt=(
                    "You are an expert news recommendation intent parser. Extract the target entity, "
                    "the user's desired stance toward that entity (AGAINST, FAVOR, or NEUTRAL_ANALYTICAL), "
                    "and an effective retrieval query for finding relevant opinion/commentary articles."
                ),
                user_prompt=user_prompt,
                schema_cls=UserStanceIntent,
            )  # type: ignore[assignment]
            with self._lock:
                self.cache[cache_key] = intent.model_dump()
                self._save_cache()
        return UserStanceIntent(**self.cache[cache_key])

    def profile_article_stance(self, doc_id: str, target_entity: str) -> ArticleStanceProfile:
        """Profile a candidate article's stance and policy frame toward `target_entity`."""
        cache_key = f"profile_v2::{target_entity.lower().strip()}::{doc_id}"
        if cache_key not in self.cache:
            row = self.df_by_id.loc[doc_id]
            words = str(row["article_clean"]).split()
            snippet = (
                " ".join(words[:320]) + " ... " + " ".join(words[-160:])
                if len(words) > 480
                else " ".join(words)
            )
            prompt = (
                f"Target Entity: {target_entity}\n\n"
                f"Article Headline: {row['headline_clean']}\n"
                f"Section: {row['Section']} | Published: {str(row['published'])[:10]}\n"
                f"Description: {row['description_clean']}\n"
                f"Article Text:\n{snippet}"
            )
            profile: ArticleStanceProfile = self.llm_client.generate_structured(
                system_prompt=(
                    "You are a political science and media framing analyst. Determine whether the "
                    "target entity is substantive to this article, classify the author's stance toward "
                    "the target entity (AGAINST, FAVOR, or NEUTRAL_ANALYTICAL), rate stance_intensity (0-5), "
                    "assign the primary policy_frame from the 12-dimension political news framing taxonomy, "
                    "and summarize the author's thesis in one sentence."
                ),
                user_prompt=prompt,
                schema_cls=ArticleStanceProfile,
            )  # type: ignore[assignment]
            with self._lock:
                self.cache[cache_key] = profile.model_dump()
                self._save_cache()
        return ArticleStanceProfile(**self.cache[cache_key])

    def recommend(
        self,
        target_entity: str,
        desired_stance: StanceLabel,
        retrieval_query: str | None = None,
        top_k: int = 5,
        seed_doc_id: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return `(mmr_recommendations_df, all_profiled_candidates_df)`."""
        query = retrieval_query or f"{target_entity} opinion commentary analysis"
        fetch_depth = (
            self.candidate_depth * 3 + 15
            if os.environ.get("BULL_CNN_JUDGE_CACHE_ONLY") == "1"
            else self.candidate_depth + 5
        )
        stage1_hits = self.base_index.search_with_scores(query, top_k=fetch_depth)
        stage1_hits = [(d, s) for d, s in stage1_hits if d != str(seed_doc_id)]
        if os.environ.get("BULL_CNN_JUDGE_CACHE_ONLY") == "1":
            ent_key = target_entity.lower().strip()
            stage1_hits = [
                (d, s) for d, s in stage1_hits if f"profile_v2::{ent_key}::{d}" in self.cache
            ]
        stage1_hits = stage1_hits[: self.candidate_depth]

        def _profile_one(item: tuple[int, tuple[str, float]]) -> dict[str, Any]:
            rank_idx, (doc_id, retr_score) = item
            prof = self.profile_article_stance(doc_id, target_entity=target_entity)
            row = self.df_by_id.loc[doc_id]
            if prof.stance_toward_target == desired_stance and prof.target_is_central:
                stance_tier = 3  # Primary: Central & exact desired stance
                base_bonus = 2.0
            elif prof.stance_toward_target == desired_stance:
                stance_tier = 2  # Secondary: Desired stance
                base_bonus = 1.0
            elif prof.stance_toward_target == "NEUTRAL_ANALYTICAL":
                stance_tier = 1  # Fallback: Neutral analytical (never opposite polarity)
                base_bonus = 0.2
            else:
                stance_tier = 0  # Opposite stance: strictly last resort
                base_bonus = -2.0

            utility = base_bonus + (
                0.65 * (prof.stance_intensity / 5.0) + 0.35 * (1.0 / (10 + rank_idx))
                if stance_tier >= 2
                else -0.10 * prof.stance_intensity
            )
            return {
                "doc_id": doc_id,
                "stage1_rank": rank_idx,
                "stage1_score": round(float(retr_score), 4),
                "headline": row["headline_clean"],
                "published": str(row["published"])[:10],
                "section": row["Section"],
                "target_is_central": prof.target_is_central,
                "stance": prof.stance_toward_target,
                "intensity": prof.stance_intensity,
                "policy_frame": prof.policy_frame,
                "argument_thesis": prof.argument_thesis,
                "stance_tier": stance_tier,
                "stance_utility": round(float(utility), 4),
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            candidate_rows = list(pool.map(_profile_one, enumerate(stage1_hits, start=1)))

        cand_df = pd.DataFrame(candidate_rows)
        if cand_df.empty:
            return cand_df, cand_df

        # Embed candidate headlines + theses for aspect/frame-diversified MMR
        texts_for_sim = [f"{r['headline']}. {r['argument_thesis']}" for r in candidate_rows]
        vecs = np.vstack([
            np.asarray(self.embedder.embed_query(t), dtype=np.float32).ravel()
            for t in texts_for_sim
        ])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        sim_matrix = vecs @ vecs.T

        # Tier-Preserving Aspect-Aware MMR diversification across the 12 Policy Frames:
        # Candidates with a higher `stance_tier` (matching stance) are ALWAYS selected before lower tiers.
        selected_indices: list[int] = []
        remaining = list(range(len(candidate_rows)))

        while remaining and len(selected_indices) < top_k:
            max_available_tier = max(candidate_rows[idx]["stance_tier"] for idx in remaining)
            tier_candidates = [
                idx for idx in remaining if candidate_rows[idx]["stance_tier"] == max_available_tier
            ]
            best_i = -1
            best_mmr = -1e9
            for idx in tier_candidates:
                rel = candidate_rows[idx]["stance_utility"]
                if not selected_indices:
                    mmr_score = rel
                else:
                    max_red = max(
                        float(sim_matrix[idx, sel_j])
                        + (
                            self.frame_redundancy_penalty
                            if candidate_rows[idx]["policy_frame"] == candidate_rows[sel_j]["policy_frame"]
                            else 0.0
                        )
                        for sel_j in selected_indices
                    )
                    mmr_score = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * max_red
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_i = idx
            selected_indices.append(best_i)
            remaining.remove(best_i)

        rec_df = cand_df.iloc[selected_indices].copy().reset_index(drop=True)
        rec_df.insert(0, "rec_rank", range(1, len(rec_df) + 1))
        return rec_df, cand_df

    def recommend_from_prompt(self, user_prompt: str, top_k: int = 5) -> pd.DataFrame:
        """End-to-end recommendation from natural language user prompt."""
        intent = self.parse_user_intent(user_prompt)
        rec_df, _ = self.recommend(
            target_entity=intent.target_entity,
            desired_stance=intent.desired_stance,
            retrieval_query=intent.retrieval_query,
            top_k=top_k,
        )
        return rec_df

    @staticmethod
    def evaluate_recommendation_quality(
        df_list: pd.DataFrame,
        desired_stance: StanceLabel,
        embedder: BaseEmbedder,
    ) -> dict[str, float]:
        """Compute quantitative Stance Precision@K, Unique Policy Frames@K, and Intra-List Diversity."""
        if df_list.empty:
            return {
                "stance_precision_at_k": 0.0,
                "central_stance_precision_at_k": 0.0,
                "mean_stance_intensity": 0.0,
                "unique_policy_frames_at_k": 0.0,
                "frame_coverage_ratio": 0.0,
                "intra_list_diversity": 0.0,
            }
        k = len(df_list)
        stance_hits = (df_list["stance"] == desired_stance).astype(float)
        central_hits = (
            (df_list["stance"] == desired_stance) & (df_list["target_is_central"] == True)
        ).astype(float)
        unique_frames = int(df_list["policy_frame"].nunique())

        texts = [f"{r['headline']}. {r['argument_thesis']}" for _, r in df_list.iterrows()]
        vecs = np.vstack([
            np.asarray(embedder.embed_query(t), dtype=np.float32).ravel() for t in texts
        ])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        sim_mat = vecs @ vecs.T
        if k > 1:
            upper_sims = sim_mat[np.triu_indices(k, k=1)]
            ild = float(1.0 - np.mean(upper_sims))
        else:
            ild = 1.0

        return {
            "stance_precision_at_k": round(float(stance_hits.mean()), 4),
            "central_stance_precision_at_k": round(float(central_hits.mean()), 4),
            "mean_stance_intensity": round(float((df_list["intensity"] * stance_hits).mean()), 4),
            "unique_policy_frames_at_k": float(unique_frames),
            "frame_coverage_ratio": round(float(unique_frames / k), 4),
            "intra_list_diversity": round(ild, 4),
        }
