"""Glass-box Dense Passage Retrieval & Unbiased Passage-to-Document Aggregation.

Method & Literature References:
- Dense Passage Retrieval (DPR):
  [Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., EMNLP 2020, arXiv:2004.04906)](https://arxiv.org/abs/2004.04906)
- MaxP & MeanP Passage-to-Document Score Aggregation:
  [Deeper Text Understanding for IR with Contextual Neural Language Modeling (Dai & Callan, SIGIR 2019, arXiv:1905.09217)](https://arxiv.org/abs/1905.09217)
- Contextual Retrieval / Contextual Header Injection:
  [Introducing Contextual Retrieval (Anthropic Research, Sept 2024)](https://www.anthropic.com/news/contextual-retrieval)
  and [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models (Günther et al., 2024, arXiv:2409.04701)](https://arxiv.org/abs/2409.04701)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.embedder.base import BaseEmbedder
from src.index.base import BaseSearchIndex
from src.tokenizer import split_bpe_tokens

ChunkingStrategy = Literal[
    "truncated_doc",      # 1 vector per article (headline + desc + first 220 tokens)
    "chunks_raw",         # Overlapping 220-token body chunks without repeating header
    "chunks_contextual",  # Overlapping 220-token body chunks with [Headline | Date | Section] header
]

TokenUnit = Literal[
    "bpe_tokens",         # Exact subword tokens via Harrier/Qwen BPE tokenizer (vocab=151,643)
    "pretokens",          # Whitespace-delimited pre-tokens (used by precomputed .npy caches)
]

AggregationMode = Literal[
    "meanp",          # Dai & Callan (2019) MeanP: Mean across all chunks of doc d
    "maxp",           # Dai & Callan (2019) MaxP: Raw max across chunks of doc d
    "debiased_maxp",  # Length-unbiased Lead + Extreme-Value-Corrected MaxP
]


@dataclass
class ChunkedCorpus:
    """Container holding passage texts and parent-document index mappings."""

    passage_texts: list[str]
    passage_doc_idx: np.ndarray
    lead_passage_idx: np.ndarray
    doc_ids: list[str]


class PassageChunker:
    """Splits a corpus DataFrame into overlapping token passages with optional contextual headers."""

    def __init__(
        self,
        strategy: ChunkingStrategy = "chunks_raw",
        chunk_size: int = 220,
        overlap: int = 40,
        token_unit: TokenUnit = "pretokens",
    ) -> None:
        self.strategy: ChunkingStrategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_unit: TokenUnit = token_unit

    def split_tokens(self, tokens: list[str]) -> list[str]:
        """Split a pre-tokenized sequence into overlapping passages of `self.chunk_size` tokens."""
        if not tokens:
            return [""]
        if len(tokens) <= self.chunk_size:
            return [" ".join(tokens)]
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            piece = tokens[start : start + self.chunk_size]
            if not piece:
                break
            chunks.append(" ".join(piece))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    # Alias for backward compatibility
    split_words = split_tokens

    def split_text(self, body_text: str) -> list[str]:
        """Split `body_text` into overlapping passages according to `self.token_unit`."""
        if self.token_unit == "bpe_tokens":
            return split_bpe_tokens(body_text, chunk_size=self.chunk_size, overlap=self.overlap)
        return self.split_tokens(str(body_text).split())

    def chunk_dataframe(self, df: pd.DataFrame, exclude_empty_flag: bool = True) -> ChunkedCorpus:
        """Transform corpus DataFrame into a `ChunkedCorpus`."""
        work_df = df[~df["flag_empty_article"]].copy() if exclude_empty_flag else df.copy()
        doc_ids: list[str] = work_df["Index"].astype(str).tolist()

        passage_texts: list[str] = []
        passage_doc_idx: list[int] = []
        lead_passage_idx: list[int] = []

        for doc_i, (_, row) in enumerate(work_df.iterrows()):
            headline = str(row["headline_clean"]).strip()
            desc = str(row["description_clean"]).strip()
            section = str(row["Section"]).strip()
            pub_date = str(row["published"])[:10]
            body_text = str(row["article_clean"])

            lead_passage_idx.append(len(passage_texts))

            if self.strategy == "truncated_doc":
                raw_chunks = self.split_text(body_text)
                lead_body = raw_chunks[0] if raw_chunks else ""
                passage_texts.append(f"{headline}. {desc}. {lead_body}".strip())
                passage_doc_idx.append(doc_i)

            elif self.strategy == "chunks_raw":
                raw_chunks = self.split_text(body_text)
                for c_idx, chunk_str in enumerate(raw_chunks):
                    if c_idx == 0:
                        passage_texts.append(f"{headline}. {desc}. {chunk_str}".strip())
                    else:
                        passage_texts.append(chunk_str)
                    passage_doc_idx.append(doc_i)

            elif self.strategy == "chunks_contextual":
                context_header = f"[Headline: {headline} | Date: {pub_date} | Section: {section}]"
                raw_chunks = self.split_text(body_text)
                for c_idx, chunk_str in enumerate(raw_chunks):
                    if c_idx == 0 and desc:
                        passage_texts.append(f"{context_header} {desc} {chunk_str}".strip())
                    else:
                        passage_texts.append(f"{context_header} {chunk_str}".strip())
                    passage_doc_idx.append(doc_i)
            else:
                raise ValueError(f"Unknown chunking strategy: {self.strategy}")

        return ChunkedCorpus(
            passage_texts=passage_texts,
            passage_doc_idx=np.asarray(passage_doc_idx, dtype=np.int32),
            lead_passage_idx=np.asarray(lead_passage_idx, dtype=np.int32),
            doc_ids=doc_ids,
        )


class DensePassageIndex(BaseSearchIndex):
    """Dense Passage Retriever supporting MeanP, MaxP, and Length-Debiased MaxP aggregation."""

    def __init__(
        self,
        passage_embeddings: np.ndarray,
        chunked_corpus: ChunkedCorpus,
        embedder: BaseEmbedder,
        aggregation: AggregationMode = "debiased_maxp",
        lead_weight: float = 0.35,
        length_penalty_gamma: float = 0.015,
    ) -> None:
        self.passage_embeddings = np.asarray(passage_embeddings, dtype=np.float32)
        self.chunked_corpus = chunked_corpus
        self.passage_doc_idx = chunked_corpus.passage_doc_idx
        self.lead_passage_idx = chunked_corpus.lead_passage_idx
        self.doc_ids = chunked_corpus.doc_ids
        self.doc_id_to_idx = {str(did): i for i, did in enumerate(self.doc_ids)}
        self.n_docs = len(self.doc_ids)
        self.embedder = embedder
        self.aggregation: AggregationMode = aggregation
        self.lead_weight = lead_weight
        self.length_penalty_gamma = length_penalty_gamma

        self.doc_chunk_counts = np.bincount(
            self.passage_doc_idx, minlength=self.n_docs
        ).astype(np.float32)
        self.doc_chunk_counts = np.maximum(self.doc_chunk_counts, 1.0)
        # Extreme-value correction sqrt(ln(M_d)) neutralizing multiple-testing length bias
        self.length_bias_correction = (
            self.length_penalty_gamma * np.sqrt(np.log(self.doc_chunk_counts))
        ).astype(np.float32)

    def get_best_passage_text(self, query: str, doc_id: str) -> str:
        """Return the highest-scoring non-lead chunk text for `(query, doc_id)` if `chunk_idx > 0`."""
        d_idx = self.doc_id_to_idx.get(str(doc_id))
        if d_idx is None:
            return ""
        start = int(self.lead_passage_idx[d_idx])
        end = (
            int(self.lead_passage_idx[d_idx + 1])
            if d_idx + 1 < self.n_docs
            else len(self.chunked_corpus.passage_texts)
        )
        if end - start <= 1:
            return ""
        q_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q_vec))
        if norm > 0:
            q_vec = q_vec / norm
        sims = self.passage_embeddings[start:end] @ q_vec
        best_local = int(np.argmax(sims))
        if best_local == 0:
            return ""
        return self.chunked_corpus.passage_texts[start + best_local]

    def score_all_documents(self, query: str) -> np.ndarray:
        """Compute aggregated document scores `(N,)` for `query`."""
        q_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q_vec))
        if norm > 0:
            q_vec = q_vec / norm

        passage_sims = self.passage_embeddings @ q_vec  # shape: (M,)

        if self.aggregation == "meanp":
            doc_scores = np.zeros(self.n_docs, dtype=np.float32)
            np.add.at(doc_scores, self.passage_doc_idx, passage_sims)
            doc_scores /= self.doc_chunk_counts

        elif self.aggregation == "maxp":
            doc_scores = np.full(self.n_docs, -1e9, dtype=np.float32)
            np.maximum.at(doc_scores, self.passage_doc_idx, passage_sims)

        elif self.aggregation == "debiased_maxp":
            max_scores = np.full(self.n_docs, -1e9, dtype=np.float32)
            np.maximum.at(max_scores, self.passage_doc_idx, passage_sims)
            debiased_peak = max_scores - self.length_bias_correction
            lead_scores = passage_sims[self.lead_passage_idx]
            doc_scores = (
                self.lead_weight * lead_scores
                + (1.0 - self.lead_weight) * debiased_peak
            )
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation}")

        return doc_scores

    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-K `(doc_id, score)` pairs."""
        doc_scores = self.score_all_documents(query)
        if top_k >= len(doc_scores):
            top_idx = np.argsort(-doc_scores)
        else:
            part = np.argpartition(-doc_scores, top_k)[:top_k]
            top_idx = part[np.argsort(-doc_scores[part])]
        return [(self.doc_ids[int(i)], float(doc_scores[int(i)])) for i in top_idx]
