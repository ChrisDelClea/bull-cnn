"""Glass-box implementation of Okapi BM25 and Fielded BM25F using SciPy CSR sparse matrices.

Method & Literature References:
- Okapi BM25 Probabilistic Retrieval Model:
  [Okapi at TREC-3 (Robertson, Walker, Jones, Hancock-Beaulieu, & Gatford, NIST 1994)](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
  and [The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, FnTIR 2009)](https://doi.org/10.1561/1500000019)
- Fielded BM25F (Per-Field Length Normalization & Linear Combination of Pseudo-TFs):
  [Simple BM25 Extension to Multiple Weighted Fields (Robertson, Zaragoza, & Taylor, CIKM 2004)](https://doi.org/10.1145/1031171.1031181)
- Standard Glasgow IR Stopword List (`ENGLISH_STOP_WORDS`):
  [Information Retrieval (Van Rijsbergen, 1979) / scikit-learn ENGLISH_STOP_WORDS](https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words)
"""

from __future__ import annotations

from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.index.base import BaseSearchIndex


def tokenize_text(text: str, remove_stopwords: bool = True) -> list[str]:
    """Deterministic alphanumeric tokenizer using scikit-learn's Glasgow IR stopword list."""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            tok = "".join(buf)
            buf.clear()
            if len(tok) > 1 and (not remove_stopwords or tok not in ENGLISH_STOP_WORDS):
                tokens.append(tok)
    if buf:
        tok = "".join(buf)
        if len(tok) > 1 and (not remove_stopwords or tok not in ENGLISH_STOP_WORDS):
            tokens.append(tok)
    return tokens


class SparseBM25Index(BaseSearchIndex):
    """Transparent sparse-matrix implementation of Flat Okapi BM25 and Fielded BM25F.

    Implements the Robertson et al. (CIKM 2004) BM25F formulation:
        tf_tilde(t, d) = sum_{f in fields} w_f * tf(t, d, f) / (1 - b_f + b_f * (dl_f / avgdl_f))
        BM25F(q, d)    = sum_{t in q} IDF(t) * (tf_tilde(t, d) * (k1 + 1)) / (tf_tilde(t, d) + k1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        field_weights: dict[str, float],
        field_b: dict[str, float] | None = None,
        k1: float = 1.5,
        exclude_empty_flag: bool = True,
    ) -> None:
        self.k1 = k1
        self.field_weights = field_weights
        self.field_b = field_b or {f: 0.75 for f in field_weights}

        work_df = df[~df["flag_empty_article"]].copy() if exclude_empty_flag else df.copy()
        self.doc_ids: np.ndarray = work_df["Index"].astype(str).to_numpy()
        self.n_docs: int = len(self.doc_ids)

        vocab: dict[str, int] = {}
        doc_pseudo_tfs: list[dict[int, float]] = [{} for _ in range(self.n_docs)]

        for field_name, weight in self.field_weights.items():
            b_val = self.field_b.get(field_name, 0.75)
            texts = work_df[field_name].fillna("").astype(str).tolist()
            tokenized_docs = [tokenize_text(t) for t in texts]
            lengths = np.array([len(toks) for toks in tokenized_docs], dtype=np.float32)
            avgdl = float(np.mean(lengths)) if self.n_docs > 0 else 1.0
            avgdl = max(avgdl, 1.0)

            for doc_idx, toks in enumerate(tokenized_docs):
                if not toks:
                    continue
                norm_denom = 1.0 - b_val + b_val * (len(toks) / avgdl)
                counts = Counter(toks)
                doc_map = doc_pseudo_tfs[doc_idx]
                for term, tf in counts.items():
                    term_id = vocab.setdefault(term, len(vocab))
                    doc_map[term_id] = doc_map.get(term_id, 0.0) + weight * (tf / norm_denom)

        self.vocab: dict[str, int] = vocab
        n_vocab = len(vocab)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        doc_freqs = np.zeros(n_vocab, dtype=np.int32)

        for doc_idx, doc_map in enumerate(doc_pseudo_tfs):
            for term_id in doc_map:
                doc_freqs[term_id] += 1

        self.idf: np.ndarray = np.log1p((self.n_docs - doc_freqs + 0.5) / (doc_freqs + 0.5)).astype(
            np.float32
        )

        for doc_idx, doc_map in enumerate(doc_pseudo_tfs):
            for term_id, pseudo_tf in doc_map.items():
                sat_tf = (pseudo_tf * (self.k1 + 1.0)) / (pseudo_tf + self.k1)
                score = float(self.idf[term_id]) * sat_tf
                rows.append(doc_idx)
                cols.append(term_id)
                vals.append(score)

        self.bm25_matrix: sp.csr_matrix = sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(self.n_docs, n_vocab),
        )

    def search_with_scores(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-K `(doc_id, bm25_score)` pairs for `query`."""
        q_tokens = tokenize_text(query)
        term_ids = [self.vocab[t] for t in q_tokens if t in self.vocab]
        if not term_ids:
            return []

        scores = np.asarray(self.bm25_matrix[:, term_ids].sum(axis=1)).ravel()
        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, top_k)[:top_k]
            top_indices = part[np.argsort(-scores[part])]

        return [
            (str(self.doc_ids[i]), float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0
        ]
