# CNN News Search & Stance-Aware Recommender

Information retrieval and stance-aware recommendation experiments on the CNN English News corpus (37,949 raw articles; 37,916 cleaned articles spanning 2011–2022).

The project investigates how lexical, dense, and hybrid retrieval architectures behave across different search intents (exact keyword lookups, semantic paraphrases, late-article fact recall, temporal queries, and opinion queries), and extends the retrieval pipeline with a stance- and policy-frame-diversified news recommender.

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_cnn_english_news_eda.ipynb              # Corpus exploration, length distributions, section/temporal analysis
│   ├── 02_preprocessing.ipynb                     # Boilerplate stripping, deduplication, token-aware cleaning
│   ├── 03_evaluation_and_lexical_baselines.ipynb  # Benchmark construction, pooled TREC grading, BM25 vs. BM25F
│   ├── 04_dense_retrieval_and_chunking.ipynb      # Passage chunking, contextual headers, MeanP/MaxP/Debiased MaxP
│   ├── 05_hybrid_and_reranking.ipynb              # Hybrid RRF, Dynamic Alpha Tuning (DAT), Cross-Encoder & LLM reranking
│   └── 06_stance_aware_recommender.ipynb          # Stance profiling and Aspect-MMR diversification across policy frames
├── src/
│   ├── tokenizer.py                               # Subword BPE tokenization utilities (151k vocabulary)
│   ├── index/
│   │   ├── base.py                                # BaseSearchIndex interface
│   │   ├── bm25.py                                # Sparse CSR implementations of flat BM25 and fielded BM25F
│   │   ├── dense.py                               # PassageChunker and DensePassageIndex (MeanP, MaxP, Debiased MaxP)
│   │   └── hybrid.py                              # HybridRRFIndex and DynamicAlphaHybridIndex (DAT)
│   ├── embedder/
│   │   ├── base.py                                # BaseEmbedder interface
│   │   └── harrier.py                             # Cached Harrier (microsoft/harrier-oss-v1-0.6b, 1024-d) client
│   ├── reranker/
│   │   ├── base.py                                # BaseReranker decorator interface
│   │   ├── cross_encoder.py                       # Qwen3-Reranker-0.6B cross-encoder reranker
│   │   └── llm_rubric.py                          # Multi-criteria pointwise LLM rubric reranker
│   ├── recommender/
│   │   ├── base.py                                # BaseNewsRecommender interface
│   │   └── stance.py                              # StanceAwareNewsRecommender (stance extraction + Aspect-MMR)
│   ├── eval/
│   │   ├── metrics.py                             # nDCG@K, MRR@K, Recall@K, MAP@K, bootstrap CIs, permutation tests
│   │   ├── benchmark.py                           # Stratified anchor sampling, 5-category query generation, TREC judge
│   │   └── tracker.py                             # Experiment runner and metric logger
│   └── llm/
│       ├── base.py                                # BaseStructuredLLMClient interface
│       ├── sglang_client.py                       # Local SGLang client for schema-constrained JSON generation
│       └── gemini_client.py                       # Google Gemini API structured output client
└── presentation/
    └── bull_cnn_nlp_stl_presentation.pptx.pdf     # Project presentation slides
```

> **Note:** The `data/` directory (raw/cleaned Parquet files, precomputed BM25F sparse matrices, and 208,187 passage embeddings) is excluded from version control due to file size.

## Methodology Overview

### 1. Corpus Exploration & Preprocessing (`01`, `02`)
- **Corpus profile:** 37,949 CNN articles across 10 primary sections (`politics`, `us`, `sport`, `world`, `opinions`, `europe`, `business`, `health`, `entertainment`, `uk`). The median article length is ~687 words (~890 BPE tokens), meaning standard 512-token dense encoders truncate roughly half of the typical article.
- **Cleaning pipeline:** Removes dateline prefixes (`(CNN) --`), wire-service boilerplate, and empty stubs (33 empty records flagged and excluded, leaving 37,916 articles), while preserving separate `headline_clean`, `description_clean`, and `article_clean` fields.

### 2. Evaluation Benchmark & Lexical Baselines (`03`)
- **Controlled Multi-Query Perturbation Benchmark:** 50 stratified anchor articles (sampled across short, median, and long length strata and editorial sections) $\times$ 5 query categories = 250 queries:
  1. `keyword_lookup` (2–4 words, entity/proper-noun heavy)
  2. `semantic_paraphrase` (4–8 words, zero lexical overlap with the headline)
  3. `deep_fact_recall` (6–12 words targeting facts in the final third of the article)
  4. `temporal_lookup` (explicit year/month + topic constraints)
  5. `stance_opinion` (requests for commentary with a specific stance toward a target entity)
- **Dual evaluation:** Reports both **Strict Source-Doc Recovery** (`Strict MRR@10`, `Strict Recall@K`) and **Pooled Graded TREC Evaluation** (`nDCG@10`, `Fair MRR@10`) using a 0–3 relevance rubric so that retrieving valid follow-up coverage on the same news event is credited rather than penalized as a false negative.
- **Lexical indexes:** Compares flat Okapi BM25 against fielded **BM25F** (`headline` weight 3.0, `description` 1.5, `body` 1.0, per-field length normalization $b_f \in \{0.30, 0.50, 0.75\}$).

### 3. Dense Passage Retrieval & Length-Bias Correction (`04`)
- **Contextual passage chunking:** Splits articles into overlapping 220-token chunks (40-token overlap), yielding 208,187 passages across the 37,916 articles, prepended with a `[Headline | Date | Section]` contextual header and encoded into 1024-dimensional vectors with `microsoft/harrier-oss-v1-0.6b`.
- **Debiased MaxP aggregation:** Naive `MaxP` favors long articles simply because an article with $M_d$ chunks has $M_d$ independent chances to produce a high cosine similarity score. We correct for extreme-value length bias and combine the lead chunk with the debiased peak passage:
  $$s_{\text{doc}}(q, d) = \lambda s_{\text{lead}}(q, d) + (1 - \lambda)\left(\max_{j=1 \dots M_d} s(q, p_{d,j}) - \gamma \sqrt{\ln M_d}\right)$$

### 4. Hybrid Fusion & Two-Stage Reranking (`05`)
- **Reciprocal Rank Fusion (`HybridRRFIndex`):** Rank-based fusion of BM25F and Debiased MaxP.
- **Dynamic Alpha Tuning (`DynamicAlphaHybridIndex`):** Estimates query lexical specificity $\alpha(q) \in [0.10, 0.65]$ from the BM25F top-1 to tail Z-score and per-token IDF confidence, shifting weight toward BM25F on exact keyword queries and toward dense MaxP on semantic paraphrases.
- **Stage-2 Reranking:** Re-scores top Stage-1 candidates with `Qwen/Qwen3-Reranker-0.6B` (feeding both the article lead and the highest-scoring interior MaxP chunk) or a pointwise multi-criteria LLM rubric reranker.

### 5. Stance-Aware & Policy-Frame-Diversified Recommender (`06`)
- Given a natural-language user prompt (e.g., requesting critical or supportive commentary on a political figure or policy), `StanceAwareNewsRecommender`:
  1. Extracts the `target_entity`, `desired_stance` (`AGAINST`, `FAVOR`, `NEUTRAL_ANALYTICAL`), and expanded `retrieval_query`.
  2. Retrieves candidate opinion/analysis articles via Stage-1 hybrid search and profiles each candidate's stance, intensity (0–5), argument thesis, and primary policy frame across a 12-dimension political framing taxonomy (`ALL_POLICY_FRAMES`).
  3. Applies tier-preserving **Aspect-MMR** to select top-$K$ articles that match the requested stance while penalizing redundant policy frames and near-duplicate theses.

## Experimental Results Summary (250 Queries, 37,916 Articles)

| Run | Family | Strict MRR@10 | Strict R@10 | Fair MRR@10 | TREC nDCG@10 | P50 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `exp01_bm25_flat` | Lexical (Flat BM25) | 0.6026 | 0.8040 | 0.7793 | 0.7371 | 12.5 ms |
| `exp02_bm25f_fielded` | Lexical (Fielded BM25F) | 0.5965 | 0.8040 | 0.7772 | 0.7401 | 12.7 ms |
| `exp03a_dense_truncated` | Dense (Truncated 220 tok) | 0.5737 | 0.7160 | 0.8055 | 0.7218 | 5.9 ms |
| `exp03b_dense_meanp` | Dense (Contextual MeanP) | 0.5731 | 0.7480 | 0.7661 | 0.7031 | 9.2 ms |
| `exp03c_dense_maxp_naive` | Dense (Contextual Naive MaxP) | 0.6070 | 0.8200 | 0.8247 | 0.7732 | 8.0 ms |
| `exp03e_dense_contextual_debiased` | Dense (Contextual Debiased MaxP) | 0.6112 | 0.7880 | 0.8319 | 0.7777 | 8.9 ms |
| `exp04a_hybrid_rrf` | Hybrid (Weighted RRF) | 0.6169 | 0.8240 | 0.8298 | 0.8004 | 27.1 ms |
| `exp04b_hybrid_dat` | Hybrid (Dynamic Alpha DAT) | 0.6670 | 0.8840 | **0.8817** | **0.8446** | 45.0 ms |
| `exp04c_hybrid_qwen_reranker` | Rerank (DAT + Qwen3-Reranker-0.6B) | **0.6775** | **0.8880** | 0.8813 | 0.8289 | 44.8 ms |
| `exp04d_search_plus_llm_rubric` | Rerank (Pointwise Rubric) | 0.6450 | 0.8520 | 0.8437 | 0.7918 | 3641.7 ms |
