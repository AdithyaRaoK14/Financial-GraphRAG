# Final Combined Report

Two sections: the final 3-way comparison (GraphRAG vs baseline vs the new vector-DB baseline), then the full EXP1-7 ablation ladder that produced the GraphRAG config used in that comparison.

## Final 3-Way Comparison

GraphRAG (from `exp7_plus_hyde_final`, your final ablation config) vs the existing Neo4j-hosted baseline vs the new FAISS vector-DB baseline. Baseline's numbers here come from `vectordb_comparison/summary.json` (reused from the same exp7_plus_hyde_final run via --baseline-from, so they should already agree with the ablation section below).

### PDF (n=14)

#### Primary Metrics (paraphrase-robust)

| Pipeline | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| GraphRAG | 78.57% | 64.29% | 78.57% | 86.31% | 88.57% | 96.43% | 78.57% | 88.36% |
| Baseline (Neo4j) | 50.00% | 42.86% | 42.86% | 58.93% | 60.00% | 64.29% | 42.86% | 86.63% |
| Vector-DB (FAISS) | 7.14% | 28.57% | 14.29% | 28.57% | 29.76% | 32.14% | 14.29% | 87.34% |

#### Secondary Metrics (token overlap)

| Pipeline | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| GraphRAG | 0.3597 | 0.4652 | 0.3212 | 0.3762 | 0.0982 |
| Baseline (Neo4j) | 0.3022 | 0.3143 | 0.1754 | 0.2571 | 0.0420 |
| Vector-DB (FAISS) | 0.3022 | 0.2515 | 0.1406 | 0.1993 | 0.0288 |

#### Retrieval Quality

| Pipeline | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 98.21% | 96.43% | 75.00% | 80.08% | 0.7815 | 0.6499 |
| Baseline (Neo4j) | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 95.83% | 69.23% | 21.21% | 76.46% | 0.7815 | 0.6499 |
| Vector-DB (FAISS) | 21.43% | 17.86% | 71.43% | 44.05% | 0.0142 | 0.0125 | 100.00% | 78.57% | 37.50% | 16.67% | 76.12% | n/a | n/a |

#### Answer Behavior

| Pipeline | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| GraphRAG | 0.00% | 100.00% | 0.0000 |
| Baseline (Neo4j) | 7.14% | 50.00% | 0.0000 |
| Vector-DB (FAISS) | 14.29% | 0.00% | 0.1429 |

#### System Efficiency

| Pipeline | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| GraphRAG | 1.282s | 169.450s | 0.0059 |
| Baseline (Neo4j) | 0.111s | 131.731s | n/a |
| Vector-DB (FAISS) | 0.058s | 355.632s | n/a |

### Audio (n=14)

#### Primary Metrics (paraphrase-robust)

| Pipeline | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| GraphRAG | 35.71% | 14.29% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 82.72% |
| Baseline (Neo4j) | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.17% |
| Vector-DB (FAISS) | 21.43% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.00% |

#### Secondary Metrics (token overlap)

| Pipeline | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| GraphRAG | 0.1464 | 0.1101 | 0.0124 | 0.0695 | 0.0031 |
| Baseline (Neo4j) | 0.0987 | 0.1592 | 0.0198 | 0.1122 | 0.0093 |
| Vector-DB (FAISS) | 0.1055 | 0.1590 | 0.0163 | 0.1114 | 0.0073 |

#### Retrieval Quality

| Pipeline | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG | 85.71% | 85.71% | 85.71% | 85.71% | 0.1466 | 0.0481 | 100.00% | 87.50% | 0.00% | 4.17% | 61.82% | 0.6687 | 0.4807 |
| Baseline (Neo4j) | 85.71% | 85.71% | 85.71% | 85.71% | 0.1466 | 0.0481 | 100.00% | 87.50% | n/a | 41.67% | 57.57% | 0.6687 | 0.4807 |
| Vector-DB (FAISS) | 71.43% | 71.43% | 71.43% | 71.43% | 0.0748 | 0.0393 | 100.00% | 87.50% | n/a | 18.75% | 56.90% | n/a | n/a |

#### Answer Behavior

| Pipeline | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| GraphRAG | 7.14% | 0.00% | 0.0000 |
| Baseline (Neo4j) | 28.57% | 0.00% | 0.0000 |
| Vector-DB (FAISS) | 28.57% | 0.00% | 0.0000 |

#### System Efficiency

| Pipeline | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| GraphRAG | 9.562s | 611.580s | 0.0016 |
| Baseline (Neo4j) | 0.045s | 191.721s | n/a |
| Vector-DB (FAISS) | 2.839s | 172.142s | n/a |

### PDF+Audio (n=12)

#### Primary Metrics (paraphrase-robust)

| Pipeline | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| GraphRAG | 66.67% | 33.33% | 41.67% | 58.33% | 54.17% | 52.78% | 58.33% | 85.79% |
| Baseline (Neo4j) | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |
| Vector-DB (FAISS) | 25.00% | 25.00% | 8.33% | 8.33% | 8.33% | 8.33% | 8.33% | 85.30% |

#### Secondary Metrics (token overlap)

| Pipeline | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| GraphRAG | 0.2254 | 0.2783 | 0.1091 | 0.1943 | 0.0311 |
| Baseline (Neo4j) | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |
| Vector-DB (FAISS) | 0.1953 | 0.2506 | 0.1096 | 0.1909 | 0.0477 |

#### Retrieval Quality

| Pipeline | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 57.58% | 43.21% | 76.52% | 0.7116 | 0.5998 |
| Baseline (Neo4j) | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |
| Vector-DB (FAISS) | 83.33% | 45.83% | 83.33% | 35.42% | 0.2546 | 0.0458 | 100.00% | 50.00% | 14.29% | 49.31% | 73.81% | n/a | n/a |

#### Answer Behavior

| Pipeline | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| GraphRAG | 0.00% | n/a | 0.3333 |
| Baseline (Neo4j) | 33.33% | n/a | 0.0000 |
| Vector-DB (FAISS) | 25.00% | n/a | 0.1667 |

#### System Efficiency

| Pipeline | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| GraphRAG | 0.151s | 166.631s | 0.0060 |
| Baseline (Neo4j) | 0.103s | 104.496s | n/a |
| Vector-DB (FAISS) | 0.057s | 192.508s | n/a |

### Combined (ALL) (n=40)

#### Primary Metrics (paraphrase-robust)

| Pipeline | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| GraphRAG | 60.00% | 37.50% | 53.33% | 63.61% | 63.00% | 66.11% | 60.00% | 85.62% |
| Baseline (Neo4j) | 30.00% | 17.50% | 23.33% | 32.50% | 33.56% | 36.67% | 23.33% | 85.09% |
| Vector-DB (FAISS) | 17.50% | 17.50% | 10.00% | 16.67% | 17.22% | 18.33% | 10.00% | 85.56% |

#### Secondary Metrics (token overlap)

| Pipeline | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| GraphRAG | 0.2448 | 0.2848 | 0.1495 | 0.2143 | 0.0448 |
| Baseline (Neo4j) | 0.1854 | 0.2318 | 0.0901 | 0.1754 | 0.0236 |
| Vector-DB (FAISS) | 0.2013 | 0.2188 | 0.0878 | 0.1660 | 0.0269 |

#### Retrieval Quality

| Pipeline | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GraphRAG | 72.50% | 63.75% | 87.50% | 66.21% | 0.2060 | 0.0883 | 100.00% | 94.17% | 70.83% | 42.59% | 72.62% | 0.7211 | 0.5756 |
| Baseline (Neo4j) | 72.50% | 63.75% | 87.50% | 66.21% | 0.2060 | 0.0883 | 100.00% | 93.06% | 57.89% | 36.55% | 68.86% | 0.7211 | 0.5756 |
| Vector-DB (FAISS) | 57.50% | 45.00% | 75.00% | 51.04% | 0.1125 | 0.0329 | 100.00% | 68.33% | 28.95% | 28.68% | 68.70% | n/a | n/a |

#### Answer Behavior

| Pipeline | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| GraphRAG | 2.50% | 66.67% | 0.1000 |
| Baseline (Neo4j) | 22.50% | 33.33% | 0.0000 |
| Vector-DB (FAISS) | 22.50% | 0.00% | 0.1000 |

#### System Efficiency

| Pipeline | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| GraphRAG | 3.841s | 323.350s | 0.0031 |
| Baseline (Neo4j) | 0.085s | 144.557s | n/a |
| Vector-DB (FAISS) | 1.031s | 242.473s | n/a |

## Ablation Ladder (EXP1 -> EXP7)

Experiments as rows, metrics as columns - read down a column to see how each metric changes as each component gets added.

### Experiment key

| Experiment | Description |
|---|---|
| EXP1 Dense | EXP1: Dense embeddings only |
| EXP2 +BM25 | EXP2: + BM25 (lexical/keyword retrieval) |
| EXP3 +FastRP | EXP3: + FastRP Graph Expansion (graph-based candidate expansion) |
| EXP4 +GraphFacts | EXP4: + Graph Facts (structured graph-derived evidence to the LLM) |
| EXP5 +Temporal | EXP5: + Temporal Graph (NEXT_VALUE/PREVIOUS_VALUE/UPDATED_TO facts, forced on for every question) |
| EXP6 +Routing | EXP6: + Question-Type Routing (real per-question pdf/audio/both + temporal-comparison detection) |
| EXP7 +HyDE (final) | EXP7 (FINAL/production config): + HyDE for audio-routed retrieval |

### PDF (n=14)

#### Primary Metrics (paraphrase-robust) — GraphRAG

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 64.29% | 57.14% | 57.14% | 70.83% | 71.90% | 76.19% | 64.29% | 88.31% |
| EXP2 +BM25 | 42.86% | 50.00% | 42.86% | 48.21% | 49.29% | 53.57% | 50.00% | 88.52% |
| EXP3 +FastRP | 71.43% | 78.57% | 71.43% | 77.38% | 78.57% | 82.14% | 71.43% | 88.88% |
| EXP4 +GraphFacts | 85.71% | 85.71% | 78.57% | 86.31% | 88.57% | 96.43% | 78.57% | 87.88% |
| EXP5 +Temporal | 85.71% | 78.57% | 71.43% | 79.17% | 81.43% | 89.29% | 71.43% | 87.49% |
| EXP6 +Routing | 78.57% | 64.29% | 78.57% | 86.31% | 88.57% | 96.43% | 78.57% | 88.36% |
| EXP7 +HyDE (final) | 78.57% | 64.29% | 78.57% | 86.31% | 88.57% | 96.43% | 78.57% | 88.36% |

#### Primary Metrics (paraphrase-robust) — Baseline

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 64.29% | 42.86% | 42.86% | 68.45% | 71.90% | 82.14% | 42.86% | 87.37% |
| EXP2 +BM25 | 57.14% | 50.00% | 42.86% | 68.45% | 71.90% | 82.14% | 42.86% | 87.30% |
| EXP3 +FastRP | 50.00% | 42.86% | 42.86% | 55.36% | 56.43% | 60.71% | 42.86% | 86.78% |
| EXP4 +GraphFacts | 50.00% | 42.86% | 42.86% | 55.36% | 56.43% | 60.71% | 42.86% | 86.78% |
| EXP5 +Temporal | 50.00% | 42.86% | 42.86% | 55.36% | 56.43% | 60.71% | 42.86% | 86.78% |
| EXP6 +Routing | 50.00% | 42.86% | 42.86% | 58.93% | 60.00% | 64.29% | 42.86% | 86.63% |
| EXP7 +HyDE (final) | 50.00% | 42.86% | 42.86% | 58.93% | 60.00% | 64.29% | 42.86% | 86.63% |

#### Secondary Metrics (token overlap) — GraphRAG

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.3897 | 0.4292 | 0.3229 | 0.3806 | 0.1556 |
| EXP2 +BM25 | 0.3460 | 0.4405 | 0.3232 | 0.3838 | 0.1296 |
| EXP3 +FastRP | 0.4317 | 0.4788 | 0.3931 | 0.4595 | 0.2003 |
| EXP4 +GraphFacts | 0.3381 | 0.4117 | 0.2689 | 0.3815 | 0.0804 |
| EXP5 +Temporal | 0.3234 | 0.3953 | 0.2641 | 0.3651 | 0.0782 |
| EXP6 +Routing | 0.3597 | 0.4652 | 0.3212 | 0.3762 | 0.0982 |
| EXP7 +HyDE (final) | 0.3597 | 0.4652 | 0.3212 | 0.3762 | 0.0982 |

#### Secondary Metrics (token overlap) — Baseline

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.3374 | 0.3535 | 0.2341 | 0.3076 | 0.0527 |
| EXP2 +BM25 | 0.3234 | 0.3260 | 0.1917 | 0.2697 | 0.0391 |
| EXP3 +FastRP | 0.3217 | 0.3017 | 0.1889 | 0.2573 | 0.0447 |
| EXP4 +GraphFacts | 0.3217 | 0.3017 | 0.1889 | 0.2573 | 0.0447 |
| EXP5 +Temporal | 0.3217 | 0.3017 | 0.1889 | 0.2573 | 0.0447 |
| EXP6 +Routing | 0.3022 | 0.3143 | 0.1754 | 0.2571 | 0.0420 |
| EXP7 +HyDE (final) | 0.3022 | 0.3143 | 0.1754 | 0.2571 | 0.0420 |

#### Retrieval Quality — GraphRAG

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 71.43% | 71.43% | 100.00% | 80.48% | 0.2845 | 0.1122 | 100.00% | 95.83% | 82.05% | 71.79% | 78.13% | n/a | n/a |
| EXP2 +BM25 | 78.57% | 75.00% | 100.00% | 81.67% | 0.2566 | 0.1122 | 100.00% | 95.83% | 57.69% | 69.23% | 78.59% | n/a | 0.5152 |
| EXP3 +FastRP | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 88.69% | 95.83% | 91.67% | 79.40% | 0.7144 | 0.6196 |
| EXP4 +GraphFacts | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 98.21% | 96.43% | 75.00% | 80.88% | 0.7144 | 0.6196 |
| EXP5 +Temporal | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 91.07% | 96.15% | 88.46% | 78.95% | 0.7144 | 0.6196 |
| EXP6 +Routing | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 98.21% | 96.43% | 75.00% | 80.08% | 0.7815 | 0.6499 |
| EXP7 +HyDE (final) | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 98.21% | 96.43% | 75.00% | 80.08% | 0.7815 | 0.6499 |

#### Retrieval Quality — Baseline

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 71.43% | 71.43% | 100.00% | 80.48% | 0.2845 | 0.1122 | 100.00% | 95.83% | 82.14% | 26.67% | 75.78% | n/a | n/a |
| EXP2 +BM25 | 78.57% | 75.00% | 100.00% | 81.67% | 0.2566 | 0.1122 | 100.00% | 95.83% | 82.14% | 18.33% | 77.90% | n/a | 0.5152 |
| EXP3 +FastRP | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 88.69% | 65.38% | 19.44% | 77.74% | 0.7144 | 0.6196 |
| EXP4 +GraphFacts | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 88.69% | 65.38% | 19.44% | 77.74% | 0.7144 | 0.6196 |
| EXP5 +Temporal | 50.00% | 46.43% | 85.71% | 60.00% | 0.1793 | 0.0714 | 100.00% | 88.69% | 65.38% | 19.44% | 77.74% | 0.7144 | 0.6196 |
| EXP6 +Routing | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 95.83% | 69.23% | 21.21% | 76.46% | 0.7815 | 0.6499 |
| EXP7 +HyDE (final) | 57.14% | 57.14% | 85.71% | 65.95% | 0.2721 | 0.0918 | 100.00% | 95.83% | 69.23% | 21.21% | 76.46% | 0.7815 | 0.6499 |

#### Answer Behavior — GraphRAG

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 7.14% | 100.00% | 0.0000 |
| EXP2 +BM25 | 7.14% | 100.00% | 0.0714 |
| EXP3 +FastRP | 14.29% | 50.00% | 0.0000 |
| EXP4 +GraphFacts | 0.00% | 100.00% | 0.0000 |
| EXP5 +Temporal | 7.14% | 100.00% | 0.0000 |
| EXP6 +Routing | 0.00% | 100.00% | 0.0000 |
| EXP7 +HyDE (final) | 0.00% | 100.00% | 0.0000 |

#### Answer Behavior — Baseline

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 14.29% | 100.00% | 0.0000 |
| EXP2 +BM25 | 14.29% | 100.00% | 0.0000 |
| EXP3 +FastRP | 28.57% | 50.00% | 0.0000 |
| EXP4 +GraphFacts | 28.57% | 50.00% | 0.0000 |
| EXP5 +Temporal | 28.57% | 50.00% | 0.0000 |
| EXP6 +Routing | 7.14% | 50.00% | 0.0000 |
| EXP7 +HyDE (final) | 7.14% | 50.00% | 0.0000 |

#### System Efficiency — GraphRAG

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.158s | 164.245s | 0.0061 |
| EXP2 +BM25 | 0.175s | 165.959s | 0.0060 |
| EXP3 +FastRP | 0.224s | 129.065s | 0.0077 |
| EXP4 +GraphFacts | 0.266s | 143.025s | 0.0070 |
| EXP5 +Temporal | 0.217s | 146.819s | 0.0068 |
| EXP6 +Routing | 0.170s | 162.206s | 0.0062 |
| EXP7 +HyDE (final) | 1.282s | 169.450s | 0.0059 |

#### System Efficiency — Baseline

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.098s | 149.021s | n/a |
| EXP2 +BM25 | 0.124s | 150.898s | n/a |
| EXP3 +FastRP | 0.169s | 119.042s | n/a |
| EXP4 +GraphFacts | 0.192s | 118.099s | n/a |
| EXP5 +Temporal | 0.155s | 117.316s | n/a |
| EXP6 +Routing | 0.110s | 132.084s | n/a |
| EXP7 +HyDE (final) | 0.111s | 131.731s | n/a |

### Audio (n=14)

#### Primary Metrics (paraphrase-robust) — GraphRAG

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 7.14% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.04% |
| EXP2 +BM25 | 35.71% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.42% |
| EXP3 +FastRP | 21.43% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 82.99% |
| EXP4 +GraphFacts | 14.29% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.54% |
| EXP5 +Temporal | 14.29% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.50% |
| EXP6 +Routing | 28.57% | 28.57% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 82.96% |
| EXP7 +HyDE (final) | 35.71% | 14.29% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 82.72% |

#### Primary Metrics (paraphrase-robust) — Baseline

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 7.14% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.09% |
| EXP2 +BM25 | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.97% |
| EXP3 +FastRP | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.72% |
| EXP4 +GraphFacts | 14.29% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.59% |
| EXP5 +Temporal | 14.29% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 83.52% |
| EXP6 +Routing | 0.00% | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.16% |
| EXP7 +HyDE (final) | 7.14% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 84.17% |

#### Secondary Metrics (token overlap) — GraphRAG

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.1085 | 0.1371 | 0.0158 | 0.0973 | 0.0061 |
| EXP2 +BM25 | 0.0802 | 0.1168 | 0.0111 | 0.0938 | 0.0058 |
| EXP3 +FastRP | 0.0731 | 0.1077 | 0.0114 | 0.0910 | 0.0064 |
| EXP4 +GraphFacts | 0.1084 | 0.1358 | 0.0193 | 0.1131 | 0.0091 |
| EXP5 +Temporal | 0.0793 | 0.1110 | 0.0145 | 0.0915 | 0.0058 |
| EXP6 +Routing | 0.1531 | 0.1171 | 0.0205 | 0.0788 | 0.0056 |
| EXP7 +HyDE (final) | 0.1464 | 0.1101 | 0.0124 | 0.0695 | 0.0031 |

#### Secondary Metrics (token overlap) — Baseline

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.0811 | 0.1529 | 0.0240 | 0.1192 | 0.0082 |
| EXP2 +BM25 | 0.0745 | 0.1410 | 0.0124 | 0.1094 | 0.0076 |
| EXP3 +FastRP | 0.0735 | 0.1331 | 0.0124 | 0.0985 | 0.0088 |
| EXP4 +GraphFacts | 0.0734 | 0.1326 | 0.0124 | 0.0952 | 0.0086 |
| EXP5 +Temporal | 0.0818 | 0.1374 | 0.0148 | 0.0997 | 0.0091 |
| EXP6 +Routing | 0.0905 | 0.1509 | 0.0105 | 0.1039 | 0.0078 |
| EXP7 +HyDE (final) | 0.0987 | 0.1592 | 0.0198 | 0.1122 | 0.0093 |

#### Retrieval Quality — GraphRAG

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 35.71% | 32.14% | 35.71% | 32.14% | 0.0893 | 0.0510 | 100.00% | 70.83% | 0.00% | 5.62% | 58.19% | n/a | n/a |
| EXP2 +BM25 | 35.71% | 32.14% | 35.71% | 32.14% | 0.0883 | 0.0510 | 100.00% | 70.83% | n/a | 20.83% | 54.45% | n/a | 0.6032 |
| EXP3 +FastRP | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | n/a | 25.00% | 54.33% | 0.7562 | 0.6422 |
| EXP4 +GraphFacts | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | 0.00% | 18.75% | 58.70% | 0.7562 | 0.6422 |
| EXP5 +Temporal | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | 0.00% | 7.14% | 55.39% | 0.7562 | 0.6422 |
| EXP6 +Routing | 85.71% | 85.71% | 85.71% | 85.71% | 0.1677 | 0.0481 | 100.00% | 87.50% | 0.00% | 6.73% | 62.82% | 0.6465 | 0.4843 |
| EXP7 +HyDE (final) | 85.71% | 85.71% | 85.71% | 85.71% | 0.1466 | 0.0481 | 100.00% | 87.50% | 0.00% | 4.17% | 61.82% | 0.6687 | 0.4807 |

#### Retrieval Quality — Baseline

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 35.71% | 32.14% | 35.71% | 32.14% | 0.0893 | 0.0510 | 100.00% | 70.83% | n/a | 20.00% | 57.24% | n/a | n/a |
| EXP2 +BM25 | 35.71% | 32.14% | 35.71% | 32.14% | 0.0883 | 0.0510 | 100.00% | 70.83% | n/a | 40.00% | 58.89% | n/a | 0.6032 |
| EXP3 +FastRP | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | n/a | 30.00% | 56.06% | 0.7562 | 0.6422 |
| EXP4 +GraphFacts | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | n/a | 30.00% | 55.88% | 0.7562 | 0.6422 |
| EXP5 +Temporal | 35.71% | 32.14% | 35.71% | 32.14% | 0.1281 | 0.0510 | 100.00% | 70.83% | n/a | 30.00% | 56.07% | 0.7562 | 0.6422 |
| EXP6 +Routing | 85.71% | 85.71% | 85.71% | 85.71% | 0.1677 | 0.0481 | 100.00% | 87.50% | n/a | 28.57% | 57.30% | 0.6465 | 0.4843 |
| EXP7 +HyDE (final) | 85.71% | 85.71% | 85.71% | 85.71% | 0.1466 | 0.0481 | 100.00% | 87.50% | n/a | 41.67% | 57.57% | 0.6687 | 0.4807 |

#### Answer Behavior — GraphRAG

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 42.86% | 0.00% | 0.0000 |
| EXP2 +BM25 | 57.14% | 0.00% | 0.0000 |
| EXP3 +FastRP | 42.86% | 0.00% | 0.0000 |
| EXP4 +GraphFacts | 21.43% | 0.00% | 0.0000 |
| EXP5 +Temporal | 28.57% | 0.00% | 0.0000 |
| EXP6 +Routing | 0.00% | 0.00% | 0.0000 |
| EXP7 +HyDE (final) | 7.14% | 0.00% | 0.0000 |

#### Answer Behavior — Baseline

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 35.71% | 0.00% | 0.0000 |
| EXP2 +BM25 | 28.57% | 0.00% | 0.0000 |
| EXP3 +FastRP | 42.86% | 0.00% | 0.0000 |
| EXP4 +GraphFacts | 42.86% | 0.00% | 0.0000 |
| EXP5 +Temporal | 42.86% | 0.00% | 0.0000 |
| EXP6 +Routing | 14.29% | 0.00% | 0.0000 |
| EXP7 +HyDE (final) | 28.57% | 0.00% | 0.0000 |

#### System Efficiency — GraphRAG

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 1.104s | 134.384s | 0.0074 |
| EXP2 +BM25 | 1.181s | 137.629s | 0.0073 |
| EXP3 +FastRP | 2.468s | 155.408s | 0.0064 |
| EXP4 +GraphFacts | 1.366s | 140.246s | 0.0071 |
| EXP5 +Temporal | 1.430s | 136.336s | 0.0073 |
| EXP6 +Routing | 1.140s | 322.604s | 0.0031 |
| EXP7 +HyDE (final) | 9.562s | 611.580s | 0.0016 |

#### System Efficiency — Baseline

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.028s | 74.394s | n/a |
| EXP2 +BM25 | 0.048s | 74.904s | n/a |
| EXP3 +FastRP | 0.061s | 73.634s | n/a |
| EXP4 +GraphFacts | 0.064s | 78.365s | n/a |
| EXP5 +Temporal | 0.053s | 76.411s | n/a |
| EXP6 +Routing | 0.044s | 189.742s | n/a |
| EXP7 +HyDE (final) | 0.045s | 191.721s | n/a |

### PDF+Audio (n=12)

#### Primary Metrics (paraphrase-robust) — GraphRAG

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 33.33% | 16.67% | 25.00% | 33.33% | 31.67% | 30.56% | 33.33% | 83.50% |
| EXP2 +BM25 | 41.67% | 25.00% | 41.67% | 41.67% | 41.67% | 41.67% | 41.67% | 84.82% |
| EXP3 +FastRP | 50.00% | 16.67% | 25.00% | 33.33% | 30.56% | 29.17% | 33.33% | 82.78% |
| EXP4 +GraphFacts | 66.67% | 33.33% | 41.67% | 58.33% | 54.17% | 52.78% | 58.33% | 85.64% |
| EXP5 +Temporal | 66.67% | 33.33% | 41.67% | 58.33% | 54.17% | 52.78% | 58.33% | 85.12% |
| EXP6 +Routing | 66.67% | 33.33% | 41.67% | 58.33% | 54.17% | 52.78% | 58.33% | 85.64% |
| EXP7 +HyDE (final) | 66.67% | 33.33% | 41.67% | 58.33% | 54.17% | 52.78% | 58.33% | 85.79% |

#### Primary Metrics (paraphrase-robust) — Baseline

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 50.00% | 16.67% | 33.33% | 37.50% | 38.89% | 41.67% | 33.33% | 85.07% |
| EXP2 +BM25 | 66.67% | 25.00% | 25.00% | 33.33% | 34.72% | 37.50% | 25.00% | 84.90% |
| EXP3 +FastRP | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |
| EXP4 +GraphFacts | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |
| EXP5 +Temporal | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |
| EXP6 +Routing | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |
| EXP7 +HyDE (final) | 33.33% | 8.33% | 8.33% | 12.50% | 13.89% | 16.67% | 8.33% | 84.35% |

#### Secondary Metrics (token overlap) — GraphRAG

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.1405 | 0.1773 | 0.0552 | 0.1327 | 0.0138 |
| EXP2 +BM25 | 0.1516 | 0.2268 | 0.0923 | 0.1640 | 0.0195 |
| EXP3 +FastRP | 0.1268 | 0.1556 | 0.0626 | 0.1165 | 0.0213 |
| EXP4 +GraphFacts | 0.2289 | 0.2767 | 0.1067 | 0.1961 | 0.0274 |
| EXP5 +Temporal | 0.2082 | 0.2532 | 0.0887 | 0.1748 | 0.0182 |
| EXP6 +Routing | 0.2289 | 0.2767 | 0.1067 | 0.1961 | 0.0274 |
| EXP7 +HyDE (final) | 0.2254 | 0.2783 | 0.1091 | 0.1943 | 0.0311 |

#### Secondary Metrics (token overlap) — Baseline

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.1347 | 0.2618 | 0.1175 | 0.1789 | 0.0213 |
| EXP2 +BM25 | 0.1014 | 0.2226 | 0.1079 | 0.1543 | 0.0167 |
| EXP3 +FastRP | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |
| EXP4 +GraphFacts | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |
| EXP5 +Temporal | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |
| EXP6 +Routing | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |
| EXP7 +HyDE (final) | 0.1502 | 0.2201 | 0.0727 | 0.1540 | 0.0188 |

#### Retrieval Quality — GraphRAG

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 91.67% | 62.50% | 91.67% | 54.17% | 0.2708 | 0.1786 | 100.00% | 95.83% | 52.38% | 44.76% | 64.55% | n/a | n/a |
| EXP2 +BM25 | 91.67% | 58.33% | 91.67% | 52.08% | 0.2375 | 0.1667 | 100.00% | 95.83% | 62.50% | 48.15% | 69.29% | n/a | 0.5061 |
| EXP3 +FastRP | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 58.33% | 41.67% | 61.60% | 0.7116 | 0.5998 |
| EXP4 +GraphFacts | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 57.58% | 44.25% | 75.46% | 0.7116 | 0.5998 |
| EXP5 +Temporal | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 63.33% | 39.18% | 73.25% | 0.7116 | 0.5998 |
| EXP6 +Routing | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 57.58% | 44.25% | 75.46% | 0.7116 | 0.5998 |
| EXP7 +HyDE (final) | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 57.58% | 43.21% | 76.52% | 0.7116 | 0.5998 |

#### Retrieval Quality — Baseline

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 91.67% | 62.50% | 91.67% | 54.17% | 0.2708 | 0.1786 | 100.00% | 95.83% | 55.56% | 31.83% | 73.26% | n/a | n/a |
| EXP2 +BM25 | 91.67% | 58.33% | 91.67% | 52.08% | 0.2375 | 0.1667 | 100.00% | 95.83% | 50.00% | 37.50% | 72.55% | n/a | 0.5061 |
| EXP3 +FastRP | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |
| EXP4 +GraphFacts | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |
| EXP5 +Temporal | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |
| EXP6 +Routing | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |
| EXP7 +HyDE (final) | 75.00% | 45.83% | 91.67% | 43.75% | 0.1980 | 0.1310 | 100.00% | 91.67% | 33.33% | 48.06% | 73.17% | 0.7116 | 0.5998 |

#### Answer Behavior — GraphRAG

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 41.67% | n/a | 0.0000 |
| EXP2 +BM25 | 33.33% | n/a | 0.1667 |
| EXP3 +FastRP | 50.00% | n/a | 0.0000 |
| EXP4 +GraphFacts | 0.00% | n/a | 0.3333 |
| EXP5 +Temporal | 8.33% | n/a | 0.3333 |
| EXP6 +Routing | 0.00% | n/a | 0.3333 |
| EXP7 +HyDE (final) | 0.00% | n/a | 0.3333 |

#### Answer Behavior — Baseline

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 16.67% | n/a | 0.0833 |
| EXP2 +BM25 | 16.67% | n/a | 0.0833 |
| EXP3 +FastRP | 33.33% | n/a | 0.0000 |
| EXP4 +GraphFacts | 33.33% | n/a | 0.0000 |
| EXP5 +Temporal | 33.33% | n/a | 0.0000 |
| EXP6 +Routing | 33.33% | n/a | 0.0000 |
| EXP7 +HyDE (final) | 33.33% | n/a | 0.0000 |

#### System Efficiency — GraphRAG

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.107s | 173.887s | 0.0058 |
| EXP2 +BM25 | 1.646s | 177.382s | 0.0056 |
| EXP3 +FastRP | 0.181s | 138.944s | 0.0072 |
| EXP4 +GraphFacts | 0.526s | 497.123s | 0.0020 |
| EXP5 +Temporal | 0.187s | 171.414s | 0.0058 |
| EXP6 +Routing | 0.162s | 157.771s | 0.0063 |
| EXP7 +HyDE (final) | 0.151s | 166.631s | 0.0060 |

#### System Efficiency — Baseline

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.067s | 148.361s | n/a |
| EXP2 +BM25 | 0.098s | 142.162s | n/a |
| EXP3 +FastRP | 0.127s | 109.427s | n/a |
| EXP4 +GraphFacts | 0.132s | 107.531s | n/a |
| EXP5 +Temporal | 0.142s | 107.360s | n/a |
| EXP6 +Routing | 0.101s | 104.172s | n/a |
| EXP7 +HyDE (final) | 0.103s | 104.496s | n/a |

### Combined (ALL) (n=40)

#### Primary Metrics (paraphrase-robust) — GraphRAG

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 35.00% | 27.50% | 36.67% | 46.39% | 46.22% | 47.78% | 43.33% | 85.37% |
| EXP2 +BM25 | 40.00% | 27.50% | 36.67% | 39.17% | 39.67% | 41.67% | 40.00% | 85.63% |
| EXP3 +FastRP | 47.50% | 32.50% | 43.33% | 49.44% | 48.89% | 50.00% | 46.67% | 84.99% |
| EXP4 +GraphFacts | 55.00% | 40.00% | 53.33% | 63.61% | 63.00% | 66.11% | 60.00% | 85.69% |
| EXP5 +Temporal | 55.00% | 37.50% | 50.00% | 60.28% | 59.67% | 62.78% | 56.67% | 85.38% |
| EXP6 +Routing | 57.50% | 42.50% | 53.33% | 63.61% | 63.00% | 66.11% | 60.00% | 85.66% |
| EXP7 +HyDE (final) | 60.00% | 37.50% | 53.33% | 63.61% | 63.00% | 66.11% | 60.00% | 85.62% |

#### Primary Metrics (paraphrase-robust) — Baseline

| Experiment | LLM Judge Correct | LLM Judge Faithful | Numeric Exact | Numeric Match (partial) | Numeric F1 | Numeric Precision | Tolerance Match (1%) | BERTScore F1 |
|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 40.00% | 22.50% | 33.33% | 46.94% | 49.11% | 55.00% | 33.33% | 85.53% |
| EXP2 +BM25 | 42.50% | 25.00% | 30.00% | 45.28% | 47.44% | 53.33% | 30.00% | 85.41% |
| EXP3 +FastRP | 30.00% | 17.50% | 23.33% | 30.83% | 31.89% | 35.00% | 23.33% | 84.98% |
| EXP4 +GraphFacts | 32.50% | 20.00% | 23.33% | 30.83% | 31.89% | 35.00% | 23.33% | 84.93% |
| EXP5 +Temporal | 32.50% | 20.00% | 23.33% | 30.83% | 31.89% | 35.00% | 23.33% | 84.91% |
| EXP6 +Routing | 27.50% | 20.00% | 23.33% | 32.50% | 33.56% | 36.67% | 23.33% | 85.08% |
| EXP7 +HyDE (final) | 30.00% | 17.50% | 23.33% | 32.50% | 33.56% | 36.67% | 23.33% | 85.09% |

#### Secondary Metrics (token overlap) — GraphRAG

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.2165 | 0.2514 | 0.1351 | 0.2071 | 0.0607 |
| EXP2 +BM25 | 0.1947 | 0.2631 | 0.1447 | 0.2163 | 0.0532 |
| EXP3 +FastRP | 0.2147 | 0.2519 | 0.1604 | 0.2276 | 0.0787 |
| EXP4 +GraphFacts | 0.2249 | 0.2746 | 0.1329 | 0.2319 | 0.0395 |
| EXP5 +Temporal | 0.2034 | 0.2532 | 0.1241 | 0.2123 | 0.0349 |
| EXP6 +Routing | 0.2481 | 0.2868 | 0.1516 | 0.2181 | 0.0446 |
| EXP7 +HyDE (final) | 0.2448 | 0.2848 | 0.1495 | 0.2143 | 0.0448 |

#### Secondary Metrics (token overlap) — Baseline

| Experiment | METEOR | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|---|
| EXP1 Dense | 0.1869 | 0.2558 | 0.1256 | 0.2031 | 0.0277 |
| EXP2 +BM25 | 0.1697 | 0.2302 | 0.1038 | 0.1790 | 0.0214 |
| EXP3 +FastRP | 0.1834 | 0.2182 | 0.0923 | 0.1707 | 0.0244 |
| EXP4 +GraphFacts | 0.1833 | 0.2180 | 0.0923 | 0.1696 | 0.0243 |
| EXP5 +Temporal | 0.1863 | 0.2197 | 0.0931 | 0.1711 | 0.0245 |
| EXP6 +Routing | 0.1825 | 0.2289 | 0.0869 | 0.1725 | 0.0230 |
| EXP7 +HyDE (final) | 0.1854 | 0.2318 | 0.0901 | 0.1754 | 0.0236 |

#### Retrieval Quality — GraphRAG

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 65.00% | 55.00% | 75.00% | 55.67% | 0.2121 | 0.1107 | 100.00% | 92.50% | 68.25% | 46.13% | 67.08% | n/a | n/a |
| EXP2 +BM25 | 67.50% | 55.00% | 75.00% | 55.46% | 0.1920 | 0.1071 | 100.00% | 92.50% | 59.52% | 52.08% | 67.35% | n/a | 0.5432 |
| EXP3 +FastRP | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 87.50% | 83.33% | 64.13% | 65.29% | 0.7282 | 0.6216 |
| EXP4 +GraphFacts | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 91.94% | 73.46% | 50.91% | 71.49% | 0.7282 | 0.6216 |
| EXP5 +Temporal | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 88.61% | 75.33% | 52.61% | 68.99% | 0.7282 | 0.6216 |
| EXP6 +Routing | 72.50% | 63.75% | 87.50% | 66.21% | 0.2133 | 0.0883 | 100.00% | 94.17% | 76.28% | 42.78% | 72.65% | 0.7133 | 0.5769 |
| EXP7 +HyDE (final) | 72.50% | 63.75% | 87.50% | 66.21% | 0.2060 | 0.0883 | 100.00% | 94.17% | 70.83% | 42.59% | 72.62% | 0.7211 | 0.5756 |

#### Retrieval Quality — Baseline

| Experiment | Hit@K | Recall@K | Source Hit@K | Source Recall@K | MRR | Context Precision | Source Type Acc | Evidence Sufficiency | Factual Consistency | Fin. Term Precision | Avg Cosine Sim | Avg FastRP Score | Avg BM25 Score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP1 Dense | 65.00% | 55.00% | 75.00% | 55.67% | 0.2121 | 0.1107 | 100.00% | 92.50% | 71.74% | 27.40% | 68.54% | n/a | n/a |
| EXP2 +BM25 | 67.50% | 55.00% | 75.00% | 55.46% | 0.1920 | 0.1071 | 100.00% | 92.50% | 69.57% | 29.44% | 69.64% | n/a | 0.5432 |
| EXP3 +FastRP | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 87.50% | 55.26% | 33.10% | 68.78% | 0.7282 | 0.6216 |
| EXP4 +GraphFacts | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 87.50% | 55.26% | 33.10% | 68.72% | 0.7282 | 0.6216 |
| EXP5 +Temporal | 52.50% | 41.25% | 70.00% | 45.38% | 0.1670 | 0.0821 | 100.00% | 87.50% | 55.26% | 33.10% | 68.78% | 0.7282 | 0.6216 |
| EXP6 +Routing | 72.50% | 63.75% | 87.50% | 66.21% | 0.2133 | 0.0883 | 100.00% | 93.06% | 57.89% | 33.67% | 68.77% | 0.7133 | 0.5769 |
| EXP7 +HyDE (final) | 72.50% | 63.75% | 87.50% | 66.21% | 0.2060 | 0.0883 | 100.00% | 93.06% | 57.89% | 36.55% | 68.86% | 0.7211 | 0.5756 |

#### Answer Behavior — GraphRAG

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 30.00% | 66.67% | 0.0000 |
| EXP2 +BM25 | 32.50% | 66.67% | 0.0750 |
| EXP3 +FastRP | 35.00% | 33.33% | 0.0000 |
| EXP4 +GraphFacts | 7.50% | 66.67% | 0.1000 |
| EXP5 +Temporal | 15.00% | 66.67% | 0.1000 |
| EXP6 +Routing | 0.00% | 66.67% | 0.1000 |
| EXP7 +HyDE (final) | 2.50% | 66.67% | 0.1000 |

#### Answer Behavior — Baseline

| Experiment | Abstain Rate | Multi-value Complete | Hallucinated Numbers |
|---|---|---|---|
| EXP1 Dense | 22.50% | 66.67% | 0.0250 |
| EXP2 +BM25 | 20.00% | 66.67% | 0.0250 |
| EXP3 +FastRP | 35.00% | 33.33% | 0.0000 |
| EXP4 +GraphFacts | 35.00% | 33.33% | 0.0000 |
| EXP5 +Temporal | 35.00% | 33.33% | 0.0000 |
| EXP6 +Routing | 17.50% | 33.33% | 0.0000 |
| EXP7 +HyDE (final) | 22.50% | 33.33% | 0.0000 |

#### System Efficiency — GraphRAG

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.474s | 156.686s | 0.0064 |
| EXP2 +BM25 | 0.969s | 159.471s | 0.0063 |
| EXP3 +FastRP | 0.997s | 141.249s | 0.0071 |
| EXP4 +GraphFacts | 0.729s | 248.281s | 0.0040 |
| EXP5 +Temporal | 0.632s | 150.528s | 0.0066 |
| EXP6 +Routing | 0.507s | 217.014s | 0.0046 |
| EXP7 +HyDE (final) | 3.841s | 323.350s | 0.0031 |

#### System Efficiency — Baseline

| Experiment | Retrieval Latency | E2E Latency | Throughput |
|---|---|---|---|
| EXP1 Dense | 0.064s | 122.704s | n/a |
| EXP2 +BM25 | 0.090s | 121.679s | n/a |
| EXP3 +FastRP | 0.119s | 100.265s | n/a |
| EXP4 +GraphFacts | 0.129s | 101.022s | n/a |
| EXP5 +Temporal | 0.115s | 100.012s | n/a |
| EXP6 +Routing | 0.084s | 143.891s | n/a |
| EXP7 +HyDE (final) | 0.085s | 144.557s | n/a |
