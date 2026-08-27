# Financial GraphRAG

A knowledge-graph-based retrieval-augmented generation (RAG) system for answering
financial questions from quarterly filings (PDF) and earnings-call recordings (audio),
benchmarked against a plain-retrieval baseline through a 7-stage ablation study.

## What this is

Most RAG systems retrieve chunks of text and hand them to a language model. This
project builds a **knowledge graph** on top of that (Neo4j) — extracting structured
facts, entities, and temporal relationships between reporting periods — and measures,
stage by stage, exactly how much each additional capability actually contributes over
plain retrieval.

The dataset covers five Indian-listed companies' quarterly filings and earnings-call
transcripts, with a 40-question ground-truth set split across three modalities: PDF-only,
audio-only, and questions requiring both sources together.

## Results — the ablation ladder

Each experiment adds exactly one capability on top of the previous one, so its individual
contribution can be measured in isolation. All numbers are on the combined 40-question set.

| Experiment | What it adds | LLM Judge Correct | Numeric Exact Match |
|---|---|---|---|
| EXP1 — Dense | Dense embedding search only | 35.0% | 36.7% |
| EXP2 — +BM25 | Keyword/lexical search | 40.0% | 36.7% |
| EXP3 — +FastRP | Graph-based candidate expansion | 47.5% | 43.3% |
| EXP4 — +Graph Facts | Structured facts extracted from the graph | 55.0% | 53.3% |
| EXP5 — +Temporal Graph | Prior/next-period relationships (forced on) | 55.0% | 50.0% |
| EXP6 — +Question-Type Routing | Routes retrieval by question type | 57.5% | 53.3% |
| EXP7 — Final configuration | + HyDE for audio-routed retrieval | **60.0%** | **53.3%** |
| *Baseline (no graph)* | *same corpus, no graph facts* | *—* | *23.3%* |

**Key finding:** structured Graph Facts (EXP4) produced the single largest jump of any
stage — confirming the graph's structured evidence, not merely its use as a data store,
is what drives the advantage over plain retrieval. The final configuration roughly
doubles the baseline's numeric accuracy and LLM-judged correctness.

Full per-modality breakdowns and the complete metric set are in `FINAL_REPORT.md`.

## Project structure

### Core pipeline
| File | What it does |
|---|---|
| `graph_rag_pipeline.py` | The full GraphRAG answer pipeline — retrieval, graph facts, temporal reasoning, routing, HyDE |
| `baseline_pipeline.py` | Plain-retrieval baseline sharing the same corpus, no graph facts |
| `retrieval.py` | Hybrid dense + BM25 + FastRP retrieval, shared by both pipelines |
| `router.py` | Rule-based question-type classifier (PDF vs. audio vs. both, temporal comparison detection) |
| `evidence_cleaning.py` | Cleans and formats retrieved text/tables before it reaches the LLM |
| `llm_client.py` | Shared LLM call wrapper (Ollama/Groq) |
| `config.py` | All tunable settings for both pipelines |

### Data ingestion
| File | What it does |
|---|---|
| `graph_builder.py` | Builds the Neo4j knowledge graph from processed chunks |
| `pdf_processor.py` / `audio_processor.py` | Extract and chunk PDF/audio source documents |
| `chunker.py` | Shared chunking logic |
| `embeddings.py` | Computes chunk embeddings |
| `fast_rp.py` | Computes FastRP graph-structural embeddings |
| `table_metrics.py` | Extracts structured metrics from financial tables |
| `temporal_utils.py` | Builds prior/next-period relationships in the graph |
| `ground_truth_generator.py` | Generates the ground-truth question/answer set |

### Evaluation
| File | What it does |
|---|---|
| `benchmark.py` | Runs both pipelines against the ground-truth set, computes all metrics |
| `run_ablation.py` | Runs the full EXP1–EXP7 ablation ladder, resumable if interrupted |
| `judge_results.py` | LLM-as-judge scoring of already-completed runs |
| `generate_report.py` / `generate_report_markdown.py` | Formats ablation results into readable reports |
| `generate_final_report.py` | Combines everything into one final report |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Neo4j connection details and LLM
   provider settings (Ollama locally, or Groq for hosted inference).
3. Ensure Neo4j is running and accessible.

## Running it

**Build the knowledge graph** (one-time, or after adding new documents):
```
python graph_builder.py
```

**Run the full benchmark** (both pipelines, all metrics):
```
python benchmark.py
```

**Run the full ablation study** (all 7 experiments, resumable):
```
python run_ablation.py
```
Add `--list` to see all experiments first, or `--only <name>` to run just one.

**Generate the combined report** after an ablation run:
```
python generate_report.py
python generate_final_report.py
```

## Ground truth

`ground_truth.json` contains 40 question/answer pairs, each annotated with the exact
supporting document, page (or audio timestamp), and expected figures — used both for
scoring and for the retrieval-quality metrics (Hit@K, Source Hit@K).
