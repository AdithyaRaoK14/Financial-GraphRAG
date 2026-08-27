"""
config.py
=========
WHAT THIS FILE DOES:
Central place for every setting used across the project — folder paths,
which companies/years/quarters to process, which Ollama model to use,
and database credentials.

Nothing else in the project should hardcode a path, model name, or
password — everything imports from here. This is the #1 fix over the
reference project (which had the Neo4j password hardcoded directly in
4 different files).

Credentials are read from environment variables (or a `.env` file, if
you install python-dotenv) — never hardcoded. Create a `.env` file
next to this one (copy `.env.example`) with your real values.
"""

import os
from pathlib import Path

# Load a .env file if python-dotenv is installed (optional but recommended)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # fine — you can also just set environment variables manually

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Companies / years / quarters to process ─────────────────────────
# Matches your folder structure: data/raw/<Company>/<Year>/<Quarter>/
COMPANIES = [
    "Credit Access Grameen",
    "InfoEdge",
    "Jindal Stainless Limited",
    "NYKAA",
    "Tata Consumer Products",
]
YEARS = ["2024", "2025", "2026"]  # sir said: only last 2 years
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

# ── Neo4j (knowledge graph database) ─────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # NO fallback default on purpose
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "graphragv4")  # <-- your database name
if NEO4J_PASSWORD is None:
    raise RuntimeError(
        "NEO4J_PASSWORD is not set. Create a .env file (see .env.example) "
        "or set the environment variable before running anything."
    )

# ── Ollama (local LLM) ────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GRAPH_EXTRACTION_MODEL = os.getenv("GRAPH_EXTRACTION_MODEL", "qwen2.5:7b")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "qwen2.5:7b")
GROUND_TRUTH_MODEL = os.getenv("GROUND_TRUTH_MODEL", "qwen2.5:7b")
# Ollama defaults to a 2048-TOKEN context window regardless of what the
# model can actually handle, unless num_ctx is set explicitly on every
# call. ground_truth_generator.py alone can send up to MAX_CONTEXT=12000
# characters (~3-4k tokens) of PDF/table/audio text, which silently blows
# past that default — the model then either loses its instructions or has
# no room left to generate, producing empty/garbled JSON no matter how
# many times you retry. 8192 comfortably covers MAX_CONTEXT plus the
# prompt scaffolding and leaves room to generate; raise it further only
# if you also raise MAX_CONTEXT.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Ollama's own default num_predict (max tokens generated per call) is small
# unless set explicitly. Without this, every generate() call was relying on
# whatever Ollama's internal default happens to be, which is risky for
# ground_truth_generator.py's 10-QA-pair JSON responses. 2048 is generous
# headroom for that; raise it if you ask for many more pairs at once.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
# graph_builder.py's extraction call asks for entities + relationships +
# metrics + guidance + risks + sentiment all in one JSON response — often
# a bigger payload than ground_truth_generator's QA pairs, especially for
# table chunks with many line items. A dedicated, larger budget avoids
# tying its truncation risk to whatever OLLAMA_NUM_PREDICT gets set to for
# other calls (e.g. shorter answer-pipeline responses).
OLLAMA_EXTRACTION_NUM_PREDICT = int(os.getenv("OLLAMA_EXTRACTION_NUM_PREDICT", "3000"))
# P3 latency fix: baseline_pipeline.py and graph_rag_pipeline.py's answer
# calls were using the full shared OLLAMA_NUM_PREDICT=2048 budget with no
# override — for a financial Q&A answer that's wildly oversized (the
# single longest ground-truth answer across all 75 benchmark questions is
# ~200 tokens). Local Ollama generation time scales roughly linearly with
# tokens produced, so a verbose comparison-question answer that ran the
# full 2048 tokens could plausibly account for most of the observed
# 400+ second outlier latencies on their own — a slow model producing an
# unnecessarily long response, not slow retrieval or a slow Neo4j call.
# P4 update: originally set to 700 (matching Groq's MAX_RESPONSE_TOKENS),
# sized against the one verbose example available at the time (~164
# tokens used). A later run surfaced comparison questions where the
# model writes an extended, self-correcting "Analysis:" section (see the
# anti-self-correction prompt rule this pairs with) and got cut off
# mid-sentence before ever reaching its FINAL ANSWER line — confirmed
# directly: a Tata Consumer Products Q1-FY26-vs-Q4-FY25 comparison
# answer was truncated at exactly 700 tokens mid "Let's check..." with
# no final line at all, despite having already correctly computed both
# figures earlier in its own reasoning. Raised to 1000 to give a real
# multi-part comparison answer room to finish even with some preamble —
# the prompt rule against narrating self-correction is the primary fix
# for the underlying verbosity, this is a safety margin on top of it.
OLLAMA_ANSWER_NUM_PREDICT = int(os.getenv("OLLAMA_ANSWER_NUM_PREDICT", "1000"))
# HTTP-level timeout (seconds) on the extraction client (see
# GraphBuilder.__init__'s self._ollama_client) — forwarded straight to
# httpx.Client, so it actually aborts a stalled request rather than
# waiting forever. 240s was cutting off calls that were genuinely still
# generating and would have succeeded shortly after — observed in
# production: an attempt killed at 240.0s, followed by a same-chunk
# retry that completed in 135.8s, i.e. strictly less than the timeout
# it was retried after. Raising this doesn't fix a truly hung request
# (still gets killed eventually and retried, same as before) — it just
# stops penalizing calls that only needed a bit more time by forcing
# them to pay for a full second generation from scratch.
OLLAMA_EXTRACTION_TIMEOUT_SEC = int(os.getenv("OLLAMA_EXTRACTION_TIMEOUT_SEC", "300"))
# Later, when you switch to Groq for speed, just set these two:
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Whisper (audio transcription) ─────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv(
    "WHISPER_MODEL_SIZE", "large-v3"
)  # tiny/base/small/medium/large-v3
# faster-whisper quantization on GPU. int8_float16 fits large-v3 on small
# VRAM cards (e.g. 4GB GTX 1650); use "float16"/"float32" if you have more
# VRAM and want max fidelity, or "int8" if you still hit out-of-memory.
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")
# beam_size/best_of of 5 (faster-whisper's library default) is noticeably
# slower than 2 for a modest speed/quality tradeoff that's hard to hear on
# clean English earnings-call audio. Raise back to 5 if you want the extra
# accuracy margin and can afford the runtime.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "2"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "2"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
# False reduces repetition-loop hallucinations on long-form audio (a known
# Whisper failure mode) at a small cost to cross-segment context.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
)
# Chunk merging targets (word counts, not characters). TARGET is where
# merging tries to cut at a sentence boundary; MAX is a hard cap that cuts
# regardless of punctuation, so one long stretch without a clean sentence
# break can't produce a runaway multi-thousand-word chunk.
WHISPER_CHUNK_TARGET_WORDS = int(os.getenv("WHISPER_CHUNK_TARGET_WORDS", "350"))
WHISPER_CHUNK_MAX_WORDS = int(os.getenv("WHISPER_CHUNK_MAX_WORDS", "700"))

# Decoding robustness thresholds (these match faster-whisper's own library
# defaults — set explicitly here so they're visible/tunable rather than
# implicit). Segments failing these checks are treated as
# silence/hallucination and get lower-quality output flagged internally by
# faster-whisper's decoder.
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(
    os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
)
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.6"))

# ── Embeddings ─────────────────────────────────────────────────────────
# Switched from all-MiniLM-L6-v2 (384-dim) to BGE (768-dim for base, or
# 1024-dim for large — pick based on GPU headroom). BGE embeddings are
# NOT comparable to MiniLM ones (different dimensionality, different
# vector space) — after changing this, every embedded node's n.embedding
# is stale until you rerun `python embeddings.py`. embeddings.py only
# fills nodes where n.embedding IS NULL, so a bare rerun after switching
# models will NOT re-embed already-embedded nodes with the new model —
# clear the stale property first, e.g.:
#   MATCH (n) WHERE n.embedding IS NOT NULL REMOVE n.embedding, n.embedding_dim
# then rerun embeddings.py. retrieval.py's in-memory cache (_load_chunks)
# also needs a fresh process (or retrieval.clear_cache()) to pick up the
# new vectors instead of serving whatever it already loaded.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = 1200  # max characters per text chunk
CHUNK_OVERLAP = 200  # overlap between adjacent chunks
MAX_SECTION_LEN = 2000  # section split threshold before recursive splitting

# ── PDF page filtering ──────────────────────────────────────────────────
# Page types (see pdf_processor.classify_page) to drop entirely instead of
# indexing — auditor legalese and security-cover certificates add almost
# no value for financial Q&A. Comma-separated env var, e.g.
# SKIP_PAGE_TYPES=auditor_report,security_cover
SKIP_PAGE_TYPES = [
    t.strip()
    for t in os.getenv("SKIP_PAGE_TYPES", "auditor_report").split(",")
    if t.strip()
]

# ── Graph extraction ─────────────────────────────────────────────────
# Some table chunks (see chunker.py's table_chunk) end up with heavily
# redundant embedding_text — the same table serialized multiple ways
# (prose paragraph, raw dump, deduped columns, transposed key/value).
# Sending all of that to a 7B model tends to make it write a prose
# summary instead of following the "return only JSON" instruction, so
# graph_builder.py truncates extraction input to this many characters.
# Chunks are still stored in full in Neo4j/embeddings — this only caps
# what's shown to the extraction LLM call.
#
# Raised from 3000 to 5000: every LLM call carries a large fixed cost
# (60-250s observed locally) mostly independent of input size within
# this range, so a chunk just over the old cutoff was paying for a
# second full round trip (see graph_builder.py's _split_text/
# "Splitting N-char chunk into M pieces" log line) just to extract the
# last few lines. Fewer, slightly larger calls costs less wall-clock
# time overall than more, smaller ones, as long as each piece still
# fits comfortably under OLLAMA_NUM_CTX with room to generate. If you
# see the model reverting to prose instead of JSON on a chunk this
# size, that's the earlier-described redundant-serialization failure
# mode returning — lower this back down rather than raising
# OLLAMA_NUM_CTX first.
GRAPH_EXTRACTION_MAX_CHARS = int(os.getenv("GRAPH_EXTRACTION_MAX_CHARS", "5000"))

# Number of parallel chunk extraction workers.
# Set to 1 to disable parallel extraction.
GRAPH_BUILDER_WORKERS = 1

# Table chunks try table_metrics.py's deterministic (no-LLM) parse of
# their structured headers/rows before ever calling Qwen — see
# graph_builder.py's process_chunk_full(). Only falls back to the LLM
# (TABLE_PROMPT) when the deterministic parse finds nothing. Set to
# false to force every table chunk through the LLM instead, e.g. while
# debugging a suspected table_metrics.py parsing gap.
USE_DETERMINISTIC_TABLE_METRICS = (
    os.getenv("USE_DETERMINISTIC_TABLE_METRICS", "true").lower() == "true"
)

# Off by default: free-text entity-to-entity relationship extraction
# ("NVIDIA DEVELOPS Blackwell") is one of the least reliable things to
# ask a 7B model for, and the graph's real structure (Company/Quarter/
# Metric/Risk/Guidance) is built deterministically, not from this. It's
# a toggle rather than a removal — see graph_builder.py's TEXT_PROMPT/
# AUDIO_PROMPT properties — so it's one flag away if you want richer
# entity-to-entity connections later.
EXTRACT_RELATIONSHIPS = os.getenv("EXTRACT_RELATIONSHIPS", "false").lower() == "true"

# ── Retrieval ─────────────────────────────────────────────────────────
TOP_K = 5  # default: how many chunks to retrieve per question
# Raised from TOP_K (5) to 7. GraphRAG is meant to make up for retrieving
# fewer chunks than baseline with structured graph facts — but when the
# graph hasn't extracted a metric for the exact figure being asked about
# (extraction gaps happen), it falls back on raw chunk text same as
# baseline, and needs comparable headroom to actually find it. Observed
# case: a clean single-line disclosure ("Net profit after tax = 60.19")
# scored 0.60-0.63 — just outside the old top-5 cutoff (~0.67+).
# Override with the TOP_K_GRAPH env var if you want to tune further.
TOP_K_GRAPH = int(os.getenv("TOP_K_GRAPH", "7"))
# Equalized with TOP_K_GRAPH rather than set independently. Previously
# this was TOP_K (5), and baseline_pipeline.py separately floored its
# effective top_k at 8 regardless of this value — i.e. baseline got
# STRICTLY MORE chunks than GraphRAG (8 vs 7), deliberately, to head off
# "GraphRAG only won because baseline was starved of context." That's a
# real concern, but fixing it by over-provisioning one side is itself an
# asymmetry a skeptical reader can point at in the other direction: any
# GraphRAG win is now confounded with "GraphRAG also got a smaller
# budget", and any baseline win is confounded with "baseline also got a
# bigger one." Equal budgets remove the objection either way — the only
# deliberate difference left between the two pipelines is that GraphRAG
# additionally receives structured graph facts (and, for comparison
# questions, temporal facts). See baseline_pipeline.py's
# DEFAULT_BASELINE_TOP_K, which now just reads this directly instead of
# flooring it at 8.
#
# Don't just trust this number because it's written down here — run
# `python benchmark.py --ablation` (or a top_k sweep of your own) to
# check whether 7 is actually a good shared budget for your dataset
# before reporting it as the final config.
TOP_K_BASELINE = TOP_K_GRAPH

# P1 fix: audio-routed retrieval specifically needs a larger budget than
# pdf/both. Measured via diagnose_audio_rank.py against the real 14
# audio questions in ground_truth.json: at the shared TOP_K_GRAPH=7
# cutoff, the correct chunk (by strict chunk_id OR document+timestamp
# overlap - HyDE's answer-bearing segment doesn't always land on the
# exact chunk boundary the annotation picked) is found for only 8/14
# (57%) questions. Recomputed directly from the measured rank of the
# correct chunk for every question (no re-running needed - the rank
# data already determines the cutoff's effect):
#   top_k=7:   8/14 (57%)
#   top_k=12: 10/14 (71%)
#   top_k=17: 13/14 (93%)
#   top_k=20: 13/14 (93%, same as 17 - clean headroom, no further gain)
#   top_k=26: 14/14 (100% - but this is the one FAR miss, rank 26, and
#     feeding the LLM 26 chunks for a single audio question is a real
#     token/latency/noise cost for one question's worth of marginal gain)
# 20 was picked as the point where the curve flattens - already
# capturing everything except the one genuine far-miss, without paying
# for the long tail past it.
#
# Applied to BOTH pipelines identically (see TOP_K_BASELINE's comment
# above on equal budgets) - giving GraphRAG alone a bigger audio budget
# would reintroduce exactly the "GraphRAG only won because baseline was
# starved of context" confound that design already rejected.
#
# Retrieval-level recall, not confirmed answer-quality: this measures
# where the CORRECT CHUNK ranks, not whether the LLM correctly picks it
# out of a larger candidate pool once it's in the prompt. More evidence
# has real cost (tokens, latency, some risk of the extra chunks being
# noise) - if benchmark.py's audio-bucket numeric_exact/hit_at_k don't
# actually improve with this change, that's real information the
# retrieval-level number alone can't tell you.
TOP_K_AUDIO = int(os.getenv("TOP_K_AUDIO", "20"))

# P1 new: config for vectordb_pipeline.py, the standalone FAISS-backed
# vector-DB baseline (genuinely decoupled from Neo4j, unlike the
# existing baseline which still reads chunk embeddings out of Neo4j -
# see export_chunks_to_faiss.py's docstring for why this is a separate,
# meaningful comparison arm).
VECTORDB_INDEX_DIR = os.getenv("VECTORDB_INDEX_DIR", "vectordb_index")
# P1 fix: confirmed live case - a first real 3-way comparison run showed
# vectordb_hit_at_k (strict chunk-id match) at a flat 0.0 for BOTH the
# pdf AND audio buckets at the original default of 7, while the looser
# source_hit_at_k (document+page/timestamp match) showed partial success
# for pdf specifically (0.43) - meaning the right page was often found,
# just under a different chunk boundary than ground truth's exact ID,
# consistent with pure dense-only search (no BM25/FastRP to help it
# converge on the same specific chunk hybrid retrieval favors) needing
# more room than a small top_k gives it. Raised uniformly for every
# question (not audio-specific like TOP_K_AUDIO) deliberately - an
# audio-routing-aware boost would mean giving this "plain vector-DB
# baseline" a piece of Question-Type Routing, the exact capability it's
# meant to NOT have by design (see vectordb_pipeline.py's docstring).
VECTORDB_TOP_K = int(os.getenv("VECTORDB_TOP_K", "20"))
VECTORDB_CANDIDATE_POOL = int(os.getenv("VECTORDB_CANDIDATE_POOL", "200"))
# hybrid retrieval: weight given to BM25 vs embedding cosine sim (0-1).
# Env-overridable so an ablation can actually turn BM25 off/up without
# editing this file — it was previously a hardcoded literal, which meant
# a BM25_WEIGHT=0.0 ablation silently changed nothing and produced rows
# identical to the default (confirmed: dense_only and dense_bm25 came out
# byte-identical on every retrieval metric in a 10-config run).
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.10"))

# ── FastRP graph embeddings ─────────────────────────────────────────────
# FastRP (Fast Random Projection) embeddings capture GRAPH STRUCTURE —
# which chunks/tables/metrics/entities are connected to which — as a
# vector, complementing the two text-similarity signals above (dense
# embedding cosine sim = MEANING, BM25 = KEYWORDS). A chunk that shares
# an Entity/Metric with lots of other evidence for this question tends to
# score higher on FastRP even if its wording doesn't closely match the
# question. Computed offline by fast_rp.py (via Neo4j GDS) and written
# onto nodes as `fastrp_embedding`; retrieval.py reads it same as it does
# `embedding`.
#
# On by default now, but it has a real prerequisite: it needs the Neo4j
# Graph Data Science (GDS) plugin installed AND fast_rp.py run at least
# once after the graph (and its temporal links) are built — until then,
# no node has a fastrp_embedding property and retrieval.py degrades
# gracefully back to dense+BM25 automatically (same fallback pattern as
# the reranker toggle below), so nothing breaks, but you also won't
# actually get FastRP's benefit until you've run fast_rp.py.
USE_FASTRP = os.getenv("USE_FASTRP", "true").lower() == "true"
FASTRP_DIMENSIONS = int(os.getenv("FASTRP_DIMENSIONS", "128"))
# gds.fastRP's iterationWeights: one weight per propagation hop out from
# each node. 4 hops covers Chunk/Table/AudioChunk -> Metric/Entity/
# Guidance/Risk -> Quarter -> sibling Quarter (temporal links) without
# over-smoothing everything toward the same vector.
FASTRP_ITERATION_WEIGHTS = [0.0, 1.0, 1.0, 0.5]
FASTRP_RANDOM_SEED = int(os.getenv("FASTRP_RANDOM_SEED", "42"))
# Node labels / relationship types projected into the GDS in-memory graph
# that FastRP runs over. Matches the schema graph_builder.py actually
# writes (see its CREATE CONSTRAINT list and _tx_write_* methods) plus
# whatever temporal_utils.py adds — projecting relationship types that
# don't exist yet (e.g. before temporal_utils.py has run) is harmless,
# GDS just finds zero of them.
FASTRP_NODE_LABELS = [
    "Company",
    "Quarter",
    "Chunk",
    "Table",
    "AudioChunk",
    "Metric",
    "Entity",
    "Guidance",
    "Risk",
]
FASTRP_REL_TYPES = [
    "HAS_QUARTER",
    "HAS_CHUNK",
    "HAS_TABLE",
    "HAS_AUDIO",
    "HAS_METRIC",
    "CONTAINS_METRIC",
    "MENTIONS",
    "HAS_GUIDANCE",
    "CONTAINS_GUIDANCE",
    "HAS_RISK",
    "CONTAINS_RISK",
    "DISCUSSES",
    "NEXT_VALUE",
    "PREVIOUS_VALUE",
    "YOY_CHANGE",
    "UPDATED_TO",
    "PERSISTED_TO",
]
FASTRP_GRAPH_NAME = os.getenv("FASTRP_GRAPH_NAME", "graphrag-fastrp")

# P1 fix (ablation support): three new flags letting run_ablation.py
# isolate Graph Facts, Temporal Graph, and Question-Type Routing as
# their own clean experiments (EXP4/EXP5/EXP6), the same way
# USE_FASTRP/BM25_WEIGHT already isolate FastRP/BM25 (EXP2/EXP3). None
# of these existed before - graph_rag_pipeline.answer() always fetched
# graph facts unconditionally, and there was no way to force temporal
# facts on for every question or force routing off, without editing
# code between ablation runs.
#
# USE_GRAPH_FACTS: default true (unchanged behavior). false skips
# _fetch_graph_facts and the period-metric lookup entirely - graph_rag_
# pipeline.answer() then runs on retrieval alone, same evidence baseline
# gets.
USE_GRAPH_FACTS = os.getenv("USE_GRAPH_FACTS", "true").lower() == "true"
# FORCE_TEMPORAL_FACTS: tri-state. Unset/empty (default) leaves
# graph_rag_pipeline.answer()'s existing behavior untouched - temporal
# facts fetch only when router.route_question() flags a comparison
# question (use_temporal_facts=None in that function's signature already
# meant this; this just makes it config-settable too). "true"/"false"
# forces temporal facts on/off for EVERY question regardless of routing
# - "true" is what EXP5 needs, so Temporal Graph's contribution is
# measured cleanly rather than only showing up on whichever subset
# routing already happened to flag (which would also leak in some of
# Question-Type Routing's own effect, muddying EXP5 vs EXP6's
# comparison).
_force_temporal_raw = os.getenv("FORCE_TEMPORAL_FACTS", "").strip().lower()
FORCE_TEMPORAL_FACTS = {"true": True, "false": False}.get(_force_temporal_raw, None)
# FORCE_SOURCE_FILTER: unset/empty (default) leaves router.route_
# question()'s real pdf/audio/both decision untouched. Set to "both" (or
# "pdf"/"audio") to override it for every question regardless of what
# routing would have picked - this is what EXP1-5 need, so Question-Type
# Routing's contribution is isolated to EXP6 alone instead of being
# baked into every earlier experiment too. Applied identically in both
# graph_rag_pipeline.py and baseline_pipeline.py for the same fairness
# reason TOP_K_AUDIO is shared between them.
FORCE_SOURCE_FILTER = os.getenv("FORCE_SOURCE_FILTER", "").strip() or None

# Hybrid retrieval score weights when FastRP data IS available on the
# retrieved candidates (see retrieval.py's _load_chunks/retrieve). Mirrors
# BM25_WEIGHT/RETRIEVAL_TYPE_BOOST_* in spirit — tunable without touching
# code. Should sum to ~1.0; retrieval.py normalizes defensively either way.
FASTRP_WEIGHT = float(os.getenv("FASTRP_WEIGHT", "0.30"))
HYBRID_DENSE_WEIGHT_WITH_FASTRP = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.45"))
HYBRID_BM25_WEIGHT_WITH_FASTRP = float(os.getenv("HYBRID_BM25_WEIGHT", "0.25"))

# How FastRP is applied when USE_FASTRP is on — two modes, see
# retrieval.py's retrieve()/_fastrp_rescore()/_fastrp_expand():
#   "rescore"   — re-weight the EXISTING dense+BM25 candidate set by
#                 proximity to the seed centroid. Cheap, but can only
#                 reorder what dense+BM25 already found; it can't surface
#                 a chunk that has no name/keyword overlap with the
#                 question but IS strongly connected in the graph (shares
#                 a Metric/Entity/temporal edge with strong matches).
#   "expand"    — additionally pulls in extra candidates from OUTSIDE the
#                 dense+BM25 top pool, based purely on graph proximity to
#                 the seed centroid, before reranking. Closer to how
#                 GraphRAG systems typically use structural embeddings
#                 (graph-neighborhood expansion feeding a reranker) rather
#                 than as a third weighted score.
# Neither is obviously correct for a given dataset — benchmark both (see
# benchmark.py's run_ablation()) before committing to one for your writeup.
FASTRP_MODE = os.getenv("FASTRP_MODE", "rescore")
# How many extra candidates "expand" mode can pull in from outside the
# dense+BM25 pool, before dedupe/rerank.
FASTRP_EXPAND_TOP_N = int(os.getenv("FASTRP_EXPAND_TOP_N", "10"))

# ── Embeddings (batching) ───────────────────────────────────────────────
EMBED_BATCH_SIZE = 32

# ── Answer pipeline tuning ────────────────────────────────────────────
# How many graph facts get sent to the LLM per question (was previously
# unbounded — some questions were sending 40-60 facts). See
# graph_rag_pipeline._select_facts().
TOP_GRAPH_FACTS = int(os.getenv("TOP_GRAPH_FACTS", "10"))

# Quota per fact category, applied instead of a flat priority cut so a
# numeric question doesn't end up with 10 metrics and zero context from
# guidance/risk/sentiment. Values should sum to roughly TOP_GRAPH_FACTS —
# they don't have to add up exactly; leftover slots are filled from
# whatever's left, prioritized the same way as before quotas existed.
GRAPH_FACT_QUOTA_NUMERIC = {
    "metric": 6,
    "entity": 2,
    "sentiment": 1,
    "guidance": 1,
    "risk": 1,
}
GRAPH_FACT_QUOTA_QUALITATIVE = {
    "sentiment": 3,
    "guidance": 3,
    "risk": 3,
    "entity": 2,
    "metric": 2,
}

# Drop retrieved chunks scoring below this before ranking. None (default)
# means no floor — existing benchmark numbers stay reproducible unless you
# opt in here or via the MIN_RETRIEVAL_SCORE env var.
_min_score_env = os.getenv("MIN_RETRIEVAL_SCORE")
MIN_RETRIEVAL_SCORE = float(_min_score_env) if _min_score_env else None

# Multiplicative score adjustments applied per chunk_type depending on
# whether a question looks numeric vs qualitative (see
# retrieval.is_numeric_question()). Numeric questions favor tables/text;
# qualitative questions favor audio commentary.
RETRIEVAL_TYPE_BOOST_NUMERIC = {
    "table": float(os.getenv("BOOST_TABLE_NUMERIC", "1.15")),
    "text": float(os.getenv("BOOST_TEXT_NUMERIC", "1.05")),
    "audio": float(os.getenv("BOOST_AUDIO_NUMERIC", "0.85")),
}
RETRIEVAL_TYPE_BOOST_QUALITATIVE = {
    "table": float(os.getenv("BOOST_TABLE_QUALITATIVE", "0.90")),
    "text": float(os.getenv("BOOST_TEXT_QUALITATIVE", "0.95")),
    "audio": float(os.getenv("BOOST_AUDIO_QUALITATIVE", "1.15")),
}
# Added to BM25_WEIGHT for numeric questions (numbers are usually found by
# keyword matching — PAT, PBT, EPS, etc.), capped at 1.0.
BM25_NUMERIC_BONUS = float(os.getenv("BM25_NUMERIC_BONUS", "0.0"))

# Cross-encoder reranking pass over the top candidates before the final
# top_k cut. Tried twice now, both measured to net-harm this corpus:
#
#   1. Original: ms-marco-MiniLM-L-6-v2, full score override.
#        dense + bm25, no reranker   hit@k 0.711   F1 0.455
#        dense + bm25 + reranker     hit@k 0.289   F1 0.411
#
#   2. Follow-up: bge-reranker-base + score blending (alpha=0.5) instead
#      of full override (see diagnose_reranker.py / diagnose_reranker_v2.py).
#      Looked like a genuine fix on a RETRIEVAL-ONLY diagnostic — hit@k
#      0.600 (no reranker) -> 0.625, stable across pool sizes 10/20/30.
#      Did NOT hold up in a full benchmark.py run with this actually
#      wired into production:
#        pdf         numeric_exact 0.591 -> 0.545  (worse)
#        pdf+audio   numeric_exact 0.250 -> 0.167  (worse)
#        ALL         numeric_exact 0.444 -> 0.389  (worse)
#        pdf retrieval_latency  ~0.12s -> ~5.0s     (40x worse, EVERY
#          bucket got slower, not just audio - the added cross-encoder
#          pass runs on every query once USE_RERANKER is on)
#      Root cause: both diagnostic scripts hardcoded source_filter="both"
#      for every test question, so the reranker's interaction with the
#      REAL routing most questions actually get in production
#      (source_filter="pdf" for most pdf-bucket questions, "audio" +
#      HyDE for audio-bucket ones - see router.py) was never actually
#      tested before this got turned on. A retrieval-only metric on the
#      wrong code path doesn't predict end-to-end behavior; confirmed
#      here the hard way rather than caught before enabling it.
#
# bge-reranker-base + blend=0.5 (RERANKER_MODEL/RERANK_BLEND_ALPHA below)
# remain the best of the configurations actually tested, in case this is
# revisited later with retrieve()'s real per-route source_filter values
# in the test loop instead of a hardcoded "both" - but reranking overall
# has now failed two separate end-to-end attempts on this corpus. Off by
# default until there's a config that's been validated against a REAL
# benchmark.py run, not just the isolated retrieval diagnostic.
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

# HyDE (Hypothetical Document Embeddings) for AUDIO questions only.
# Measured on the 6 real audio questions - mean rank of the chunk that
# actually contains the answer, and how many land inside top_k:
#     raw question    9.2   2/6
#     declarative     7.8   2/6   (free template rewrite - did NOT help)
#     keywords        9.3   2/6   (free - did not help)
#     HyDE            5.7   4/6   (one short LLM call per query)
# The audio failure was a semantic gap (interrogative question vs
# declarative transcript), NOT chunk size - audio chunks already contain
# each answer whole. PDF retrieval is deliberately excluded: it performs
# well already and does not need the extra call.
USE_HYDE_FOR_AUDIO = os.getenv("USE_HYDE_FOR_AUDIO", "true").lower() == "true"

# Run the LLM-as-judge inside benchmark.py (faithfulness + correctness per
# question). Costs roughly one extra LLM call per question - about 30 min
# per 40-question config - so it is opt-in rather than always on. Every
# other extended metric (ROUGE/BLEU/METEOR/BERTScore, MRR, context
# precision, source-type accuracy) is computed regardless and is nearly
# free. The judge can also be run offline afterwards on saved results
# (full_metrics.py --llm), since it only needs question / ground truth /
# answer / evidence, all of which benchmark.py already stores.
BENCHMARK_LLM_JUDGE = os.getenv("BENCHMARK_LLM_JUDGE", "false").lower() == "true"
# Swapped from ms-marco-MiniLM-L-6-v2 (trained on short web-QA passages,
# a domain mismatch against dense financial tables/call transcripts —
# see USE_RERANKER above) to bge-reranker-base, which measured
# meaningfully more robust on this corpus at the same pool sizes. Larger
# download than MiniLM; first use will fetch it.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# Left at 20 (the original value) rather than assumed up to 30 — pool size
# is a tuning knob, not something to change on the strength of a plan
# doc's "Top-30 -> Top-8" framing. Benchmark both on your actual dataset
# (run_ablation() in benchmark.py sweeps RERANK_CANDIDATE_POOL alongside
# the retrieval-signal combinations) and keep whichever wins.
RERANK_CANDIDATE_POOL = int(os.getenv("RERANK_CANDIDATE_POOL", "30"))
# Blend weight between the ORIGINAL hybrid (embedding+BM25+FastRP) score
# and the reranker's score for everything inside RERANK_CANDIDATE_POOL,
# both min-max normalized within the pool first (a cross-encoder logit
# isn't on the same 0-1-ish scale as the hybrid score, so blending them
# unnormalized would let whichever happens to have larger raw magnitude
# silently dominate). alpha=1.0 disables the reranker's effect entirely
# (pure original order); alpha=0.0 is full override (the old, measured-
# harmful behavior). 0.5 is the only setting tested so far that beat the
# no-reranker baseline, and did so consistently across pool=10/20/30 —
# see USE_RERANKER's comment above for the numbers. Full override alone
# (even with the better model) still measured below baseline on hit@k,
# so this blend is doing real work, not just riding the model swap.
RERANK_BLEND_ALPHA = float(os.getenv("RERANK_BLEND_ALPHA", "0.5"))

# P1 fix: guarantees a minimum number of top-ranked audio candidates
# survive retrieval.retrieve()'s final top_k cut when source_filter==
# "both" (comparison/"why did X change" questions - see router.py's
# COMPARISON_KEYWORDS), so numeric-intent ranking boosts can't crowd
# audio commentary out entirely for a question that genuinely needs
# both a figure and an explanation. See retrieve()'s comment at the
# final order[] assembly for the confirmed live case this fixes. Set to
# 0 to disable and restore the old flat-cut behavior.
MIN_AUDIO_SLOTS_WHEN_BOTH = int(os.getenv("MIN_AUDIO_SLOTS_WHEN_BOTH", "1"))

# P1 fix: scoped pdf-only synonym expansion, measured via
# diagnose_query_expansion.py to move pdf-bucket hit@k 0.5455 -> 0.6364
# with no effect on audio and a regression on pdf+audio that scoping to
# source_filter=="pdf" avoids — see retrieval.py's
# _expand_query_for_pdf() for the full numbers and mechanism. Default
# on; set false to fall back to the unexpanded query.
USE_PDF_QUERY_EXPANSION = (
    os.getenv("USE_PDF_QUERY_EXPANSION", "true").lower() == "true"
)

# ── Ground truth generation ─────────────────────────────────────────────
MAX_CONTEXT = 12000  # char budget for ground_truth_generator's context
# After generating QA pairs, re-ask the LLM per-pair "is this answer
# explicitly supported by the text?" and drop any that come back false.
# Roughly doubles ground_truth_generator's LLM calls (one extra call per
# surviving pair) but catches hallucinated/unsupported pairs the generation
# prompt's own self-check missed. Turn off with GROUND_TRUTH_VERIFY=false
# if you want faster (if slightly less reliable) regeneration runs.
GROUND_TRUTH_VERIFY = os.getenv("GROUND_TRUTH_VERIFY", "true").lower() == "true"

# ── Graph fact display ordering ──────────────────────────────────────────
# Within the "Metrics" section of the prompt, facts are sorted by how
# important each metric name is (most important first), falling back to
# alphabetical for anything not in this list — rather than pure
# alphabetical, which buries key figures like Revenue behind things like
# "Attrition Rate". Matched as a substring against the metric name
# (case-insensitive), in priority order, so "Net Revenue" and "Revenue
# Growth" both match "revenue".
METRIC_IMPORTANCE_ORDER = [
    "revenue",
    "ebitda",
    "net profit",
    "profit after tax",
    "pat",
    "profit before tax",
    "pbt",
    "eps",
    "margin",
    "growth",
    "expense",
    "cost",
    "tax",
]

# ── Question-intent keywords ─────────────────────────────────────────────
# Used by retrieval.is_numeric_question() to nudge ranking (BM25 weight,
# per-chunk-type boosts) toward tables/text for numeric-looking questions
# and toward audio commentary for qualitative ones. Moved here (rather
# than hardcoded in retrieval.py) so the heuristic can be tuned without
# touching code — e.g. adding a new metric name or a company-specific
# acronym.
#
# P0 fix: these constants were defined here but retrieval.py never
# actually read them — it compiled its own separate, hardcoded
# _NUMERIC_KEYWORDS/_QUALITATIVE_KEYWORDS regexes instead, so tuning
# this file silently did nothing (confirmed: grepping the whole codebase
# found zero references to these names outside this file). Wired
# retrieval.py to read from here now (see its is_numeric_question()),
# so the docstring above is actually true.
#
# P1 fix: "q[1-4]" and "fy ?\d{2,4}" used to be in NUMERIC_QUESTION_KEYWORDS.
# Almost every question in this benchmark — numeric AND qualitative alike
# — names a fiscal period ("...in Q1 FY25?"), so those two patterns fired
# on nearly every question regardless of actual intent, pushing
# numeric_score above qualitative_score by default. Confirmed: 5 of 6
# audio-bucket qualitative questions were misclassified as numeric purely
# because they named a quarter. Removed both patterns.
#
# P2 fix (regression from the above): removing the period patterns fixed
# the audio bucket but broke "why did revenue increase quarter-on-quarter
# in Q4 FY24?" / "What are the standalone and consolidated figures...?" —
# these only ever scored ONE numeric point ("revenue") against ONE
# qualitative point ("why"), a tie that resolves as qualitative, which is
# wrong for a question that explicitly needs a specific figure out of a
# table. Added QoQ/YoY comparison phrasing and "what are"/"figures" as
# their own numeric-intent signals — a targeted fix for the actual
# missing signal, not a reintroduction of the overly-broad period-mention
# pattern (a bare "Q1 FY25" mention still doesn't count). Verified
# against all 40 benchmark questions: pdf 22/22 numeric, audio 0/6,
# pdf+audio 12/12 — all correctly classified.
NUMERIC_QUESTION_KEYWORDS = (
    r"\b(how much|how many|what was|what is|what are|total|revenue|profit|"
    r"income|margin|ebitda|growth|percentage|percent|rate|amount|figures?|"
    r"value|pat|pbt|eps|disburs|volume|gmv|expense|cost|tax|billing|"
    r"yoy|qoq|quarter[\s-]on[\s-]quarter|quarter[\s-]over[\s-]quarter|"
    r"year[\s-]on[\s-]year|year[\s-]over[\s-]year|sequentially|"
    r"rs\.?|million|billion|crore|lakh)\b"
    r"|₹"
)
QUALITATIVE_QUESTION_KEYWORDS = (
    r"\b(what did management say|why|outlook|risk|guidance|highlight|"
    r"strategy|opinion|sentiment|milestone|initiative|management)\b"
)

# ── Evidence text cleanup (graph_rag_pipeline) ───────────────────────────
# PDF header/footer noise that shows up on nearly every page (company
# address, CIN, email/website, "Regd. & Corporate Office", etc.) and adds
# nothing to answering a question — stripped line-by-line before the
# evidence goes in the prompt. Moved here so new noise patterns can be
# added without touching graph_rag_pipeline.py.
NOISE_PATTERNS = [
    r"\S+@\S+\.\S+",  # emails
    r"\bwww\.\S+",  # website
    r"\bCIN\s*:?\s*[A-Z0-9]{15,25}",
    r"Regd\.?\s*&?\s*Corporate Office.*",
    r"Registered Office.*",
    r"Copyright\s*(\(c\)|©)?.*",
    r"Page\s+\d+\s+of\s+\d+",
]

# Common mojibake from PDF text extracted as Latin-1/Windows-1252 and
# re-decoded as UTF-8 (curly quotes, dashes, ® / © turning into "Â®" /
# "Â©", etc.). Fixed explicitly for the common cases; graph_rag_pipeline's
# catch-all regex mops up any other "â€X" / stray "Â" sequences.
OCR_ARTIFACT_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€¢": "-",
    "Â®": "®",
    "Â©": "©",
    "Â°": "°",
    "Â": "",
}

# ── Logging ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
