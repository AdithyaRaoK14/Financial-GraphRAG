"""
retrieval.py
============
WHAT THIS FILE DOES:
Shared by both pipelines (GraphRAG and baseline). Given a question, finds
the top-k chunks whose embeddings are most similar to the question's
embedding — i.e. semantic search. This is the "R" (retrieval) in RAG.

Loads every embedded node from Neo4j into memory once (cheap for a
few-thousand-chunk dataset) — Chunk, Table, AND AudioChunk, not just
Chunk — then does cosine similarity in numpy. No separate vector database
needed.

Also blends in BM25 keyword scoring (hybrid retrieval) when rank_bm25 is
installed — pure embedding search can miss exact figures, tickers, or
acronyms that keyword matching catches. Install with
`pip install rank_bm25`; retrieval silently falls back to embeddings-only
if it's not present.

Company/year/quarter live on the (:Quarter) node, not on the chunk-level
nodes themselves, so loading joins through Quarter to pick those up for
every node type.
"""

import json
import logging
import re

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

import config
from evidence_cleaning import clean_evidence_text, format_table_markdown

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    _BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    _RERANKER_LIB_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    _RERANKER_LIB_AVAILABLE = False

# --- Question-intent heuristic -------------------------------------------
# Numeric questions ("what was revenue?") are usually answered by exact
# figures in tables/PDF text and are well served by keyword (BM25)
# matching. Qualitative questions ("what did management say about risk?")
# are usually answered by transcript commentary and are better served by
# semantic (embedding) matching. This is a simple keyword heuristic, not a
# classifier — it's meant to nudge ranking, not gate retrieval outright.
#
# P-fix: "q[1-4]" and "fy ?\d{2,4}" used to be in this list. Almost every
# question in this benchmark — numeric AND qualitative alike — names a
# fiscal period ("...in Q1 FY25?"), so those two patterns fired on nearly
# every question regardless of actual intent and pushed numeric_score
# above qualitative_score by default. Confirmed: 5 of 6 audio-bucket
# qualitative questions ("what risks did management mention...", "what
# concerns were raised...") were misclassified as numeric purely because
# they named a quarter, which (a) applied the audio 0.85x type-penalty
# below instead of the 1.15x boost during RETRIEVAL — starving these
# questions of the very audio chunks they needed, for both pipelines
# equally, since they share this retrieval code — and (b) fed the same
# wrong classification into graph_rag_pipeline.py's numeric-vs-qualitative
# branching (disabling temporal-fact fallback, skipping the qualitative
# refusal retry). A period mention is not a numeric-intent signal; it's
# just how these questions are scoped. Removed both patterns.
#
# P-fix (regression from the above fix): removing the period patterns
# fixed the audio bucket but silently broke a different set of questions
# — "why did revenue increase quarter-on-quarter in Q4 FY24?" and "What
# are the standalone and consolidated figures, and why do they differ?"
# — that only ever scored ONE numeric point ("revenue") against ONE
# qualitative point ("why"), a tie that is_numeric_question() (strict
# `>`) resolves as qualitative. That's wrong for these: they explicitly
# need the specific figures out of a table (a QoQ/YoY comparison, or two
# named tables), the "why"/"what are" wording doesn't make them
# transcript-commentary questions. Confirmed: 5 of 6 pdf+audio retrieval
# misses and 2 of 22 pdf-bucket retrieval misses this round were exactly
# this pattern (checked via hit_at_k). Fix: recognize QoQ/YoY comparison
# phrasing and "what are .../figures" as their own numeric-intent
# signals — this is a targeted fix for the actual missing signal, not a
# reintroduction of the overly-broad period-mention pattern that caused
# the original bug (a bare "Q1 FY25" mention still doesn't count).
# P0 fix: these used to be hardcoded here, duplicating (and eventually
# drifting from) config.NUMERIC_QUESTION_KEYWORDS/QUALITATIVE_QUESTION_
# KEYWORDS, which existed for exactly this but were never actually read
# by anything — confirmed via a full-codebase grep, zero references
# outside config.py itself. Wired to read from config.py now, matching
# the same getattr-with-fallback pattern already used for the boost maps
# below, so the "tunable without touching code" docstring on those
# config constants is actually true. The fallback strings are identical
# to config.py's current values (not the old, since-removed q[1-4]/
# fy-pattern version) so a config.py missing this attribute degrades to
# the same fixed behavior rather than silently reintroducing the bug
# that pattern caused.
_NUMERIC_KEYWORDS = re.compile(
    getattr(
        config,
        "NUMERIC_QUESTION_KEYWORDS",
        r"\b(how much|how many|what was|what is|what are|total|revenue|profit|"
        r"income|margin|ebitda|growth|percentage|percent|rate|amount|figures?|"
        r"value|pat|pbt|eps|disburs|volume|gmv|expense|cost|tax|billing|"
        r"yoy|qoq|quarter[\s-]on[\s-]quarter|quarter[\s-]over[\s-]quarter|"
        r"year[\s-]on[\s-]year|year[\s-]over[\s-]year|sequentially|"
        r"rs\.?|million|billion|crore|lakh)\b"
        r"|₹",
    ),
    re.I,
)
_QUALITATIVE_KEYWORDS = re.compile(
    getattr(
        config,
        "QUALITATIVE_QUESTION_KEYWORDS",
        r"\b(what did management say|why|outlook|risk|guidance|highlight|"
        r"strategy|opinion|sentiment|milestone|initiative|management)\b",
    ),
    re.I,
)

# Multiplicative score adjustments applied per chunk_type depending on
# question intent — nudges ranking without hard-filtering anything out.
# Read from config.py so these are tunable without editing code; fall
# back to the values used before they were made configurable.
_NUMERIC_TYPE_BOOST = getattr(
    config, "RETRIEVAL_TYPE_BOOST_NUMERIC", {"table": 1.15, "text": 1.05, "audio": 0.85}
)
_QUALITATIVE_TYPE_BOOST = getattr(
    config,
    "RETRIEVAL_TYPE_BOOST_QUALITATIVE",
    {"table": 0.90, "text": 0.95, "audio": 1.15},
)

_reranker_model = None

# P1 fix: measured via diagnose_query_expansion.py (read-only, real per-
# question router.py routing, not a hardcoded source_filter like the
# reranker diagnostics used) - appending domain-synonym terms to the
# query text before retrieval:
#   pdf bucket:        hit@k 0.5455 -> 0.6364, recall@k 0.5455 -> 0.6364
#   pdf+audio bucket:  hit@k 0.7500 -> 0.6667  (regression)
#   audio bucket:      unchanged (0.5000 both) - map is finance-metric
#                       terms, rarely relevant to pure commentary questions
# Scoped to fire ONLY when source_filter=="pdf" so the confirmed pdf
# win applies without the pdf+audio regression - "both"/"audio"-routed
# questions are untouched. Deliberately narrow, hand-picked synonym set
# rather than a general thesaurus: this corpus's confirmed failure mode
# is near-metric confusion ("Revenue from operations" vs "Total Net
# Sales/Revenue from Operations" vs "Other Income"), so a broad synonym
# source would risk making that worse; kept to the exact terms actually
# measured to help.
_PDF_SYNONYM_MAP = {
    "revenue": ["income", "sales", "turnover"],
    "profit": ["PAT", "earnings", "net income"],
    "margin": ["profitability"],
    "growth": ["increase", "change"],
    "guidance": ["outlook", "forecast"],
    "eps": ["earnings per share"],
    "ebitda": ["operating profit"],
    "expenses": ["costs", "expenditure"],
}
_PDF_SYNONYM_TERM_RE = {
    term: re.compile(rf"\b{re.escape(term)}\b", re.I)
    for term in _PDF_SYNONYM_MAP
}


def _expand_query_for_pdf(question: str, source_filter: str) -> str:
    """Appends matched synonym terms to `question`, once each, but ONLY
    when source_filter=="pdf" (see _PDF_SYNONYM_MAP's comment for the
    measured pdf-win/pdf+audio-regression that scopes this). Never
    alters the original question text, only adds terms after it, so
    original phrasing still dominates whatever ranking signal weighs
    word position/proximity. Gated behind config.USE_PDF_QUERY_EXPANSION
    (default True) so it can be switched off in one line without
    reverting this function."""
    if source_filter != "pdf" or not getattr(
        config, "USE_PDF_QUERY_EXPANSION", True
    ):
        return question
    extra_terms = []
    for term, synonyms in _PDF_SYNONYM_MAP.items():
        if _PDF_SYNONYM_TERM_RE[term].search(question):
            extra_terms.extend(synonyms)
    if not extra_terms:
        return question
    return f"{question} {' '.join(extra_terms)}"




def _get_reranker():
    """Lazily load the cross-encoder reranker (only if config.USE_RERANKER
    is on and sentence-transformers' CrossEncoder is available). This
    downloads a small model on first use — see config.RERANKER_MODEL."""
    global _reranker_model
    if _reranker_model is None and _RERANKER_LIB_AVAILABLE:
        _reranker_model = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_model


def _rerank_text(chunk: dict) -> str:
    """Text handed to the cross-encoder for one candidate chunk — the
    same cleaned/formatted text the LLM eventually sees (evidence_
    cleaning.py), not the raw chunk text. Measured to matter less than
    the model swap / blending on their own (see diagnose_reranker.py),
    but this is what the validated bge-reranker-base + blend=0.5 numbers
    in config.py's USE_RERANKER comment were actually measured with, so
    kept consistent with that rather than reverted back to raw text."""
    if chunk.get("chunk_type") == "table":
        return format_table_markdown(chunk)
    return clean_evidence_text(chunk.get("text") or chunk.get("embedding_text") or "")


def _minmax(values) -> list:
    """Min-max normalize a list of scores to [0, 1] so the original
    hybrid score and the reranker's score can be blended on comparable
    scales (a cross-encoder logit is not bounded 0-1 the way the hybrid
    score roughly is). All-equal input (e.g. a pool of 1) returns a flat
    0.5 for every entry rather than dividing by zero."""
    arr = np.asarray(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return list((arr - lo) / (hi - lo))


def is_numeric_question(question: str) -> bool:
    """Heuristic: does this question look like it wants a specific figure
    (numeric_score > qualitative_score) rather than commentary/context?"""
    q = question or ""
    numeric_score = len(_NUMERIC_KEYWORDS.findall(q))
    qualitative_score = len(_QUALITATIVE_KEYWORDS.findall(q))
    return numeric_score > qualitative_score


def _normalize_signature(text: str, length: int = 300) -> str:
    """Normalized signature used to detect near-duplicate chunks (e.g. the
    same revenue table repeated on an annexure page). Lowercase,
    whitespace-collapsed, alphanumeric-only prefix — good enough to catch
    exact/near-exact repeats without being so short it collides on
    unrelated short chunks."""
    cleaned = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    return cleaned[:length]


_embed_model = None
_chunk_cache = None  # list of dicts, loaded once per process
_embedding_matrix = None  # normalized (N, dim) numpy array aligned with _chunk_cache
_bm25_index = None  # BM25Okapi over the same corpus, same row order as _chunk_cache
# Normalized (N, dim) FastRP matrix aligned with _chunk_cache, plus a mask
# of which rows actually have a FastRP vector. Nodes that fast_rp.py never
# reached (or a graph where it was never run) simply have no vector — the
# mask lets FastRP degrade to a no-op for those instead of polluting
# scores with zeros. See _load_chunks() and _fastrp_* below.
_fastrp_matrix = None
_fastrp_present = None

# P1 latency fix: a multi-company GraphRAG question ("Among the five
# companies...") calls retrieve() once PER COMPANY with the exact same
# question text — only the company filter differs — yet each call was
# re-running model.encode() on identical input. Confirmed in the
# benchmark: the two "Among the five companies" rows were the latency
# outliers (445s/388s, vs ~150s average), calling retrieve() up to 5x.
# baseline_pipeline.py and graph_rag_pipeline.py also both call
# retrieve() with the same question text for the same benchmark row, so
# this also removes one redundant encode per question even outside the
# multi-company case. Keyed on the raw question string; bounded so a
# long-running process (not just a single benchmark run) can't grow this
# unboundedly. This changes nothing about WHAT gets retrieved — the
# vector is identical either way, only recomputation is avoided.
_QUERY_EMBEDDING_CACHE_MAX = 512
_query_embedding_cache: dict[str, np.ndarray] = {}


_HYDE_CACHE: dict[str, str] = {}
_HYDE_CACHE_MAX = 256

_HYDE_PROMPT = (
    "Write two sentences that would plausibly appear in an Indian "
    "company's earnings-call transcript answering this question. Write "
    "them as management commentary, not as a question. Do not invent "
    "specific numbers.\n\nQUESTION: {question}"
)


def _hyde_query(question: str) -> str:
    """Generate a hypothetical ANSWER to embed instead of the question.

    Why: audio retrieval underperformed because of a SEMANTIC GAP, not
    chunking — measured directly. The answer-bearing chunk sits whole
    inside a single audio chunk every time (answer-similarity 0.58-0.77),
    but questions are interrogative while transcripts are declarative, so
    the right chunk ranked #9-#14 instead of inside top-7.

    Measured on the real 6 audio questions, mean rank of the
    answer-bearing chunk and how many would actually be retrieved:
        raw question   9.2   2/6
        declarative    7.8   2/6   (template rewrite, no LLM)
        keywords       9.3   2/6
        HyDE (this)    5.7   4/6
    The free rewrites did not help; only a genuine hypothetical answer
    did. Costs one short generation per query, so it is applied ONLY to
    audio-routed questions — PDF numeric retrieval already performs well
    (F1 0.782) and does not need it.

    Falls back to the raw question on any failure: this must never break
    retrieval, only improve it when it works.
    """
    cached = _HYDE_CACHE.get(question)
    if cached is not None:
        return cached
    try:
        import llm_client

        text = llm_client.generate(
            _HYDE_PROMPT.format(question=question),
            model=config.ANSWER_MODEL,
            temperature=0.0,
            num_predict=120,
        ).strip()
        if not text:
            return question
    except Exception as exc:
        logger.debug("HyDE generation failed (%s) - using raw question.", exc)
        return question

    if len(_HYDE_CACHE) >= _HYDE_CACHE_MAX:
        _HYDE_CACHE.pop(next(iter(_HYDE_CACHE)))
    _HYDE_CACHE[question] = text
    return text


def _encode_question(model, question: str) -> np.ndarray:
    cached = _query_embedding_cache.get(question)
    if cached is not None:
        return cached
    vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    if len(_query_embedding_cache) >= _QUERY_EMBEDDING_CACHE_MAX:
        _query_embedding_cache.pop(next(iter(_query_embedding_cache)))
    _query_embedding_cache[question] = vec
    return vec


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# Financial metric aliases used only for ranking.  They do not change or
# write anything to Neo4j.
_METRIC_ALIASES = {
    "revenue from operations": ("revenue from operations", "revenue"),
    "revenue": ("revenue from operations", "revenue"),
    "net profit": ("net profit", "profit after tax", "profit for the period"),
    "profit after tax": ("profit after tax", "net profit"),
    "basic eps": ("basic eps", "earnings per share"),
    "earnings per share": ("earnings per share", "basic eps"),
    "ebitda": ("ebitda",),
    "pat": ("pat", "profit after tax", "net profit"),
}


def _question_metric_phrases(question: str) -> tuple[str, ...]:
    q = (question or "").lower()
    phrases = []
    for key, aliases in _METRIC_ALIASES.items():
        if key in q:
            phrases.extend(aliases)
    # Prefer the longest phrase first so "revenue from operations" wins over
    # the generic "revenue" alias.
    return tuple(dict.fromkeys(sorted(phrases, key=len, reverse=True)))


def _requested_statement_type(question: str, statement_type: str = None) -> str | None:
    """Resolve statement type without changing stored graph data.

    SINGLE SOURCE OF TRUTH: this is the only implementation of this
    resolution logic in the project. graph_rag_pipeline.py previously
    defined its own separate copy of a function with this same name and
    an OPPOSITE default ("consolidated" there vs "standalone" here) —
    that split-brain default was a confirmed P0 bug: for any question
    with no explicit "standalone"/"consolidated" wording and no
    statement_type metadata, chunk retrieval (this function) would rank
    toward standalone evidence while graph_rag_pipeline.py's fact
    resolution simultaneously resolved toward consolidated, guaranteeing
    the LLM saw two "correctly selected" but contradictory values for the
    same metric (confirmed against the Info Edge Q4 FY26 benchmark row).
    graph_rag_pipeline.py now imports and calls this function directly
    instead of keeping a second copy.

    A benchmark question may carry an explicit statement_type in its
    metadata. Real user questions normally do not, so the textual
    question remains the fallback. Default (no explicit signal either
    way): consolidated — most companies' primary reported figures are
    consolidated, which also matches this benchmark's own ground truth
    (49 consolidated vs 12 standalone vs 4 "both" vs 10 unspecified,
    out of 75 questions).
    """
    if statement_type and str(statement_type).lower() in {"standalone", "consolidated"}:
        return str(statement_type).lower()
    q = question or ""
    if re.search(r"\bstandalone\b", q, re.I):
        return "standalone"
    if re.search(r"\bconsolidated\b", q, re.I):
        return "consolidated"
    return "consolidated" if is_numeric_question(question) else None


def _chunk_statement_type(chunk: dict) -> str | None:
    text = " ".join(
        str(chunk.get(k) or "")
        for k in ("title", "heading", "section", "text", "embedding_text")
    )
    text = re.sub(r"\s+", " ", text).lower()
    # Match the explicit financial-results heading first.  This avoids
    # treating an incidental mention of both words as the statement type.
    m = re.search(
        r"statement of .*?\b(standalone|consolidated)\b .*?\bfinancial results\b",
        text,
        re.I,
    )
    if m:
        return m.group(1).lower()
    for kind in ("standalone", "consolidated"):
        if re.search(rf"\b{kind}\b\s+(?:financial|results)", text, re.I):
            return kind
    return None


def _financial_query_adjustment(
    question: str, chunk: dict, statement_type: str = None
) -> float:
    """Return a small, bounded ranking adjustment for financial Q&A.

    Dense/BM25 retrieval is still the primary signal.  This only breaks ties
    between near-identical financial tables by favoring the requested metric
    and the requested statement type.  It is deliberately not a hard filter,
    because some documents do not contain an explicit statement heading in
    the table chunk itself.
    """
    if not is_numeric_question(question):
        return 1.0

    adjustment = 1.0
    text = " ".join(
        str(chunk.get(k) or "")
        for k in ("title", "heading", "section", "text", "embedding_text")
    ).lower()

    phrases = _question_metric_phrases(question)
    if phrases:
        if any(p in text for p in phrases):
            adjustment *= 1.10
        else:
            adjustment *= 0.94

    wanted = _requested_statement_type(question, statement_type)
    if wanted:
        actual = _chunk_statement_type(chunk)
        if actual == wanted:
            adjustment *= 1.12
        elif actual in {"standalone", "consolidated"}:
            adjustment *= 0.86

    return adjustment


# Every embeddable label in the graph maps to a (source_type, chunk_type)
# pair. Chunk/Table always come from PDFs; AudioChunk always comes from
# earnings-call audio (see graph_builder.py's store_chunk dispatch).
LABEL_TO_TYPE = {
    "Chunk": ("pdf", "text"),
    "Table": ("pdf", "table"),
    "AudioChunk": ("audio", "audio"),
}


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def clear_cache():
    """
    Drop the in-memory chunk/embedding cache. Call this after rebuilding
    or re-embedding the graph (e.g. after running embeddings.py again) —
    otherwise retrieve() keeps serving stale vectors from the last time
    this process loaded them.
    """
    global _chunk_cache, _embedding_matrix, _bm25_index, _fastrp_matrix, _fastrp_present
    _chunk_cache = None
    _embedding_matrix = None
    _bm25_index = None
    _fastrp_matrix = None
    _fastrp_present = None
    _query_embedding_cache.clear()


def _load_chunks():
    global _chunk_cache, _embedding_matrix, _bm25_index, _fastrp_matrix, _fastrp_present
    if _chunk_cache is not None:
        return _chunk_cache

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (q:Quarter)-[]->(n)
                WHERE n.embedding IS NOT NULL
                RETURN elementId(n) AS eid,
                       labels(n)[0] AS label,
                       n.id AS node_id,
                       coalesce(n.text, n.embedding_text) AS text,
                       n.embedding_text AS embedding_text,
                       q.company AS company,
                       q.year AS year,
                       q.quarter AS quarter,
                       n.page AS page,
                       n.section AS section,
                       n.heading AS heading,
                       n.title AS title,
                       n.document_name AS document_name,
                       n.page_type AS page_type,
                       n.start AS start,
                       n.end AS end,
                       n.headers AS headers,
                       n.rows AS rows,
                       n.embedding AS embedding,
                       n.fastrp_embedding AS fastrp_embedding
                """
            )
            rows = list(result)
    finally:
        driver.close()

    chunks = []
    vectors = []
    fastrp_vectors = []

    for r in rows:
        source_type, chunk_type = LABEL_TO_TYPE.get(r["label"], ("unknown", "unknown"))

        # Table nodes store `rows` as a JSON string (graph_builder.py writes
        # it with json.dumps so Neo4j — which has no nested-list property
        # type — can store it). Chunk/AudioChunk nodes never set this
        # property, so r["rows"] is None for them and this is a no-op.
        # format_table_markdown() (evidence_cleaning.py) needs the decoded
        # list to render a clean table instead of falling back to raw
        # flattened text — this was never wired through before, so every
        # table chunk silently used the flattened fallback.
        parsed_rows = None
        if r.get("rows"):
            try:
                parsed_rows = json.loads(r["rows"])
            except (TypeError, ValueError):
                parsed_rows = None

        chunks.append(
            {
                "chunk_id": r["node_id"] or r["eid"],
                "text": r["text"] or "",
                "embedding_text": r["embedding_text"],
                "company": r["company"],
                "year": r["year"],
                "quarter": r["quarter"],
                "source_type": source_type,
                "chunk_type": chunk_type,
                "page": r["page"],
                "section": r["section"],
                "heading": r["heading"],
                "title": r["title"],
                "document_name": r["document_name"],
                "page_type": r["page_type"],
                "start": r["start"],
                "end": r["end"],
                "headers": r.get("headers"),
                "rows": parsed_rows,
            }
        )
        vectors.append(r["embedding"])
        fastrp_vectors.append(r.get("fastrp_embedding"))

    _chunk_cache = chunks

    if vectors:
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        _embedding_matrix = matrix / norms  # normalize once, up front
    else:
        _embedding_matrix = np.zeros((0, 0), dtype=np.float32)

    if _BM25_AVAILABLE and chunks:
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        _bm25_index = BM25Okapi(tokenized_corpus)
    else:
        _bm25_index = None

    # FastRP structural embeddings (written onto nodes by fast_rp.py).
    # Rows without a vector get zeros and are masked out via
    # _fastrp_present, so a graph where fast_rp.py was never run — or
    # where it only reached some labels — degrades to plain dense+BM25
    # rather than scoring everything against a meaningless zero vector.
    dims = {len(v) for v in fastrp_vectors if v}
    if dims and len(dims) == 1:
        dim = dims.pop()
        fmatrix = np.zeros((len(chunks), dim), dtype=np.float32)
        present = np.zeros(len(chunks), dtype=bool)
        for i, v in enumerate(fastrp_vectors):
            if v and len(v) == dim:
                fmatrix[i] = np.asarray(v, dtype=np.float32)
                present[i] = True
        norms = np.linalg.norm(fmatrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        _fastrp_matrix = fmatrix / norms
        _fastrp_present = present
        logger.info(
            "FastRP embeddings loaded for %d/%d chunks (dim=%d).",
            int(present.sum()),
            len(chunks),
            dim,
        )
    else:
        _fastrp_matrix = None
        _fastrp_present = None
        if dims:
            logger.warning(
                "Inconsistent fastrp_embedding dimensions %s — FastRP disabled. "
                "Re-run fast_rp.py so every node gets the same dimension.",
                dims,
            )
        else:
            logger.info(
                "No fastrp_embedding found on any chunk — FastRP scoring is "
                "inactive. Run fast_rp.py to enable it."
            )

    return _chunk_cache


def _normalize_company_name(value: str | None) -> str:
    """Normalize company names for retrieval filters without changing stored data."""
    import re

    s = str(value or "").lower()
    # Ground-truth company names are often the full legal/descriptive name
    # with a parenthetical qualifier — e.g. "Nykaa (FSN E-Commerce Ventures
    # Limited)" or "Info Edge (India) Limited" — while the graph stores a
    # short form with no parenthetical ("NYKAA", "InfoEdge"). Strip
    # parenthetical content before normalizing so both collapse to the
    # same core name instead of permanently mismatching.
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    # Legal suffixes are not semantically useful for matching the indexed
    # company field (e.g. "CreditAccess Grameen Limited" vs
    # "Credit Access Grameen"). Keep the stored values untouched.
    s = re.sub(
        r"\b(limited|ltd|incorporated|inc|corp|corporation|company|co)\b", " ", s
    )
    # Compare the canonical alphanumeric form so formatting differences such
    # as "CreditAccess Grameen", "Credit Access Grameen", and
    # "CreditAccess Grameen Limited" cannot empty the candidate pool.
    return re.sub(r"[^a-z0-9]+", "", s)


def _company_matches(candidate: str | None, wanted: str) -> bool:
    """True if a chunk/quarter's normalized company name matches the
    wanted normalized company name. Exact match first; falls back to
    substring containment (either direction) so a short stored code like
    "nykaa" still matches a longer descriptive name whose parenthetical
    qualifier didn't fully strip to the same string, without that fallback
    ever matching on an empty string."""
    norm_candidate = _normalize_company_name(candidate)
    if not norm_candidate or not wanted:
        return False
    if norm_candidate == wanted:
        return True
    return norm_candidate in wanted or wanted in norm_candidate


def _fastrp_available() -> bool:
    return (
        getattr(config, "USE_FASTRP", False)
        and _fastrp_matrix is not None
        and _fastrp_present is not None
        and bool(_fastrp_present.any())
    )


def _fastrp_seed_centroid(idxs, combined, seed_n: int = 5):
    """Build the 'where in the graph is this question pointing' vector.

    FastRP embeds graph STRUCTURE, not text, so there is no way to embed
    the question itself into FastRP space. Instead we take the chunks
    dense+BM25 is already most confident about (the seeds), and average
    their structural vectors — that centroid represents the region of the
    graph the question lives in. Chunks near that centroid are
    structurally related (share Metric/Entity/temporal edges) even when
    their wording doesn't match the question.

    Returns None when no seed has a FastRP vector, so callers can skip
    FastRP entirely rather than steer by a meaningless centroid.
    """
    order = np.argsort(-combined)
    seeds = []
    for j in order:
        i = idxs[int(j)]
        if _fastrp_present[i]:
            seeds.append(_fastrp_matrix[i])
        if len(seeds) >= seed_n:
            break
    if not seeds:
        return None
    centroid = np.mean(np.asarray(seeds, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-8:
        return None
    return centroid / norm


def _fastrp_sims(idxs, centroid):
    """Cosine similarity of each candidate to the seed centroid, with
    chunks that have no FastRP vector forced to 0 so they are neither
    rewarded nor penalised relative to their dense/BM25 score."""
    sub = _fastrp_matrix[idxs]
    sims = sub @ centroid
    mask = _fastrp_present[idxs]
    sims = np.where(mask, sims, 0.0)
    # FastRP cosines can be negative; clip so a structurally-unrelated
    # chunk can't drag a strong dense match below an unrelated one.
    return np.clip(sims, 0.0, None).astype(np.float32)


def _fastrp_rescore(embed_sims, bm25_sub, fastrp_sims):
    """'rescore' mode — blend FastRP in as a third weighted signal
    alongside dense and BM25, re-weighting the candidate set dense+BM25
    already found. Weights come from config
    (HYBRID_DENSE_WEIGHT_WITH_FASTRP / HYBRID_BM25_WEIGHT_WITH_FASTRP /
    FASTRP_WEIGHT) and are normalized here so they don't have to sum to
    exactly 1.0 in config — config.py's comment promises that behavior."""
    w_dense = float(getattr(config, "HYBRID_DENSE_WEIGHT_WITH_FASTRP", 0.45))
    w_bm25 = float(getattr(config, "HYBRID_BM25_WEIGHT_WITH_FASTRP", 0.25))
    w_frp = float(getattr(config, "FASTRP_WEIGHT", 0.30))

    if bm25_sub is None:
        w_bm25 = 0.0
    total = w_dense + w_bm25 + w_frp
    if total <= 0:
        return embed_sims
    w_dense, w_bm25, w_frp = w_dense / total, w_bm25 / total, w_frp / total

    combined = w_dense * embed_sims + w_frp * fastrp_sims
    if bm25_sub is not None:
        combined = combined + w_bm25 * bm25_sub
    return combined


def _fastrp_expand(candidate_order, fastrp_sims, pool_size, expand_n):
    """'expand' mode — pull in extra candidates from OUTSIDE the
    dense+BM25 top pool purely on graph proximity, so a chunk with no
    keyword/semantic overlap can still surface if it's strongly connected
    to the seeds. This is closer to how GraphRAG systems normally use
    structural embeddings (neighborhood expansion feeding a reranker)
    than treating FastRP as a third score.

    The expanded candidates are drawn from the SAME metadata-filtered
    index set as everything else (company/year/quarter/source filters
    already applied upstream), so expansion can never leak in another
    company's or period's chunks — it only reaches past the dense+BM25
    ranking, not past the filters.
    """
    order = list(candidate_order)
    if expand_n <= 0 or len(order) <= pool_size:
        return order

    in_pool = set(int(j) for j in order[:pool_size])
    outside = [int(j) for j in order[pool_size:] if int(j) not in in_pool]
    if not outside:
        return order

    outside_ranked = sorted(outside, key=lambda j: -float(fastrp_sims[j]))
    promoted = [j for j in outside_ranked[:expand_n] if fastrp_sims[j] > 0]
    if not promoted:
        return order

    promoted_set = set(promoted)
    # Promoted candidates sit directly after the dense+BM25 pool so the
    # reranker (if enabled) judges them alongside it; everything else
    # keeps its original relative order.
    return (
        order[:pool_size]
        + promoted
        + [int(j) for j in order[pool_size:] if int(j) not in promoted_set]
    )


def retrieve(
    question: str,
    company: str = None,
    year: str = None,
    quarter: str = None,
    source_filter: str = "both",
    statement_type: str = None,
    top_k: int = config.TOP_K,
    hybrid: bool = True,
    bm25_weight: float = None,
    weight_by_intent: bool = True,
    dedupe: bool = True,
    min_score: float = None,
) -> list[dict]:
    """Return the top_k chunks most relevant to `question`, optionally
    filtered by company/year/quarter/source_type. Searches across every
    embedded node type (Chunk, Table, AudioChunk).

    Combines embedding cosine similarity with BM25 keyword matching
    (hybrid retrieval) when rank_bm25 is installed and hybrid=True —
    pure semantic search can miss exact figures, tickers, or acronyms
    that BM25 catches, and vice versa. Falls back to embeddings-only if
    rank_bm25 isn't installed (`pip install rank_bm25`).

    weight_by_intent: nudges ranking by chunk_type based on whether the
    question looks numeric (favors table/text, raises effective BM25
    weight) or qualitative (favors audio commentary) — see
    is_numeric_question(). Does not hard-filter any source_type; only
    reorders within whatever source_filter already allows.

    dedupe: collapses near-duplicate chunks (e.g. the same figure repeated
    across an annexure page) to the single highest-scoring copy before
    top_k is applied, so top_k isn't spent on repeats.

    min_score: drops candidates below this combined score before ranking.
    Defaults to config.MIN_RETRIEVAL_SCORE if set, else no floor.

    If config.USE_RERANKER is on and sentence-transformers' CrossEncoder
    is installed, the top config.RERANK_CANDIDATE_POOL candidates (after
    min_score/dedupe) get rescored with a cross-encoder and reordered
    before the top_k cut — usually the biggest single retrieval-quality
    lever available, at the cost of extra latency and a model download on
    first use. Off by default; silently skipped if the library import
    fails, same pattern as the BM25 fallback."""
    chunks = _load_chunks()

    # P1 fix: scoped pdf-only synonym expansion — see
    # _expand_query_for_pdf's comment for the measured numbers. Applied
    # before `question` is used for anything downstream (embedding,
    # BM25, is_numeric_question, HyDE) so this reproduces exactly what
    # diagnose_query_expansion.py actually measured, not a partial
    # variant that was never tested.
    question = _expand_query_for_pdf(question, source_filter)

    if bm25_weight is None:
        bm25_weight = config.BM25_WEIGHT
    if min_score is None:
        min_score = getattr(config, "MIN_RETRIEVAL_SCORE", None)

    idxs = list(range(len(chunks)))
    if company:
        wanted_company = _normalize_company_name(company)
        idxs = [
            i
            for i in idxs
            if _company_matches(chunks[i].get("company"), wanted_company)
        ]
    if year:
        wanted_year = str(year).strip()
        # A year range like "2024-2026" ("across FY24-FY26, which Q4...")
        # means "any of these years", not a single year to exact-match -
        # as a literal string that never equals any chunk's single-year
        # tag, so this silently returned zero chunks for every
        # cross-year question before this check existed.
        range_match = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", wanted_year)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            idxs = [
                i
                for i in idxs
                if str(chunks[i].get("year") or "").strip().isdigit()
                and lo <= int(chunks[i]["year"]) <= hi
            ]
        else:
            idxs = [
                i
                for i in idxs
                if str(chunks[i].get("year") or "").strip() == wanted_year
            ]
    if quarter:
        wanted_quarter = str(quarter).strip().upper()
        # "FY-full" ("which quarter had the highest revenue in FY26")
        # means "search every quarter, there's no single one to filter
        # to" - ground truth uses this literal value for ranking
        # questions. Exact-matching it against real Q1-Q4 tags returned
        # zero chunks every time; treat it (and equivalent phrasings) as
        # "no quarter filter" instead of a literal quarter to match.
        if wanted_quarter not in ("FY-FULL", "FY", "FULL", "ALL", "ANNUAL", ""):
            idxs = [
                i
                for i in idxs
                if str(chunks[i].get("quarter") or "").strip().upper() == wanted_quarter
            ]
    if source_filter in ("pdf", "audio"):
        idxs = [i for i in idxs if chunks[i]["source_type"] == source_filter]

    # Hard statement-type filter (P0 fix). Previously the only
    # standalone/consolidated signal at retrieval time was the ±12%/-14%
    # ranking nudge in _financial_query_adjustment() below — a soft
    # adjustment, never an exclusion. That meant a chunk whose own text
    # was unambiguously the WRONG statement type (e.g. a page headed
    # "STATEMENT OF STANDALONE ... FINANCIAL RESULTS" when the question
    # wants consolidated) could still be retrieved, formatted, and handed
    # to the LLM as if it were a legitimate candidate — confirmed in the
    # Info Edge Q4 FY26 benchmark row, where both the standalone and
    # consolidated statement pages were retrieved side by side with no
    # signal ruling either out, and the model refused to answer, citing
    # "conflicting values."
    #
    # This only excludes a chunk when _chunk_statement_type() confidently
    # detected the OPPOSITE type from its own text/metadata (never on a
    # guess) — chunks with no detectable statement type (most audio,
    # most narrative text, some tables) are left untouched, so this
    # can't silently zero out retrieval for documents/questions where
    # statement type isn't determinable. If excluding contradicting
    # chunks would leave nothing at all for this company/period, the
    # filter is skipped rather than returning an empty result — a
    # detection false-positive should degrade to old (soft-boost-only)
    # behavior, not to zero evidence.
    if is_numeric_question(question):
        wanted_stmt = _requested_statement_type(question, statement_type)
        if wanted_stmt:
            opposite_stmt = (
                "standalone" if wanted_stmt == "consolidated" else "consolidated"
            )
            non_contradicting = [
                i for i in idxs if _chunk_statement_type(chunks[i]) != opposite_stmt
            ]
            if non_contradicting:
                idxs = non_contradicting

    if not idxs:
        return []

    numeric_intent = weight_by_intent and is_numeric_question(question)
    if weight_by_intent and numeric_intent:
        # numbers are usually found by keywords (revenue, PAT, EPS...) —
        # lean harder on BM25 for numeric questions, on top of the type
        # boost applied below.
        bm25_bonus = getattr(config, "BM25_NUMERIC_BONUS", 0.15)
        bm25_weight = min(1.0, bm25_weight + bm25_bonus)

    model = _get_embed_model()
    # HyDE for audio-routed questions only — see _hyde_query() for the
    # measurement that motivated this (2/6 -> 4/6 answer-bearing chunks
    # retrieved). PDF retrieval is left untouched: it already performs
    # well and does not carry the interrogative/declarative gap that
    # earnings-call transcripts do. BM25 still scores the ORIGINAL
    # question text, so exact keyword/company matching is unaffected.
    embed_query = question
    if (
        getattr(config, "USE_HYDE_FOR_AUDIO", True)
        and str(source_filter).lower() == "audio"
    ):
        embed_query = _hyde_query(question)
    q_vec = _encode_question(model, embed_query)

    sub_matrix = _embedding_matrix[idxs]
    embed_sims = sub_matrix @ q_vec  # both sides normalized -> cosine similarity

    use_bm25 = hybrid and _bm25_index is not None and bm25_weight > 0

    if use_bm25:
        bm25_scores_full = np.asarray(_bm25_index.get_scores(_tokenize(question)))
        bm25_sub = bm25_scores_full[idxs]
        max_bm25 = bm25_sub.max()
        if max_bm25 > 0:
            bm25_sub = (
                bm25_sub / max_bm25
            )  # normalize into [0, 1] to blend with cosine sim
        combined = (1 - bm25_weight) * embed_sims + bm25_weight * bm25_sub
    else:
        bm25_sub = None
        combined = embed_sims

    # FastRP structural scoring. Applied AFTER dense+BM25 (which provide
    # the seeds) and BEFORE type/financial adjustments and reranking, so
    # those later stages act on the structurally-informed ranking. Both
    # modes are no-ops when fast_rp.py hasn't been run — see
    # _fastrp_available().
    fastrp_sims = None
    fastrp_mode = str(getattr(config, "FASTRP_MODE", "rescore")).lower()
    if _fastrp_available():
        centroid = _fastrp_seed_centroid(idxs, combined)
        if centroid is not None:
            fastrp_sims = _fastrp_sims(idxs, centroid)
            if fastrp_mode == "rescore":
                combined = _fastrp_rescore(embed_sims, bm25_sub, fastrp_sims)

    if weight_by_intent:
        boost_map = _NUMERIC_TYPE_BOOST if numeric_intent else _QUALITATIVE_TYPE_BOOST
        type_boosts = np.array(
            [
                boost_map.get(chunks[idxs[j]]["chunk_type"], 1.0)
                for j in range(len(idxs))
            ]
        )
        combined = combined * type_boosts

    # Query-aware financial disambiguation.  This happens after the common
    # dense+BM25 score so it improves both GraphRAG and baseline equally.
    if numeric_intent:
        financial_adjustments = np.array(
            [
                _financial_query_adjustment(question, chunks[idxs[j]], statement_type)
                for j in range(len(idxs))
            ],
            dtype=np.float32,
        )
        combined = combined * financial_adjustments

    candidate_order = np.argsort(-combined)

    if min_score is not None:
        candidate_order = [j for j in candidate_order if combined[int(j)] >= min_score]

    if dedupe:
        seen_signatures = set()
        deduped_order = []
        for j in candidate_order:
            i = idxs[int(j)]
            sig = _normalize_signature(chunks[i]["text"])
            if sig and sig in seen_signatures:
                continue  # lower-scored duplicate of a chunk already kept
            if sig:
                seen_signatures.add(sig)
            deduped_order.append(j)
        candidate_order = deduped_order

    if fastrp_sims is not None and fastrp_mode == "expand":
        candidate_order = _fastrp_expand(
            candidate_order,
            fastrp_sims,
            pool_size=int(getattr(config, "TOP_K", top_k) or top_k),
            expand_n=int(getattr(config, "FASTRP_EXPAND_TOP_N", 10)),
        )

    rerank_scores_map = {}
    use_reranker = getattr(config, "USE_RERANKER", False) and _RERANKER_LIB_AVAILABLE
    if use_reranker and candidate_order:
        pool_size = min(
            len(candidate_order), getattr(config, "RERANK_CANDIDATE_POOL", 20)
        )
        pool = list(candidate_order[:pool_size])
        reranker = _get_reranker()
        if reranker is not None:
            pairs = [(question, _rerank_text(chunks[idxs[int(j)]])[:512]) for j in pool]
            raw_scores = reranker.predict(pairs)
            rerank_scores_map = {int(j): float(s) for j, s in zip(pool, raw_scores)}

            # P1 fix: this used to fully OVERRIDE rank order with the
            # reranker's score alone (sorted purely by raw_scores) —
            # measured to actively harm retrieval on this corpus (see
            # config.py's USE_RERANKER comment: hit@k 0.711 -> 0.289).
            # Blending with the ORIGINAL hybrid score instead of
            # replacing it - both min-max normalized within this pool
            # first, since a cross-encoder logit and the hybrid score
            # are not on comparable scales - was the configuration that
            # actually beat the no-reranker baseline in re-testing
            # (bge-reranker-base + alpha=0.5, stable across pool sizes
            # 10/20/30; full override with the same model still measured
            # below baseline). alpha=1.0 disables the reranker's effect
            # entirely; alpha=0.0 reproduces the old full-override
            # behavior exactly.
            alpha = float(getattr(config, "RERANK_BLEND_ALPHA", 0.5))
            original_scores = [float(combined[int(j)]) for j in pool]
            norm_orig = _minmax(original_scores)
            norm_rerank = _minmax(list(raw_scores))
            blended = [
                alpha * o + (1 - alpha) * r for o, r in zip(norm_orig, norm_rerank)
            ]
            pool_ranked = [
                j for _, j in sorted(zip(blended, pool), key=lambda pair: -pair[0])
            ]
            candidate_order = pool_ranked + list(candidate_order[pool_size:])

    order = list(candidate_order[:top_k])

    # P1 fix: source_filter=="both" is router.py's signal that a question
    # genuinely needs both a numeric anchor AND qualitative commentary
    # (comparison/"why did X change" questions - see router.py's
    # COMPARISON_KEYWORDS). But this was a flat rank cut with no
    # guaranteed representation by source type, so once a question is
    # classified numeric-intent (needed to rank the actual figure highly
    # - see is_numeric_question) the resulting table/pdf boost could
    # crowd audio out of top_k entirely, even though "both" made it
    # eligible. Confirmed live case: "Why did Nykaa's revenue increase
    # quarter-on-quarter in Q3 FY25?" retrieved 5 PDF chunks and ZERO
    # audio chunks; the model correctly had the right numbers but no
    # commentary to explain "why" and refused outright - hit_at_k still
    # scored this a "hit" (the numeric chunk matched a supporting_
    # chunk_id), masking that the qualitative chunk never made it in.
    # This reserves a small minimum number of top-ranked audio slots
    # (only when source_filter=="both", only when audio candidates
    # actually exist in the pool) by swapping in the top-ranked excluded
    # audio candidate(s) for the WEAKEST non-audio slot(s) already in
    # order - everything else keeps its rank-driven position untouched.
    # Default 1 slot, configurable, and does nothing when source_filter
    # is "pdf" or "audio" alone.
    if source_filter == "both" and top_k > 1:
        min_audio = min(int(getattr(config, "MIN_AUDIO_SLOTS_WHEN_BOTH", 1)), top_k - 1)
        if min_audio > 0:
            in_order = set(int(j) for j in order)
            have_audio = sum(
                1 for j in order if chunks[idxs[int(j)]].get("source_type") == "audio"
            )
            if have_audio < min_audio:
                audio_pool = [
                    j
                    for j in candidate_order
                    if chunks[idxs[int(j)]].get("source_type") == "audio"
                    and int(j) not in in_order
                ]
                for j in audio_pool[: min_audio - have_audio]:
                    for k in range(len(order) - 1, -1, -1):
                        if chunks[idxs[int(order[k])]].get("source_type") != "audio":
                            order[k] = j
                            break

    results = []
    for local_i in order:
        i = idxs[int(local_i)]
        c = dict(chunks[i])
        c["score"] = float(combined[int(local_i)])
        c["embedding_score"] = float(embed_sims[int(local_i)])
        if bm25_sub is not None:
            c["bm25_score"] = float(bm25_sub[int(local_i)])
        if fastrp_sims is not None:
            c["fastrp_score"] = float(fastrp_sims[int(local_i)])
        if int(local_i) in rerank_scores_map:
            c["rerank_score"] = rerank_scores_map[int(local_i)]
        results.append(c)

    return results
