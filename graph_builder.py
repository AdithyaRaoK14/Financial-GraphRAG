"""
graph_builder.py
================

Builds the Neo4j knowledge graph from structured chunks.

Pipeline

Chunks
    ↓
Qwen
    ↓
Entities
    ↓
Relationships
    ↓
Neo4j

Design notes
------------
- One Neo4j session per document (quarter), not one per write. Every
  write within that session is its own explicit transaction
  (session.execute_write), and chunk writes are batched with UNWIND
  instead of one round-trip per chunk.
- Dynamic relationship names (LLM output) are validated against
  [A-Z0-9_]+ before being interpolated into Cypher.
- Every LLM JSON response is parsed defensively — malformed output from
  one chunk logs a warning and yields an empty result instead of taking
  down the whole build.
- Multiple LLM calls happen per chunk (entities, metrics, guidance, risk,
  sentiment). That's the dominant runtime cost on large reports; if it
  becomes a bottleneck, the next step is combining these into one
  structured prompt per chunk rather than five.
- Chunks within a quarter are processed by a small pool of worker
  threads (config.GRAPH_BUILDER_WORKERS), each opening its own Neo4j
  session — the driver is thread-safe, sessions are not. Threads (not
  processes) are enough here since the wait is on network I/O to
  Ollama/Neo4j, not CPU-bound Python.
- Oversized chunks are split into pieces and each piece extracted
  separately (results merged), instead of being truncated and losing
  everything past the cutoff.
- Chunks whose exact text was already seen (common across quarters —
  Safe Harbor pages, boilerplate accounting policy notes) reuse the
  cached extraction instead of another LLM call; a short chunk that's
  ~entirely a known disclaimer/nav marker is skipped with no LLM call
  at all.
- progress.json is written after every completed chunk by default
  (PROGRESS_SAVE_EVERY=1) — a chunk-level LLM call is far more
  expensive than one small JSON write, so batching the writes wasn't
  worth the resumability it gives up. Configurable via
  config.PROGRESS_SAVE_EVERY if that trade-off ever flips.
- GRAPH_BUILDER_WORKERS > 1 only helps if the Ollama *server* is also
  configured to serve requests in parallel (OLLAMA_NUM_PARALLEL,
  usually set to at least the worker count on whatever host/process
  is running `ollama serve`). If it isn't, Ollama queues requests
  internally and extra worker threads just wait on each other instead
  of getting concurrent LLM throughput. This is server-side config,
  not something graph_builder.py can set — build_all() only logs a
  best-effort warning if it can't confirm it's configured (see
  _warn_if_ollama_not_parallel()).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase
import ollama

import config
import table_metrics
import temporal_utils

logger = logging.getLogger(__name__)

# Cypher relationship types can't be parameterized, so any relation name
# that comes from LLM output gets validated against this before being
# interpolated into a query string.
RELATION_NAME_RE = re.compile(r"^[A-Z0-9_]+$")

# Chunks that are almost entirely legal/administrative boilerplate carry
# no extractable financial signal, but still cost a full LLM call (tens
# of seconds) if sent through normally. A chunk is only skipped when one
# of these markers appears AND the chunk is short-ish (see
# _is_boilerplate) — a long chunk that happens to mention "Safe Harbor"
# in passing while also discussing real numbers is NOT skipped.
BOILERPLATE_MARKERS = (
    "safe harbor statement",
    "forward-looking statements",
    "forward looking statements",
    "table of contents",
    "disclaimer",
)

# Below this length, a marker match is treated as "this chunk IS the
# boilerplate" rather than "this chunk mentions boilerplate in passing".
# Tune via config if a report's disclaimer pages run longer/shorter.
BOILERPLATE_MAX_CHARS = getattr(config, "BOILERPLATE_MAX_CHARS", 600)

# Chunks at or below this length essentially never carry extractable
# graph information — page headers/footers like "Page 2", section
# markers like "Contents", stray running heads, etc. Skipped with no
# LLM call, same as boilerplate. This is independent of the marker-based
# boilerplate check above: it doesn't look for specific phrases, just
# treats "too short to say anything" as its own reason to skip.
TINY_CHUNK_MAX_CHARS = getattr(config, "TINY_CHUNK_MAX_CHARS", 15)


def _progress_bar(current, total, width=20):
    if not total:
        return "░" * width
    filled = int(width * current / total)
    return "█" * filled + "░" * (width - filled)


def _warn_if_ollama_not_parallel(workers):
    """Best-effort sanity check, not a guarantee.

    GRAPH_BUILDER_WORKERS > 1 only buys concurrent LLM throughput if
    the Ollama *server* is also configured (via OLLAMA_NUM_PARALLEL)
    to handle that many requests at once — otherwise it serializes
    them internally and the extra worker threads just queue.

    This process can only see OLLAMA_NUM_PARALLEL if it happens to be
    set in *this* environment, which won't be the case if `ollama
    serve` is running elsewhere (another shell, a systemd service, a
    different container). So: warn on anything short of positive
    confirmation, but phrase it as "couldn't confirm", not "it's
    wrong" — this check can't tell the two apart.
    """
    if workers <= 1:
        return
    raw = os.environ.get("OLLAMA_NUM_PARALLEL")
    if raw is None:
        logger.warning(
            "GRAPH_BUILDER_WORKERS=%d but OLLAMA_NUM_PARALLEL isn't set "
            "in this process's environment. If the Ollama server wasn't "
            "started with OLLAMA_NUM_PARALLEL>=%d (e.g. in whatever shell "
            "runs `ollama serve`), it may serve chunks one at a time "
            "regardless of worker count. Can't confirm either way from "
            "here — just flagging it.",
            workers,
            workers,
        )
        return
    try:
        configured = int(raw)
    except ValueError:
        return
    if configured < workers:
        logger.warning(
            "GRAPH_BUILDER_WORKERS=%d but OLLAMA_NUM_PARALLEL=%d — the "
            "Ollama server may bottleneck extraction below what the "
            "worker count implies.",
            workers,
            configured,
        )


class GraphBuilder:
    # Shared building blocks. Kept short deliberately — this is what
    # actually determines output quality (schema, entity types, table/
    # OCR/period rules, "return only JSON"); the long example lists and
    # repeated wording that used to surround this were prompt padding,
    # not signal. See _JSON_SCHEMA / ENTITY_TYPES / _TABLE_AND_OCR_RULES
    # below for what's reused between the text and table prompts.
    #
    # Relationships (free-text "NVIDIA DEVELOPS Blackwell" extraction) are
    # off by default — see config.EXTRACT_RELATIONSHIPS — because it's
    # one of the least reliable things to ask a 7B model for, and the
    # graph's real structural relationships (Company/Quarter/Metric/
    # Risk/Guidance) are already built deterministically from the schema,
    # not inferred from LLM output. It's a toggle, not a deletion: set
    # config.EXTRACT_RELATIONSHIPS = True to turn it back on for entity-
    # to-entity connections beyond that deterministic structure (e.g.
    # "NVIDIA DEVELOPS Blackwell") without touching any code here.
    # TEXT_PROMPT/TABLE_PROMPT/AUDIO_PROMPT are properties (not fixed
    # strings) so they pick this up from config at call time. TABLE_PROMPT
    # never includes relationships regardless of the flag — a raw table
    # has entities and numbers, not the kind of prose a relationship
    # would come from.

    _JSON_SCHEMA = """{
  "entities": [{"name": "NVIDIA", "type": "Company"}],
  "metrics": [{"name": "Revenue", "value": "26.3", "unit": "Billion USD", "period": "Q1 FY2025"}],
  "guidance": [{"topic": "Revenue", "statement": "Revenue expected to grow in H2", "timeframe": "H2 FY2025"}],
  "risks": [{"type": "Supply Chain", "statement": "Supply chain constraints remain."}],
  "sentiment": {"sentiment": "Positive", "confidence": 0.91}
}"""

    _JSON_SCHEMA_WITH_RELATIONSHIPS = """{
  "entities": [{"name": "NVIDIA", "type": "Company"}],
  "relationships": [{"source": "NVIDIA", "target": "Blackwell", "relation": "DEVELOPS"}],
  "metrics": [{"name": "Revenue", "value": "26.3", "unit": "Billion USD", "period": "Q1 FY2025"}],
  "guidance": [{"topic": "Revenue", "statement": "Revenue expected to grow in H2", "timeframe": "H2 FY2025"}],
  "risks": [{"type": "Supply Chain", "statement": "Supply chain constraints remain."}],
  "sentiment": {"sentiment": "Positive", "confidence": 0.91}
}"""

    _RELATIONSHIPS_DISABLED_RULE = """- Do NOT extract relationships between entities (e.g. "NVIDIA DEVELOPS
  Blackwell") — there is no "relationships" key in the format above.
  Free-text relationship extraction is one of the least reliable things
  asked of a 7B model, so it's off by default; set
  config.EXTRACT_RELATIONSHIPS = True to turn it back on."""

    _RELATIONSHIPS_ENABLED_RULE = """- relationships = notable factual relationships between two entities you
  already listed in "entities" (e.g. NVIDIA -[DEVELOPS]-> Blackwell).
  Keep "relation" a short SCREAMING_SNAKE_CASE verb phrase. Only extract
  ones you're confident about — when unsure, omit rather than guess.
  Deterministic graph structure (Company/Quarter/Metric/Risk/Guidance)
  is still built directly from the schema, not from this list; this is
  only for extra entity-to-entity connections beyond that."""

    ENTITY_TYPES = (
        "Company, Person, Organization, Subsidiary, Product, Technology, "
        "Metric, Risk, Guidance, Country, Region, Industry, Currency, "
        "Competitor"
    )

    # OCR + period-extraction + table rules. These directly affect
    # extraction quality on financial statements, so they're kept in
    # full (not shortened) and reused by both prompts below.
    _TABLE_AND_OCR_RULES = """- Financial statement line items are often OCR-extracted with NO
  repeated column headers — a single line may read like:
    "Cash and cash equivalents 763.02 520.09 1,032.45"
  where the three numbers are three DIFFERENT time periods for the
  SAME metric (e.g. current quarter, same quarter last year, and prior
  year-end), not three different metrics. Look earlier in the text for
  the period labels these numbers belong to (e.g. "As at September 30,
  2025" / "As at September 30, 2024" / "As at March 31, 2025") and
  extract EACH number as its own metric entry with that period filled
  in — do not skip a line just because it has multiple numbers or
  minor OCR noise (e.g. "8. 27" instead of "8.27").
- Financial tables are column-based: each column is usually one
  reporting period. If a row reports N numeric values for one line
  item, that is N periods of the SAME metric — return N separate
  metric entries, never one entry holding a list of values.
- Only fill in "period" when its label is explicitly present in the
  text (a column header, "As at ...", "Q1 FY2026", etc.); otherwise
  leave "period" as an empty string rather than guessing.
- Only fill in "unit" with a currency/unit explicitly stated in the
  text. If the text says "₹ in Crore", use "Crore INR" for every
  metric from that table. Never default to "USD" or any other
  currency when none is stated."""

    @property
    def _schema(self):
        return (
            self._JSON_SCHEMA_WITH_RELATIONSHIPS
            if getattr(config, "EXTRACT_RELATIONSHIPS", False)
            else self._JSON_SCHEMA
        )

    @property
    def _relationship_rule(self):
        return (
            self._RELATIONSHIPS_ENABLED_RULE
            if getattr(config, "EXTRACT_RELATIONSHIPS", False)
            else self._RELATIONSHIPS_DISABLED_RULE
        )

    @property
    def TEXT_PROMPT(self):
        return f"""You are an expert financial analyst and knowledge graph extractor.

Extract structured information from the financial text below and return
ONLY valid JSON — no explanation, no markdown fences.

Format:

{self._schema}

Entity types:
{self.ENTITY_TYPES}

Rules:
- guidance = ONLY forward-looking statements.
- risks = ONLY business risks.
- sentiment = overall management/tone sentiment for this text, or
  {{"sentiment": null, "confidence": null}} if not applicable.
- If a category has nothing to extract, return an empty list for it —
  never omit a key.
{self._relationship_rule}
{self._TABLE_AND_OCR_RULES}

No explanation. Return only the JSON object above."""

    # Table chunks get their own prompt, but as of graph_builder.py's
    # deterministic table-metrics path (see table_metrics.py), most table
    # chunks never reach an LLM at all — headers/rows are already
    # structured data, and Python can align "row label + column period ->
    # value" without asking a model to re-derive something it can already
    # see. TABLE_PROMPT is now a FALLBACK, used only when
    # table_metrics.extract_metrics_from_table() returns nothing (e.g. a
    # malformed/irregular table pdfplumber didn't extract cleanly) — see
    # process_chunk_full(). Always uses the no-relationships schema,
    # regardless of config.EXTRACT_RELATIONSHIPS — a raw table isn't
    # prose a relationship would sensibly come from.
    @property
    def TABLE_PROMPT(self):
        return f"""You are an expert financial analyst and knowledge graph extractor.

This table couldn't be parsed automatically from its raw rows/columns
(unusual layout or extraction noise), so you're seeing it as text instead.
Extract structured information from it and return ONLY valid JSON — no
explanation, no markdown fences.

Format:

{self._JSON_SCHEMA}

Only populate "entities" and "metrics" — this is a table, not narrative
text. Always return empty lists for "guidance" and "risks", and
{{"sentiment": null, "confidence": null}} for "sentiment"; do not infer or
invent them from a table.

Entity types:
{self.ENTITY_TYPES}

Rules:
- If a category has nothing to extract, return an empty list for it —
  never omit a key.
{self._TABLE_AND_OCR_RULES}

No explanation. Return only the JSON object above."""

    # Audio (earnings-call transcript) chunks get their own prompt too:
    # no OCR/table rules (nothing here was scanned from a PDF), and
    # metrics are deliberately EXCLUDED — the PDF financial statements
    # are the source of truth for reported figures. When management
    # says "revenue grew 12%" on the call, that's colour on a number the
    # PDF already has precisely; asking Qwen to also extract it from
    # audio risks a second, slightly different value for the same metric
    # (a transcription/rounding mismatch) that graph_rag_pipeline.py then
    # has to reconcile. Audio is for the qualitative side PDFs don't
    # capture well: guidance, risk, sentiment, and who-said-what.
    @property
    def AUDIO_PROMPT(self):
        return f"""You are an expert financial analyst and knowledge graph extractor,
listening to an earnings call transcript.

Extract structured information from the transcript segment below and
return ONLY valid JSON — no explanation, no markdown fences.

Format:

{self._schema}

Entity types:
{self.ENTITY_TYPES}

Rules:
- "metrics" must ALWAYS be an empty list. The PDF financial statements
  are the source of truth for reported figures — do not extract
  numbers/percentages mentioned verbally here, even if management states
  them explicitly, to avoid a second, possibly-mismatched value for the
  same metric.
- entities = people (executives, analysts by name if stated), products,
  competitors, and partners mentioned — not numeric figures.
- guidance = ONLY forward-looking statements (outlook, targets, planned
  launches, strategic direction).
- risks = ONLY business risks, challenges, or headwinds mentioned.
- sentiment = overall management/tone sentiment for this segment, or
  {{"sentiment": null, "confidence": null}} if not applicable (e.g. a
  purely procedural operator segment).
- If a category has nothing to extract, return an empty list for it —
  never omit a key.
{self._relationship_rule}

No explanation. Return only the JSON object above."""

    # Kept as an alias — extract_all() picks TEXT_PROMPT or TABLE_PROMPT
    # per chunk_type now, but anything still referencing COMBINED_PROMPT
    # directly (e.g. a stray script) keeps working against the text
    # prompt, which is the closest equivalent to the old combined one.
    @property
    def COMBINED_PROMPT(self):
        return self.TEXT_PROMPT

    def __init__(self, uri, username, password, model="qwen2.5:7b"):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = model

        # Dedicated client (not the bare ollama.chat() module-level
        # function, which has no way to pass a timeout) with a real
        # HTTP-level timeout — forwarded straight to httpx.Client, so it
        # actually aborts the request when exceeded, rather than just
        # abandoning a Python-side wait while the server keeps
        # generating. Extraction (_extract_json) is the caller that
        # needs this: a production run hit 3 attempts at 750-900s EACH
        # (~42 minutes) on one large table-fallback chunk with no
        # timeout at all. Safe to share across build_all()'s worker
        # threads — httpx.Client is documented thread-safe for
        # concurrent requests.
        self._ollama_client = ollama.Client(
            host=getattr(config, "OLLAMA_HOST", "http://localhost:11434"),
            timeout=getattr(config, "OLLAMA_EXTRACTION_TIMEOUT_SEC", 240),
        )
        # Chunks whose LLM extraction ultimately failed (unparseable JSON
        # after all retries, or a non-dict response) or hit an unhandled
        # exception during processing. Populated by _record_failed_chunk()
        # and flushed to results/failed_chunks.json via save_failed_chunks().
        self.failed_chunks = []
        # Guards self.failed_chunks and progress dict mutation now that
        # process_chunk_full() can be called from multiple worker threads
        # (see build_all()'s ThreadPoolExecutor).
        self._state_lock = threading.Lock()

        # extract_all() result cache, keyed by sha256 of the chunk text
        # actually sent to the LLM. Quarterly reports repeat entire pages
        # (Safe Harbor, company description, accounting policy notes)
        # near-verbatim across quarters — an exact text match reuses the
        # prior JSON instead of paying for another ollama.chat() call.
        # Loaded from / flushed to results/extraction_cache.json so the
        # cache also pays off across separate runs, not just within one.
        self._extraction_cache = {}

        # Per-text-hash locks so that if two worker threads hit a cache
        # miss for the *same* text_hash at (almost) the same time, only
        # one of them actually calls extract_all() — the other waits on
        # the lock and then reuses the result the first thread cached,
        # instead of both paying for a redundant Qwen call. Guarded by
        # _state_lock since dict access itself needs to be thread-safe.
        self._extraction_locks = {}

    # ----------------------------------------------------
    # Extraction cache (see _extraction_cache above)
    # ----------------------------------------------------

    @staticmethod
    def _extraction_cache_path():
        return config.RESULTS_DIR / "extraction_cache.json"

    def load_extraction_cache(self):
        """Call once at startup, mirroring load_progress()."""
        path = self._extraction_cache_path()
        if not path.exists():
            return
        try:
            self._extraction_cache = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read existing %s — starting fresh.", path)
            self._extraction_cache = {}

    def save_extraction_cache(self):
        path = self._extraction_cache_path()
        path.write_text(
            json.dumps(self._extraction_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _hash_text(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    _WHITESPACE_RE = re.compile(r"\s+")

    @classmethod
    def _normalize_whitespace(cls, text):
        """Collapses runs of whitespace (spaces, tabs, newlines) to a
        single space and strips the ends. Used ONLY when computing the
        extraction-cache key — two chunks that differ solely in
        whitespace (re-wrapped PDF extraction, trailing blank lines,
        etc.) now hit the same cache entry instead of missing on a
        byte-for-byte difference that carries no meaning. The text sent
        to the LLM itself is untouched, so this can't change what the
        model sees or produces — it only changes which chunks share a
        cache hit."""
        return cls._WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def _is_boilerplate(text):
        """True only when the chunk is SHORT and ~entirely a known
        disclaimer/nav marker — never skips a long chunk just because
        it contains one of these phrases in passing (e.g. a narrative
        paragraph that references "forward-looking statements" while
        also discussing real guidance numbers still gets extracted)."""
        if len(text) > BOILERPLATE_MAX_CHARS:
            return False
        lowered = text.lower()
        return any(marker in lowered for marker in BOILERPLATE_MARKERS)

    @staticmethod
    def _is_tiny(text):
        """True when there's just not enough text here to extract
        anything — running heads, page numbers, bare section markers
        ("Page 2", "Contents"). Independent of _is_boilerplate: no
        phrase match required, just a length floor."""
        return len(text.strip()) <= TINY_CHUNK_MAX_CHARS

    def close(self):
        self.driver.close()

    # ----------------------------------------------------
    # LLM extraction (JSON-parsing is defensive: malformed output from
    # one chunk logs a warning and returns {} instead of raising)
    # ----------------------------------------------------

    # qwen2.5:7b frequently wraps JSON in ```json ... ``` fences despite
    # being told not to — this strips a leading/trailing fence line if
    # present. Safe no-op on already-clean output.
    _FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

    @staticmethod
    def _attempt_json_repair(text: str) -> str | None:
        """Best-effort repair for JSON that was cut off mid-generation —
        i.e. num_predict was hit before the model finished, not a
        genuinely malformed/hallucinated document. With format="json"
        constraining decoding (see _extract_json below), truncation is
        overwhelmingly the actual failure mode: the model can't emit a
        stray comma or a misplaced bracket, it can only stop early. A
        production run hit this on a single large table fallback —
        3 full retries at 750-900s EACH (the same truncation point every
        time, since nothing about the input changed) for output this
        function repairs in milliseconds.

        Walks the string tracking bracket/string nesting, and the last
        point where a complete top-level array element ended (a comma
        appearing with the innermost open bracket being '[', i.e.
        between array elements, not mid-object) — that point is safe to
        cut at without losing a partially-written element. Everything
        after it is discarded, and every bracket still open at that
        point is closed. Returns None (repair not attempted, caller
        falls through to retrying the model) if no such safe point was
        found — e.g. the output was truncated before even one array
        element completed, or wasn't truncated JSON at all.

        This does NOT attempt to fix arbitrary syntax errors (a genuine
        misplaced comma, an unescaped quote elsewhere in the document) —
        only this specific, extremely common truncation shape. The
        caller still validates the result with json.loads(); a
        mis-repair simply fails that check and falls through to a
        normal retry, same as if this function didn't exist.
        """
        stack: list[str] = []
        in_string = False
        escape = False
        last_safe_cut = None
        stack_at_cut = None

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
            elif ch == "," and stack and stack[-1] == "[":
                last_safe_cut = i
                stack_at_cut = list(stack)

        if last_safe_cut is None or stack_at_cut is None:
            return None

        repaired = text[:last_safe_cut].rstrip()
        closers = {"{": "}", "[": "]"}
        for opener in reversed(stack_at_cut):
            repaired += closers[opener]
        return repaired

    def _extract_json(self, prompt, text, max_retries=2):
        last_content = None

        for attempt in range(max_retries + 1):
            try:
                start = time.perf_counter()

                response = self._ollama_client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    options={
                        "num_ctx": config.OLLAMA_NUM_CTX,
                        # Extraction JSON (entities/metrics/guidance/
                        # risks/sentiment together) tends to run larger
                        # than other callers, hence the dedicated budget
                        # rather than the shared default.
                        "num_predict": config.OLLAMA_EXTRACTION_NUM_PREDICT,
                    },
                    # Constrains decoding so the model MUST emit
                    # syntactically valid JSON — doesn't guarantee our
                    # schema (still handled by the isinstance(dict) check
                    # below), but it stops the model from ignoring the
                    # instruction entirely and writing prose instead,
                    # which is what was happening on large table chunks.
                    format="json",
                )

                logger.info(
                    "ollama.chat() took %.1f s (attempt %d/%d, %d chars in)",
                    time.perf_counter() - start,
                    attempt + 1,
                    max_retries + 1,
                    len(text),
                )

                content = response["message"]["content"]
            except Exception:
                # Covers connection errors AND self._ollama_client's HTTP
                # timeout (an httpx timeout exception) — both are
                # "this attempt didn't produce output in time", handled
                # identically: log, retry if attempts remain.
                logger.warning(
                    "LLM call failed after %.1f s (attempt %d/%d)",
                    time.perf_counter() - start,
                    attempt + 1,
                    max_retries + 1,
                    exc_info=True,
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                self._last_extraction_ok = False
                self._last_raw_response = None
                return {}

            last_content = content
            cleaned = self._FENCE_RE.sub("", content).strip()

            try:
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"Expected a JSON object, got {type(parsed).__name__}"
                    )
                self._last_extraction_ok = True
                self._last_raw_response = None
                return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                # Try an in-process repair before burning a full retry
                # (another ollama.chat() call, i.e. potentially several
                # more minutes) — see _attempt_json_repair()'s docstring.
                # This is the fix for the exact case that cost 42 minutes
                # in production: 3 retries that each truncated at the
                # same point, for output this repairs in milliseconds.
                repaired = self._attempt_json_repair(cleaned)
                if repaired is not None:
                    try:
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            logger.info(
                                "Repaired truncated JSON output in-process "
                                "(attempt %d/%d) — no retry needed",
                                attempt + 1,
                                max_retries + 1,
                            )
                            self._last_extraction_ok = True
                            self._last_raw_response = None
                            return parsed
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass  # repair didn't produce valid JSON — fall through to retry

                if attempt < max_retries:
                    logger.info(
                        "Retrying unparseable LLM output (attempt %d/%d)",
                        attempt + 1,
                        max_retries + 1,
                    )
                    time.sleep(2)
                    continue

        logger.warning(
            "Could not parse LLM JSON output after %d attempts: %r",
            max_retries + 1,
            (last_content or "")[:200],
        )
        self._last_extraction_ok = False
        self._last_raw_response = last_content
        return {}

    @staticmethod
    def _split_text(text, max_chars):
        """Split oversized text into <= max_chars pieces, breaking on a
        paragraph boundary near the limit when one exists so a table row
        or sentence isn't sliced in half. Used instead of truncating so
        content past the old cutoff still gets extracted rather than
        silently dropped."""
        if len(text) <= max_chars:
            return [text]

        pieces = []
        remaining = text
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            split_at = window.rfind("\n\n")
            if split_at < max_chars * 0.5:  # too early / no good break
                split_at = window.rfind("\n")
            if split_at < max_chars * 0.5:
                split_at = max_chars  # no reasonable boundary — hard cut
            pieces.append(remaining[:split_at])
            remaining = remaining[split_at:]
        if remaining.strip():
            pieces.append(remaining)
        return pieces

    def extract_all(self, text, chunk_type="text"):
        """One LLM call per (piece of) text, returning entities, metrics,
        guidance, risks, and sentiment together (relationships are no
        longer requested — see TEXT_PROMPT's rules). Missing keys default
        to empty so callers don't need to special-case a partial/
        malformed response.

        chunk_type picks the prompt:
          - "table": TABLE_PROMPT (entities + metrics only). This is a
            FALLBACK now — most table chunks are handled by
            table_metrics.extract_metrics_from_table() in
            process_chunk_full() and never reach this method at all; it
            only runs when that deterministic parse comes back empty.
          - "audio": AUDIO_PROMPT (no metrics — the PDF is the source of
            truth for reported figures; audio is qualitative-only).
          - anything else ("text"): TEXT_PROMPT, the full schema.
        Same JSON schema shape either way, so merging below doesn't need
        to special-case which prompt ran.

        Oversized chunks are split into multiple pieces (see _split_text)
        rather than truncated, and each piece's extraction is merged —
        content past the old GRAPH_EXTRACTION_MAX_CHARS cutoff is no
        longer silently dropped. This costs one extra LLM call per extra
        piece, so it trades some speed for completeness on the (rare)
        oversized chunks; most chunks are a single piece and unaffected.
        """
        if chunk_type == "table":
            prompt = self.TABLE_PROMPT
        elif chunk_type == "audio":
            prompt = self.AUDIO_PROMPT
        else:
            prompt = self.TEXT_PROMPT
        pieces = self._split_text(text, config.GRAPH_EXTRACTION_MAX_CHARS)
        if len(pieces) > 1:
            logger.info(
                "Splitting %d-char chunk into %d pieces for extraction "
                "(was truncated before)",
                len(text),
                len(pieces),
            )

        merged = {
            "entities": [],
            # No longer requested in any prompt (see TEXT_PROMPT's rules —
            # free-text relationship extraction was dropped as one of the
            # least reliable things a 7B model was being asked to do).
            # Kept in the merged shape and the write path below
            # (_tx_write_entities_and_relationships) as a defensive
            # no-op rather than ripped out — if a prompt ever asks for it
            # again, or a model hallucinates the key anyway, this still
            # handles it instead of KeyError-ing.
            "relationships": [],
            "metrics": [],
            "guidance": [],
            "risks": [],
            "sentiment": {},
        }
        all_ok = True
        last_raw = None

        for piece in pieces:
            result = self._extract_json(prompt, piece)
            if not isinstance(result, dict):
                logger.warning(
                    "extract_all got a non-dict result (%s) after retries "
                    "— treating as empty.",
                    type(result).__name__,
                )
                result = {}

            piece_ok = getattr(self, "_last_extraction_ok", True)
            all_ok = all_ok and piece_ok
            if not piece_ok:
                last_raw = getattr(self, "_last_raw_response", None)

            merged["entities"].extend(result.get("entities") or [])
            merged["relationships"].extend(result.get("relationships") or [])
            merged["metrics"].extend(result.get("metrics") or [])
            merged["guidance"].extend(result.get("guidance") or [])
            merged["risks"].extend(result.get("risks") or [])
            # Sentiment is a single value, not a list — keep the first
            # piece that actually has one rather than the last, since
            # opening paragraphs are more likely to carry overall tone.
            piece_sentiment = result.get("sentiment") or {}
            if piece_sentiment.get("sentiment") and not merged["sentiment"].get(
                "sentiment"
            ):
                merged["sentiment"] = piece_sentiment

        merged["_extraction_ok"] = all_ok
        merged["_raw_response"] = last_raw
        return merged

    # ----------------------------------------------------
    # Failed-chunk tracking — so a bad LLM response or unhandled
    # exception costs you one retry later instead of a full rebuild.
    # ----------------------------------------------------

    def _record_failed_chunk(
        self,
        chunk,
        company,
        year,
        quarter,
        raw_response,
        error="Could not parse LLM JSON output after retries",
    ):
        entry = {
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type"),
            "company": company,
            "year": year,
            "quarter": quarter,
            "text": chunk.get("embedding_text") or chunk.get("text", ""),
            # Saved so retry_failed_chunks.py / reprocess_low_metric_
            # chunks.py can reconstruct a chunk dict that still carries
            # provenance — without these, a retry would stamp
            # source_document/source_page as None onto Metric/Guidance/
            # Risk nodes, silently overwriting the correct values from
            # the original (failed) attempt with nothing.
            "document_name": chunk.get("document_name"),
            "page": chunk.get("page"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "raw_response": raw_response,
            "error": error,
        }
        with self._state_lock:
            self.failed_chunks.append(entry)

    def save_failed_chunks(self):
        """Merge self.failed_chunks into results/failed_chunks.json,
        keyed by chunk_id (a later failure for the same chunk overwrites
        an earlier one). Safe to call even if nothing failed this run —
        it's then a no-op and leaves any existing file untouched."""
        if not self.failed_chunks:
            return

        path = config.RESULTS_DIR / "failed_chunks.json"

        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read existing %s — starting fresh.", path)
                existing = []

        by_id = {c["chunk_id"]: c for c in existing if c.get("chunk_id")}
        for c in self.failed_chunks:
            if c.get("chunk_id"):
                by_id[c["chunk_id"]] = c

        merged = list(by_id.values())
        path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.warning(
            "Saved %d failed chunk(s) (this run: %d) to %s",
            len(merged),
            len(self.failed_chunks),
            path,
        )

    # ----------------------------------------------------
    # Resume support — a small progress.json (same pattern as
    # failed_chunks.json) tracking which chunks in which quarter have
    # already had their LLM extraction + Neo4j write done. Neo4j writes
    # themselves are all MERGE-based and safe to redo, so the only thing
    # worth skipping on a rerun is the expensive ollama.chat() call.
    #
    # Structure on disk:
    #   {
    #     "completed_quarters": ["Company|2026|Q1", ...],
    #     "in_progress": {"Company|2026|Q2": ["chunk_id1", "chunk_id2"]}
    #   }
    #
    # A quarter moves from in_progress to completed_quarters once every
    # chunk in it has been attempted and link_pdf_audio has run — after
    # that, build_all() skips the whole quarter without even opening its
    # chunk JSON files. While a quarter is still in progress, only the
    # chunk_ids already in its list are skipped; everything else in that
    # quarter is (re)processed normally.
    # ----------------------------------------------------

    @staticmethod
    def _progress_path():
        return config.RESULTS_DIR / "progress.json"

    def load_progress(self):
        """Call once at the start of build_all(). Returns a fresh,
        empty structure if no progress file exists yet or it's
        unreadable — same "start clean rather than crash" behavior as
        save_failed_chunks()."""
        path = self._progress_path()
        if not path.exists():
            return {"completed_quarters": set(), "in_progress": {}}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read existing %s — starting fresh.", path)
            return {"completed_quarters": set(), "in_progress": {}}
        return {
            "completed_quarters": set(raw.get("completed_quarters", [])),
            "in_progress": {k: set(v) for k, v in raw.get("in_progress", {}).items()},
        }

    def save_progress(self, progress):
        """Overwrites progress.json with the full current state. Called
        after every chunk (not just at the end of a quarter) so a
        Ctrl+C mid-run loses at most the one chunk that was in flight —
        the write itself is a small JSON file, negligible next to a
        30+ second LLM call."""
        path = self._progress_path()
        serializable = {
            "completed_quarters": sorted(progress["completed_quarters"]),
            "in_progress": {k: sorted(v) for k, v in progress["in_progress"].items()},
        }
        path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ----------------------------------------------------
    # Constraints
    # ----------------------------------------------------

    def create_constraints(self, session):
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Quarter) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (x:Chunk) REQUIRE x.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:AudioChunk) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Guidance) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Risk) REQUIRE r.id IS UNIQUE",
        ]
        for q in queries:
            session.run(q)

    # ----------------------------------------------------
    # Company / Quarter
    # ----------------------------------------------------

    def create_company(self, session, company):
        session.execute_write(self._tx_create_company, company)

    @staticmethod
    def _tx_create_company(tx, company):
        tx.run("MERGE (c:Company{name:$name})", name=company)

    def create_quarter(self, session, company, year, quarter):
        qid = f"{company}_{year}_{quarter}"
        session.execute_write(self._tx_create_quarter, company, qid, year, quarter)
        return qid

    @staticmethod
    def _tx_create_quarter(tx, company, qid, year, quarter):
        tx.run(
            """
            MATCH (c:Company{name:$company})
            MERGE (q:Quarter{id:$id})
            SET
                q.company=$company,
                q.year=$year,
                q.quarter=$quarter
            MERGE (c)-[:HAS_QUARTER]->(q)
            """,
            company=company,
            id=qid,
            year=year,
            quarter=quarter,
        )

    # ----------------------------------------------------
    # Chunk writes — grouped by type and written with UNWIND, one
    # transaction per type per document instead of one per chunk.
    # Every node type stores document_name, matching the PDF schema.
    # ----------------------------------------------------

    def store_chunks(self, session, quarter_id, chunks):
        text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
        table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
        audio_chunks = [c for c in chunks if c["chunk_type"] == "audio"]

        known = len(text_chunks) + len(table_chunks) + len(audio_chunks)
        if known != len(chunks):
            unknown_types = sorted(
                {
                    c["chunk_type"]
                    for c in chunks
                    if c["chunk_type"] not in ("text", "table", "audio")
                }
            )
            raise ValueError(f"Unknown chunk type(s): {unknown_types}")

        if text_chunks:
            session.execute_write(self._tx_write_text_chunks, quarter_id, text_chunks)
        if table_chunks:
            session.execute_write(self._tx_write_table_chunks, quarter_id, table_chunks)
        if audio_chunks:
            session.execute_write(self._tx_write_audio_chunks, quarter_id, audio_chunks)

    def store_chunk(self, session, quarter_id, chunk):
        """Single-chunk convenience wrapper around store_chunks()."""
        self.store_chunks(session, quarter_id, [chunk])

    @staticmethod
    def _tx_write_text_chunks(tx, quarter_id, chunks):
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            UNWIND $rows AS row
            MERGE (c:Chunk{id: row.id})
            SET
                c.type="text",
                c.text=row.text,
                c.page=row.page,
                c.page_type=row.page_type,
                c.section=row.section,
                c.heading=row.heading,
                c.document_name=row.document_name,
                c.source_file_hash=row.source_file_hash,
                c.processed_at=row.processed_at,
                c.embedding_text=row.embedding_text
            MERGE (q)-[:HAS_CHUNK]->(c)
            """,
            qid=quarter_id,
            rows=[
                {
                    "id": c["chunk_id"],
                    "text": c["text"],
                    "page": c.get("page"),
                    "page_type": c.get("page_type"),
                    "section": c.get("section"),
                    "heading": c.get("heading"),
                    "document_name": c.get("document_name"),
                    "source_file_hash": c.get("source_file_hash"),
                    "processed_at": c.get("processed_at"),
                    "embedding_text": c.get("embedding_text"),
                }
                for c in chunks
            ],
        )

    @staticmethod
    def _tx_write_table_chunks(tx, quarter_id, chunks):
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            UNWIND $rows AS row
            MERGE (t:Table{id: row.id})
            SET
                t.title=row.title,
                t.headers=row.headers,
                t.rows=row.rows,
                t.page=row.page,
                t.page_type=row.page_type,
                t.section=row.section,
                t.heading=row.heading,
                t.document_name=row.document_name,
                t.source_file_hash=row.source_file_hash,
                t.processed_at=row.processed_at,
                t.embedding_text=row.embedding_text
            MERGE (q)-[:HAS_TABLE]->(t)
            """,
            qid=quarter_id,
            rows=[
                {
                    "id": c["chunk_id"],
                    "title": c.get("title"),
                    "headers": c.get("headers"),
                    "rows": json.dumps(c.get("rows")),
                    "page": c.get("page"),
                    "page_type": c.get("page_type"),
                    "section": c.get("section"),
                    "heading": c.get("heading"),
                    "document_name": c.get("document_name"),
                    "source_file_hash": c.get("source_file_hash"),
                    "processed_at": c.get("processed_at"),
                    "embedding_text": c.get("embedding_text"),
                }
                for c in chunks
            ],
        )

    @staticmethod
    def _tx_write_audio_chunks(tx, quarter_id, chunks):
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            UNWIND $rows AS row
            MERGE (a:AudioChunk{id: row.id})
            SET
                a.text=row.text,
                a.start=row.start,
                a.end=row.end,
                a.company=row.company,
                a.year=row.year,
                a.quarter=row.quarter,
                a.document_name=row.document_name,
                a.speaker=row.speaker,
                a.source_file_hash=row.source_file_hash,
                a.processed_at=row.processed_at,
                a.embedding_text=row.embedding_text
            MERGE (q)-[:HAS_AUDIO]->(a)
            """,
            qid=quarter_id,
            rows=[
                {
                    "id": c["chunk_id"],
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "company": c["company"],
                    "year": c["year"],
                    "quarter": c["quarter"],
                    "document_name": c.get("document_name"),
                    "speaker": c.get("speaker"),
                    "source_file_hash": c.get("source_file_hash"),
                    "processed_at": c.get("processed_at"),
                    "embedding_text": c["embedding_text"],
                }
                for c in chunks
            ],
        )

    def store_document(self, session, company, year, quarter, chunks):
        self.create_company(session, company)
        qid = self.create_quarter(session, company, year, quarter)
        self.store_chunks(session, qid, chunks)
        return qid

    @staticmethod
    def build_page_text_lookup(chunks) -> dict:
        """{page_number: concatenated "text"-type chunk text} for one
        document's chunks, in chunk order. Used only as last-resort
        context for table_metrics.py's period recovery (see
        process_chunk_full above) — a table chunk's own headers/rows
        are always tried first, this is purely a fallback source of
        the period labels a table chunk doesn't carry itself."""
        lookup: dict = {}
        for c in chunks:
            if c.get("chunk_type") != "text":
                continue
            page = c.get("page")
            if page is None:
                continue
            text = c.get("text") or ""
            if not text:
                continue
            lookup[page] = f"{lookup[page]} {text}" if page in lookup else text
        return lookup

    # --------------------------------------------------
    # Entity / Relationship Extraction (batched per chunk: every entity
    # and relationship pulled from one chunk is written in a single
    # transaction, and relationships of the same type are UNWOUND
    # together since the relation name itself can't be parameterized)
    # --------------------------------------------------

    def process_chunk_full(
        self, session, quarter_id, company, year, quarter, chunk, page_text_lookup=None
    ):
        """Table chunks try table_metrics.py's deterministic parse first
        (no LLM call at all) before falling back to one extract_all() LLM
        call. Guidance/risk/sentiment are always skipped for table
        chunks — whichever path filled in metrics, a raw balance sheet
        isn't a source for qualitative narrative.

        page_text_lookup: optional {page_number: concatenated_text} map
        built once per quarter (see build_page_text_lookup below) from
        this document's own "text"-type chunks. Passed through to
        table_metrics.extract_metrics_from_table() as last-resort
        context for recovering period labels a table's own headers/rows
        didn't capture (e.g. a multi-line header pdfplumber read by
        physical line instead of by column) — see that module's
        _recover_periods_from_context for exactly what this is and
        isn't used for. None reproduces the exact prior behavior."""
        text = chunk.get("embedding_text") or chunk.get("text", "")
        chunk_type = chunk.get("chunk_type")

        # Provenance: what specific chunk/page/document/timestamp an
        # extracted fact came from, so "show me where this came from" is
        # answerable without re-reading every source document. Stamped
        # directly onto Metric/Guidance/Risk nodes below (see
        # _tx_write_metrics/_tx_write_guidance/_tx_write_risks) in
        # addition to the CONTAINS_METRIC/CONTAINS_GUIDANCE/CONTAINS_RISK
        # edges to the source Chunk/Table/AudioChunk node (which already
        # carries the same fields, plus embedding_text) — the denormalized
        # copy means a single-hop property read answers "where's this
        # from" without a graph traversal at answer time.
        provenance = {
            "document_name": chunk.get("document_name"),
            "page": chunk.get("page"),  # PDF text/table chunks only
            "audio_start": chunk.get("start"),  # audio chunks only
            "audio_end": chunk.get("end"),  # audio chunks only
        }

        if self._is_boilerplate(text):
            logger.info(
                "Skipping boilerplate chunk %s (no LLM call)", chunk["chunk_id"]
            )
            return

        if self._is_tiny(text):
            logger.info("Skipping tiny chunk %s (no LLM call)", chunk["chunk_id"])
            return

        # Deterministic fast path for table chunks — headers/rows are
        # already structured data (see chunker.py's table_chunk()), so
        # table_metrics.py aligns "row label + column period -> value"
        # in Python instead of paying for an LLM call. This is the
        # overwhelming majority of table chunks; only genuinely
        # irregular ones (returns []) fall through to the LLM-based
        # TABLE_PROMPT below, exactly as if this fast path didn't exist.
        if chunk_type == "table" and getattr(
            config, "USE_DETERMINISTIC_TABLE_METRICS", True
        ):
            context_text = None
            if page_text_lookup is not None:
                context_text = page_text_lookup.get(chunk.get("page"))
            det_metrics = table_metrics.extract_metrics_from_table(
                chunk, context_text=context_text
            )
            if det_metrics:
                logger.info(
                    "Table chunk %s parsed deterministically (%d metrics, no LLM call)",
                    chunk["chunk_id"],
                    len(det_metrics),
                )
                session.execute_write(
                    self._tx_write_metrics,
                    quarter_id,
                    chunk["chunk_id"],
                    det_metrics,
                    company,
                    year,
                    quarter,
                    provenance,
                )
                return
            logger.info(
                "Table chunk %s didn't parse deterministically — "
                "falling back to LLM extraction",
                chunk["chunk_id"],
            )

        # chunk_type is folded into the key (not just the text) since
        # TEXT_PROMPT and TABLE_PROMPT can produce different results for
        # identical text — without this, a cache hit could reuse a
        # table-style extraction for a text chunk or vice versa.
        text_hash = self._hash_text(f"{chunk_type}:{self._normalize_whitespace(text)}")
        with self._state_lock:
            cached = self._extraction_cache.get(text_hash)
            if cached is None:
                # Get (or create) the lock for this specific text_hash so
                # that concurrent cache-misses on the same text serialize
                # on extract_all() rather than both calling it.
                hash_lock = self._extraction_locks.setdefault(
                    text_hash, threading.Lock()
                )

        if cached is not None:
            result = cached
        else:
            with hash_lock:
                # Re-check: another thread may have populated the cache
                # for this text_hash while we were waiting on hash_lock.
                with self._state_lock:
                    cached = self._extraction_cache.get(text_hash)
                if cached is not None:
                    result = cached
                else:
                    result = self.extract_all(text, chunk_type=chunk_type)
                    # Only cache extractions that actually parsed — an
                    # empty result from a retry-exhausted failure
                    # shouldn't be reused for every future occurrence of
                    # the same text.
                    if result.get("_extraction_ok", True):
                        with self._state_lock:
                            self._extraction_cache[text_hash] = result

        if not result.get("_extraction_ok", True):
            self._record_failed_chunk(
                chunk, company, year, quarter, result.get("_raw_response")
            )

        # ---- entities + relationships ----
        entities = [e for e in result["entities"] if e.get("name") and e.get("type")]

        relationships = []
        for rel in result["relationships"]:
            relation = str(rel.get("relation", "")).strip().upper().replace(" ", "_")
            if not RELATION_NAME_RE.match(relation):
                logger.warning(
                    "Dropping relationship with invalid name: %r", rel.get("relation")
                )
                continue

            source_id = (
                f"{rel['source_type']}:{rel['source']}"
                if "source_type" in rel
                else rel.get("source")
            )
            target_id = (
                f"{rel['target_type']}:{rel['target']}"
                if "target_type" in rel
                else rel.get("target")
            )
            if not source_id or not target_id:
                continue

            relationships.append(
                {"relation": relation, "source_id": source_id, "target_id": target_id}
            )

        if entities or relationships:
            session.execute_write(
                self._tx_write_entities_and_relationships,
                chunk["chunk_id"],
                entities,
                relationships,
            )

        # ---- metrics ----
        metrics = [m for m in result["metrics"] if m.get("name")]
        if metrics:
            session.execute_write(
                self._tx_write_metrics,
                quarter_id,
                chunk["chunk_id"],
                metrics,
                company,
                year,
                quarter,
                provenance,
            )

        # Guidance/risk/sentiment are about qualitative commentary — skip
        # raw table chunks, same as before.
        if chunk.get("chunk_type") == "table":
            return

        # ---- guidance ----
        guidance = [g for g in result["guidance"] if g.get("statement")]
        if guidance:
            session.execute_write(
                self._tx_write_guidance,
                quarter_id,
                chunk["chunk_id"],
                guidance,
                provenance,
            )

        # ---- risks ----
        risks = [r for r in result["risks"] if r.get("statement")]
        if risks:
            session.execute_write(
                self._tx_write_risks,
                quarter_id,
                chunk["chunk_id"],
                risks,
                provenance,
            )

        # ---- sentiment ----
        sentiment = result["sentiment"]
        if sentiment.get("sentiment"):
            session.execute_write(self._tx_write_sentiment, quarter_id, sentiment)

    @staticmethod
    def _tx_write_entities_and_relationships(tx, chunk_id, entities, relationships):
        if entities:
            tx.run(
                """
                MATCH (c) WHERE c.id = $chunk_id
                WITH c
                UNWIND $entities AS entity
                MERGE (e:Entity{id: entity.id})
                SET e.name = entity.name, e.type = entity.type
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                entities=[
                    {
                        "id": f"{e['type']}:{e['name']}",
                        "name": e["name"],
                        "type": e["type"],
                    }
                    for e in entities
                ],
            )

        # Relation type can't be parameterized in Cypher, so group pairs
        # by relation and issue one UNWIND query per distinct relation
        # type instead of one query per pair.
        by_relation = {}
        for rel in relationships:
            by_relation.setdefault(rel["relation"], []).append(rel)

        for relation, rels in by_relation.items():
            tx.run(
                f"""
                UNWIND $pairs AS pair
                MATCH (a:Entity{{id: pair.source_id}})
                MATCH (b:Entity{{id: pair.target_id}})
                MERGE (a)-[:{relation}]->(b)
                """,
                pairs=[
                    {"source_id": r["source_id"], "target_id": r["target_id"]}
                    for r in rels
                ],
            )

    # --------------------------------------------------
    # Metric write (batched per chunk)
    #
    # Provenance (source_chunk_id/source_document/source_page/
    # source_audio_start/source_audio_end) is denormalized directly onto
    # Metric/Guidance/Risk nodes below so "where did this come from" is a
    # single property read, not a graph traversal, at answer time. Each
    # of these node types is MERGEd on a key that's already specific to
    # one occurrence (Metric: company+year+quarter+name+period; Guidance/
    # Risk: a hash of the exact statement text), so in the normal case
    # there's exactly one source chunk and this property is unambiguous.
    # In the rare case the identical metric/statement is re-extracted
    # from a second chunk (e.g. the same table repeated on an annexure
    # page), the denormalized property reflects whichever write ran
    # last — the CONTAINS_METRIC/CONTAINS_GUIDANCE/CONTAINS_RISK edges
    # are the ground truth for "every chunk this came from" in that case,
    # since MERGE accumulates edges from all sources rather than
    # overwriting them.
    # --------------------------------------------------

    @staticmethod
    def _tx_write_metrics(
        tx, quarter_id, chunk_id, metrics, company, year, quarter, provenance=None
    ):
        provenance = provenance or {}
        rows = [
            {
                # period is part of the id on purpose: COMBINED_PROMPT
                # explicitly instructs the LLM to extract one metric entry
                # PER PERIOD when a line reports several time periods for
                # the same metric name (e.g. "Cash and cash equivalents
                # 763.02 520.09 1,032.45" -> 3 entries, one per period).
                # Without period in the id, all 3 MERGE onto the same
                # Metric node and each SET overwrites the last — silently
                # keeping only the most recently written period's value
                # and discarding the rest, even though extraction worked
                # correctly. period defaults to "" (not None) so metrics
                # from a chunk with no stated period still get a stable,
                # distinct id instead of colliding on a shared "None".
                "metric_id": (
                    f"{company}|{year}|{quarter}|{m['name']}|{m.get('period') or ''}"
                ),
                "name": m["name"],
                "value": m.get("value"),
                "unit": m.get("unit"),
                "period": m.get("period"),
            }
            for m in metrics
        ]

        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            MATCH (c) WHERE c.id = $chunk_id
            WITH q, c
            UNWIND $rows AS row
            MERGE (m:Metric{id: row.metric_id})
            SET
                m.name=row.name,
                m.value=row.value,
                m.unit=row.unit,
                m.period=row.period,
                m.company=$company,
                m.year=$year,
                m.quarter=$quarter,
                m.source_chunk_id=$chunk_id,
                m.source_document=$document_name,
                m.source_page=$page,
                m.source_audio_start=$audio_start,
                m.source_audio_end=$audio_end
            MERGE (q)-[:HAS_METRIC]->(m)
            MERGE (c)-[:CONTAINS_METRIC]->(m)
            """,
            qid=quarter_id,
            chunk_id=chunk_id,
            rows=rows,
            company=company,
            year=year,
            quarter=quarter,
            document_name=provenance.get("document_name"),
            page=provenance.get("page"),
            audio_start=provenance.get("audio_start"),
            audio_end=provenance.get("audio_end"),
        )

    # --------------------------------------------------
    # Guidance / Risk / Sentiment write (batched per chunk)
    # --------------------------------------------------

    @staticmethod
    def _tx_write_guidance(tx, quarter_id, chunk_id, items, provenance=None):
        provenance = provenance or {}
        rows = [
            {
                "id": hashlib.sha256(g["statement"].encode()).hexdigest(),
                "topic": g.get("topic"),
                "statement": g["statement"],
                "timeframe": g.get("timeframe"),
            }
            for g in items
        ]

        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            MATCH (c) WHERE c.id = $chunk_id
            WITH q, c
            UNWIND $rows AS row
            MERGE (g:Guidance{id: row.id})
            SET g.topic=row.topic, g.statement=row.statement, g.timeframe=row.timeframe,
                g.source_chunk_id=$chunk_id,
                g.source_document=$document_name,
                g.source_page=$page,
                g.source_audio_start=$audio_start,
                g.source_audio_end=$audio_end
            MERGE (q)-[:HAS_GUIDANCE]->(g)
            MERGE (c)-[:CONTAINS_GUIDANCE]->(g)
            """,
            qid=quarter_id,
            chunk_id=chunk_id,
            rows=rows,
            document_name=provenance.get("document_name"),
            page=provenance.get("page"),
            audio_start=provenance.get("audio_start"),
            audio_end=provenance.get("audio_end"),
        )

    @staticmethod
    def _tx_write_risks(tx, quarter_id, chunk_id, items, provenance=None):
        provenance = provenance or {}
        rows = [
            {
                "id": hashlib.sha256(r["statement"].encode()).hexdigest(),
                "type": r.get("type"),
                "statement": r["statement"],
            }
            for r in items
        ]

        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            MATCH (c) WHERE c.id = $chunk_id
            WITH q, c
            UNWIND $rows AS row
            MERGE (r:Risk{id: row.id})
            SET r.type=row.type, r.statement=row.statement,
                r.source_chunk_id=$chunk_id,
                r.source_document=$document_name,
                r.source_page=$page,
                r.source_audio_start=$audio_start,
                r.source_audio_end=$audio_end
            MERGE (q)-[:HAS_RISK]->(r)
            MERGE (c)-[:CONTAINS_RISK]->(r)
            """,
            qid=quarter_id,
            chunk_id=chunk_id,
            rows=rows,
            document_name=provenance.get("document_name"),
            page=provenance.get("page"),
            audio_start=provenance.get("audio_start"),
            audio_end=provenance.get("audio_end"),
        )

    @staticmethod
    def _tx_write_sentiment(tx, quarter_id, sentiment):
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            SET
                q.sentiment=$sentiment,
                q.confidence=$confidence
            """,
            qid=quarter_id,
            sentiment=sentiment["sentiment"],
            confidence=sentiment.get("confidence"),
        )

    # --------------------------------------------------
    # Cross-link PDF and audio content (GraphRAG's actual advantage over
    # plain RAG: connect a PDF chunk/table to audio commentary that talks
    # about the same thing). Run once per quarter after all chunks in
    # that quarter have had entity/metric extraction — links nodes that
    # mention the same Entity or the same Metric name, e.g.
    #     (PDF Revenue table) -[:DISCUSSES]-> (Audio Revenue commentary)
    # --------------------------------------------------

    def link_pdf_audio(self, session, quarter_id):
        session.execute_write(self._tx_link_pdf_audio, quarter_id)

    @staticmethod
    def _tx_link_pdf_audio(tx, quarter_id):
        # Shared entities
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            MATCH (q)-[:HAS_CHUNK|HAS_TABLE]->(pdf)-[:MENTIONS]->(e:Entity)
            MATCH (q)-[:HAS_AUDIO]->(audio:AudioChunk)-[:MENTIONS]->(e)
            MERGE (pdf)-[:DISCUSSES]->(audio)
            """,
            qid=quarter_id,
        )

        # Shared metrics (by metric name, since Metric nodes are unique
        # per company/year/quarter/name -- a PDF table and an audio
        # commentary segment both citing "Revenue" for the same quarter
        # will point at the same Metric node)
        tx.run(
            """
            MATCH (q:Quarter{id:$qid})
            MATCH (q)-[:HAS_CHUNK|HAS_TABLE]->(pdf)-[:CONTAINS_METRIC]->(m:Metric)
            MATCH (q)-[:HAS_AUDIO]->(audio:AudioChunk)-[:CONTAINS_METRIC]->(m)
            MERGE (pdf)-[:DISCUSSES]->(audio)
            """,
            qid=quarter_id,
        )

    # --------------------------------------------------
    # Temporal linking — connects the graph across time (NEXT_QUARTER,
    # SAME_QUARTER_LAST_YEAR, metric/guidance/risk evolution). See
    # temporal_utils.py for the implementation. Unlike link_pdf_audio()
    # (which runs once per quarter, right after that quarter's chunks
    # are loaded), this needs to see every quarter already in the graph
    # to compute "what's next" correctly, so it's called once at the end
    # of build_all() rather than per-quarter — see below.
    # --------------------------------------------------

    def build_temporal_links(self, session):
        return temporal_utils.link_all(session)


# --------------------------------------------------
# Build Everything (used by run_all.py)
# --------------------------------------------------


def build_all():
    """
    Reads every processed chunk JSON (produced by data_processing.py) and
    loads it into Neo4j: company/quarter nodes, chunk/table/audio nodes,
    entities + relationships, metrics, guidance, risks, and sentiment.

    Opens ONE Neo4j session per document (quarter) instead of one per
    write — every write inside that session is still its own explicit
    transaction (session.execute_write).
    """

    builder = GraphBuilder(
        config.NEO4J_URI,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD,
        model=config.GRAPH_EXTRACTION_MODEL,
    )

    progress = builder.load_progress()
    builder.load_extraction_cache()
    _warn_if_ollama_not_parallel(max(1, getattr(config, "GRAPH_BUILDER_WORKERS", 2)))
    if progress["completed_quarters"] or progress["in_progress"]:
        logger.warning(
            "Loaded %s at startup: %d quarter(s) marked complete, "
            "%d quarter(s) partially done. These will be skipped/resumed "
            "accordingly. Delete this file first if you want a fully "
            "fresh rebuild.",
            builder._progress_path(),
            len(progress["completed_quarters"]),
            len(progress["in_progress"]),
        )

    try:
        with builder.driver.session(database=config.NEO4J_DATABASE) as session:
            builder.create_constraints(session)

        for company in config.COMPANIES:
            for year in config.YEARS:
                for quarter in config.QUARTERS:
                    q_dir = config.PROCESSED_DATA_DIR / company / year / quarter

                    if not q_dir.exists():
                        continue

                    quarter_key = f"{company}|{year}|{quarter}"
                    if quarter_key in progress["completed_quarters"]:
                        logger.info(
                            "Skipping %s — already completed in a prior run.",
                            quarter_key,
                        )
                        continue

                    chunks = []

                    for jf in q_dir.glob("*.json"):
                        chunks.extend(json.loads(jf.read_text(encoding="utf-8")))

                    if not chunks:
                        continue

                    # chunk_ids already processed for this quarter in an
                    # earlier, interrupted run — everything else in the
                    # loop below runs exactly as before.
                    done_chunk_ids = progress["in_progress"].get(quarter_key, set())

                    logger.info(
                        "Building graph for %s %s-%s (%d chunks%s)",
                        company,
                        quarter,
                        year,
                        len(chunks),
                        f", resuming — {len(done_chunk_ids)} already done"
                        if done_chunk_ids
                        else "",
                    )

                    with builder.driver.session(
                        database=config.NEO4J_DATABASE
                    ) as session:
                        try:
                            qid = builder.store_document(
                                session, company, year, quarter, chunks
                            )
                        except Exception:
                            logger.error(
                                "Failed to store document %s %s-%s",
                                company,
                                quarter,
                                year,
                                exc_info=True,
                            )
                            continue

                        pending = [
                            c for c in chunks if c.get("chunk_id") not in done_chunk_ids
                        ]

                        page_text_lookup = builder.build_page_text_lookup(chunks)

                        chunk_times = []
                        completed = 0
                        progress_lock = threading.Lock()
                        # progress.json is saved after every completed
                        # chunk by default — a single LLM call (30-60s)
                        # dwarfs the cost of one small JSON write, so
                        # there's little to gain from batching, and
                        # batching means a crash can lose more than one
                        # completed chunk. Override via config if a
                        # given run's I/O profile makes batching
                        # worthwhile. Always saved at the end of the
                        # quarter regardless (below).
                        PROGRESS_SAVE_EVERY = getattr(config, "PROGRESS_SAVE_EVERY", 1)

                        def _run_chunk(chunk):
                            """Runs in a worker thread. Opens its own
                            Neo4j session — sessions aren't thread-safe,
                            but the driver is, so each worker gets one."""
                            cid = chunk.get("chunk_id")
                            t0 = time.perf_counter()
                            try:
                                with builder.driver.session(
                                    database=config.NEO4J_DATABASE
                                ) as worker_session:
                                    builder.process_chunk_full(
                                        worker_session,
                                        qid,
                                        company,
                                        year,
                                        quarter,
                                        chunk,
                                        page_text_lookup=page_text_lookup,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "Extraction failed for chunk %s",
                                    cid,
                                    exc_info=True,
                                )
                                builder._record_failed_chunk(
                                    chunk,
                                    company,
                                    year,
                                    quarter,
                                    None,
                                    error=f"Unhandled exception during processing: {e}",
                                )
                            return cid, time.perf_counter() - t0

                        workers = max(1, getattr(config, "GRAPH_BUILDER_WORKERS", 2))
                        executor = ThreadPoolExecutor(max_workers=workers)
                        try:
                            futures = {
                                executor.submit(_run_chunk, chunk): chunk
                                for chunk in pending
                            }
                            for future in as_completed(futures):
                                cid, elapsed = future.result()
                                with progress_lock:
                                    completed += 1
                                    i = completed
                                    chunk_times.append(elapsed)
                                    avg = sum(chunk_times) / len(chunk_times)
                                    done_chunk_ids.add(cid)
                                    progress["in_progress"][quarter_key] = (
                                        done_chunk_ids
                                    )
                                    should_save = completed % PROGRESS_SAVE_EVERY == 0
                                    if should_save:
                                        builder.save_progress(progress)

                                logger.info(
                                    "Chunk %d/%d %s %.1fs (avg %.1fs) %s",
                                    i,
                                    len(pending),
                                    _progress_bar(i, len(pending)),
                                    elapsed,
                                    avg,
                                    cid,
                                )
                        except KeyboardInterrupt:
                            # cancel_futures only drops futures that
                            # hadn't started yet — futures already
                            # running keep running until wait=True
                            # returns. Those chunks got fully written to
                            # Neo4j but, since as_completed() never got
                            # to hand them back to us, their chunk_ids
                            # never made it into done_chunk_ids. Sweep
                            # every future here (not just the ones the
                            # for-loop already consumed) so completed
                            # work isn't silently redone next run.
                            executor.shutdown(wait=True, cancel_futures=True)
                            with progress_lock:
                                for fut in futures:
                                    if fut.done() and not fut.cancelled():
                                        try:
                                            cid, _ = fut.result()
                                            done_chunk_ids.add(cid)
                                        except Exception:
                                            # Chunk raised inside
                                            # _run_chunk — already
                                            # recorded via
                                            # _record_failed_chunk, not
                                            # "done".
                                            pass
                                progress["in_progress"][quarter_key] = done_chunk_ids
                                builder.save_progress(progress)
                            raise
                        else:
                            executor.shutdown(wait=True)
                            # Make sure the final batch (which may not
                            # land on a PROGRESS_SAVE_EVERY boundary) is
                            # persisted before moving on.
                            with progress_lock:
                                progress["in_progress"][quarter_key] = done_chunk_ids
                                builder.save_progress(progress)

                        try:
                            builder.link_pdf_audio(session, qid)
                        except Exception:
                            logger.warning(
                                "PDF/audio cross-linking failed for %s %s-%s",
                                company,
                                quarter,
                                year,
                                exc_info=True,
                            )

                        # Every chunk in this quarter has been attempted
                        # at least once — mark it complete so future runs
                        # skip it outright instead of reopening its JSON
                        # files and checking chunk_ids one by one.
                        progress["completed_quarters"].add(quarter_key)
                        progress["in_progress"].pop(quarter_key, None)
                        builder.save_progress(progress)

        # Every quarter that's going to be in the graph this run has been
        # loaded — now connect them across time. This has to run once,
        # after the full per-(company, year, quarter) loop above, rather
        # than per-quarter: computing "what's the next quarter" correctly
        # requires seeing every quarter already in the graph, including
        # ones from a previous run that were skipped above because
        # they're already in completed_quarters. Safe to re-run any time
        # (see temporal_utils.py — every relationship type it owns is
        # rebuilt from scratch on each call).
        logger.info(
            "Building temporal links (NEXT_QUARTER, SAME_QUARTER_LAST_YEAR, "
            "metric/guidance/risk evolution) across the full graph..."
        )
        with builder.driver.session(database=config.NEO4J_DATABASE) as session:
            temporal_stats = builder.build_temporal_links(session)
        logger.info("Temporal linking done: %s", temporal_stats)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user — progress saved to %s. "
            "Rerun to resume where you left off.",
            builder._progress_path(),
        )
        raise
    finally:
        builder.save_extraction_cache()
        builder.save_failed_chunks()
        if builder.failed_chunks:
            logger.warning(
                "%d chunk(s) failed this run — see results/failed_chunks.json. "
                "Run `python retry_failed_chunks.py` to retry just those.",
                len(builder.failed_chunks),
            )
        builder.close()


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")
    build_all()
