"""
evidence_cleaning.py
=====================
WHAT THIS FILE DOES:
Shared text cleanup applied to retrieved chunk text before it goes into
either pipeline's prompt: fixes common PDF-extraction mojibake (curly
quotes/dashes turned into "â€"/"Â" sequences), strips repetitive
header/footer noise (emails, CIN numbers, "Registered Office", page
numbers), and collapses excess blank lines/spaces.

This is shared between graph_rag_pipeline.py and baseline_pipeline.py
(rather than living only in one of them) for two reasons:
  1. It saves prompt tokens either way.
  2. Applying it identically to both pipelines keeps the GraphRAG vs
     baseline comparison about the graph, not about which pipeline
     happened to get tidier evidence text.

Also provides format_table_markdown(): renders a Table chunk's
structured headers/rows (see retrieval.py, sourced from the headers/rows
graph_builder.py already stores on every Table node) as a clean markdown
table instead of the flattened row-by-row text a plain PDF table
extraction produces. Same sharing rationale as clean_evidence_text —
both pipelines get the same table formatting quality.

Patterns/mappings live in config.py (NOISE_PATTERNS, OCR_ARTIFACT_MAP) so
they're tunable without touching this file.
"""

import re

import config
from table_metrics import (
    _label_column_index,
    _looks_like_actual_period,
    _normalize_ocr_period,
    _recover_periods_from_context,
    _resolve_headers,
    detect_statement_type,
)

_NOISE_PATTERNS = [re.compile(p, re.I) for p in config.NOISE_PATTERNS]
_OCR_ARTIFACT_MAP = config.OCR_ARTIFACT_MAP
_OCR_ARTIFACT_CATCHALL = re.compile(r"â€.")


def _clean_ocr_artifacts(text: str) -> str:
    cleaned = text or ""
    for bad, good in _OCR_ARTIFACT_MAP.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = _OCR_ARTIFACT_CATCHALL.sub("", cleaned)
    return cleaned


def clean_evidence_text(text: str) -> str:
    """Fix OCR/mojibake artifacts, strip header/footer noise patterns, and
    collapse repeated blank lines/spaces before evidence text is sent to
    the LLM. Saves prompt tokens and removes distracting boilerplate."""
    cleaned = _clean_ocr_artifacts(text or "")
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _cell(row, i):
    return str(row[i]).strip() if i < len(row) and row[i] is not None else ""


def statement_type_tag(chunk: dict) -> str:
    """Return a "**[Standalone figures]**\\n" / "**[Consolidated figures]**\\n"
    prefix when the chunk's own metadata/text confidently indicates its
    statement type (via table_metrics.detect_statement_type), or "" when
    it can't be determined.

    P0 fix: this tagging previously only applied to Table chunks (inline
    inside format_table_markdown below). Narrative TEXT chunks got no
    such signal at all, even though the exact same standalone/
    consolidated ambiguity applies to them — confirmed in a real
    benchmark row (Info Edge Q4 FY26) where the full text of both the
    standalone and consolidated statement pages were retrieved as
    Chunk-type (not Table-type) nodes and shown to the LLM with nothing
    to tell them apart. Both baseline_pipeline.py's and
    graph_rag_pipeline.py's _format_chunk() now call this for every
    chunk, not just tables, so the disambiguation is consistent
    regardless of how a given filing happened to get chunked.
    """
    statement_type = detect_statement_type(
        chunk.get("title"),
        chunk.get("section"),
        chunk.get("document_name"),
        (chunk.get("text") or "")[:500],
    )
    return f"**[{statement_type} figures]**\n" if statement_type else ""


def format_table_markdown(chunk: dict, max_rows: int = 40) -> str:
    """Render a Table chunk's structured headers/rows as a clean markdown
    table — 'Revenue from operations | 2345.98 | 2154.94' with the period
    each value belongs to right there in the same row, instead of the
    flattened text a naive PDF table extraction produces ('Revenue from
    operations\\n= 2345.98\\n= 2154.94\\n...' with the column mapping
    separated from the values, forcing the LLM to guess which figure goes
    with which quarter).

    Reuses table_metrics.py's _resolve_headers/_label_column_index (the
    same header-cleanup and label-column-detection logic graph_builder.py
    already relies on for numeric metric extraction) rather than
    reimplementing table-shape detection here — one source of truth for
    "which column is the label" instead of two that could disagree.

    P0 fix: this used to stop at _resolve_headers and render a bare
    "Col 2 | Col 3 | ..." whenever a column header didn't resolve to a
    real period (blank, OCR-corrupted, or a split multi-row date) —
    table_metrics.extract_metrics_from_table() already has period
    normalization (_normalize_ocr_period) and last-resort context
    recovery (_recover_periods_from_context) for exactly this, but only
    the Neo4j-write path (graph_builder.py) was calling it; the evidence
    shown to the LLM never got the same treatment. Confirmed live case:
    a CreditAccess Grameen revenue table rendered with "Col 2".."Col 7"
    headers, so the LLM had no way to tell which column was Q2 FY25 —
    the correct value (1,453.29) was sitting right there in the table,
    but with no label to point at it, the model fell back to a
    Metric fact tagged "confidence=low_possible_subitem_mismatch" and
    answered 4,900 instead. Recovering real period labels here (using
    the same, already-existing table_metrics.py functions the
    Neo4j-write path relies on — nothing about table_metrics.py itself
    changes) gives the model a correctly-labeled column to point at
    instead, so the existing prompt rule ("trust the source text over a
    low-confidence fact") has something to actually act on.

    Falls back to the flattened, cleaned chunk text — same as before this
    function existed — whenever headers/rows are missing (older data,
    audio/text chunks, or a table that doesn't resolve into at least a
    label column + one value column). Same graceful-degradation pattern
    used everywhere else in this codebase; a table that can't be neatly
    rendered still reaches the LLM as raw text rather than being dropped.
    """
    headers = chunk.get("headers")
    rows = chunk.get("rows")
    fallback = clean_evidence_text(chunk.get("text") or "")[:1200]

    if not headers or not rows:
        return fallback

    try:
        resolved_headers, resolved_rows = _resolve_headers(headers, rows)
        if len(resolved_headers) < 2 or not resolved_rows:
            return fallback
        label_col = _label_column_index(resolved_headers, resolved_rows)
    except Exception:
        return fallback

    excluded_cols = {label_col}
    # Match table_metrics.py: when the real label is column 1, column 0 is
    # a serial-number prefix and must not be rendered as a financial value.
    if label_col != 0:
        excluded_cols.add(0)
    value_cols = [i for i in range(len(resolved_headers)) if i not in excluded_cols]
    if not value_cols:
        return fallback

    # Normalize OCR-corrupted month tokens ("31-0ec-25" -> "31-Dec-25")
    # first — cheap, and fixes some headers outright without needing
    # context recovery at all.
    period_headers = [_normalize_ocr_period(resolved_headers[i]) for i in value_cols]

    # If any value column's header doesn't look like a real period, try
    # to recover all of them from the surrounding page text — same
    # strategy, same helper, as table_metrics.extract_metrics_from_table
    # uses for the Neo4j-write path. Only replaces headers when recovery
    # returns a confident, unambiguous match for every value column at
    # once (see _recover_periods_from_context's docstring) — otherwise
    # this falls through to the existing "Col N" fallback unchanged, so
    # a table this can't confidently label still reaches the LLM rather
    # than being dropped or mislabeled.
    if not all(_looks_like_actual_period(h) for h in period_headers):
        first_row_label = None
        for row in resolved_rows:
            if label_col < len(row) and str(row[label_col] or "").strip():
                first_row_label = str(row[label_col]).strip()
                break
        recovered = _recover_periods_from_context(
            chunk.get("text") or chunk.get("embedding_text") or "",
            len(value_cols),
            first_row_label,
        )
        if recovered is not None:
            period_headers = recovered

    header_cells = ["Metric"] + [
        # P1 fix: a garbled-but-non-empty header ("-24", "3" — OCR
        # fragments, page-number leakage, whatever survived pdfplumber)
        # used to pass straight through here because `x or "Col N"` only
        # catches EMPTY strings, not garbage ones. Confirmed live case:
        # a CreditAccess Grameen table rendered "| Metric | -24 | Col 3 |
        # 3 |" — "-24" and "3" look enough like real values that they're
        # actively misleading (worse than an honest "Col N" placeholder),
        # and recovery had already failed to replace them by this point.
        # Gate on the same strict period-shape check used above so only
        # a header that's either a genuine original period OR a
        # successfully-recovered one ever reaches the LLM as a label.
        period_headers[pos]
        if _looks_like_actual_period(period_headers[pos])
        else f"Col {i + 1}"
        for pos, i in enumerate(value_cols)
    ]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] * len(header_cells)) + "|",
    ]
    for row in resolved_rows[:max_rows]:
        label = _cell(row, label_col)
        if not label:
            continue
        values = [_cell(row, i) for i in value_cols]
        lines.append("| " + " | ".join([label] + values) + " |")

    if len(lines) <= 2:
        return fallback  # header only, no usable data rows

    table_md = "\n".join(lines)
    if len(resolved_rows) > max_rows:
        table_md += f"\n(+{len(resolved_rows) - max_rows} more rows truncated)"

    # Tag the table as Standalone/Consolidated when it can be told
    # confidently from the chunk's own metadata/text — this is exactly
    # the distinction the answer prompt's standalone-vs-consolidated
    # rule depends on, and a "Col 2 | Col 3 | ..." table gives the model
    # no way to tell them apart on its own. See statement_type_tag()
    # above (shared with narrative text chunks as of the P0 fix).
    table_md = statement_type_tag(chunk) + table_md

    return table_md
