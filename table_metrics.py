"""
table_metrics.py
=================

Deterministic (no-LLM) parser that turns a table chunk's raw `headers`/
`rows` — as produced by chunker.py's table_chunk(), sourced from
pdf_processor.py's raw pdfplumber extraction — into a flat list of
metric facts ready for Neo4j:

    [{"name": "Revenue from operations", "value": "1105.17",
      "unit": "Crore INR", "period": "30-Jun-24"}, ...]

Why this exists
----------------
Financial-statement tables are already structured data: one column is
usually the metric label, every other column is a reporting period,
and every other cell is a number. Asking a 7B LLM to re-derive "row
label + column period -> value" from that (graph_builder.py's
TABLE_PROMPT, kept as a fallback) is slower and strictly no better
than just reading it off directly when the table parsed cleanly.

Called from graph_builder.py's process_chunk_full():
    table_metrics.extract_metrics_from_table(chunk) -> list[dict]

IMPORTANT — the chunk's `headers`/`rows` here are NOT the same thing
as pdf_processor.py's table_to_text() output. table_to_text() resolves
multi-row headers and period alignment for the *embedding text*, but
chunker.py's table_chunk() stores the raw, unresolved headers/rows
alongside it (see chunker.py's table_chunk()). So this module redoes
that resolution itself, from scratch, rather than trying to re-parse
the already-formatted embedding_text string.

Contract: return [] (never raise) for anything this parser can't
confidently handle — blank/ambiguous headers, non-numeric columns,
character-level OCR corruption it can't safely repair. process_chunk_full()
falls back to the LLM path whenever this returns nothing, exactly as if
this module didn't exist. A wrong deterministic metric silently written
to Neo4j is worse than one that goes through the slower LLM fallback —
so every check below is written to bail out rather than guess.
"""

from __future__ import annotations

import re

# -------------------------------------------------------------
# Period-label detection (mirrors pdf_processor.py's
# _looks_like_period_label — reimplemented here rather than imported
# so this module has no dependency on pdf_processor.py's OCR/fitz/
# pdfplumber import chain; it only ever touches the already-extracted
# chunk dict).
# -------------------------------------------------------------

_STANDALONE_RE = re.compile(r"\bstandalone\b", re.IGNORECASE)
_CONSOLIDATED_RE = re.compile(r"\bconsolidated\b", re.IGNORECASE)


def detect_statement_type(*texts: str | None) -> str | None:
    """Scans the given text fields (title, section, nearby narrative —
    whatever's available on the chunk) for a "Standalone"/"Consolidated"
    marker, the same way a human reader would tell the two apart in an
    Indian financial results filing (both variants are near-universally
    labeled explicitly, e.g. "STATEMENT OF STANDALONE AUDITED FINANCIAL
    RESULTS"). Returns "Standalone", "Consolidated", or None if neither
    marker is present in any of the given text.

    Checks Consolidated first: some filings' running header repeats
    "...for both the Standalone and Consolidated results..." in
    boilerplate above the actual table, and if that boilerplate also
    contains "Consolidated" (which it usually does, since both words
    tend to appear together in that boilerplate), preferring whichever
    one appears more specifically isn't reliable — but a table that
    matches BOTH words is far more likely to be a page-level running
    header/boilerplate than an actual single statement, so this only
    returns a confident answer when exactly one of the two markers is
    present, and returns None (ambiguous) rather than guess when both
    or neither are found."""
    combined = " ".join(t for t in texts if t)
    if not combined:
        return None
    has_standalone = bool(_STANDALONE_RE.search(combined))
    has_consolidated = bool(_CONSOLIDATED_RE.search(combined))
    if has_standalone and not has_consolidated:
        return "Standalone"
    if has_consolidated and not has_standalone:
        return "Consolidated"
    return None


_PURE_NUMBER_RE = re.compile(r"[-+(]?[\d,]*\.?\d*[)%]?")

# A bare 1-2 digit day-of-month (1-31) or a bare 4-digit year
# (1900-2099) — these are exactly the shape pdfplumber leaves behind
# when a multi-row date header ("30 Sep 2025") gets split across
# physical rows, one fragment per row ("30" on one row, "2025" on the
# next). They parse as plain numbers just like real financial data
# does, but a real revenue/profit/EPS figure essentially never lands
# on a bare "30" or "2025" — so counting these against the "is this a
# real data row" check below causes _resolve_headers to stop merging
# the header one row too early and leak "Sr." / "No." rows straight
# into the table body as if they were metric rows (the exact failure
# behind tables rendering as "Col 2 | Col 3 | ..." with a stray
# "Sr. | 30 | 30 | 30" row underneath instead of a real date header).
_BARE_DAY_RE = re.compile(r"^\(?\d{1,2}\)?$")
_BARE_YEAR_RE = re.compile(r"^\(?(19|20)\d{2}\)?$")


def _looks_like_date_fragment(value) -> bool:
    if not value:
        return False
    stripped = str(value).strip()
    if not stripped:
        return False
    if _BARE_DAY_RE.match(stripped):
        day = int(re.sub(r"[()]", "", stripped))
        return 1 <= day <= 31
    return bool(_BARE_YEAR_RE.match(stripped))


def _looks_like_period_label(value) -> bool:
    """True for short strings that name a reporting period ('30 Sep
    2025', 'FY26', 'Q2 FY26', 'Unaudited') as opposed to a numeric data
    value ('1591.04', '(500)'). A period label always has letters; a
    pure number (however formatted) never does."""
    if not value:
        return False
    stripped = str(value).strip()
    if not stripped or len(stripped) > 25:
        return False
    if _PURE_NUMBER_RE.fullmatch(stripped):
        return False
    return any(c.isalpha() for c in stripped)


def _resolve_headers(headers, rows):
    """Consumes leading rows that are themselves non-data (letterhead,
    CIN/address lines, "As at <date1> / <date2>" period rows, an
    audit-status row) rather than real table data. Without this, an
    ambiguous/blank header ("", "", "") would otherwise be the only
    thing available to build a period label from, and the real
    period-per-column info sitting a few rows down would silently
    become a stray data row instead.

    The first candidate row is only consumed if the header row itself
    was ambiguous (blank or duplicate labels) — the actual signal that
    a multi-row header exists — so a genuinely simple table whose first
    real row happens to be all-text isn't eaten by mistake. The cap on
    how many rows this will consume is generous (letterhead stacks in
    real filings have run 10+ rows deep before the actual period
    header appears) because the per-row content check — not the cap —
    is what stops it: the very first row containing a genuine numeric
    data cell breaks the loop immediately, regardless of how high the
    cap is set, so raising it doesn't create a way to accidentally
    consume real data.
    """
    if not rows:
        return list(headers), rows

    resolved = list(headers)
    remaining = list(rows)
    consumed = 0
    max_leading_label_rows = 20

    while remaining and consumed < max_leading_label_rows:
        candidate = remaining[0]
        non_empty = [v for v in candidate if v and str(v).strip()]
        if not non_empty:
            break

        # The stopping signal is at the ROW level, not per-cell: a
        # genuine data row always has several numeric cells (one per
        # period column), while a letterhead/address/CIN row has at
        # most one incidental number in it (a street number, a phone
        # digit run) — cell-content alone can't tell "49," in an
        # address apart from a real value, since both parse as a
        # plain number; how MANY such cells are on the same row can.
        # Bare day-of-month ("30") or bare year ("2025") fragments are
        # excluded from this count — see _looks_like_date_fragment —
        # because a split multi-row date header produces exactly this
        # shape (several small/round numbers repeated across columns)
        # and would otherwise look just as "numeric" as a real data
        # row, causing the header merge to stop one row too early.
        numeric_cells = sum(
            1
            for v in non_empty
            if _parse_numeric_cell(str(v).strip())[0] is not None
            and not _looks_like_date_fragment(v)
        )
        if numeric_cells >= 2:
            break

        if consumed == 0:
            non_blank = [h for h in resolved if h and str(h).strip()]
            ambiguous = len(set(non_blank)) < len(non_blank) or any(
                not (h and str(h).strip()) for h in resolved
            )
            if not ambiguous:
                break

        width = max(len(resolved), len(candidate))
        merged = []
        for i in range(width):
            h = resolved[i] if i < len(resolved) else ""
            sub = candidate[i] if i < len(candidate) else ""
            h, sub = (str(h).strip() if h else ""), (str(sub).strip() if sub else "")
            if not sub:
                merged.append(h)
            elif consumed == 0:
                merged.append(sub)
            elif h and h != sub:
                merged.append(f"{h} ({sub})")
            else:
                merged.append(sub)
        resolved = merged
        remaining = remaining[1:]
        consumed += 1

    return resolved, remaining


# -------------------------------------------------------------
# Strict period-shape validation — used both for a lone value column's
# header (see extract_metrics_from_table below) AND, more importantly,
# as a hard gate on every multi-column period before any metric is
# written. _looks_like_period_label above and the row-level numeric-
# ask "is this text, not a number" — that's the right (loose) bar for
# deciding whether a row is part of a header stack, but nowhere near
# strict enough to trust as an actual period value: "Guru" and "r's
# Review Report on Consolidated Unau" both pass that test too. This is
# the fix for a real, observed failure mode — pdfplumber's borderless
# "text" column-detection strategy occasionally mistakes an ordinary
# prose paragraph for a table, slicing sentences into fake columns
# ("Results of Jindal St" | "ainless Limited..."). The real financial
# table on the same page still extracts correctly alongside it (this
# doesn't lose data), but graph_builder.py's process_chunk_full() would
# otherwise happily write whatever silently pattern-matched cells that
# prose-shaped "table" turned up as if they were real metric facts.
# A genuine period always has one of a small number of concrete shapes
# (a quarter/year label, a real date, an audit-status word) — this
# requires the FULL string to be one of those shapes, not just contain
# a stray digit somewhere.
# -------------------------------------------------------------

_STRICT_PERIOD_RE = re.compile(
    r"""^\(?\s*(
        q[1-4]\s?['\u2019]?(fy)?\s?\d{2,4}                          # Q1FY24, Q1 FY2024
      | fy\s?['\u2019]?\d{2,4}                                       # FY24, FY 2024
      | \d{1,2}\s?[\-/.]\s?[a-z]{3,9}\.?\s?['\u2019]?[\-/.]?\s?\d{2,4}   # 30-Jun-23, 30 June '23
      | \d{1,2}\s?[\-/.]\s?\d{1,2}\s?[\-/.]\s?\d{2,4}                 # 30-06-2023, 30/06/23
      | [a-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}                           # June 30, 2023
      | (year|quarter)\s?(ended|end)?\s.{0,20}?\d{2,4}                # Year ended 31-Mar-24
      | as\s+(at|on)\s.{0,20}?\d{2,4}                                 # As at 30 June 2023
      | (un)?audited
      | restated
    )\s*\)?$""",
    re.IGNORECASE | re.VERBOSE,
)


_OCR_DATE_MONTH_FIXES = [
    # Targeted OCR fixes ONLY for month tokens inside date-shaped strings.
    # Do not apply these globally to numeric cells.
    (re.compile(r"(?<=\\d[-/.])0ec(?=[-/.]\\d{2,4}\\b)", re.IGNORECASE), "Dec"),
    (re.compile(r"(?<=\\d[-/.])0ct(?=[-/.]\\d{2,4}\\b)", re.IGNORECASE), "Oct"),
    (re.compile(r"(?<=\\d[-/.])0ov(?=[-/.]\\d{2,4}\\b)", re.IGNORECASE), "Nov"),
    (re.compile(r"(?<=\\d[-/.])0ar(?=[-/.]\\d{2,4}\\b)", re.IGNORECASE), "Mar"),
]


def _normalize_ocr_period(text):
    """Repair only narrowly recognizable OCR-corrupted month tokens.

    Example:
        31-0ec-25 -> 31-Dec-25

    This is intentionally restricted to date-shaped strings so financial
    numeric values are never altered by these OCR repairs.
    """
    if not text:
        return text

    normalized = str(text).strip()
    for pattern, replacement in _OCR_DATE_MONTH_FIXES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _looks_like_actual_period(text) -> bool:
    """Strict: the FULL string must be one of a small set of concrete
    period shapes (quarter/year label, real date, audit-status word),
    not just contain a digit or keyword somewhere in it — see the
    module note above for why the looser check isn't safe to trust for
    an actual output value."""
    if not text:
        return False
    stripped = _normalize_ocr_period(text)
    if not stripped or len(stripped) > 30:
        return False
    return bool(_STRICT_PERIOD_RE.match(stripped))


# -------------------------------------------------------------
# Label column detection — usually column 0, but some reports prefix
# a serial-number column ("1.", "2.", "3.") before the real metric
# label. Deciding this from the data (not just assuming column 0)
# avoids extracting "1", "2", "3" as metric names.
# -------------------------------------------------------------

_SERIAL_CELL_RE = re.compile(r"^\(?\d{1,3}\)?\.?$")


def _label_column_index(headers, rows) -> int:
    if len(headers) < 3 or not rows:
        return 0
    sample = [r for r in rows[:6] if r]
    if not sample:
        return 0
    numeric_first = sum(
        1 for r in sample if r and _SERIAL_CELL_RE.match(str(r[0]).strip())
    )
    return 1 if numeric_first / len(sample) >= 0.6 else 0


# -------------------------------------------------------------
# Metric-label validation — rejects structural rows (section markers,
# "Notes", serial-number headers) and gibberish, but deliberately does
# NOT reject a bare "Total" — "Total Revenue" / "Total" lines are
# legitimate aggregate metrics in financial statements.
# -------------------------------------------------------------

_NON_METRIC_LABEL_RE = re.compile(
    r"^(note[s]?|continued|contd\.?|particulars|sr\.?\s*no\.?|s\.?\s*no\.?)$",
    re.IGNORECASE,
)


def _looks_like_metric_label(label: str) -> bool:
    if not label:
        return False
    stripped = label.strip()
    if len(stripped) < 2 or len(stripped) > 150:
        return False
    if _NON_METRIC_LABEL_RE.match(stripped):
        return False
    if not any(c.isalpha() for c in stripped):
        return False
    return True


# -------------------------------------------------------------
# Numeric cell parsing. Handles the common OCR/formatting corruption
# this pipeline actually sees (stray spaces around commas/decimals,
# apostrophes splicing digits apart, parenthesized negatives, currency
# symbols, percent signs) but deliberately stops short of trying to
# repair character-level misreads (e.g. "1't.92" for "11.92") — that
# kind of corruption can't be fixed without guessing, so the cell is
# just dropped instead, same as if this module never saw it.
# -------------------------------------------------------------

_OCR_SPACING_FIXES = [
    (re.compile(r"(\d)\s+,"), r"\1,"),
    (re.compile(r",\s+(\d)"), r",\1"),
    (re.compile(r"(\d)\s+\.\s+(\d)"), r"\1.\2"),
    (re.compile(r"(\d)'(\d)"), r"\1\2"),
]

_NO_DATA_TOKENS = {"-", "--", "—", "–", "na", "n/a", "nil", "nan", ""}

_CURRENCY_PREFIX_RE = re.compile(r"^(₹|rs\.?|inr|usd|\$)\s*", re.IGNORECASE)

_CLEAN_NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")


def _parse_numeric_cell(raw) -> tuple[str | None, str | None]:
    """Returns (value_string, unit_or_None), or (None, None) if the
    cell isn't confidently a plain number."""
    if raw is None:
        return None, None

    text = str(raw).strip()
    if text.lower() in _NO_DATA_TOKENS:
        return None, None

    negative = False

    paren = re.fullmatch(r"\((.*)\)", text)
    if paren:
        negative = True
        text = paren.group(1).strip()

    unit = None
    if text.endswith("%"):
        unit = "%"
        text = text[:-1].strip()

    text = _CURRENCY_PREFIX_RE.sub("", text).strip()

    for pattern, repl in _OCR_SPACING_FIXES:
        text = pattern.sub(repl, text)
    text = text.replace(" ", "")

    if text.endswith("-"):
        negative = True
        text = text[:-1]
    if text.startswith("-"):
        negative = True
        text = text[1:]

    text = text.replace(",", "")

    if not text or not _CLEAN_NUMBER_RE.fullmatch(text):
        # Whatever's left (stray letters, mixed junk, an OCR misread
        # like "1't92") isn't safely recoverable — bail rather than
        # guess at what digit it was supposed to be.
        return None, None

    value = f"-{text}" if negative else text
    return value, unit


# -------------------------------------------------------------
# Table-level unit detection — financial reports state the scale once
# ("₹ in Crore", "Rs. in Lakhs") in the table title or the heading
# above it, not per cell. Mirrors the canonical unit strings
# graph_builder.py's TABLE_PROMPT rule asks the LLM path to use, so
# both paths produce comparable unit values in Neo4j.
# -------------------------------------------------------------

_UNIT_PATTERNS = [
    (re.compile(r"(₹|rs\.?)\s*(in\s*)?crore", re.IGNORECASE), "Crore INR"),
    (re.compile(r"(₹|rs\.?)\s*(in\s*)?lakh", re.IGNORECASE), "Lakh INR"),
    (re.compile(r"(₹|rs\.?|inr)\s*(in\s*)?million", re.IGNORECASE), "Million INR"),
    (re.compile(r"(₹|rs\.?|inr)\s*(in\s*)?billion", re.IGNORECASE), "Billion INR"),
    (re.compile(r"(usd|\$)\s*(in\s*)?million", re.IGNORECASE), "Million USD"),
    (re.compile(r"(usd|\$)\s*(in\s*)?billion", re.IGNORECASE), "Billion USD"),
    (re.compile(r"\bcrore\b", re.IGNORECASE), "Crore INR"),
    (re.compile(r"\blakh[s]?\b", re.IGNORECASE), "Lakh INR"),
]


# -------------------------------------------------------------
# Context-text period recovery — last resort, used ONLY when the
# table's own headers/rows didn't yield a valid period per value
# column. Two real-world observed failure shapes so far (both
# CreditAccess Grameen Q2-2026):
#
#   1. chunk cdc48dd41f12b577 (balance sheet, "As at ..." style):
#      pdfplumber read a 3-line header by physical line instead of by
#      column, so each column's date got split — month+day landed on
#      one line ("As at Sep 30,"), the year on another, later line
#      ("2025"). _recover_split_as_at below reassembles these.
#
#   2. chunk c0375f90e5fac45d (cash-flow statement, "For the ... ended
#      ..." style): same misread, but here the three full dates happen
#      to survive as contiguous "<Month> <Day>,<Year>" text — no split
#      needed, just three dates sitting in a row in the page text.
#      _recover_contiguous_dates below handles this, simpler shape.
#
# Both strategies are deliberately narrow — each only recognizes one
# specific, common Indian financial-statement header shape, bounded to
# a header_window that stops before the table's own first real data
# row starts (see _header_window below) so neither can wander into
# real row values and mistake a number there for part of a date. If
# neither strategy produces exactly one confident, validly-shaped
# period per value column, this returns None and the caller falls back
# to whatever it would have done anyway (fail closed, same as before
# either of these existed) — it never guesses.
# -------------------------------------------------------------

_MONTH_ABBR = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}

_AS_AT_FRAGMENT_RE = re.compile(
    r"as\s+at\s+([A-Za-z]{3,9})\s*(\d{1,2})?\s*,?", re.IGNORECASE
)

# A "day,year" or bare "year" continuation token, e.g. "31,2025" or
# "2025" — the leftover half of a header cell that got split onto a
# different physical line than its "As at <Month>" fragment.
_YEAR_CONTINUATION_RE = re.compile(r"\b(?:(\d{1,2})\s*,\s*)?(\d{4})\b")

# A full, contiguous date — "September 30,2025", "March 31, 2025" —
# for header shapes where the date didn't get split across lines.
_FULL_DATE_RE = re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2})\s*,\s*(\d{4})", re.IGNORECASE)

_HEADER_WINDOW_START_RE = re.compile(r"particulars", re.IGNORECASE)
# Fallback boundary for where header fragments stop and real row data
# begins, used only when the caller can't supply the table's own first
# row label (see _header_window below) to find that boundary exactly.
_HEADER_WINDOW_END_FALLBACK_RE = re.compile(
    r"\bassets\b|\bliabilities\b|\bincome\b|\bexpenses\b|\bcash\s+flow\b|\(a\)",
    re.IGNORECASE,
)


def _header_window(context_text: str, first_row_label: str | None) -> str:
    """Slices context_text down to just the header block: after
    "Particulars" (if present) and before the table's own first real
    data row starts. Bounding the end is what keeps a real data-row
    number from ever being mistaken for part of a split-off header
    date — using the table's own first row label to find that boundary
    is far more reliable than guessing from a fixed keyword list, since
    it's the exact text that comes right after the header in the
    source PDF, whatever kind of statement this is."""
    start_match = _HEADER_WINDOW_START_RE.search(context_text)
    window = context_text[start_match.end() :] if start_match else context_text

    if first_row_label:
        # OCR/pdfplumber whitespace is unreliable (extra/missing spaces
        # around punctuation), so match the label with flexible
        # whitespace rather than requiring an exact substring — and
        # drop a leading "(a)"/"1."-style serial prefix, which is
        # table-cell structure that doesn't appear in the flowing page
        # text the same way.
        core = re.sub(r"^\(?\w{1,3}\)?[.)]?\s*", "", first_row_label).strip()
        if len(core) >= 4:
            pattern = re.escape(core)
            pattern = re.sub(r"\\\s+", r"\\s*", pattern)
            label_match = re.search(pattern, window, re.IGNORECASE)
            if label_match:
                return window[: label_match.start()]

    end_match = _HEADER_WINDOW_END_FALLBACK_RE.search(window)
    return window[: end_match.start()] if end_match else window


def _recover_contiguous_dates(header_window: str, n_periods: int) -> list[str] | None:
    """Strategy for header shapes where each column's full date
    ("September 30,2025") survived as one contiguous match — no
    reassembly needed, just read them off in order."""
    matches = list(_FULL_DATE_RE.finditer(header_window))
    if len(matches) != n_periods:
        return None

    periods = []
    for m in matches:
        month = _MONTH_ABBR.get(m.group(1).lower())
        if month is None:
            return None
        periods.append(f"{month} {int(m.group(2))}, {m.group(3)}")

    if len(set(periods)) != n_periods or not all(
        _looks_like_actual_period(p) for p in periods
    ):
        return None
    return periods


def _recover_split_as_at(header_window: str, n_periods: int) -> list[str] | None:
    """Strategy for header shapes where each column's date got split
    across two physical lines: "As at <Month> <Day>," on one line, the
    matching "<Day>,<Year>" or bare "<Year>" continuation later on."""
    as_at_matches = list(_AS_AT_FRAGMENT_RE.finditer(header_window))
    if len(as_at_matches) != n_periods:
        return None

    months = [_MONTH_ABBR.get(m.group(1).lower()) for m in as_at_matches]
    if any(mo is None for mo in months):
        return None
    days = [m.group(2) for m in as_at_matches]

    # Look for year-continuation tokens after the last "as at" fragment
    # (where the split-off day/year half of each header cell lands),
    # skipping past a lone leftover serial-column token like "No.", but
    # still bounded to the header block so a real data-row number can't
    # be mistaken for one.
    tail = header_window[as_at_matches[-1].end() :]
    year_matches = list(_YEAR_CONTINUATION_RE.finditer(tail))

    periods = []
    year_idx = 0
    for month, day in zip(months, days):
        if day is None:
            if year_idx >= len(year_matches):
                return None
            day, year = year_matches[year_idx].groups()
            year_idx += 1
            if day is None:
                return None
        else:
            if year_idx >= len(year_matches):
                return None
            _, year = year_matches[year_idx].groups()
            year_idx += 1
        # "Sep 30, 2025" shape — matches _looks_like_actual_period's
        # "[a-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4}" branch.
        periods.append(f"{month} {int(day)}, {year}")

    if len(set(periods)) != n_periods or not all(
        _looks_like_actual_period(p) for p in periods
    ):
        return None
    return periods


def _recover_periods_from_context(
    context_text, n_periods: int, first_row_label: str | None = None
) -> list[str] | None:
    """Try to reconstruct exactly `n_periods` real period labels from
    nearby page text, trying each known header shape in turn. Returns
    None (never []) unless one strategy produces a confident,
    unambiguous match — see module note above."""
    if not context_text or n_periods < 1:
        return None

    header_window = _header_window(context_text, first_row_label)

    return _recover_contiguous_dates(header_window, n_periods) or _recover_split_as_at(
        header_window, n_periods
    )


def _detect_table_unit(chunk) -> str | None:
    context = " ".join(str(chunk.get(field) or "") for field in ("title", "heading"))
    if not context.strip():
        return None
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(context):
            return unit
    return None


# -------------------------------------------------------------
# Public API
# -------------------------------------------------------------


def extract_metrics_from_table(chunk, context_text: str | None = None) -> list[dict]:
    """
    chunk: a table-type chunk dict from chunker.py's table_chunk(), i.e.
    one with "headers" (list[str]) and "rows" (list[list[str]]) keys
    holding the raw, unresolved pdfplumber output.

    context_text: optional nearby page text (e.g. the sibling text
    chunks graph_builder.py extracted from the same PDF page) used ONLY
    as a last resort to recover real period labels when the table's own
    headers/rows come back blank or corrupted — see
    _recover_periods_from_context above. Passing None reproduces the
    exact prior behavior (fail closed, no recovery attempted).

    Returns a list of {"name", "value", "unit", "period"} dicts, or []
    if the table can't be confidently parsed this way (graph_builder.py
    then falls back to the LLM-based TABLE_PROMPT for this chunk).
    """
    headers = chunk.get("headers") or []
    rows = chunk.get("rows") or []

    if not rows:
        return []

    headers, rows = _resolve_headers(headers, rows)

    if len(headers) < 2:
        return []  # no separate label + value column(s) to align

    label_col = _label_column_index(headers, rows)

    # _label_column_index shifting to column 1 means column 0 looks
    # like a serial-number prefix ("1.", "2.") — that column isn't the
    # label AND isn't a real period/value column either, so it has to
    # be excluded from both, not just from being the label. Leaving it
    # in value_cols meant a bogus "period" like the literal header text
    # "Sr No" was being written as if it were a real reporting period.
    excluded_cols = {label_col}
    if label_col != 0:
        excluded_cols.add(0)
    value_cols = [i for i in range(len(headers)) if i not in excluded_cols]

    if not value_cols:
        return []

    # First real data row's label — used only as a boundary marker for
    # context-text period recovery below (see _header_window): it's
    # the exact text that immediately follows the header in the source
    # PDF, so it's a far more reliable "where the header ends" signal
    # than any fixed keyword list, regardless of statement type.
    first_row_label = None
    for row in rows:
        if row and label_col < len(row) and str(row[label_col] or "").strip():
            first_row_label = str(row[label_col]).strip()
            break

    # Multiple value columns need distinct, non-blank periods — writing
    # metrics from two columns under the same (or blank) period would
    # collide onto the same metric_id in Neo4j and silently overwrite
    # all but the last column's value (see graph_builder.py's
    # _tx_write_metrics comment on why period is part of the id). A
    # single value column has no such ambiguity even with a blank period.
    #
    # Distinctness alone isn't enough, though: two DIFFERENT fragments
    # of a mis-sliced prose sentence are also "distinct and non-blank"
    # while being nowhere near real period labels. Requiring every
    # period to additionally match a real period SHAPE is what actually
    # catches that case (see _looks_like_actual_period's docstring) —
    # and if the header row for this table is genuinely just chopped
    # prose, that's true of the whole row, not one column, so failing
    # the whole table closed here (rather than dropping just the bad
    # columns) is the same fail-closed contract used everywhere else in
    # this module.
    if len(value_cols) > 1:
        periods = [_normalize_ocr_period(headers[i]) for i in value_cols]

        if all(not p for p in periods):
            # EVERY period is blank — a different, unambiguous signal
            # from "some blank, some not" (still handled below, still
            # fails closed). This is pdfplumber losing the header row
            # entirely on a borderless table: a real financial table
            # (label + N numbers per row, same N across almost every
            # row) with no header text at all. Observed in production —
            # a clean 6-column quarter/half-year/year table came back
            # as headers=["","","","","","",""] and was sent through
            # the LLM fallback, which took 42 minutes and still failed
            # to parse (see graph_builder.py's TABLE_PROMPT fallback
            # path) for the exact numbers visible right here the whole
            # time. We still don't know what each column MEANS
            # semantically — refusing to guess a real period label is
            # correct — but positional labels ("col_1".."col_N") keep
            # every value distinct and attributable to the right column
            # without asserting a calendar period this parser can't see.
            #
            # Gated on row-shape consistency (not just "headers are
            # blank") because ragged per-row value-cell counts is the
            # actual signature of the mis-sliced-prose failure mode
            # _looks_like_actual_period defends against — a real table
            # has nearly the same column count on almost every row; a
            # paragraph sliced into fake columns does not.
            row_value_counts = [
                sum(
                    1
                    for c in value_cols
                    if c < len(row) and _parse_numeric_cell(row[c])[0] is not None
                )
                for row in rows
                if row and label_col < len(row) and str(row[label_col] or "").strip()
            ]
            if not row_value_counts:
                return []
            mode_count = max(set(row_value_counts), key=row_value_counts.count)
            if mode_count < max(2, len(value_cols) - 2):
                # Even the most common row doesn't look like it's
                # actually filling most of the value columns — too
                # sparse to trust as "one real table with a lost header"
                # rather than something more broken.
                return []
            consistent = sum(1 for c in row_value_counts if abs(c - mode_count) <= 1)
            if consistent / len(row_value_counts) < 0.8:
                return []
            headers = list(headers)
            for position, col in enumerate(value_cols, start=1):
                headers[col] = f"col_{position}"
        elif len(set(periods)) != len(periods) or any(not p for p in periods):
            recovered = _recover_periods_from_context(
                context_text, len(value_cols), first_row_label
            )
            if recovered is None:
                return []
            headers = list(headers)
            for col, period in zip(value_cols, recovered):
                headers[col] = period
        elif not all(_looks_like_actual_period(p) for p in periods):
            recovered = _recover_periods_from_context(
                context_text, len(value_cols), first_row_label
            )
            if recovered is None:
                return []
            headers = list(headers)
            for col, period in zip(value_cols, recovered):
                headers[col] = period

    table_unit = _detect_table_unit(chunk)

    metrics = []

    for row in rows:
        if not row or label_col >= len(row):
            continue

        label = str(row[label_col] or "").strip()
        if not _looks_like_metric_label(label):
            continue

        for col in value_cols:
            raw_cell = row[col] if col < len(row) else ""
            value, cell_unit = _parse_numeric_cell(raw_cell)
            if value is None:
                continue

            header_text = str(headers[col]).strip() if col < len(headers) else ""
            if len(value_cols) == 1:
                # A lone value column's header is often a generic label
                # ("Value", "Amount") rather than an actual period — the
                # multi-column ambiguity check above doesn't apply here
                # since there's nothing to disambiguate, so nothing
                # forces this header to be a real period. Only use it
                # when it actually looks like one; otherwise leave period
                # blank rather than store a misleading value like "Value".
                period = header_text if _looks_like_actual_period(header_text) else ""
            else:
                period = _normalize_ocr_period(header_text)

            metrics.append(
                {
                    "name": label,
                    "value": value,
                    "unit": cell_unit or table_unit,
                    "period": period,
                }
            )

    return metrics


# -------------------------------------------------------------
# CLI (quick manual check against one saved chunk JSON, e.g. a single
# entry copied out of data/processed/<Company>/<Year>/<Quarter>/*.json)
# -------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("Usage: python table_metrics.py chunk.json")
        raise SystemExit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)

    test_chunk = data[0] if isinstance(data, list) else data

    result = extract_metrics_from_table(test_chunk)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n{len(result)} metric(s) extracted", file=sys.stderr)
