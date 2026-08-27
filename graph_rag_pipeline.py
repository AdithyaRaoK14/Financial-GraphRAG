"""
graph_rag_pipeline.py
======================
WHAT THIS FILE DOES:
This is your MAIN pipeline — the one you're trying to prove is better than
plain RAG. For a given question, it:

  1. Routes the question (router.py) to decide pdf/audio/both.
  2. Retrieves the top-k most relevant chunks (retrieval.py) — across
     Chunk, Table, AND AudioChunk nodes.
  3. Pulls related FACTS from the Neo4j knowledge graph — entities,
     metrics, and (for the quarters those chunks belong to) guidance,
     risks, and management sentiment — as a safety net in case the exact
     figure or statement wasn't in the top-k chunks themselves.
  4. Builds a prompt combining the question + retrieved facts + retrieved
     tables + retrieved commentary, and asks the LLM to answer using ONLY
     that evidence.

The key difference from baseline_pipeline.py: this one gives the LLM
structured facts from the graph IN ADDITION to raw chunk text. That's the
whole point of the comparison you're building.
"""

import re
import time

from neo4j import GraphDatabase

import config
import router
import retrieval
import llm_client
from evidence_cleaning import (
    clean_evidence_text,
    format_table_markdown,
    statement_type_tag,
)

_driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
)

_WIDE_TREND_RE = re.compile(
    r"\b(trend(?:ed)?|over the (?:full|past|last|entire)|three.?year|3.?year|"
    r"multi.?year|full period|entire period)\b",
    re.I,
)
_QUARTER_MENTION_RE = re.compile(r"\bQ[1-4]\s*FY\s*'?\d{2,4}\b", re.I)

# P-fix: "revenue increase quarter-on-quarter in Q2 FY26" and "revenue
# increase year-on-year in Q3 FY25" only ever name ONE explicit period —
# the comparison period ("the prior quarter", "the same quarter last
# year") is implied, never spelled out as its own "Qn FYnn". Before this
# fix, _extract_fiscal_periods() only ever returned periods that are
# literally written in the question text, so _fetch_period_metric_facts()
# had no way to look up the implied prior-period figure, and the ground
# truth for these questions always requires BOTH figures. Matches both
# the "-on-" and "-over-"/"qoq"/"yoy" phrasings so this stays in sync
# with router.py's COMPARISON_KEYWORDS regardless of which variant a
# question happens to use.
_QOQ_RE = re.compile(
    r"\bquarter[\s-]on[\s-]quarter\b|\bquarter[\s-]over[\s-]quarter\b|\bqoq\b|"
    r"\bsequential(?:ly)?\b",
    re.I,
)
_YOY_RE = re.compile(
    r"\byear[\s-]on[\s-]year\b|\byear[\s-]over[\s-]year\b|\byoy\b", re.I
)
_QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def _adjacent_fiscal_quarter(qtr: str, year: str, offset: int) -> tuple[str, str]:
    """(quarter, year) `offset` quarters away from (qtr, year).

    offset=-1 -> the immediately preceding quarter (for QoQ questions).
    offset=-4 -> the same quarter one fiscal year earlier (for YoY questions).
    Uses floor division so this correctly rolls back across a fiscal-year
    boundary, e.g. Q1 FY26 with offset=-1 -> Q4 FY25.
    """
    idx = _QUARTER_ORDER.index(qtr) + offset
    yr = int(year) + (idx // 4)
    return _QUARTER_ORDER[idx % 4], str(yr)
_COMPANY_SPLIT_RE = re.compile(r"\s+(?:vs\.?|versus|and)\s+|\s*/\s*|\s*,\s*", re.I)
_ALL_COMPANIES_RE = re.compile(r"^\s*all(?:\s+companies)?\s*$", re.I)

_company_roster_cache: list[str] | None = None


def _fact_budget(question: str, routing: dict) -> int:
    """A "how did revenue trend from Q1 FY24 through Q4 FY26" question
    needs one data point per quarter in that range — up to 12 for a
    three-year span — not just the single current value a normal lookup
    needs. The default fact cap (sized for "what was revenue this
    quarter") was truncating these before the LLM ever saw enough of the
    series to describe a trend. Widen the budget only when the question
    actually spans multiple quarters or uses explicit trend language,
    rather than raising it for every question — that would just
    reintroduce noise into simple single-value lookups."""
    base = getattr(config, "TOP_GRAPH_FACTS", 10)
    quarter_mentions = len(set(_QUARTER_MENTION_RE.findall(question or "")))
    if _WIDE_TREND_RE.search(question or "") or quarter_mentions >= 3:
        return max(base, 24)
    if routing.get("question_type") == "temporal" and quarter_mentions >= 2:
        return max(base, 16)
    return base


def _split_companies(company: str) -> list[str]:
    """Ground-truth company fields for comparison questions come as
    "Nykaa (...) vs Tata Consumer Products Limited" or "Jindal Stainless
    Limited / NYKAA" rather than a single name. The combined string
    happens to still pass retrieval.py's substring-fallback company
    matcher often enough by accident (both names survive normalization
    concatenated together), but that's not something to depend on — it
    silently breaks for 3+ companies or any name whose normalized form
    isn't cleanly retained in the combined string. Splitting explicitly
    and fetching each company's facts separately is the deliberate,
    reliable version of the same idea."""
    if not company:
        return [company]
    parts = [p.strip() for p in _COMPANY_SPLIT_RE.split(company) if p.strip()]
    return parts if len(parts) > 1 else [company]


def _company_roster() -> list[str]:
    """Distinct company names actually present in the graph (from Quarter
    nodes), for genuine "All companies" / "among the five companies"
    questions — these need every company's figure to rank or compare, not
    a single company filter that (correctly) returns nothing for a
    company value like "All companies". Cached for the process lifetime;
    read-only, and the roster does not change without a rebuild anyway."""
    global _company_roster_cache
    if _company_roster_cache is not None:
        return _company_roster_cache
    with _driver.session(database=config.NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH (q:Quarter) WHERE q.company IS NOT NULL "
            "RETURN DISTINCT q.company AS company"
        )
        _company_roster_cache = [r["company"] for r in result if r["company"]]
    return _company_roster_cache


FINAL_ANSWER_MARKER = "FINAL ANSWER:"

ANSWER_PROMPT = """You are a financial analyst assistant. Answer the question
using ONLY the evidence below. Be precise with numbers.

RULES:
- Answer directly and concisely, citing the specific figures/quotes used.
- FINAL ANSWER LINE: only the figures the question actually asks for.
  Never add a computed change/difference/percentage unless the question
  explicitly requests one (e.g. "by how much", "what growth rate") —
  "why did revenue increase" and "what was revenue in X and Y" do not
  ask for a computed delta, even if the arithmetic would be correct.
- COPY FIGURES EXACTLY. Never rescale, re-round, move a decimal point,
  or reorder digits — this has repeatedly turned a correct source value
  into a wrong one (e.g. 2873.26 written as 287.326). With any number
  that has a thousands digit, recount the digits before/after the
  decimal against the source before finalizing.
- WHICH VALUE TO USE when the evidence has several similar figures:
  prefer the one whose (a) metric name and (b) period both match the
  question. If exactly one matches on both, use it — other rows (a
  vaguer label, a different period, a sub-line rather than the total)
  are not conflicts and must not stop you answering. This applies to
  both structured "Metrics:" facts and raw table/text rows — a Metrics
  fact is auto-extracted and can carry the wrong value or period for a
  row; when a fact disagrees with its own source table/text, the source
  wins. In these filings a results row runs current-quarter first, then
  earlier periods (e.g. "Basic (in Rs) 21.20 15.76 2.96 ..." → current
  quarter is 21.20) — trust that position over a fact naming a later
  column as current.
- Declining when a well-matched value IS present is as wrong as
  inventing one. Only decline when the evidence genuinely lacks the
  answer, not because it's messy or has unrelated rows alongside it.
- QUALITATIVE questions (what management discussed/highlighted/
  explained): the matching rules above are about NUMBERS, not wording.
  If the evidence contains commentary that is clearly the substance
  asked about, summarize it even if it doesn't use the question's exact
  words. The "Guidance:/Risks:/Sentiment:" lines are a short pre-
  extracted SUMMARY, not the full call — before concluding the evidence
  lacks an answer, check the full retrieved PDF/AUDIO text below them
  too, not just that summary.
- FORWARD-LOOKING/QUALITATIVE COMMENTARY OFTEN APPEARS IN Q&A, NOT ONLY
  IN PREPARED REMARKS. An analyst's specific question and management's
  direct reply to it counts as commentary on that topic just as much as
  an unprompted statement does — do not require an explicit "our
  outlook is..." framing. Confirmed miss: an analyst asked about a
  capex/volume-guidance update, management replied with the revised
  numbers and timeline, and this was answered "management did not
  provide forward-looking commentary" even though that reply was
  exactly the commentary being asked about — it just came as an answer
  to a question rather than a volunteered statement. Read Q&A exchanges
  for their content, not for whether they were prompted or spontaneous.
- "WHY" QUESTIONS THAT NEED BOTH A FIGURE AND A REASON (e.g. "why did
  revenue increase QoQ"): answer in two parts — state the figures from
  the Metrics facts/table (as above), THEN give the reason from the
  PDF/AUDIO commentary. Having both pieces of evidence present is
  enough to answer fully — it is not a harder case that needs more
  certainty than either part alone, and giving only the numbers without
  attempting the reason (when the commentary is right there) is an
  incomplete answer, not a safe one. Only decline the REASON part
  specifically if the commentary genuinely doesn't address it — that
  does not mean declining the whole answer when the figures are known.
- Multiple facts with the IDENTICAL value for the same metric/period are
  the same underlying figure reported twice (e.g. once with a period
  tag, once with a source-row tag) — not a conflict, not a reason for
  caution. Treat them as one fact.
- Match the metric precisely: "revenue" means "Revenue from operations",
  not segment revenue, other income, or total income, unless the
  evidence only offers one of those and clearly labels it as the answer.
- UNITS: never convert currencies or units — copy the number and unit
  exactly as stated, in the evidence's own units. EXCEPT: a unit tag
  is sometimes simply wrong at the source (confirmed cases: a figure
  tagged "Million INR" sitting right next to same-metric neighbors
  tagged "Crore INR" for adjacent periods with matching order of
  magnitude; an Indian company's filing with one fact tagged "Billion
  USD" while every other figure on the same filing is Crore INR). When
  a unit tag looks inconsistent with neighboring same-metric facts or
  every other figure on the same filing, AND a plausible unit would put
  the number back in the same order of magnitude as those neighbors,
  trust the number and treat the unit as whatever the filing actually
  uses (typically Crore INR for an Indian filing) — don't decline or
  keep an implausible unit over one suspicious tag alone.
- ROUNDING is not a conflict: a fact reading "2267" and its source row
  reading "2,267.21" are the same figure — use the more precise one.
  Both this and the unit-tag rule above REQUIRE something to verify
  against (a table row, or a clearly-plausible neighboring figure) —
  they say when NOT to let a superficial mismatch block an answer you
  can otherwise confirm, they do not mean "always answer." If the ONLY
  evidence for a period is a fact tagged
  "confidence=low_possible_subitem_mismatch" and nothing else (no clean
  fact, no readable table row) gives that period's figure, say that
  figure is uncertain — don't state it plainly just because it's the
  only number available. A second fact with the same value from the
  SAME source document/page is not independent corroboration.
- FLATTENED TABLES (no markdown "|" columns, just running text): some
  pages appear as a row of period-label phrases ("3 months period ended
  30/06/2025 / Preceding 3 months ended 31/03/2025 / ... / Year ended
  31/03/2025") followed by a row of numbers in the SAME order. Match by
  strict left-to-right position, and note the LAST number in this shape
  is very often a full-YEAR total, not a quarter — a quarterly question
  should essentially never resolve to whichever number is largest/last;
  find the column whose label actually says "3 months"/"quarter".
- Never substitute a similar-sounding metric for the one actually asked
  about. When a table and narrative text both report the same metric,
  prefer the table unless the narrative explicitly says it supersedes
  or restates the table's figure. When a fact is tagged "company=X",
  it belongs ONLY to X — never mix figures across companies in a
  comparison or ranking.
- If the evidence has multiple EQUALLY well-matched values for the same
  metric/period (not standalone-vs-consolidated — see below), report
  each: if they agree for a stated reason (e.g. a restatement), present
  both and say why; if they genuinely conflict with no stated reason,
  say they conflict rather than silently picking one. This is only for
  genuine ties — a precisely-matched value beside a vaguer one is not a
  tie, use the precise one.
- STANDALONE vs CONSOLIDATED: unless the question explicitly asks for
  standalone, use ONLY the consolidated value — don't report both.
- Fiscal-year labels ("Q2-2026"/"Q2 FY26") follow the company's fiscal
  year (commonly April-March) and won't match the calendar year of the
  underlying dates (e.g. "Q2 FY26" ≈ July-Sept 2025) — a fact tagged
  "(Q2-2026)" whose period text says "September 30, 2025" is the SAME
  period, not a conflict.
- COMPARISON/RANKING/TREND questions: give the requested conclusion
  (which period/company, the specific figures and any explicitly-
  requested difference) in a few sentences — don't restate every
  candidate's evidence line or show arithmetic step by step, compute
  silently. For a RANKING question, state the winner and its figure,
  and include other periods' figures only where the evidence actually
  gives them — don't pad with numbers you didn't find.
- Don't restate a fact's own metadata tags ("(Q2-2025)", "source=",
  "chunk=") as if they were part of your answer.
- REASONING: you may briefly explain first (e.g. why you rejected a
  similar figure), but decide once — don't narrate a multi-round self-
  correction ("wait, actually", "re-evaluating") in the visible
  response. End every response, including a refusal, with exactly one
  line: {final_answer_marker} <only the figure(s)/fact(s) actually
  asked for — no dates, no source references, no rejected candidates>.
  If no explanation is needed, this line can be your entire response.
- If the evidence does not contain the answer, reply with EXACTLY:
  "The retrieved evidence does not contain this information." Do not
  guess or offer a "closest relevant" figure as a stand-in.
- TWO-OR-MORE-VALUE questions (a comparison, QoQ/YoY change, "how did X
  compare to Y"): your final line must include every value asked for,
  not just one side — but only state a figure you actually found; if
  one period's figure is missing, say so rather than inventing it.
  Label each value clearly with punctuation, e.g. "Q2 FY25: 9,745.65
  crore; Q2 FY26: 10,880.89 crore" — never run numbers together with
  only a comma or no label ("9,745.65, 10,880.89" or "98,10880.89" are
  both wrong).

QUESTION:
{question}

GRAPH FACTS (metrics, entities, guidance, risks, sentiment):
{facts}

PDF — TABLES:
{tables}

PDF — NARRATIVE TEXT:
{pdf_text}

AUDIO — MANAGEMENT COMMENTARY / Q&A:
{audio_text}

Re-read the question before answering — it's easy to lose track of exactly
what's being asked after reading through the evidence above.

QUESTION (repeated):
{question}

ANSWER:"""


def _dedupe_facts(facts: set[str]) -> list[str]:
    """Collapse near-duplicate facts that mean the same thing but are
    phrased slightly differently — e.g. the exact same figure showing up
    once because a PDF table reported it and again because the audio
    commentary also mentioned it. We compare a normalized signature
    (lowercase, punctuation stripped, spaces collapsed) and keep only the
    first occurrence of each signature."""
    seen_signatures = set()
    deduped = []
    for fact in sorted(facts):
        signature = re.sub(r"[^a-z0-9]", "", fact.lower())
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(fact)
    return deduped


_MISMATCHED_PERIOD_RE = re.compile(
    r"\b(half.?year|nine.?months?|year ended|annual|twelve.?months?|"
    r"6.?months?|six.?months?|9.?months?|12.?months?)\b",
    re.I,
)
_PRIOR_YEAR_REFERENCE_RE = re.compile(
    r"\b(previous year|prior year|year ago|last year)\b|"
    r"corresponding.{0,20}(previous|prior|last)",
    re.I,
)
_BAD_PERIOD_LABEL_RE = re.compile(r"^\s*col(?:umn)?[_\s]?\d+\s*$", re.I)

# P0 fix: catches a confirmed corrupted-extraction shape — "Q4 FY204" —
# found in a real benchmark row (Info Edge Q4 FY26, segment-table
# artifact). A genuine fiscal-year digit run is either 2 digits ("FY24")
# or 4 digits ("FY2024"); 1, 3, or 5+ digits means a table-parsing/OCR
# error mangled the year (e.g. "2026" losing a leading digit, or a
# neighboring cell's digit bleeding into this one), not a real period.
# This period label previously passed through unflagged because
# _MISMATCHED_PERIOD_RE only looks for aggregation-window words
# (half-year/nine-months/etc.), not malformed digit counts.
_FY_YEAR_DIGITS_RE = re.compile(r"\bFY\s*'?(\d+)\b", re.I)


def _has_implausible_fy_digits(period_text: str) -> bool:
    return any(
        len(m.group(1)) not in (2, 4)
        for m in _FY_YEAR_DIGITS_RE.finditer(str(period_text or ""))
    )


def _has_bad_period_label(period_text: str) -> bool:
    """True if `period` is just a generic column placeholder like "col_3"
    — extraction fell back to a raw column index because it couldn't
    resolve a real period header, meaning the metric's name/value pairing
    for that row is itself unreliable, not just the period label — OR if
    the period text contains an implausible fiscal-year digit count (see
    _has_implausible_fy_digits). Applies regardless of what kind of
    period the question asked about."""
    text = str(period_text or "").strip()
    return bool(_BAD_PERIOD_LABEL_RE.match(text)) or _has_implausible_fy_digits(text)


def _period_aggregation_mismatch(quarter: str, period_text: str) -> bool:
    """True if a metric tagged to a specific quarter (Q1-Q4) has `period`
    text that actually says "half year"/"6 months ended"/"nine months"/
    "year ended"/etc — a different aggregation window than the quarter it
    claims to be, from a table column that got mis-mapped at build time.
    Only meaningful for single-quarter questions — an FY/annual question
    legitimately wants "year ended" period text on what's stored as a
    Q4-tagged record, so callers handling FY questions should skip this
    check."""
    qtr = str(quarter or "").upper()
    if qtr not in {"Q1", "Q2", "Q3", "Q4"}:
        return False
    text = str(period_text or "")
    # A fact whose own period text says "previous year"/"corresponding
    # period... previous" is describing a different fiscal year than
    # whatever this record's quarter/year tag claims to represent - a
    # self-contradiction visible in the text itself, independent of
    # whether the phrase also mentions an aggregation window. Confirmed
    # pattern: "corresponding 3 months ended previous year" tagged as if
    # it were the current Q2-2025 value.
    if _PRIOR_YEAR_REFERENCE_RE.search(text):
        return True
    return bool(_MISMATCHED_PERIOD_RE.search(text))


_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# "September 30, 2025" / "31 March 2026" / "30-June-2025"
_PERIOD_DATE_NAMED_RE = re.compile(
    r"\b(?:(\d{1,2})\s*[-/ ]\s*)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"\s*[-/ ,]\s*(?:(\d{1,2})\s*[-/ ,]\s*)?(\d{4})\b",
    re.I,
)
# "31/03/2025" or "30-06-2024"
_PERIOD_DATE_SLASHED_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
# "30062024" — an unseparated DDMMYYYY run, seen in real corrupted labels
_PERIOD_DATE_RUNON_RE = re.compile(r"\b(\d{2})(\d{2})(\d{4})\b")


_QUARTER_STYLE_PERIOD_RE = re.compile(r"\b(Q[1-4])\s*FY\s*(\d{2}|\d{4})\b", re.I)


def _parse_quarter_style_period(period_text: str):
    """Extract ('Q1', 2024) from a quarter-style period label like
    'Q1 FY2024' or 'Q1 FY24'.

    P16 fix: _period_label_contradicts_tag() only understood DATE-style
    labels ('December 31, 2024'), so quarter-style labels sailed through
    unchecked. Confirmed live on CreditAccess Grameen Basic EPS, where
    three records were all tagged (Q4-2026) but carried period labels
    'Q1 FY2024' / 'Q1 FY2025' — plainly contradicting their own quarter
    tag. Nothing flagged them, and the model picked one of the wrong
    ones (15.76 instead of the ground truth 21.20)."""
    m = _QUARTER_STYLE_PERIOD_RE.search(str(period_text or ""))
    if not m:
        return None
    qtr = m.group(1).upper()
    yr = m.group(2)
    year = int(yr) if len(yr) == 4 else 2000 + int(yr)
    return qtr, year


def _parse_period_label_month_year(period_text: str):
    """Extract (month, calendar_year) from a period label when it states a
    specific date. Returns None when no confident date is present — a
    label with no parseable date is left alone, never guessed at."""
    text = str(period_text or "")

    m = _PERIOD_DATE_NAMED_RE.search(text)
    if m:
        month = _MONTH_NAMES.get(m.group(2)[:3].lower())
        if month:
            return month, int(m.group(4))

    for pattern in (_PERIOD_DATE_SLASHED_RE, _PERIOD_DATE_RUNON_RE):
        m = pattern.search(text)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31 and 1990 <= year <= 2100:
                return month, year
    return None


def _fiscal_quarter_year(month: int, calendar_year: int):
    """Indian fiscal calendar (April-March), matching this dataset's
    convention where year=2026 means FY26 (April 2025 - March 2026):
    Apr-Jun = Q1, Jul-Sep = Q2, Oct-Dec = Q3, Jan-Mar = Q4."""
    if 4 <= month <= 6:
        return "Q1", calendar_year + 1
    if 7 <= month <= 9:
        return "Q2", calendar_year + 1
    if 10 <= month <= 12:
        return "Q3", calendar_year + 1
    return "Q4", calendar_year


def _period_label_contradicts_tag(quarter: str, year: str, period_text: str) -> bool:
    """True when a fact's period label states a specific date that maps to
    a DIFFERENT fiscal quarter/year than the fact's own (quarter, year)
    tag — i.e. the stored period tag is mislabeled.

    P8 fix, the highest-impact issue found in the 45-question run.
    Confirmed on multiple real rows: the graph contains Metric records
    whose (Qn-YYYY) tag disagrees with their own period label, e.g.
      "Revenue from operations = 2,267.21 (Q3-2026) [period=December 31, 2024]"
    December 2024 is Q3 FY25, not Q3 FY26 — and 2,267.21 is indeed the
    Q3 FY25 figure. These mislabeled records sit alongside the CORRECT,
    cleanly-recovered facts for the same question, and the LLM
    repeatedly picked the mislabeled one, producing the wrong prior/
    current period value on comparison questions (confirmed on Nykaa
    Q3 FY26, Info Edge Q1 FY26, Nykaa Q2 FY25, CreditAccess Q1 FY25).

    Deliberately conservative: only fires when a full date can be parsed
    confidently from the label. A label with no date, or an aggregation
    phrase without one, is left to the existing
    _period_aggregation_mismatch / _has_bad_period_label checks."""
    qtr = str(quarter or "").strip().upper()
    if not qtr.startswith("Q"):
        return False  # FY/annual rows are handled by the aggregation check
    try:
        tag_year = int(str(year))
    except (TypeError, ValueError):
        return False

    # P16 fix: check quarter-style labels ('Q1 FY2024') first — these were
    # previously invisible to this function, which only understood
    # date-style labels. See _parse_quarter_style_period().
    quarter_style = _parse_quarter_style_period(period_text)
    if quarter_style:
        return quarter_style != (qtr, tag_year)

    parsed = _parse_period_label_month_year(period_text)
    if not parsed:
        return False

    label_qtr, label_year = _fiscal_quarter_year(*parsed)
    return (label_qtr, label_year) != (qtr, tag_year)


def _period_is_suspect(quarter: str, period_text: str, year: str = None) -> bool:
    """Combined check for _fetch_graph_facts, which has no question-level
    context to know whether an FY question is in play — see
    _has_bad_period_label / _period_aggregation_mismatch /
    _period_label_contradicts_tag for what each part catches and why
    _fetch_period_metric_facts (which does know the requested period
    type) applies them separately instead."""
    return (
        _has_bad_period_label(period_text)
        or _period_aggregation_mismatch(quarter, period_text)
        or (
            year is not None
            and _period_label_contradicts_tag(quarter, year, period_text)
        )
    )


_RATE_METRIC_NAME_RE = re.compile(
    r"\b(growth|margin|change|yoy|qoq|rate|ratio|percentage|ROI|ROE|CAGR)\b", re.I
)


def _unit_is_implausible(unit_val: str, source_text: str) -> bool:
    """True if `unit` mentions USD/$ but that never appears anywhere in
    the metric's own source text - extraction has occasionally
    hallucinated a unit (e.g. "Billion USD" on an INR-crore filing) that
    was never actually printed in the filing."""
    return bool(re.search(r"\busd\b|\$", str(unit_val or ""), re.I)) and not bool(
        re.search(r"\busd\b|\$", str(source_text or ""), re.I)
    )


def _value_is_mismatched_percentage(name: str, value: str) -> bool:
    """True if the raw value looks like a bare percentage (e.g. "17%")
    but the metric's own name gives no indication it's a rate/ratio/
    growth figure (e.g. "Revenue from operations" is an absolute Crore
    figure, not a percentage) - a growth-rate row from a nearby table
    column that got misattributed to the absolute-value metric name
    during extraction."""
    if not re.match(r"^\s*-?\d+(\.\d+)?\s*%\s*$", str(value or "")):
        return False
    return not _RATE_METRIC_NAME_RE.search(str(name or ""))


# P5 fix: narrowly scoped to "revenue"/"sales"/"turnover" only — the one
# metric family where every company in this corpus has consistently
# reported values in the thousands-of-crore range (confirmed across all
# 5 companies' ground-truth figures), so a single/near-zero digit value
# under this name is essentially always a corrupted extraction, not a
# real headline figure. Deliberately NOT applied to profit or other
# absolute-currency metrics, which can legitimately be small (a weak
# quarter, a near-breakeven result) — rejecting those on a value-size
# heuristic risks discarding a genuinely correct small number. This is
# a narrow safety net for one confirmed failure mode (a recovered
# "Revenue from operations = 1"), not a general plausibility model.
_REVENUE_METRIC_NAME_RE = re.compile(r"\b(revenue|sales|turnover)\b", re.I)


def _value_absent_from_source(value: str, source_text: str) -> bool:
    """True when a Metric node's value does not appear anywhere in the
    chunk text it was extracted from.

    P23 fix. Confirmed on 6 of the 7 rows where GraphRAG lost to
    baseline: evidence sufficiency was 1.0 (the correct figure WAS in the
    retrieved text), baseline read it correctly from that text, and
    GraphRAG instead used a graph fact carrying a wrong value — in one
    case 35,697.03 for a row whose true figure was 9,720.35. Others were
    subtler: 1,437.15 instead of 1,512.03, 3,803.92 instead of 4,214.45.

    A prompt rule (P19) telling the model to prefer the source table over
    a contradicting fact did NOT fix this — an authoritative-looking
    "Metrics:" line outweighs a table the model has to read. So this
    drops the fact instead: if the extractor produced a number that
    isn't even present in its own source chunk, the extraction is wrong
    and the fact should never reach the prompt.

    Deliberately conservative:
      - only drops when source_text is non-empty (no text, no judgement)
      - compares with separators stripped, so 1,512.03 matches "1512.03"
        and "1,512.03"
      - a value that appears anywhere in the source is KEPT, even if it
        is on the wrong row; picking the wrong row is a different problem
        handled by the period/statement filters, and dropping those too
        would throw away good facts.
    """
    v = str(value or "").strip()
    if not v:
        return False
    src = str(source_text or "")
    if not src.strip():
        return False  # nothing to validate against - keep the fact
    flat_src = re.sub(r"[\s,]", "", src)
    flat_v = re.sub(r"[\s,]", "", v)
    if not flat_v:
        return False
    if flat_v in flat_src:
        return False
    # Also accept a trailing-zero variant ("21.2" stored, "21.20" printed)
    if flat_v.endswith(".0") and flat_v[:-2] in flat_src:
        return False
    if "." in flat_v and flat_v.rstrip("0").rstrip(".") in flat_src:
        return False
    return True


def _revenue_value_is_implausibly_small(name: str, value: str) -> bool:
    """True if `name` is a revenue/sales/turnover metric and `value` is
    a suspiciously tiny number (< 10) for it. Confirmed real case: a
    recovered "Revenue from operations = 1" (Q3-2026) fact for
    CreditAccess Grameen — a table-extraction error grabbed a stray
    digit (likely a footnote marker or an unrelated sub-line-item, see
    the "(a)/(b)/(c)..." lettered sub-item mislabeling found in the same
    table) instead of the actual headline figure, which should have been
    in the hundreds or thousands of crore."""
    name = str(name or "")
    if not _REVENUE_METRIC_NAME_RE.search(name):
        return False
    try:
        numeric_value = float(str(value or "").replace(",", "").strip())
    except (TypeError, ValueError):
        return False
    return abs(numeric_value) < 10


def _fetch_graph_facts(chunk_ids: list[str]) -> list[str]:
    """
    Pulls structured facts connected to the retrieved chunks: metrics and
    entity relationships attached directly to those chunk/table/audio
    nodes, plus guidance, risks, and management sentiment recorded on the
    Quarter each chunk belongs to.

    Node ids in this graph are stored as `n.id` (the stable chunk_id),
    not `n.chunk_id` as a property name — and metrics are connected via
    :CONTAINS_METRIC, not :MENTIONS (see graph_builder.py).
    """
    if not chunk_ids:
        return []

    facts = set()

    with _driver.session(database=config.NEO4J_DATABASE) as session:
        # Metric facts directly attached to these nodes (any label —
        # Chunk, Table, or AudioChunk can all have metrics attached)
        result = session.run(
            """
                MATCH (n)-[:CONTAINS_METRIC]->(m:Metric)
                WHERE n.id IN $ids
                OPTIONAL MATCH (src) WHERE src.id = m.source_chunk_id
                RETURN m.name AS name, m.value AS value, m.unit AS unit,
                       m.quarter AS quarter, m.year AS year,
                       m.period AS period, m.source_chunk_id AS source_chunk_id,
                       m.source_document AS source_document, m.source_page AS source_page,
                       m.company AS company,
                       coalesce(src.text, src.embedding_text, n.text, n.embedding_text, '') AS source_text
                """,
            ids=chunk_ids,
        )
        for r in result:
            source_text = r.get("source_text") or ""

            # See _period_is_suspect() docstring for what this catches.
            # Read-only skip — nothing is written back to Neo4j.
            if _period_is_suspect(r.get("quarter"), r.get("period"), r.get("year")):
                continue

            # Sanity-check the unit and value against the metric's own
            # source text / name. See _unit_is_implausible /
            # _value_is_mismatched_percentage docstrings for what each
            # catches. Drop rather than hand the LLM a confidently-wrong
            # number/unit pair.
            if _unit_is_implausible(r.get("unit"), source_text):
                continue
            if _value_is_mismatched_percentage(r.get("name"), r.get("value")):
                continue
            if _revenue_value_is_implausibly_small(r.get("name"), r.get("value")):
                continue
            # P23: drop facts whose value isn't even in their own source
            # chunk - a wrong extraction, not a wrong row.
            if _value_absent_from_source(r.get("value"), source_text):
                continue

            # P7 fix: confirmed live bug — this function (used whenever a
            # chunk with an attached Metric node gets retrieved normally,
            # regardless of question wording) has NO recovery path at all;
            # it always used the stored m.value directly. The P6 recovery
            # fix only lives in _fetch_period_metric_facts(), a completely
            # separate function keyed on an EXACT match between the
            # query-side metric name ("Basic EPS") and the stored node
            # name — but the real stored name is "Earning per share (EPS)
            # - Basic", which never exactly matches, so that function can
            # never find this record at all (confirmed: the exact-match
            # Cypher query returns zero rows for this metric/company/
            # period). This is the actual live path supplying the wrong
            # value (11.23 instead of 10.82) for Jindal Stainless Q4 FY26
            # Basic EPS. Recovery here uses the record's OWN stored name
            # (r["name"]) rather than a query-side candidate — the EPS
            # fallback pattern in _source_metric_current_period_value()
            # only needs "basic"/"diluted" to appear as a word in
            # whatever name is passed, which the stored name satisfies
            # even though it's phrased differently from the candidate.
            recovered_value = None
            recovered_unit = None
            if r.get("quarter") and str(r.get("quarter")).upper() != "FY":
                recovered_value = _source_metric_current_period_value(
                    source_text, r.get("name") or ""
                )
                if recovered_value is not None:
                    if _value_is_mismatched_percentage(r.get("name"), recovered_value):
                        recovered_value = None
                    elif _revenue_value_is_implausibly_small(
                        r.get("name"), recovered_value
                    ):
                        recovered_value = None
                    elif not _unit_is_implausible(r.get("unit"), source_text):
                        recovered_unit = r.get("unit")

            use_value = recovered_value if recovered_value is not None else r["value"]
            unit = f" {recovered_unit}" if recovered_unit else ""
            meta = []
            # Tagged first — the most important disambiguator once
            # retrieval spans multiple companies (comparison/ranking
            # questions), and harmless context otherwise.
            if r.get("company"):
                meta.append(f"company={r['company']}")
            statement = _infer_statement_type(source_text)
            if statement:
                meta.append(f"statement={statement}")
            if r.get("period"):
                meta.append(f"period={r['period']}")
            if recovered_value is not None:
                meta.append("derived_from=raw_chunk")
            elif _metric_source_has_ambiguous_breakdown(r.get("name"), source_text):
                meta.append("confidence=low_possible_subitem_mismatch")
            if r.get("source_document"):
                meta.append(f"source={r['source_document']}")
            if r.get("source_page") is not None:
                meta.append(f"page={r['source_page']}")
            if r.get("source_chunk_id"):
                meta.append(f"chunk={r['source_chunk_id']}")
            suffix = f" [{'; '.join(meta)}]" if meta else ""
            facts.add(
                f"{r['name']} = {use_value}{unit} ({r['quarter']}-{r['year']}){suffix}"
            )

        # Entity relationships mentioned in these nodes
        result = session.run(
            """
                MATCH (n)-[:MENTIONS]->(e:Entity)-[rel]->(t:Entity)
                WHERE n.id IN $ids
                RETURN e.name AS source, type(rel) AS relation, t.name AS target
                LIMIT 30
                """,
            ids=chunk_ids,
        )
        for r in result:
            facts.add(f"{r['source']} -[{r['relation']}]-> {r['target']}")

        # Guidance, risks, and sentiment for the quarters these
        # chunks belong to (attached to the parent Quarter node)
        result = session.run(
            """
                MATCH (q:Quarter)-[]->(n)
                WHERE n.id IN $ids
                WITH DISTINCT q
                OPTIONAL MATCH (q)-[:HAS_GUIDANCE]->(g:Guidance)
                OPTIONAL MATCH (q)-[:HAS_RISK]->(r:Risk)
                RETURN q.sentiment AS sentiment, q.confidence AS confidence,
                       collect(DISTINCT g.statement) AS guidance,
                       collect(DISTINCT r.statement) AS risks
                """,
            ids=chunk_ids,
        )
        for r in result:
            if r.get("sentiment"):
                conf = r.get("confidence")
                conf_str = f" (confidence {conf})" if conf is not None else ""
                facts.add(f"Management sentiment: {r['sentiment']}{conf_str}")
            for statement in r.get("guidance") or []:
                if statement:
                    facts.add(f"Guidance: {statement}")
            for statement in r.get("risks") or []:
                if statement:
                    facts.add(f"Risk: {statement}")

    return _dedupe_facts(facts)


def _extract_fiscal_periods(
    question: str, year: str = None, quarter: str = None
) -> list[tuple[str, str]]:
    """Extract the fiscal periods that the question actually asks about.

    Explicit Qn FYn pairs are preserved.  Questions such as "highest revenue
    in FY26" or "revenue trend from FY24 to FY26" are expanded to the
    quarter-level records needed for the calculation.
    """
    qtext = question or ""
    found: list[tuple[str, str]] = []

    for q, fy in re.findall(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", qtext, flags=re.I):
        yy = "20" + fy if len(fy) == 2 else fy
        found.append((f"Q{q}", yy))

    # Only one explicit period was named, but the question's phrasing
    # ("quarter-on-quarter" / "year-on-year" etc.) implies a second,
    # unstated comparison period — add it so the period-metric lookup
    # below can fetch both figures deterministically instead of the LLM
    # having to find the prior value unaided in raw retrieved text (see
    # _adjacent_fiscal_quarter for why). Skip when a wide multi-quarter
    # trend is also being asked for; that's handled by needs_quarters
    # below and adding one more single period here would be redundant.
    if len(found) == 1 and not _WIDE_TREND_RE.search(qtext):
        qtr0, yr0 = found[0]
        if _QOQ_RE.search(qtext):
            found.append(_adjacent_fiscal_quarter(qtr0, yr0, -1))
        elif _YOY_RE.search(qtext):
            found.append(_adjacent_fiscal_quarter(qtr0, yr0, -4))

    fy_mentions: list[int] = []
    for fy in re.findall(r"\bFY\s*(\d{2,4})\b", qtext, flags=re.I):
        yy = int("20" + fy) if len(fy) == 2 else int(fy)
        if yy not in fy_mentions:
            fy_mentions.append(yy)

    needs_quarters = bool(
        re.search(
            r"\b(highest|lowest|trend|across|full three-year|full three year|"
            r"full two-year|full two year|all quarters|quarterly figures|"
            r"peak|quarter recorded|two different quarters|different quarters)\b",
            qtext,
            re.I,
        )
    )
    needs_annual = bool(
        re.search(
            r"\b(annual|yearly|fiscal year|year-on-year revenue growth|"
            r"strongest year-on-year)\b",
            qtext,
            re.I,
        )
    )

    if needs_quarters:
        for yy in sorted(fy_mentions):
            for q in ("Q1", "Q2", "Q3", "Q4"):
                found.append((q, str(yy)))

    if needs_annual and not needs_quarters:
        # Annual figures in this graph are stored on the Q4 quarter node with
        # a period label such as "Year ended ...".  FY is an internal lookup
        # marker; _fetch_period_metric_facts maps it to Q4 and filters the
        # returned period labels.
        for yy in sorted(fy_mentions):
            found.append(("FY", str(yy)))

    if year and quarter and str(quarter).upper() in {"Q1", "Q2", "Q3", "Q4"}:
        found.append((str(quarter).upper(), str(year)))

    out, seen = [], set()
    for qtr, yr in found:
        key = (str(qtr).upper(), str(yr))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _metric_name_candidates(question: str) -> list[str]:
    q = (question or "").lower()
    mappings = [
        ("revenue from operations", "Revenue from operations"),
        ("revenue", "Revenue from operations"),
        ("basic earnings per share", "Basic EPS"),
        ("basic eps", "Basic EPS"),
        ("diluted earnings per share", "Diluted EPS"),
        ("diluted eps", "Diluted EPS"),
        ("earnings per share", "Basic EPS"),
        ("profit after tax", "Profit after tax"),
        ("net profit for the period", "Net profit for the period"),
        ("net profit", "Net profit for the period"),
        ("pat", "PAT"),
        ("ebitda", "EBITDA"),
        ("interest income", "Interest income"),
    ]
    return list(dict.fromkeys(metric for needle, metric in mappings if needle in q))


# P0 fix: this module used to define its own copy of
# _requested_statement_type() with a DIFFERENT default (consolidated)
# than retrieval.py's copy (standalone) — a confirmed split-brain bug
# where retrieval ranking and graph-fact resolution silently disagreed
# on which statement type to prefer whenever a question didn't say
# "standalone"/"consolidated" explicitly. retrieval.py's version is now
# the single source of truth (see its docstring); every call site here
# uses retrieval._requested_statement_type directly instead of a local
# copy, so there is exactly one place this default can ever change.


def _infer_statement_type(text: str) -> str | None:
    """Infer statement type from the already-stored source chunk text."""
    t = re.sub(r"\s+", " ", text or "").lower()
    m = re.search(
        r"statement of .*?\b(standalone|consolidated)\b .*?\bfinancial results\b",
        t,
        re.I,
    )
    if m:
        return m.group(1).lower()
    for kind in ("standalone", "consolidated"):
        if re.search(rf"\b{kind}\b\s+(?:financial|results)", t, re.I):
            return kind
    return None


_LETTERED_SUBITEM_MARKER_RE = re.compile(r"\(a\)\s*\(b\)\s*\(c\)", re.I)

# P10 fix: confirmed on real data that a second, structurally different
# CreditAccess text format exists — a "Columns:" key-value dump where
# each lettered sub-item sits on its own line, separated from the next
# by several "= value" lines, rather than "(a) (b) (c)..." appearing
# together on one line:
#     (a) lnterest income
#      = 1105.17
#      = 964.79
#     (b) Fees and commission
#      = 6.05
#     ...
# The adjacent-marker regex above correctly does not match this (there
# is real content between the letters), so a record in this format
# silently got NO ambiguous-breakdown tag at all — including a case
# where the row itself was correctly identified ("Total revenue from
# operations") but its value was OCR-misread (170.03 instead of
# 1,170.03, the leading digit lost in the scan). This checks for the
# same three letters present anywhere in order, without requiring
# adjacency, so a table in either observed format gets flagged.
# Bounded to 300 chars — the real confirmed case has (a)...(c) only 121
# chars apart; unbounded matching risked flagging unrelated documents
# where footnote markers "(a)", "(b)", "(c)" happen to appear far apart
# with no real connection to a revenue breakdown (confirmed: an
# unrelated document with ~250 chars of filler between them false-
# triggered before this bound was added).
_LETTERED_SUBITEM_SCATTERED_RE = re.compile(
    r"\(a\).{0,150}?\(b\).{0,150}?\(c\)", re.I | re.S
)


def _has_lettered_subitem_breakdown(text: str) -> bool:
    """True if EITHER observed lettered-breakdown format is present — see
    _LETTERED_SUBITEM_SCATTERED_RE's docstring for why two patterns exist."""
    return bool(
        _LETTERED_SUBITEM_MARKER_RE.search(text)
        or _LETTERED_SUBITEM_SCATTERED_RE.search(text)
    )


def _metric_source_has_ambiguous_breakdown(metric_name: str, source_text: str) -> bool:
    """True when the metric's own source text shows a lettered sub-item
    breakdown ('(a) (b) (c) (d) (e)') near the metric label AND a clean
    'Total {metric_name}' phrase can't be found nearby.

    P6 fix, confirmed on a real case: CreditAccess Grameen's revenue
    tables list "Revenue from operations" as a SECTION HEADER with five
    sub-items underneath (interest income, fees, etc., each individually
    much smaller than the real total), and this specific company's PDFs
    OCR badly enough that the "Total revenue from operations" row label
    is frequently corrupted past recognition (e.g. "lotal" instead of
    "Total", "levenue"/"opetations" for "Revenue"/"operations") and the
    numbers after it appear in a scrambled, non-linear reading order.
    A naive "first number after the label" recovery in that situation
    reliably grabs a sub-item's value instead of the total — confirmed:
    "Revenue from operations" recovered as 4,900.11, which is actually
    the Interest income sub-item's full-year column, not any quarter's
    total revenue. Rather than guess at a fix for OCR corruption we
    can't reliably reverse across every scan variant, this flags the
    situation so the caller can decline to state a number with false
    confidence."""
    name = str(metric_name or "")
    if not _REVENUE_METRIC_NAME_RE.search(name):
        return False
    text = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not _has_lettered_subitem_breakdown(text):
        return False
    label = re.escape(name)
    has_clean_total = bool(re.search(rf"\btotal\s+{label}\b", text, re.I))
    return not has_clean_total


def _source_metric_current_period_value(
    source_text: str, metric_name: str
) -> str | None:
    """Recover the current-quarter value directly from the source chunk text.

    Some older Metric nodes were built from malformed table columns and can
    therefore contain the wrong period/value pairing. The PDF text chunk still
    contains the original row. For a quarter-specific question, the first
    numeric value on the *total* metric row is the current quarter value.
    This is a read-only recovery path; it never writes to Neo4j.

    P6 fix: when the metric name is an EPS variant (e.g. "Basic EPS"), the
    metric name itself never appears verbatim in the source table — filings
    label the section "Earning per share (EPS)" and the actual data row
    just "a) Basic" / "b) Diluted", so the original literal-phrase match
    silently never fired for EPS at all (confirmed: Jindal Stainless Q4
    FY26 "Basic EPS" — recovery never triggered, so the wrong stored value
    11.23 was used unchanged, even though the raw text's own first-column
    value, 10.82, was correct and recoverable with the right label pattern).

    P6 fix: when the label match lands on a lettered sub-item breakdown
    ("(a) (b) (c)...", see _metric_source_has_ambiguous_breakdown) and no
    clean "Total {label}" phrase can be found, this returns None rather
    than guessing — the numbers in that situation are confirmed to appear
    in a scrambled, unreliable order for at least one real filing.
    """
    import re

    text = re.sub(r"\s+", " ", str(source_text or "")).strip()
    if not text:
        return None

    label = re.escape(metric_name)
    eps_match = re.search(r"\b(basic|diluted)\b", metric_name, re.I)
    # Prefer an explicit total row, because the first occurrence of a metric
    # name in OCR text may be the column heading rather than the data row.
    patterns = [
        rf"\btotal\s+{label}\b(.*?)(?=\bother income\b|\bexpenses?\b|$)",
        rf"\b{label}\b(.*?)(?=\bother income\b|\bexpenses?\b|$)",
    ]
    if eps_match:
        # EPS-specific fallback: match the "Earning per share" section
        # header, then the Basic/Diluted sub-label as its own row —
        # these never appear as one contiguous phrase with the metric
        # name in the source table.
        which = eps_match.group(1)
        patterns.append(
            rf"\bearning\s*per\s*share\b.*?\b{which}\b\s*(.*?)"
            rf"(?=\b(?:basic|diluted)\b|\(eps for the quarter|$)"
        )
    num = r"(?<![A-Za-z])(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Za-z])"
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        span = m.group(1)
        if _has_lettered_subitem_breakdown(span) and not re.match(
            r"^\s*total\b", pat, re.I
        ):
            # We only matched the bare-label (section header) pattern and
            # landed on a lettered sub-item breakdown — the numbers here
            # are confirmed unreliable for at least one real filing.
            # Don't guess; let the caller fall back to declining rather
            # than stating a wrong number confidently.
            continue
        values = re.findall(num, span)
        if values:
            # Return the first value on the current-period row.
            return values[0].replace(",", "")
    return None


_PERIOD_TAG_RE = re.compile(r"\((Q[1-4]|FY)-(\d{4})\)")


def _fact_period_key(fact: str) -> str:
    m = _PERIOD_TAG_RE.search(fact)
    return m.group(0) if m else ""


_PERIOD_TAG_VALUE_RE = re.compile(r"period=([^;\]]+)")


def _fact_has_period_or_statement_tag(fact: str) -> bool:
    """True if this fact carries a tag that legitimately backs its claim
    to a specific period or statement type.

    P0 fix: a bare "period=" substring used to count as legitimate
    regardless of its content — but a corrupted value like
    "period=Q4 FY204" (see _has_bad_period_label) is not a real claim
    to a period, it's an extraction error. Before this fix, any fact
    carrying such a tag was exempt from _prefer_recovered_facts()'s
    fuzzy overlap-based dedup below, letting a corrupted-period
    duplicate of a correctly-recovered fact survive into the LLM's
    context unchallenged (confirmed: Info Edge Q4 FY26 "Total Net
    Sales/Revenue from Operations = 8,050.96 ... period=Q4 FY204").
    _fetch_graph_facts()/_fetch_period_metric_facts() already filter
    these out via _has_bad_period_label before this point, so this is
    a second line of defense for any other fact-producing path (e.g.
    _fetch_temporal_facts(), which does not call _period_is_suspect)
    that might still emit one — a bad period tag here is treated the
    same as no period tag at all, not as a free pass from dedup."""
    m = _PERIOD_TAG_VALUE_RE.search(fact)
    has_valid_period = bool(m) and not _has_bad_period_label(m.group(1).strip())
    return has_valid_period or "statement=" in fact


def _metric_name_tokens(fact: str) -> set[str]:
    name = fact.split(" = ", 1)[0]
    name = re.sub(r"\b(standalone|consolidated)\b", "", name, flags=re.I)
    return {t for t in re.findall(r"[a-z]+", name.lower()) if len(t) > 4}


def _prefer_recovered_facts(facts: list[str]) -> list[str]:
    """A fact tagged "derived_from=raw_chunk" was recovered by matching
    the metric name directly against the "current period" column of its
    source table row — it's been cross-validated in a way a plain
    chunk-attached metric fact (from _fetch_graph_facts, which just reads
    whatever value the graph extraction stored) has not. When both exist
    for the same (company, metric, quarter-year) and disagree, the
    untagged one is more likely the extraction error — e.g. a
    mislabeled/wrong-row value under the same metric name. Drop the
    untagged competitor rather than hand the LLM two candidate numbers
    with no signal for which one to trust.

    Also catches the same problem under a differently-worded metric name:
    an untagged fact (no period=, no statement= — nothing backing its
    claim to be the current period at all) whose name shares substantial
    words with a validated fact for the same company/period — e.g. a
    segment table's "Total Net Sales/Revenue from Operations" row
    competing with the main statement's validated "Revenue from
    operations". Confirmed case: these can even be on a different unit
    scale (Million vs Crore) — not a duplicate, just an unrelated stray
    number with no cross-validation, presented as if it were a real
    competing candidate. A fact that DOES carry its own period/statement
    tag is making its own claim and is left to the standalone/
    consolidated resolver instead of being dropped here."""
    validated = [
        (_fact_company(f), _metric_name_tokens(f), _fact_period_key(f))
        for f in facts
        if "derived_from=raw_chunk" in f
    ]
    if not validated:
        return facts

    recovered_keys = {
        (_fact_company(f), _metric_base_name(f), _fact_period_key(f))
        for f in facts
        if "derived_from=raw_chunk" in f
    }

    result = []
    for f in facts:
        if "derived_from=raw_chunk" in f:
            result.append(f)
            continue
        key = (_fact_company(f), _metric_base_name(f), _fact_period_key(f))
        if key in recovered_keys:
            continue  # exact metric-name match — original behavior
        if _fact_category(f) == "metric" and not _fact_has_period_or_statement_tag(f):
            my_company = _fact_company(f)
            my_period = _fact_period_key(f)
            my_tokens = _metric_name_tokens(f)
            if any(
                my_company == vc and my_period == vp and len(my_tokens & vt) >= 2
                for vc, vt, vp in validated
            ):
                continue  # untagged, unvalidated, name overlaps a validated fact
        result.append(f)
    return result


def _fetch_period_metric_facts(
    question: str,
    company: str = None,
    year: str = None,
    quarter: str = None,
    statement_type: str = None,
) -> list[str]:
    """Read exact metric/period pairs from Neo4j for requested periods.

    This is read-only.  It fixes the ambiguity where a company/quarter has
    both standalone and consolidated values for the same metric.  The old
    implementation returned both, but the fact string did not identify the
    statement type, so the later standalone/consolidated resolver could not
    distinguish them.
    """
    if not company:
        return []

    periods = _extract_fiscal_periods(question, year, quarter)
    metrics = _metric_name_candidates(question)
    if not periods or not metrics:
        return []

    wanted_statement = retrieval._requested_statement_type(question, statement_type)
    facts = set()

    with _driver.session(database=config.NEO4J_DATABASE) as session:
        for metric_name in metrics:
            for qtr, yr in periods:
                query_quarter = "Q4" if qtr == "FY" else qtr
                wanted_company_norm = retrieval._normalize_company_name(company)
                result = session.run(
                    """
                    MATCH (m:Metric)
                    WHERE toUpper(toString(m.quarter)) = toUpper($quarter)
                      AND toString(m.year) = $year
                      AND toLower(toString(m.name)) = toLower($metric)
                    OPTIONAL MATCH (src)
                    WHERE src.id = m.source_chunk_id
                    RETURN m.name AS name, m.value AS value, m.unit AS unit,
                           m.quarter AS quarter, m.year AS year, m.period AS period,
                           m.source_chunk_id AS source_chunk_id,
                           m.source_document AS source_document,
                           m.source_page AS source_page,
                           m.company AS company,
                           coalesce(src.text, src.embedding_text, '') AS source_text
                    ORDER BY m.source_page
                    LIMIT 200
                    """,
                    quarter=query_quarter,
                    year=yr,
                    metric=metric_name,
                )
                # Company match happens in Python (same normalizer/fallback
                # retrieval.py uses) rather than in the Cypher WHERE clause —
                # see retrieval._company_matches for why the naive
                # replace()-chain equality this used to do in Cypher can't
                # handle a ground-truth name like "Nykaa (FSN E-Commerce
                # Ventures Limited)" against a stored "NYKAA".
                records = [
                    r
                    for r in result
                    if retrieval._company_matches(r.get("company"), wanted_company_norm)
                    and not _has_bad_period_label(r.get("period"))
                    and not _period_label_contradicts_tag(
                        r.get("quarter"), r.get("year"), r.get("period")
                    )
                    and (
                        qtr == "FY"
                        or not _period_aggregation_mismatch(
                            r.get("quarter"), r.get("period")
                        )
                    )
                ]

                # P9 fix: confirmed live gap — the P7 fix (EPS metric names
                # never appear verbatim in source text, e.g. stored as
                # "Earning per share (EPS) - Basic" when the candidate is
                # "Basic EPS") was only applied to _fetch_graph_facts(),
                # which only runs when chunk-based retrieval happens to
                # surface the record. THIS function is supposed to be the
                # backup path independent of chunk ranking — but its own
                # Cypher MATCH still requires an exact m.name match, so
                # when chunk retrieval misses (confirmed: Jindal Stainless
                # Q3 FY24 Basic EPS — 0 of 6 supporting chunks retrieved),
                # there was no fallback at all. Widen the match here too,
                # the same way, when the exact match found nothing for an
                # EPS-family candidate.
                if not records and re.search(r"\b(basic|diluted)\b", metric_name, re.I):
                    eps_result = session.run(
                        """
                        MATCH (m:Metric)
                        WHERE toUpper(toString(m.quarter)) = toUpper($quarter)
                          AND toString(m.year) = $year
                          AND toLower(toString(m.name)) CONTAINS 'eps'
                        OPTIONAL MATCH (src)
                        WHERE src.id = m.source_chunk_id
                        RETURN m.name AS name, m.value AS value, m.unit AS unit,
                               m.quarter AS quarter, m.year AS year, m.period AS period,
                               m.source_chunk_id AS source_chunk_id,
                               m.source_document AS source_document,
                               m.source_page AS source_page,
                               m.company AS company,
                               coalesce(src.text, src.embedding_text, '') AS source_text
                        ORDER BY m.source_page
                        LIMIT 200
                        """,
                        quarter=query_quarter,
                        year=yr,
                    )
                    which = re.search(r"\b(basic|diluted)\b", metric_name, re.I).group(
                        1
                    )
                    records = [
                        r
                        for r in eps_result
                        if retrieval._company_matches(
                            r.get("company"), wanted_company_norm
                        )
                        and re.search(rf"\b{which}\b", str(r.get("name") or ""), re.I)
                        and not _has_bad_period_label(r.get("period"))
                        and not _period_label_contradicts_tag(
                            r.get("quarter"), r.get("year"), r.get("period")
                        )
                        and (
                            qtr == "FY"
                            or not _period_aggregation_mismatch(
                                r.get("quarter"), r.get("period")
                            )
                        )
                    ]

                # P11 fix: same class of bug as P9, confirmed on a second
                # metric family — CreditAccess Grameen's real stored name
                # for the bottom-line profit figure is "Profit for the
                # period / year (V-VI)", not "Net profit for the period"
                # (the candidate _metric_name_candidates generates), so
                # the exact-match query above returned nothing even
                # though the correct value (349.21, matching ground
                # truth exactly) exists in Neo4j. Broadened the same way
                # P9 did for EPS — but this table has SEVERAL other
                # "profit"-named rows (before tax, margin, comprehensive
                # income, gross profit) that must NOT be matched, so the
                # filter is an explicit exclusion list rather than a
                # loose CONTAINS.
                #
                # Real, more cautious difference from the EPS case:
                # broadening surfaced TWO records for the same nominal
                # quarter with genuinely different values (349.21 and
                # 826.03, likely a single-quarter figure and a half-
                # year/YTD aggregate that both got tagged with this
                # quarter during extraction) and no period text to tell
                # them apart. Guessing between them would trade an
                # honest miss for a confident wrong answer, so this only
                # returns a broadened match when exactly ONE distinct
                # value survives the exclusion filter — ambiguous cases
                # are left exactly as unanswered as they are today,
                # rather than resolved by a coin flip.
                if not records and re.search(r"\bprofit\b", metric_name, re.I):
                    profit_result = session.run(
                        """
                        MATCH (m:Metric)
                        WHERE toUpper(toString(m.quarter)) = toUpper($quarter)
                          AND toString(m.year) = $year
                          AND toLower(toString(m.name)) CONTAINS 'profit'
                          AND (
                            toLower(toString(m.name)) CONTAINS 'profit for the period'
                            OR toLower(toString(m.name)) CONTAINS 'profit for the year'
                            OR toLower(toString(m.name)) CONTAINS 'net profit'
                          )
                          AND NOT toLower(toString(m.name)) CONTAINS 'before tax'
                          AND NOT toLower(toString(m.name)) CONTAINS 'margin'
                          AND NOT toLower(toString(m.name)) CONTAINS 'comprehensive'
                          AND NOT toLower(toString(m.name)) CONTAINS 'gross'
                          AND NOT toLower(toString(m.name)) CONTAINS 'reclassif'
                          AND NOT toLower(toString(m.name)) CONTAINS 'fair value'
                        OPTIONAL MATCH (src)
                        WHERE src.id = m.source_chunk_id
                        RETURN m.name AS name, m.value AS value, m.unit AS unit,
                               m.quarter AS quarter, m.year AS year, m.period AS period,
                               m.source_chunk_id AS source_chunk_id,
                               m.source_document AS source_document,
                               m.source_page AS source_page,
                               m.company AS company,
                               coalesce(src.text, src.embedding_text, '') AS source_text
                        ORDER BY m.source_page
                        LIMIT 200
                        """,
                        quarter=query_quarter,
                        year=yr,
                    )
                    candidates = [
                        r
                        for r in profit_result
                        if retrieval._company_matches(
                            r.get("company"), wanted_company_norm
                        )
                        and not _has_bad_period_label(r.get("period"))
                        and not _period_label_contradicts_tag(
                            r.get("quarter"), r.get("year"), r.get("period")
                        )
                        and (
                            qtr == "FY"
                            or not _period_aggregation_mismatch(
                                r.get("quarter"), r.get("period")
                            )
                        )
                    ]
                    distinct_values = {
                        str(c.get("value")).strip()
                        for c in candidates
                        if c.get("value") is not None
                    }
                    if len(distinct_values) == 1:
                        records = candidates
                    # else: 0 or 2+ distinct values — leave records empty
                    # rather than guess between conflicting figures.

                if qtr == "FY":
                    records = [
                        r
                        for r in records
                        if re.search(
                            r"\b(year|annual|12 months|twelve months)\b",
                            str(r.get("period") or ""),
                            re.I,
                        )
                    ]

                typed = [
                    r
                    for r in records
                    if _infer_statement_type(r.get("source_text") or "")
                    == wanted_statement
                ]
                selected = typed if typed else records

                # The existing Metric nodes may contain a bad period/value
                # alignment from an earlier table parse. For a concrete
                # quarter, recover the current-quarter value from the raw
                # source chunk as an additional authoritative fact. This is
                # read-only and leaves the existing Neo4j database untouched.
                if qtr != "FY":
                    for r in selected:
                        recovered = _source_metric_current_period_value(
                            r.get("source_text") or "", metric_name
                        )
                        if recovered is None:
                            continue
                        source_statement = _infer_statement_type(
                            r.get("source_text") or ""
                        )
                        if source_statement and source_statement != wanted_statement:
                            continue
                        if _value_is_mismatched_percentage(metric_name, recovered):
                            continue
                        if _revenue_value_is_implausibly_small(metric_name, recovered):
                            continue
                        # P1 fix: confirmed on a real case (NYKAA Q1/Q2 FY25
                        # "Net profit for the period") — the ORIGINAL node's
                        # unit can be part of the SAME bad extraction that
                        # made recovery necessary in the first place (here:
                        # tagged "Billion USD" on a plain-INR-crore filing
                        # with no "USD"/"$" anywhere in its own source
                        # text). Recovery re-derives the VALUE fresh from
                        # raw text, independent of that node's metadata —
                        # gating recovery itself on the old node's already-
                        # known-untrustworthy unit discarded a verified-
                        # correct number (13.64, matching ground truth
                        # exactly) over a separate, already-bad field. Keep
                        # the recovered value; only drop the unit display
                        # when it's the implausible one, rather than either
                        # discarding the value or mislabeling it.
                        node_unit = r.get("unit")
                        unit_is_bad = _unit_is_implausible(
                            node_unit, r.get("source_text")
                        )
                        meta = [
                            f"statement={source_statement or wanted_statement}",
                            "source_row=current-period",
                            "derived_from=raw_chunk",
                        ]
                        if unit_is_bad:
                            meta.append("unit_omitted=implausible_source_unit")
                        if r.get("company"):
                            meta.insert(0, f"company={r['company']}")
                        if r.get("source_document"):
                            meta.append(f"source={r['source_document']}")
                        if r.get("source_page") is not None:
                            meta.append(f"page={r['source_page']}")
                        if r.get("source_chunk_id"):
                            meta.append(f"chunk={r['source_chunk_id']}")
                        unit = f" {node_unit}" if node_unit and not unit_is_bad else ""
                        facts.add(
                            f"{metric_name} = {recovered}{unit} "
                            f"({qtr}-{yr}) [{'; '.join(meta)}]"
                        )
                        break

                for r in selected:
                    if _unit_is_implausible(r.get("unit"), r.get("source_text")):
                        continue
                    if _value_is_mismatched_percentage(r.get("name"), r.get("value")):
                        continue
                    if _revenue_value_is_implausibly_small(
                        r.get("name"), r.get("value")
                    ):
                        continue
                    if _value_absent_from_source(r.get("value"), r.get("source_text")):
                        continue
                    meta = []
                    if r.get("company"):
                        meta.append(f"company={r['company']}")
                    statement = _infer_statement_type(r.get("source_text") or "")
                    if statement:
                        meta.append(f"statement={statement}")
                    if r.get("period"):
                        meta.append(f"period={r['period']}")
                    # P6 fix: flag (don't silently trust) a value whose own
                    # source text shows the same lettered sub-item breakdown
                    # pattern confirmed unreliable above — the stored value
                    # may be a sub-item rather than the true total. Keeping
                    # the fact (rather than dropping it) preserves recall
                    # for cases where it's actually fine; the tag lets the
                    # LLM (and a reader) see this is lower-confidence rather
                    # than presenting it as equally solid as everything else.
                    if _metric_source_has_ambiguous_breakdown(
                        r.get("name"), r.get("source_text")
                    ):
                        meta.append("confidence=low_possible_subitem_mismatch")
                    if r.get("source_document"):
                        meta.append(f"source={r['source_document']}")
                    if r.get("source_page") is not None:
                        meta.append(f"page={r['source_page']}")
                    if r.get("source_chunk_id"):
                        meta.append(f"chunk={r['source_chunk_id']}")
                    suffix = f" [{'; '.join(meta)}]" if meta else ""
                    unit = f" {r['unit']}" if r.get("unit") else ""
                    facts.add(
                        f"{r['name']} = {r['value']}{unit} "
                        f"({r['quarter']}-{r['year']}){suffix}"
                    )

    return sorted(facts)


def _fetch_temporal_facts(chunk_ids: list[str]) -> list[str]:
    """
    Adds "what changed" facts by walking the temporal links that
    temporal_utils.py builds after graph_builder.py finishes loading a
    quarter (NEXT_VALUE/PREVIOUS_VALUE/YOY_CHANGE on Metric, UPDATED_TO
    on Guidance, PERSISTED_TO on Risk) — one hop out from whatever this
    quarter's chunks already surfaced, not a wholesale dump of a
    neighboring quarter's facts. Only called for comparison/trend-
    looking questions (see answer()) since it's extra DB round trips
    that a plain single-quarter lookup doesn't need.

    Percent-change figures come straight from the relationship's stored
    percent_change property (computed once by temporal_utils.py) rather
    than being recomputed here from the raw value strings — same number,
    computed once, not re-derived per question.

    Graceful on a graph built before temporal_utils.py existed: the
    OPTIONAL MATCHes just find nothing and this returns [], same as if
    it were never called.
    """
    if not chunk_ids:
        return []

    facts = set()

    def _pct_suffix(pct):
        if pct is None:
            return ""
        sign = "+" if pct >= 0 else ""
        return f" ({sign}{pct:.1f}%)"

    with _driver.session(database=config.NEO4J_DATABASE) as session:
        # Metric trend: one hop back (PREVIOUS_VALUE) and one hop
        # forward (NEXT_VALUE) from whatever metric this chunk/table/
        # audio segment already contains, plus the true year-over-year
        # pair (YOY_CHANGE) if one exists — NEXT_VALUE's neighbor isn't
        # necessarily a year away if there's a gap in the data.
        result = session.run(
            """
            MATCH (n)-[:CONTAINS_METRIC]->(m:Metric)
            WHERE n.id IN $ids
            OPTIONAL MATCH (m)-[prev_rel:PREVIOUS_VALUE]->(prev:Metric)
            OPTIONAL MATCH (m)-[next_rel:NEXT_VALUE]->(nxt:Metric)
            OPTIONAL MATCH (m)-[yoy_rel:YOY_CHANGE]->(yoy_prev:Metric)
            RETURN m.name AS name, m.value AS value, m.unit AS unit,
                   m.quarter AS quarter, m.year AS year,
                   m.period AS period, m.source_document AS source_document,
                   m.source_page AS source_page,
                   prev.value AS prev_value, prev.unit AS prev_unit,
                   prev.quarter AS prev_quarter, prev.year AS prev_year,
                   prev.period AS prev_period,
                   prev.source_document AS prev_source_document,
                   prev.source_page AS prev_source_page,
                   prev_rel.percent_change AS prev_pct,
                   prev_rel.change_type AS prev_change_type,
                   nxt.value AS next_value, nxt.unit AS next_unit,
                   nxt.quarter AS next_quarter, nxt.year AS next_year,
                   nxt.period AS next_period,
                   nxt.source_document AS next_source_document,
                   nxt.source_page AS next_source_page,
                   next_rel.percent_change AS next_pct,
                   next_rel.change_type AS next_change_type,
                   yoy_prev.value AS yoy_value, yoy_prev.unit AS yoy_unit,
                   yoy_prev.quarter AS yoy_quarter, yoy_prev.year AS yoy_year,
                   yoy_prev.period AS yoy_period,
                   yoy_prev.source_document AS yoy_source_document,
                   yoy_prev.source_page AS yoy_source_page,
                   yoy_rel.percent_change AS yoy_pct
            """,
            ids=chunk_ids,
        )
        for r in result:
            cur_meta = []
            if r.get("period"):
                cur_meta.append(f"period={r['period']}")
            if r.get("source_document"):
                cur_meta.append(f"source={r['source_document']}")
            if r.get("source_page") is not None:
                cur_meta.append(f"page={r['source_page']}")
            cur_suffix = f" [{'; '.join(cur_meta)}]" if cur_meta else ""
            cur = f"{r['value']}{' ' + r['unit'] if r.get('unit') else ''} ({r['quarter']}-{r['year']}){cur_suffix}"
            if r.get("prev_value") is not None:
                prev_meta = []
                if r.get("prev_period"):
                    prev_meta.append(f"period={r['prev_period']}")
                if r.get("prev_source_document"):
                    prev_meta.append(f"source={r['prev_source_document']}")
                if r.get("prev_source_page") is not None:
                    prev_meta.append(f"page={r['prev_source_page']}")
                prev_suffix = f" [{'; '.join(prev_meta)}]" if prev_meta else ""
                prev = (
                    f"{r['prev_value']}{' ' + r['prev_unit'] if r.get('prev_unit') else ''} "
                    f"({r['prev_quarter']}-{r['prev_year']}){prev_suffix}"
                )
                change_type = r.get("prev_change_type") or "sequential"
                facts.add(
                    f"{r['name']} trend ({change_type}): {prev} -> {cur}"
                    f"{_pct_suffix(r.get('prev_pct'))}"
                )
            if r.get("next_value") is not None:
                next_meta = []
                if r.get("next_period"):
                    next_meta.append(f"period={r['next_period']}")
                if r.get("next_source_document"):
                    next_meta.append(f"source={r['next_source_document']}")
                if r.get("next_source_page") is not None:
                    next_meta.append(f"page={r['next_source_page']}")
                next_suffix = f" [{'; '.join(next_meta)}]" if next_meta else ""
                nxt = (
                    f"{r['next_value']}{' ' + r['next_unit'] if r.get('next_unit') else ''} "
                    f"({r['next_quarter']}-{r['next_year']}){next_suffix}"
                )
                change_type = r.get("next_change_type") or "sequential"
                facts.add(
                    f"{r['name']} trend ({change_type}): {cur} -> {nxt}"
                    f"{_pct_suffix(r.get('next_pct'))}"
                )
            if r.get("yoy_value") is not None:
                yoy_meta = []
                if r.get("yoy_period"):
                    yoy_meta.append(f"period={r['yoy_period']}")
                if r.get("yoy_source_document"):
                    yoy_meta.append(f"source={r['yoy_source_document']}")
                if r.get("yoy_source_page") is not None:
                    yoy_meta.append(f"page={r['yoy_source_page']}")
                yoy_suffix = f" [{'; '.join(yoy_meta)}]" if yoy_meta else ""
                yoy_prev = (
                    f"{r['yoy_value']}{' ' + r['yoy_unit'] if r.get('yoy_unit') else ''} "
                    f"({r['yoy_quarter']}-{r['yoy_year']}){yoy_suffix}"
                )
                facts.add(
                    f"{r['name']} trend (YoY): {yoy_prev} -> {cur}"
                    f"{_pct_suffix(r.get('yoy_pct'))}"
                )

        # Guidance continuity: did this quarter's guidance on a topic
        # change from the previous quarter that mentioned the same topic?
        result = session.run(
            """
            MATCH (q:Quarter)-[]->(n)
            WHERE n.id IN $ids
            WITH DISTINCT q
            MATCH (q)-[:HAS_GUIDANCE]->(g:Guidance)
            OPTIONAL MATCH (prev:Guidance)-[:UPDATED_TO]->(g)
            WHERE prev IS NOT NULL
            RETURN g.topic AS topic, prev.statement AS prev_statement,
                   g.statement AS statement
            """,
            ids=chunk_ids,
        )
        for r in result:
            if r.get("prev_statement"):
                facts.add(
                    f'Guidance update on {r["topic"]}: was "{r["prev_statement"]}" '
                    f'-> now "{r["statement"]}"'
                )

        # Risk continuity: was this quarter's risk also flagged in the
        # immediately preceding quarter that mentioned the same type?
        result = session.run(
            """
            MATCH (q:Quarter)-[]->(n)
            WHERE n.id IN $ids
            WITH DISTINCT q
            MATCH (q)-[:HAS_RISK]->(r:Risk)
            OPTIONAL MATCH (prev:Risk)-[:PERSISTED_TO]->(r)
            WHERE prev IS NOT NULL
            RETURN r.type AS type, r.statement AS statement
            """,
            ids=chunk_ids,
        )
        for r in result:
            facts.add(
                f"Risk persisted from prior quarter: {r['type']} — {r['statement']}"
            )

    return _dedupe_facts(facts)


_STANDALONE_RE = re.compile(r"\bstandalone\b", re.I)
_CONSOLIDATED_RE = re.compile(r"\bconsolidated\b", re.I)


_COMPANY_TAG_RE = re.compile(r"company=([^;\]]+)")


def _fact_company(fact: str) -> str:
    """Extract the company= tag from a fact string, if present, so
    cross-company facts for "the same" metric name (comparison/ranking
    questions) are never grouped together as if they were competing
    variants of one company's single fact."""
    m = _COMPANY_TAG_RE.search(fact)
    return m.group(1).strip().lower() if m else ""


def _metric_base_name(fact: str) -> str:
    """Metric name with a 'standalone'/'consolidated' qualifier stripped,
    so 'PAT (Standalone) = 60.19' and 'PAT (Consolidated) = 47.21' group
    under the same base name ('pat') for resolution below."""
    name = fact.split(" = ", 1)[0]
    name = re.sub(r"\b(standalone|consolidated)\b", "", name, flags=re.I)
    name = re.sub(r"[()\s]+", " ", name).strip().lower()
    return name


def _resolve_standalone_vs_consolidated(
    facts: list[str], question: str, statement_type: str = None
) -> list[str]:
    """Fix for the biggest correctness bug seen in the benchmark: when the
    graph has BOTH a standalone and a consolidated figure for the same
    metric (e.g. PAT = 60.19 standalone vs PAT = 47.21 consolidated),
    sending both to the LLM and hoping the prompt rules sort it out is
    exactly what produces answers that mix the two up (60.19 vs 47.21).

    Resolve it here, before facts are ranked or sent to the LLM at all.
    Resolution order (delegated entirely to retrieval._requested_statement_type
    — the single source of truth, see its docstring for the P0 rationale):
      1. An explicit statement_type (e.g. from ground-truth eval metadata) —
         this was previously ignored here, so a fact fetched for the right
         statement type upstream could still be overridden by this
         function's own, separate, question-text-only guess.
      2. The question explicitly saying "standalone" or "consolidated".
      3. Default to consolidated.
    Metrics with only a single variant (or no standalone/consolidated
    marker at all — most metrics) pass through unchanged."""
    wanted_kind = (
        retrieval._requested_statement_type(question, statement_type) or "consolidated"
    )

    groups: dict[str, list[tuple[str, str]]] = {}
    passthrough = []
    for f in facts:
        if _fact_category(f) != "metric":
            passthrough.append(f)
            continue
        is_standalone = bool(_STANDALONE_RE.search(f))
        is_consolidated = bool(_CONSOLIDATED_RE.search(f))
        if not is_standalone and not is_consolidated:
            passthrough.append(f)
            continue
        kind = "consolidated" if is_consolidated else "standalone"
        groups.setdefault((_fact_company(f), _metric_base_name(f)), []).append(
            (f, kind)
        )

    resolved = list(passthrough)
    for variants in groups.values():
        kinds_present = {kind for _, kind in variants}
        if len(kinds_present) == 1:
            resolved.extend(f for f, _ in variants)  # nothing to resolve
            continue
        chosen = [f for f, kind in variants if kind == wanted_kind]
        resolved.extend(chosen if chosen else [f for f, _ in variants])

    return resolved


def _fact_category(fact: str) -> str:
    # startswith("Guidance")/("Risk") rather than the exact "Guidance:"/
    # "Risk:" prefix so temporal facts from _fetch_temporal_facts()
    # ("Guidance update on ...", "Risk persisted from prior quarter: ...")
    # sort into the same section as their non-temporal counterparts.
    if fact.startswith("Guidance"):
        return "guidance"
    if fact.startswith("Risk"):
        return "risk"
    if fact.startswith("Management sentiment:"):
        return "sentiment"
    if " -[" in fact:
        return "entity"
    return "metric"


# Lower number = sent to the LLM first / kept when facts are truncated.
# Numeric questions ("what was revenue?") want metrics first; qualitative
# questions ("what did management say about risk?") want
# guidance/risk/sentiment first. Both keep some of everything — this
# re-prioritizes, it doesn't drop categories outright.
_NUMERIC_FACT_PRIORITY = {
    "metric": 0,
    "entity": 1,
    "sentiment": 2,
    "guidance": 3,
    "risk": 3,
}
_QUALITATIVE_FACT_PRIORITY = {
    "sentiment": 0,
    "guidance": 0,
    "risk": 0,
    "entity": 1,
    "metric": 2,
}

# Common filler words stripped out before comparing a question's wording
# against a metric fact's name — not an exhaustive stopword list, just
# enough noise removed that keyword overlap reflects the METRIC being
# asked about ("revenue", "PAT", "EPS") rather than sentence scaffolding.
_METRIC_RELEVANCE_STOPWORDS = {
    "what",
    "was",
    "is",
    "the",
    "for",
    "in",
    "of",
    "a",
    "an",
    "to",
    "did",
    "how",
    "much",
    "were",
    "are",
    "this",
    "that",
    "company",
    "companys",
    "limited",
    "during",
    "quarter",
    "year",
    "fy",
}


def _question_keywords(question: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", (question or "").lower())
    return {t for t in tokens if t not in _METRIC_RELEVANCE_STOPWORDS and len(t) > 2}


def _filter_metrics_by_relevance(facts: list[str], question: str) -> list[str]:
    """For a numeric question, narrow the 'metric' fact pool down to
    facts whose metric NAME shares a keyword with the question — before
    quota-based selection in _select_facts spends its metric slots on
    them. Fixes the specific failure pattern seen in the benchmark: a
    question about revenue getting facts like "Basic = 0.04" or "Changes
    in inventories = 10.32" mixed in alongside (or instead of) the actual
    revenue figure, just because those metrics also happened to be
    attached to the retrieved chunks.

    Deliberately conservative: only filters when at least one metric fact
    actually has keyword overlap with the question. If none do — the
    graph's metric names are phrased very differently from how the
    question asks for them — this leaves the pool untouched rather than
    risk dropping the one relevant metric over a wording mismatch; an
    empty overlap set is evidence the check isn't reliable here, not
    evidence nothing is relevant."""
    q_keywords = _question_keywords(question)
    if not q_keywords:
        return facts

    metrics = [f for f in facts if _fact_category(f) == "metric"]
    non_metrics = [f for f in facts if _fact_category(f) != "metric"]

    relevant = [
        f
        for f in metrics
        if q_keywords & set(re.findall(r"[a-z]+", f.split(" = ", 1)[0].lower()))
    ]

    if not relevant:
        return facts  # no overlap anywhere — don't filter blind

    return non_metrics + relevant


def _select_facts(facts: list[str], question: str, max_facts: int = None) -> list[str]:
    """Limit + prioritize graph facts before they go in the prompt. Sending
    40-60 undifferentiated facts (metrics mixed with a dozen guidance/risk
    statements) for a simple "what was revenue?" question buries the right
    answer in noise.

    Uses a per-category quota (config.GRAPH_FACT_QUOTA_NUMERIC /
    _QUALITATIVE) rather than a flat priority cut, so a numeric question
    doesn't end up with 10 metrics and zero guidance/risk/sentiment
    context. Leftover slots (quota totals don't have to add up to
    max_facts exactly) are filled from whatever's left, in the same
    priority order used before quotas existed."""
    if max_facts is None:
        max_facts = getattr(config, "TOP_GRAPH_FACTS", 10)
    numeric = retrieval.is_numeric_question(question)
    if numeric:
        facts = _filter_metrics_by_relevance(facts, question)
    priority = _NUMERIC_FACT_PRIORITY if numeric else _QUALITATIVE_FACT_PRIORITY
    quotas = getattr(
        config,
        "GRAPH_FACT_QUOTA_NUMERIC" if numeric else "GRAPH_FACT_QUOTA_QUALITATIVE",
        None,
    )

    by_category = {}
    for f in sorted(facts):
        by_category.setdefault(_fact_category(f), []).append(f)

    if not quotas:
        ranked = sorted(facts, key=lambda f: priority.get(_fact_category(f), 5))
        return ranked[:max_facts]

    selected = []
    for cat, quota in quotas.items():
        selected.extend(by_category.get(cat, [])[:quota])

    if len(selected) < max_facts:
        selected_set = set(selected)
        remaining = [f for f in facts if f not in selected_set]
        remaining.sort(key=lambda f: priority.get(_fact_category(f), 5))
        selected.extend(remaining[: max_facts - len(selected)])

    return selected[:max_facts]


# Category display order + section headers for the prompt — grouping
# facts under headings (rather than one flat bullet list) gives the LLM
# clearer structure to work from.
_FACT_SECTION_LABELS = [
    ("metric", "Metrics"),
    ("entity", "Entities"),
    ("guidance", "Guidance"),
    ("risk", "Risks"),
    ("sentiment", "Sentiment"),
]


def _metric_sort_key(fact: str) -> tuple:
    """Sort key for a metric fact ('name = value ...'): rank by position
    in config.METRIC_IMPORTANCE_ORDER (substring match against the metric
    name, case-insensitive), falling back to alphabetical for anything not
    in that list, so key figures like Revenue/EBITDA/PAT surface first
    instead of wherever they land alphabetically."""
    name = fact.split(" = ", 1)[0].strip().lower()
    for rank, keyword in enumerate(getattr(config, "METRIC_IMPORTANCE_ORDER", [])):
        if keyword in name:
            return (rank, fact)
    return (len(getattr(config, "METRIC_IMPORTANCE_ORDER", [])), fact)


def _format_facts(facts: list[str]) -> str:
    if not facts:
        return "(no graph facts found)"
    by_category = {}
    for f in facts:
        by_category.setdefault(_fact_category(f), []).append(f)
    sections = []
    for cat, label in _FACT_SECTION_LABELS:
        items = by_category.get(cat)
        if items:
            # Sort within the section (not just across sections) so the
            # same set of facts always renders in the same order.
            # Metrics get a domain-aware priority order (Revenue/EBITDA/
            # PAT first); everything else is plain alphabetical.
            sort_key = _metric_sort_key if cat == "metric" else (lambda f: f)
            lines = "\n".join(f"- {f}" for f in sorted(items, key=sort_key))
            sections.append(f"{label}:\n{lines}")
    return "\n\n".join(sections)


# Header/footer noise stripping and OCR/mojibake cleanup now live in
# evidence_cleaning.py (shared with baseline_pipeline.py) — see
# _format_chunk() below, which calls clean_evidence_text() directly.


def _format_chunk(c: dict) -> str:
    label_bits = [
        c["source_type"].upper(),
        f"{c['quarter']}-{c['year']}",
    ]

    if c.get("company"):
        label_bits.append(c["company"])

    if c.get("score") is not None:
        label_bits.append(f"score={c['score']:.3f}")

    if c.get("document_name"):
        label_bits.append(c["document_name"])

    if c.get("section"):
        label_bits.append(c["section"])

    if c.get("page") is not None:
        label_bits.append(f"p.{c['page']}")

    if c.get("start") is not None and c.get("end") is not None:
        label_bits.append(f"{c['start']:.0f}s-{c['end']:.0f}s")

    if c.get("chunk_type") == "table":
        cleaned_text = format_table_markdown(c)
    else:
        # P0 fix: narrative text chunks now get the same standalone/
        # consolidated tag table chunks already had — see
        # evidence_cleaning.statement_type_tag() docstring.
        cleaned_text = statement_type_tag(c) + clean_evidence_text(c["text"])[:1200]
    return f"[{' | '.join(str(b) for b in label_bits)}]\n{cleaned_text}"


_REFUSAL_RE = re.compile(
    r"\b(does not contain|no information|not (?:explicitly )?(?:mentioned|"
    r"available|stated|found|provided)|cannot find|could not find|"
    r"unable to (?:find|determine)|not present in the (?:evidence|text|data))\b",
    re.I,
)


_DELTA_REQUESTED_RE = re.compile(
    r"\b(by how much|how much (?:did|has|is)|what (?:was|is) the change|"
    r"what growth rate|growth rate|percentage change|% change|"
    r"the difference|change in)\b",
    re.I,
)
_TRAILING_DELTA_RE = re.compile(
    r"\s*[;,]?\s*(?:change|delta|difference)\s*[:=].*$", re.I
)
_TRAILING_PCT_PAREN_RE = re.compile(r"\s*\([+\-]\s*[\d.,]+\s*%\)\s*$")
# P2 fix: confirmed live case where the ANSWER_PROMPT's tightened
# "don't decline over a suspicious-but-superficial mismatch" rule (see
# below) made the model MORE willing to commit to a full explanatory
# answer instead of a refusal - and that fuller answer style brought a
# compound delta shape neither existing regex catches: "1,065.41
# (+226.79 crore (+17.81%))" - a crore-delta paren NESTING a percent
# paren, not a bare "(+17.81%)" alone. _TRAILING_PCT_PAREN_RE only ever
# matched the bare form. This is a strict superset of that shape (an
# outer paren with a signed number, an optional unit word, and an
# OPTIONAL nested percent paren) - run before it, and still leaves the
# original bare-percent case for _TRAILING_PCT_PAREN_RE below to handle
# unchanged.
_TRAILING_DELTA_PAREN_RE = re.compile(
    r"\s*\([+\-]\s*[\d.,]+(?:\s*[a-zA-Z]+)?\s*"
    r"(?:\([+\-]\s*[\d.,]+\s*%\))?\s*\)\s*$"
)


def _fix_decimal_shift(response: str, evidence_text: str) -> str:
    """Deterministic backstop for a confirmed, recurring generation error:
    the model copies the right DIGITS but puts the decimal point in the
    wrong place - 2873.26 -> 287.326, 2154.94 -> 1549.44*, 102.83 ->
    1028.3. (*that one also reordered digits, so it's caught differently
    below.) A prompt rule already tells the model "copy figures exactly,
    never move a decimal point" (see ANSWER_PROMPT) - it still recurred a
    third time even with that rule in place, so this closes the gap
    deterministically instead of hoping a stronger instruction is enough.

    Only touches the FINAL ANSWER line. For every number there, strips
    the decimal point and thousands separators to get a raw digit
    string, then looks for a number ANYWHERE in the evidence with the
    EXACT SAME digit string but a decimal point in a different position.
    If found, replaces the answer's number with the evidence's version
    verbatim - not a guess at the "right" value, just restoring the
    decimal placement the evidence itself already shows. Numbers with no
    digit-identical match in the evidence are left untouched entirely -
    this never invents or moves a number that isn't independently
    confirmed digit-for-digit in the source."""
    if FINAL_ANSWER_MARKER not in response:
        return response
    head, marker, tail = response.rpartition(FINAL_ANSWER_MARKER)

    evidence_numbers = re.findall(r"\d[\d,]*\.?\d*", evidence_text or "")
    # Map: raw digit string (no comma, no decimal point) -> the FIRST
    # verbatim evidence number seen with that digit string. First-seen
    # is deliberate - later occurrences of the same figure (e.g. in a
    # different table further down) are extremely unlikely to disagree
    # on decimal placement, so there's no meaningful "which one" choice
    # to make here.
    digits_to_evidence_number = {}
    for ev_num in evidence_numbers:
        raw_digits = ev_num.replace(",", "").replace(".", "")
        if len(raw_digits) < 4:
            continue  # too short for a decimal-shift to be distinguishable/meaningful
        if raw_digits not in digits_to_evidence_number:
            digits_to_evidence_number[raw_digits] = ev_num

    def _fix_one(m: re.Match) -> str:
        answer_num = m.group(0)
        raw_digits = answer_num.replace(",", "").replace(".", "")
        if len(raw_digits) < 4:
            return answer_num
        evidence_match = digits_to_evidence_number.get(raw_digits)
        if evidence_match is None or evidence_match == answer_num:
            return answer_num
        return evidence_match

    new_tail = re.sub(r"\d[\d,]*\.?\d*", _fix_one, tail)
    if new_tail != tail:
        return f"{head}{marker}{new_tail}"
    return response



def _strip_unrequested_delta(response: str, question: str) -> str:
    """The ANSWER_PROMPT already tells the model not to append a computed
    change/difference/percentage to the FINAL ANSWER line unless the
    question explicitly asks for one — confirmed cases of exactly this
    happening are even quoted in the prompt as a warning. It still
    recurs (e.g. "1,508.35 Crore INR; 1,462.89 Crore INR; change: +45.46
    Crore INR" for a plain "why did revenue increase" question), so this
    is a deterministic backstop rather than relying solely on the model
    to keep following the instruction. Only touches the tail after
    FINAL_ANSWER_MARKER, and only when the question itself contains no
    cue that a computed delta was actually requested.
    """
    if _DELTA_REQUESTED_RE.search(question or ""):
        return response
    if FINAL_ANSWER_MARKER not in response:
        return response
    head, marker, tail = response.rpartition(FINAL_ANSWER_MARKER)
    new_tail = _TRAILING_DELTA_RE.sub("", tail)
    new_tail = _TRAILING_DELTA_PAREN_RE.sub("", new_tail)
    new_tail = _TRAILING_PCT_PAREN_RE.sub("", new_tail)
    if new_tail != tail:
        return f"{head}{marker}{new_tail}"
    return response


def _ensure_final_answer_line(prompt: str, response: str) -> str:
    """Confirmed failure pattern distinct from a refusal: a compound
    question ("revenue grew 3.9% QoQ per the filing; does management's
    commentary corroborate this?") got a full, on-topic answer that
    correctly used the retrieved evidence, but never emitted the
    required FINAL_ANSWER_MARKER line at all — so it never restated the
    actual figures the ground truth needs. _is_refusal() doesn't catch
    this; there's no "does not contain" phrase to match, the model just
    didn't follow the required output format. One bounded retry, reusing
    the exact same prompt and evidence, naming what was missing from the
    previous reply. Shared by graph_rag_pipeline.py and
    baseline_pipeline.py so this is a fair, identical rule for both.
    """
    if FINAL_ANSWER_MARKER in response:
        return response
    fixup_prompt = (
        prompt
        + f"\n\nYour previous answer did not end with the required "
        f"'{FINAL_ANSWER_MARKER}' line:\n{response}\n\nAnswer the same "
        f"question again using the same evidence above. Your response "
        f"MUST end with one '{FINAL_ANSWER_MARKER}' line containing "
        f"every value the question asks for."
    )
    fixup_response = llm_client.generate(
        fixup_prompt,
        model=config.ANSWER_MODEL,
        num_predict=config.OLLAMA_ANSWER_NUM_PREDICT,
    )
    return fixup_response if FINAL_ANSWER_MARKER in fixup_response else response


def _is_refusal(response: str) -> bool:
    """True if the model's answer is a "not found"-style refusal rather
    than an actual attempt at the number."""
    return bool(_REFUSAL_RE.search(response or ""))


def _find_unambiguous_fact(
    facts: list[str], question: str, quarter: str, year: str
) -> str | None:
    """Safety net for a confirmed failure pattern: the model sometimes
    refuses ("does not contain this information") even when the final,
    already-noise-filtered evidence contains exactly one metric fact
    matching the requested metric name and period — nothing left to
    disambiguate. Verified against real cases in this project where that
    single fact held the correct, ground-truth-matching value every time.

    Returns None (no override) whenever there is genuine ambiguity — zero
    matches, or more than one — so this never substitutes a guess. It only
    ever fires when the evidence itself has already resolved the question
    to a single candidate and the model declined to use it anyway.

    P1 fix: a fact tagged "confidence=low_possible_subitem_mismatch" is
    exactly the kind of guess this function's own docstring says never
    to substitute — the tag exists because the source table's row
    structure made the value impossible to fully verify (see
    _metric_source_has_ambiguous_breakdown / the matching prompt rule).
    Before this fix, being the ONLY candidate for a period was enough to
    override a refusal with it regardless of that tag, silently
    defeating the whole point of flagging it: a refusal is the model
    declining to guess, and overriding that refusal with a fact the
    extraction pipeline itself already flagged as an unverified guess
    would make things worse, not better. Excluded from the match pool
    entirely (not just deprioritized) — a low-confidence fact left as
    the sole candidate should surface as "no fact found" (returns None,
    same as the zero-match case) so the response stays a refusal rather
    than a confidently wrong substitution."""
    metric_candidates = {m.lower() for m in _metric_name_candidates(question)}
    if not metric_candidates:
        return None
    period_tag = f"({quarter}-{year})" if quarter and year else None
    matches = [
        f
        for f in facts
        if _fact_category(f) == "metric"
        and _metric_base_name(f) in metric_candidates
        and (not period_tag or period_tag in f)
        and "confidence=low" not in f
    ]
    return matches[0] if len(matches) == 1 else None


def _fact_to_direct_answer(fact: str) -> str:
    """Render a single fact string ("Metric = value unit (Qn-YYYY)
    [tags...]") as a plain-prose answer."""
    core = fact.split(" [", 1)[0]
    name, rest = core.split(" = ", 1)
    return f"{name.strip()}: {rest.strip()}"


_QUALITATIVE_RETRY_PROMPT = """You are a financial analyst assistant. Answer
the question using ONLY the retrieved text below. This text is the full
transcript/document excerpt retrieved for this question — read all of it
before deciding whether it answers the question.

QUESTION: {question}

RETRIEVED TEXT:
{raw_text}

Summarize what management actually said that is relevant to the question,
in your own words. Only say the text doesn't address the question if,
having read all of it, it genuinely contains no relevant commentary. End
your response with one line in EXACTLY this format:
{final_answer_marker} <a concise answer to the question>"""


def _qualitative_refusal_retry(
    question: str, pdf_text_str: str, audio_text_str: str
) -> str | None:
    """Second-chance pass for a qualitative question the first pass
    refused. The main ANSWER_PROMPT puts the Guidance/Risks/Sentiment
    SUMMARY facts ahead of the raw retrieved text, and the model
    sometimes anchors on that (necessarily incomplete) summary and
    declines without reading the raw text below it — confirmed: most
    refusals on audio-sourced qualitative questions had a directly
    relevant audio chunk sitting right there in the same prompt, unused.
    A single prompt-instruction fix isn't reliably enough on its own to
    change this kind of anchoring (the analogous instruction against
    appending unrequested deltas is already in ANSWER_PROMPT with a
    "confirmed" example and still recurs), so this retries once with a
    short, facts-free prompt containing only the raw retrieved text.
    Returns None (no change) if there's no raw text worth retrying on,
    or if the retry also refuses — this never fabricates an answer, it
    only gives the model a cleaner second look at evidence it already had.
    """
    raw_text = "\n\n".join(
        s
        for s in (pdf_text_str, audio_text_str)
        if s and not s.startswith("(no ")
    )
    if not raw_text.strip():
        return None
    prompt = _QUALITATIVE_RETRY_PROMPT.format(
        question=question,
        raw_text=raw_text,
        final_answer_marker=FINAL_ANSWER_MARKER,
    )
    retry_response = llm_client.generate(
        prompt, model=config.ANSWER_MODEL, num_predict=config.OLLAMA_ANSWER_NUM_PREDICT
    )
    if _is_refusal(retry_response):
        return None
    return retry_response


def answer(
    question: str,
    company: str = None,
    year: str = None,
    quarter: str = None,
    use_temporal_facts: bool = None,
    statement_type: str = None,
) -> dict:
    """
    use_temporal_facts: None (default) keeps the current behavior — only
    fetch temporal facts (NEXT_VALUE/PREVIOUS_VALUE/UPDATED_TO/
    PERSISTED_TO) when router.route_question() flags the question as a
    comparison. Pass False to force them off regardless of routing, or
    True to force them on — mainly for benchmark.py's graph/temporal
    ablation, which needs to isolate "graph facts, no temporal" from
    "graph facts + temporal" as two clean conditions rather than relying
    on which questions the router happens to flag as comparisons.
    """
    routing = router.route_question(question)
    # P1 fix (ablation support): FORCE_SOURCE_FILTER lets an ablation
    # config override routing's source_filter decision entirely (e.g.
    # "both" for every question, to isolate Question-Type Routing's own
    # contribution as its own experiment rather than baking routing's
    # effect into every earlier experiment). None (default, unset env
    # var) leaves routing's real decision untouched - existing behavior
    # is unchanged unless a config explicitly opts into this.
    force_source_filter = getattr(config, "FORCE_SOURCE_FILTER", None)
    if force_source_filter:
        routing = dict(routing)
        routing["source_filter"] = force_source_filter

    if company and _ALL_COMPANIES_RE.match(company):
        companies = _company_roster()
    else:
        companies = _split_companies(company)
    multi_company = len(companies) > 1

    t0 = time.time()
    chunks = []
    seen_chunk_ids = set()
    # P1 fix: audio-routed questions get a larger base budget - see
    # config.TOP_K_AUDIO's comment for the measured 57% -> 93% recall
    # gain this closes. top_k scales with the number of companies in
    # play on top of that base, same as before, so a 2-way comparison
    # still doesn't have to fight a single-company's worth of slots.
    base_top_k = (
        config.TOP_K_AUDIO
        if routing["source_filter"] == "audio"
        else config.TOP_K_GRAPH
    )
    per_company_top_k = (
        base_top_k
        if not multi_company
        else max(
            base_top_k,
            (base_top_k * len(companies) + len(companies) - 1)
            // len(companies),
        )
    )
    for c in companies:
        for chunk in retrieval.retrieve(
            question,
            company=c,
            year=year,
            quarter=quarter,
            source_filter=routing["source_filter"],
            statement_type=statement_type,
            top_k=per_company_top_k,
        ):
            if chunk["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["chunk_id"])
                chunks.append(chunk)
    retrieval_sec = time.time() - t0

    # Merging per-company retrieval means chunk count scales with the
    # number of companies (up to 5 for an "All companies" ranking
    # question) - left unchecked that's up to base_top_k * 5 chunks in
    # one prompt, easily enough to exceed a hosted answer model's
    # per-request context limit or eat a disproportionate share of a
    # rate-limited daily token budget. Cap the total, but allocate
    # roughly evenly per company first (keeping each company's
    # highest-scoring chunks up to its share) rather than a pure global
    # score sort - a global sort could starve a company of any
    # representation at all if its chunks happen to score lower, which
    # would make a genuine ranking/comparison question unanswerable for
    # whichever companies got cut entirely.
    #
    # P1 fix: this used config.TOP_K_GRAPH directly - hardcoded to 7,
    # completely blind to base_top_k above (which becomes
    # config.TOP_K_AUDIO=20 for audio-routed questions). For a SINGLE
    # company the "elif" branch below fires regardless of multi_company,
    # so this was silently re-clipping every audio question's retrieval
    # back down to 7 immediately after correctly pulling 20 - undoing
    # the TOP_K_AUDIO fix entirely without any error or warning.
    # Confirmed: graphrag_retrieved_chunk_ids was exactly 7 on every
    # single one of 14 real audio questions, while baseline (which has
    # no equivalent cap) correctly showed ~20. Using base_top_k here
    # instead keeps this cap's original purpose (bounding multi-company
    # blowup) while actually scaling with whatever budget retrieval was
    # really given, audio or not.
    max_total_chunks = base_top_k * min(len(companies), 2)
    if len(chunks) > max_total_chunks and multi_company:
        per_company_share = max(2, max_total_chunks // len(companies))
        by_company: dict[str, list[dict]] = {}
        for c in chunks:
            by_company.setdefault(c.get("company") or "", []).append(c)
        for company_chunks in by_company.values():
            company_chunks.sort(key=lambda c: c.get("score") or 0, reverse=True)
        capped = []
        for company_chunks in by_company.values():
            capped.extend(company_chunks[:per_company_share])
        # If under budget after even allocation, fill remaining slots
        # with the next-best leftover chunks regardless of company.
        if len(capped) < max_total_chunks:
            leftovers = sorted(
                (
                    c
                    for company_chunks in by_company.values()
                    for c in company_chunks[per_company_share:]
                ),
                key=lambda c: c.get("score") or 0,
                reverse=True,
            )
            capped.extend(leftovers[: max_total_chunks - len(capped)])
        chunks = capped
    elif len(chunks) > max_total_chunks:
        chunks.sort(key=lambda c: c.get("score") or 0, reverse=True)
        chunks = chunks[:max_total_chunks]

    chunk_ids = [c["chunk_id"] for c in chunks]

    t0 = time.time()
    # P1 fix (ablation support): USE_GRAPH_FACTS lets an ablation config
    # disable graph-derived facts entirely - both the metric/guidance/
    # risk/sentiment facts from _fetch_graph_facts AND the period-metric
    # lookup below, since both are graph-derived evidence conceptually
    # ("Graph Facts" as its own ablation stage, isolated from Temporal
    # Graph which is a separate, later stage). Default true - unset env
    # var means unchanged behavior. t0/timing is still measured either
    # way so graph_fact_fetch_sec stays meaningful (near-zero when
    # skipped) rather than undefined.
    if getattr(config, "USE_GRAPH_FACTS", True):
        all_facts = _fetch_graph_facts(chunk_ids)
        period_facts: set[str] = set()
        for c in companies:
            period_facts |= set(
                _fetch_period_metric_facts(
                    question,
                    company=c,
                    year=year,
                    quarter=quarter,
                    statement_type=statement_type,
                )
            )
        if period_facts:
            all_facts = _dedupe_facts(set(all_facts) | period_facts)
    else:
        all_facts = []
    # Comparison/trend questions ("compare X vs Y", "how has revenue
    # grown") get extra facts pulled from the temporal links
    # temporal_utils.py builds (metric NEXT_VALUE/PREVIOUS_VALUE,
    # guidance UPDATED_TO, risk PERSISTED_TO) — skipped otherwise since
    # it's extra DB round trips a plain single-quarter lookup doesn't
    # need. routing["is_comparison"] comes from router.route_question().
    # use_temporal_facts overrides that routing-based default when set
    # explicitly (see answer()'s docstring).
    # P1 fix (ablation support): FORCE_TEMPORAL_FACTS lets an ablation
    # config force this on/off for every question, so "Temporal Graph"
    # can be measured as its own clean experiment instead of only ever
    # firing on whichever questions routing happens to flag as
    # comparisons (which would also leak in Question-Type Routing's
    # effect). Only applies when the caller didn't already pass
    # use_temporal_facts explicitly - an explicit caller argument still
    # wins, same precedence as any other parameter-vs-config default.
    if use_temporal_facts is None:
        use_temporal_facts = getattr(config, "FORCE_TEMPORAL_FACTS", None)
    fetch_temporal = (
        routing.get("is_comparison")
        if use_temporal_facts is None
        else use_temporal_facts
    )
    # For numeric metric comparisons, the exact period-aware lookup above is
    # more reliable than adding one-hop temporal metric facts from whichever
    # chunk happened to rank highly.  Keep temporal traversal for qualitative
    # comparisons (guidance/risk/sentiment), and let the structured period
    # lookup handle numeric comparisons deterministically.
    #
    # P-fix: this used to disable temporal traversal for ANY numeric
    # comparison question, trusting the period lookup unconditionally.
    # But the period lookup only fetches periods _extract_fiscal_periods()
    # can actually resolve from the question text — for a comparison
    # phrasing it doesn't recognize, that can be just the ONE explicit
    # period, silently leaving the second value unfetched by either path
    # (structured lookup has no second period to query; temporal
    # traversal was turned off here). Only disable temporal traversal
    # when the structured lookup actually resolved 2+ periods, so it has
    # a real shot at the comparison — otherwise keep temporal traversal
    # on as the fallback for whatever period the lookup couldn't resolve.
    resolved_periods = _extract_fiscal_periods(question, year, quarter)
    if (
        fetch_temporal
        and retrieval.is_numeric_question(question)
        and _metric_name_candidates(question)
        and len(resolved_periods) >= 2
    ):
        fetch_temporal = False
    if fetch_temporal:
        temporal_facts = _fetch_temporal_facts(chunk_ids)
        if temporal_facts:
            all_facts = _dedupe_facts(set(all_facts) | set(temporal_facts))
    graph_fact_fetch_sec = time.time() - t0
    all_facts = _resolve_standalone_vs_consolidated(all_facts, question, statement_type)
    all_facts = _prefer_recovered_facts(all_facts)
    facts = _select_facts(
        all_facts, question, max_facts=_fact_budget(question, routing)
    )

    tables = [c for c in chunks if c["chunk_type"] == "table"]
    pdf_text = [c for c in chunks if c["chunk_type"] == "text"]
    audio_text = [c for c in chunks if c["chunk_type"] == "audio"]

    facts_str = _format_facts(facts)
    tables_str = (
        "\n\n".join(_format_chunk(c) for c in tables)
        if tables
        else "(no tables retrieved)"
    )
    pdf_text_str = (
        "\n\n".join(_format_chunk(c) for c in pdf_text)
        if pdf_text
        else "(no PDF narrative retrieved)"
    )
    audio_text_str = (
        "\n\n".join(_format_chunk(c) for c in audio_text)
        if audio_text
        else "(no audio commentary retrieved)"
    )

    prompt = ANSWER_PROMPT.format(
        question=question,
        facts=facts_str,
        tables=tables_str,
        pdf_text=pdf_text_str,
        audio_text=audio_text_str,
        final_answer_marker=FINAL_ANSWER_MARKER,
    )

    t0 = time.time()
    response = llm_client.generate(
        prompt, model=config.ANSWER_MODEL, num_predict=config.OLLAMA_ANSWER_NUM_PREDICT
    )
    llm_generation_sec = time.time() - t0
    response = _ensure_final_answer_line(prompt, response)

    response = _fix_decimal_shift(
        response, f"{facts_str}\n\n{tables_str}\n\n{pdf_text_str}\n\n{audio_text_str}"
    )
    response = _strip_unrequested_delta(response, question)

    answer_overridden = False
    if _is_refusal(response):
        override_fact = _find_unambiguous_fact(facts, question, quarter, year)
        if override_fact is not None:
            response = _fact_to_direct_answer(override_fact)
            answer_overridden = True
        elif not retrieval.is_numeric_question(question):
            retry_response = _qualitative_refusal_retry(
                question, pdf_text_str, audio_text_str
            )
            if retry_response is not None:
                response = _strip_unrequested_delta(retry_response, question)
                answer_overridden = True

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "source_type": c["source_type"],
            "chunk_type": c["chunk_type"],
            "document_name": c.get("document_name"),
            "page": c.get("page"),
            "section": c.get("section"),
            "title": c.get("title"),
            "start": c.get("start"),
            "end": c.get("end"),
            "score": c.get("score"),
            "embedding_score": c.get("embedding_score"),
            "bm25_score": c.get("bm25_score"),
            "fastrp_score": c.get("fastrp_score"),
        }
        for c in chunks
    ]

    evidence_text = f"{facts_str}\n\n{tables_str}\n\n{pdf_text_str}\n\n{audio_text_str}"
    facts_reduction_pct = (
        round((1 - len(facts) / len(all_facts)) * 100, 1) if all_facts else 0.0
    )

    return {
        "question": question,
        "answer": response,
        "answer_overridden": answer_overridden,
        "routing": routing,
        "num_facts_used": len(facts),
        "num_facts_available": len(all_facts),
        "used_temporal_facts": bool(fetch_temporal),
        "facts_reduction_pct": facts_reduction_pct,
        "facts": facts,
        "num_chunks_used": len(chunks),
        "sources": sources,
        "prompt": prompt,
        "evidence_text": evidence_text,
        "evidence_chars": len(evidence_text),
        "timing": {
            "retrieval_sec": round(retrieval_sec, 4),
            "graph_fact_fetch_sec": round(graph_fact_fetch_sec, 4),
            "llm_generation_sec": round(llm_generation_sec, 4),
            "total_sec": round(
                retrieval_sec + graph_fact_fetch_sec + llm_generation_sec, 4
            ),
        },
    }


if __name__ == "__main__":
    result = answer(
        "What was the revenue growth for Q2 2026?",
        company="Tata Consumer Products",
        year="2026",
        quarter="Q2",
    )
    print(result["answer"])
