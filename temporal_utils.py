"""
temporal_utils.py
==================
WHAT THIS FILE DOES:
Turns the knowledge graph from a set of isolated per-quarter snapshots
into a connected timeline, so questions like "compare Q2 2026 vs Q2 2025"
or "which risks persisted across quarters" can be answered by TRAVERSING
the graph instead of relying on retrieval to happen to pull both quarters'
chunks into the same top-k.

This is called once, after graph_builder.py has finished loading every
company/year/quarter's chunks (see build_all() in graph_builder.py) — not
per-chunk and not per-quarter, because computing "what's the next quarter"
correctly requires seeing every quarter that's actually in the graph
first.

Relationships created (all directed EARLIER -> LATER unless noted):

    Quarter -[:NEXT_QUARTER]-> Quarter            (chronologically next)
    Quarter -[:PREVIOUS_QUARTER]-> Quarter         (inverse of the above)
    Quarter -[:SAME_QUARTER_LAST_YEAR]-> Quarter   (LATER -> EARLIER: this
                                                     year's Q-N -> last
                                                     year's Q-N)
    Metric  -[:NEXT_VALUE]-> Metric                (same company + metric
                                                     name, next quarter —
                                                     carries delta,
                                                     percent_change,
                                                     change_type props)
    Metric  -[:PREVIOUS_VALUE]-> Metric            (inverse of the above,
                                                     same properties)
    Metric  -[:YOY_CHANGE]-> Metric                (same company + metric
                                                     name, true year-over-
                                                     year pair via
                                                     SAME_QUARTER_LAST_YEAR
                                                     — carries delta,
                                                     percent_change)
    Guidance -[:UPDATED_TO]-> Guidance             (same company + topic,
                                                     next quarter that
                                                     mentioned it)
    Risk    -[:PERSISTED_TO]-> Risk                (same company + risk
                                                     type, next quarter
                                                     that mentioned it)

The change properties (delta, percent_change) mean a question like "which
metric grew fastest" can be answered with one Cypher query sorted by
percent_change instead of re-parsing every Metric.value string at query
time. change_type on NEXT_VALUE/PREVIOUS_VALUE is "QoQ" when the two
quarters are literally adjacent (same year, consecutive quarter index) or
"YoY" when they're the same quarter label one year apart; anything else
(a gap in the data — e.g. Q3 was never ingested) is honestly labeled
"sequential" rather than mislabeled QoQ. YOY_CHANGE is computed
separately, straight from Quarter-level SAME_QUARTER_LAST_YEAR pairs,
rather than by counting hops in the NEXT_VALUE chain — a NEXT_VALUE chain
can be anywhere from 1 to 4+ hops apart depending on gaps in the data, so
"4 hops away" isn't a reliable stand-in for "same quarter last year";
going through SAME_QUARTER_LAST_YEAR directly is.


Design notes
------------
- Fully generic: nothing here hardcodes a company name, year, or metric.
  Every grouping key (company, metric name, guidance topic, risk type) is
  read off the nodes already in the graph, so adding a 6th company or a
  3rd year needs zero changes to this file.
- "Next quarter" means the next quarter *that actually exists in the
  graph* for that company, not the next quarter on a calendar — if Q3
  2025 was never ingested, Q2 2025 links straight to Q4 2025. This
  reflects the real data you have instead of assuming a gap-free feed.
- Idempotent and safe to re-run: every relationship type this module
  owns is deleted and rebuilt from scratch each call (see
  `_reset_relationship`), rather than MERGE-only. MERGE alone would
  leave stale links after re-running once a quarter gets inserted
  between two that were previously adjacent (e.g. loading Q2 2025 after
  Q1/Q3 were already linked directly to each other) — deleting first
  means the chain is always recomputed correctly instead of silently
  keeping the old, now-wrong edge.
- Ordering is done in Python, not Cypher: sorting "Q1" < "Q2" < "Q3" <
  "Q4" (and across year boundaries) is fiddly to express correctly in
  Cypher and easy to get subtly wrong; pulling the small number of rows
  involved (one per quarter/metric/guidance-topic/risk-type) into Python
  and sorting there is simpler to read and to test.
- Grouping keys are lowercased/stripped where they come from LLM output
  (guidance topic, risk type) since the extraction model isn't
  perfectly consistent about capitalization ("Supply Chain" vs "supply
  chain") — without normalizing, two mentions of the same real-world
  risk would end up as separate chains that never connect.
"""

from __future__ import annotations

import logging
import re

import config

logger = logging.getLogger(__name__)

# Metric.value is stored as the raw string the source gave us (see
# table_metrics.py / graph_builder.py's TABLE_PROMPT) — "1,591.04",
# "(500)" (accounting negative), "12%". Computing delta/percent_change
# needs an actual number, so this strips exactly those formatting
# conventions and nothing else: no unit conversion, no currency
# handling (a value already carries its own "unit" property separately;
# mixing units when diffing isn't attempted here).
_NUMERIC_STRIP_RE = re.compile(r"[,%\s]")


def _parse_numeric(value) -> float | None:
    """Best-effort parse of a stored Metric.value string into a float.
    Returns None (not 0) when it can't confidently parse — callers must
    treat None as "unknown", not "zero", or a delta would silently
    compare against a wrong baseline."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    s = _NUMERIC_STRIP_RE.sub("", s)
    if not s or not re.fullmatch(r"[+-]?\d*\.?\d+", s):
        return None
    try:
        n = float(s)
    except ValueError:
        return None
    return -n if negative else n


# Canonical within-year ordering. Anything not in this list (a label the
# extraction/ingestion pipeline didn't expect, e.g. "H1") sorts after all
# of these rather than raising — better to place it last than to crash
# the whole linking pass over one odd label.
_QUARTER_ORDER = {
    q: i for i, q in enumerate(getattr(config, "QUARTERS", ["Q1", "Q2", "Q3", "Q4"]))
}


def _quarter_sort_key(year, quarter):
    try:
        y = int(year)
    except (TypeError, ValueError):
        y = 0
    return (y, _QUARTER_ORDER.get(quarter, len(_QUARTER_ORDER)), str(quarter))


def _normalize_key(value: str | None) -> str:
    """Grouping key for topic/type strings that come from LLM output —
    lowercased and whitespace-collapsed so near-identical labels
    ("Supply Chain", " supply chain ") group together. Returns "" for
    None/empty so callers can filter it out explicitly rather than
    silently grouping every un-labeled item into one giant chain."""
    return " ".join((value or "").split()).lower()


def _reset_relationship(session, rel_type: str) -> None:
    """Delete every existing relationship of this type before rebuilding
    it. See module docstring for why this is safer than MERGE-only on
    re-runs."""
    session.run(f"MATCH ()-[r:{rel_type}]->() DELETE r")


# ----------------------------------------------------------------------
# Quarter sequencing
# ----------------------------------------------------------------------


def link_quarter_sequence(session) -> dict:
    """Links each company's Quarter nodes into a chronological chain:
    NEXT_QUARTER forward, PREVIOUS_QUARTER backward. "Next" = the next
    quarter that exists in the graph for that company, not the next
    quarter on the calendar (see module docstring)."""
    _reset_relationship(session, "NEXT_QUARTER")
    _reset_relationship(session, "PREVIOUS_QUARTER")

    rows = session.run(
        "MATCH (q:Quarter) RETURN q.id AS id, q.company AS company, "
        "q.year AS year, q.quarter AS quarter"
    ).data()

    by_company: dict[str, list[dict]] = {}
    for r in rows:
        by_company.setdefault(r["company"], []).append(r)

    pairs = []
    for company, quarters in by_company.items():
        quarters.sort(key=lambda r: _quarter_sort_key(r["year"], r["quarter"]))
        for earlier, later in zip(quarters, quarters[1:]):
            pairs.append({"earlier_id": earlier["id"], "later_id": later["id"]})

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Quarter{id: pair.earlier_id})
            MATCH (b:Quarter{id: pair.later_id})
            MERGE (a)-[:NEXT_QUARTER]->(b)
            MERGE (b)-[:PREVIOUS_QUARTER]->(a)
            """,
            pairs=pairs,
        )

    logger.info(
        "Linked %d quarter-sequence pair(s) across %d compan(y/ies).",
        len(pairs),
        len(by_company),
    )
    return {"quarter_sequence_pairs": len(pairs)}


def link_same_quarter_last_year(session) -> dict:
    """Links (this year's QN) -[:SAME_QUARTER_LAST_YEAR]-> (last year's
    QN) for every company/quarter label where both years exist in the
    graph. Directed LATER -> EARLIER, matching how the question is
    usually asked ("how does this quarter compare to the same quarter
    last year")."""
    _reset_relationship(session, "SAME_QUARTER_LAST_YEAR")

    rows = session.run(
        "MATCH (q:Quarter) RETURN q.id AS id, q.company AS company, "
        "q.year AS year, q.quarter AS quarter"
    ).data()

    # Index by (company, quarter_label, year) -> id, so each row can look
    # up "does (company, same quarter, year-1) exist?" directly.
    index: dict[tuple, str] = {}
    for r in rows:
        try:
            year_int = int(r["year"])
        except (TypeError, ValueError):
            continue
        index[(r["company"], r["quarter"], year_int)] = r["id"]

    pairs = []
    for (company, quarter_label, year_int), this_id in index.items():
        prev_id = index.get((company, quarter_label, year_int - 1))
        if prev_id:
            pairs.append({"later_id": this_id, "earlier_id": prev_id})

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (later:Quarter{id: pair.later_id})
            MATCH (earlier:Quarter{id: pair.earlier_id})
            MERGE (later)-[:SAME_QUARTER_LAST_YEAR]->(earlier)
            """,
            pairs=pairs,
        )

    logger.info("Linked %d same-quarter-last-year pair(s).", len(pairs))
    return {"same_quarter_last_year_pairs": len(pairs)}


# ----------------------------------------------------------------------
# Metric evolution
# ----------------------------------------------------------------------


def _change_type(earlier: dict, later: dict) -> str:
    """Classifies a metric-evolution pair as 'QoQ' (literally adjacent —
    same year, consecutive quarter index), 'YoY' (same quarter label,
    one year apart), or 'sequential' (whatever's actually next in the
    graph, spanning a gap — e.g. Q3 was never ingested). Honest labeling
    matters more than a tidy two-way split: mislabeling a gapped pair as
    QoQ would make a "quarterly growth" query silently wrong."""
    try:
        ey, ly = int(earlier["year"]), int(later["year"])
    except (TypeError, ValueError, KeyError):
        return "sequential"
    eq, lq = earlier.get("quarter"), later.get("quarter")
    if ey == ly and (_QUARTER_ORDER.get(lq, -1) - _QUARTER_ORDER.get(eq, -2)) == 1:
        return "QoQ"
    if ly == ey + 1 and eq == lq:
        return "YoY"
    return "sequential"


def link_metric_evolution(session) -> dict:
    """Links same-name Metric nodes for the same company chronologically:
    NEXT_VALUE forward, PREVIOUS_VALUE backward. Lets a question like
    "how has revenue changed over the last 4 quarters" be answered by
    walking NEXT_VALUE from the earliest Revenue node instead of
    depending on retrieval to surface every quarter's revenue chunk.

    Each relationship also carries delta, percent_change, and
    change_type ('QoQ' / 'YoY' / 'sequential' — see _change_type())
    properties, computed once here rather than left for every query to
    re-derive from the raw value strings — so "which metric grew
    fastest" is one Cypher query sorted by percent_change, not a
    recomputation per question. delta/percent_change are omitted (left
    unset) rather than set to 0 when either value can't be parsed as a
    number (_parse_numeric() returns None) — a missing property reads
    as "unknown" in Cypher; a 0 would read as "no change", which is a
    different, wrong claim.

    Grouped by exact m.name (not a fuzzy/base name) — "PAT (Standalone)"
    and "PAT (Consolidated)" are extracted as distinct names already, so
    grouping by exact name keeps standalone/consolidated as separate
    chains rather than interleaving two different figures into one
    "evolution"."""
    _reset_relationship(session, "NEXT_VALUE")
    _reset_relationship(session, "PREVIOUS_VALUE")

    rows = session.run(
        "MATCH (m:Metric) RETURN m.id AS id, m.company AS company, "
        "m.year AS year, m.quarter AS quarter, m.name AS name, "
        "m.value AS value"
    ).data()

    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        if not r["name"]:
            continue
        by_key.setdefault((r["company"], r["name"]), []).append(r)

    pairs = []
    for (_company, _name), items in by_key.items():
        items.sort(key=lambda r: _quarter_sort_key(r["year"], r["quarter"]))
        for earlier, later in zip(items, items[1:]):
            # Skip same-quarter duplicates (shouldn't normally happen —
            # metric ids already include company/year/quarter/name/period
            # — but two different periods within one quarter, e.g. a
            # restated figure, would otherwise "link" to themselves).
            if earlier["id"] == later["id"]:
                continue
            earlier_val = _parse_numeric(earlier.get("value"))
            later_val = _parse_numeric(later.get("value"))
            delta = None
            percent = None
            if earlier_val is not None and later_val is not None:
                delta = later_val - earlier_val
                if earlier_val != 0:
                    percent = (delta / abs(earlier_val)) * 100
            pairs.append(
                {
                    "earlier_id": earlier["id"],
                    "later_id": later["id"],
                    "delta": delta,
                    "percent_change": percent,
                    "change_type": _change_type(earlier, later),
                }
            )

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Metric{id: pair.earlier_id})
            MATCH (b:Metric{id: pair.later_id})
            MERGE (a)-[fwd:NEXT_VALUE]->(b)
            SET fwd.delta = pair.delta,
                fwd.percent_change = pair.percent_change,
                fwd.change_type = pair.change_type
            MERGE (b)-[bwd:PREVIOUS_VALUE]->(a)
            SET bwd.delta = pair.delta,
                bwd.percent_change = pair.percent_change,
                bwd.change_type = pair.change_type
            """,
            pairs=pairs,
        )

    logger.info(
        "Linked %d metric-evolution pair(s) across %d (company, metric) chain(s).",
        len(pairs),
        len(by_key),
    )
    return {"metric_evolution_pairs": len(pairs)}


def link_metric_yoy_change(session) -> dict:
    """Adds a direct Metric -[:YOY_CHANGE]-> Metric relationship for the
    same company + metric name at quarters connected by
    SAME_QUARTER_LAST_YEAR — a true year-over-year comparison, computed
    independently of the NEXT_VALUE chain. NEXT_VALUE links whatever
    quarter is next *in the data*, which is exactly one year away only
    when there are no gaps; going through SAME_QUARTER_LAST_YEAR
    directly (same underlying logic as link_same_quarter_last_year())
    gets YoY right regardless of gaps elsewhere in the timeline.

    Same delta/percent_change properties as NEXT_VALUE (see
    link_metric_evolution()'s docstring for the None-vs-0 rule), plus a
    fixed change_type='YoY' — this relationship type only ever means
    that."""
    _reset_relationship(session, "YOY_CHANGE")

    rows = session.run(
        "MATCH (m:Metric) RETURN m.id AS id, m.company AS company, "
        "m.year AS year, m.quarter AS quarter, m.name AS name, "
        "m.value AS value"
    ).data()

    index: dict[tuple, dict] = {}
    for r in rows:
        if not r["name"]:
            continue
        try:
            year_int = int(r["year"])
        except (TypeError, ValueError):
            continue
        index[(r["company"], r["name"], r["quarter"], year_int)] = r

    pairs = []
    for (company, name, quarter, year_int), later in index.items():
        earlier = index.get((company, name, quarter, year_int - 1))
        if not earlier or earlier["id"] == later["id"]:
            continue
        earlier_val = _parse_numeric(earlier.get("value"))
        later_val = _parse_numeric(later.get("value"))
        delta = None
        percent = None
        if earlier_val is not None and later_val is not None:
            delta = later_val - earlier_val
            if earlier_val != 0:
                percent = (delta / abs(earlier_val)) * 100
        pairs.append(
            {
                "earlier_id": earlier["id"],
                "later_id": later["id"],
                "delta": delta,
                "percent_change": percent,
            }
        )

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Metric{id: pair.earlier_id})
            MATCH (b:Metric{id: pair.later_id})
            MERGE (a)-[r:YOY_CHANGE]->(b)
            SET r.delta = pair.delta,
                r.percent_change = pair.percent_change,
                r.change_type = 'YoY'
            """,
            pairs=pairs,
        )

    logger.info("Linked %d YoY metric-change pair(s).", len(pairs))
    return {"metric_yoy_change_pairs": len(pairs)}


# ----------------------------------------------------------------------
# Guidance updates
# ----------------------------------------------------------------------


def link_guidance_updates(session) -> dict:
    """Links Guidance nodes that share a company + topic chronologically
    via UPDATED_TO, so "did guidance on X change" can be answered by
    walking the chain instead of hoping both quarters' guidance chunks
    got retrieved together."""
    _reset_relationship(session, "UPDATED_TO")

    rows = session.run(
        """
        MATCH (q:Quarter)-[:HAS_GUIDANCE]->(g:Guidance)
        WHERE g.topic IS NOT NULL AND g.topic <> ''
        RETURN g.id AS id, q.company AS company, q.year AS year,
               q.quarter AS quarter, g.topic AS topic
        """
    ).data()

    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        topic_key = _normalize_key(r["topic"])
        if not topic_key:
            continue
        by_key.setdefault((r["company"], topic_key), []).append(r)

    pairs = []
    for (_company, _topic), items in by_key.items():
        items.sort(key=lambda r: _quarter_sort_key(r["year"], r["quarter"]))
        for earlier, later in zip(items, items[1:]):
            if earlier["id"] == later["id"]:
                continue
            pairs.append({"earlier_id": earlier["id"], "later_id": later["id"]})

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Guidance{id: pair.earlier_id})
            MATCH (b:Guidance{id: pair.later_id})
            MERGE (a)-[:UPDATED_TO]->(b)
            """,
            pairs=pairs,
        )

    logger.info(
        "Linked %d guidance-update pair(s) across %d (company, topic) chain(s).",
        len(pairs),
        len(by_key),
    )
    return {"guidance_update_pairs": len(pairs)}


# ----------------------------------------------------------------------
# Risk persistence
# ----------------------------------------------------------------------


def link_risk_persistence(session) -> dict:
    """Links Risk nodes that share a company + risk type chronologically
    via PERSISTED_TO, so "which risks persisted across quarters" or
    "when did this risk first/last appear" can be answered directly."""
    _reset_relationship(session, "PERSISTED_TO")

    rows = session.run(
        """
        MATCH (q:Quarter)-[:HAS_RISK]->(r:Risk)
        WHERE r.type IS NOT NULL AND r.type <> ''
        RETURN r.id AS id, q.company AS company, q.year AS year,
               q.quarter AS quarter, r.type AS type
        """
    ).data()

    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        type_key = _normalize_key(r["type"])
        if not type_key:
            continue
        by_key.setdefault((r["company"], type_key), []).append(r)

    pairs = []
    for (_company, _type), items in by_key.items():
        items.sort(key=lambda r: _quarter_sort_key(r["year"], r["quarter"]))
        for earlier, later in zip(items, items[1:]):
            if earlier["id"] == later["id"]:
                continue
            pairs.append({"earlier_id": earlier["id"], "later_id": later["id"]})

    if pairs:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Risk{id: pair.earlier_id})
            MATCH (b:Risk{id: pair.later_id})
            MERGE (a)-[:PERSISTED_TO]->(b)
            """,
            pairs=pairs,
        )

    logger.info(
        "Linked %d risk-persistence pair(s) across %d (company, risk type) chain(s).",
        len(pairs),
        len(by_key),
    )
    return {"risk_persistence_pairs": len(pairs)}


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def link_all(session) -> dict:
    """Runs every temporal-linking pass, in order, and returns a combined
    stats dict. Call this once after all quarters have been loaded — see
    build_all() in graph_builder.py, which calls this automatically."""
    stats = {}
    stats.update(link_quarter_sequence(session))
    stats.update(link_same_quarter_last_year(session))
    stats.update(link_metric_evolution(session))
    stats.update(link_metric_yoy_change(session))
    stats.update(link_guidance_updates(session))
    stats.update(link_risk_persistence(session))
    return stats


if __name__ == "__main__":
    # Standalone entry point: `python temporal_utils.py` re-links an
    # already-built graph without re-running extraction — handy after
    # manually editing Neo4j, or the first time you add this file to an
    # existing graph-rag-v1 database.
    import logging as _logging

    from neo4j import GraphDatabase

    _logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            result = link_all(session)
        print(result)
    finally:
        driver.close()
