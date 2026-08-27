"""
benchmark.py — fixed version for the current GraphRAG benchmark.

Changes:
1. Reads a single normal ground_truth.json array.
2. Also tolerates the old accidentally-concatenated-array format.
3. Keeps the existing retrieval diagnostics: scores, Hit@K, Recall@K,
   retrieval overlap, hallucination count, facts used, latency.
4. Adds numeric precision, recall and F1 so extra/wrong numbers are penalized.
5. Carries metric / statement_type / unit metadata when present.
6. Does NOT change Neo4j or the retrieval/GraphRAG/baseline pipelines.
"""

import json
import os
import re
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
import graph_rag_pipeline
import baseline_pipeline

# ---------------------------------------------------------------------
# Extended evaluation metrics (ROUGE / BLEU / METEOR / BERTScore, richer
# retrieval metrics, optional LLM judge).
#
# WHY: strict exact-match is the harshest measure available and reporting
# it alone understates the system - "2,345.98 Crore INR" against a ground
# truth "Rs2,345.98 crore" is correct but scores 0 on token overlap, and
# a correctly computed 3.6% against a ground-truth 3.7% scores 0 on exact
# match. These are the metrics comparable published RAG benchmarks report,
# so results become comparable instead of pessimistically self-scored.
#
# Every library here is OPTIONAL: a missing one leaves its columns blank
# rather than breaking a multi-hour run.
#     pip install rouge-score nltk bert-score
# ---------------------------------------------------------------------
try:
    from rouge_score import rouge_scorer as _rouge_mod
    _ROUGE = _rouge_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
except Exception:
    _ROUGE = None

try:
    import nltk as _nltk
    from nltk.translate.bleu_score import sentence_bleu as _sentence_bleu
    from nltk.translate.bleu_score import SmoothingFunction as _Smooth
    from nltk.translate.meteor_score import meteor_score as _meteor_score
    for _pkg in ("wordnet", "omw-1.4", "punkt"):
        try:
            _nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    _SMOOTH = _Smooth().method1
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

_BERT_SCORER = None
_INDEX_SIZE = None


def _tok_for_overlap(text):
    return re.findall(r"[a-z0-9.]+", str(text or "").lower())


def text_overlap_metrics(pred, gt):
    """ROUGE / BLEU / METEOR. These PENALISE paraphrasing, so they belong
    in a secondary block of the report, not the headline."""
    out = {"rouge1": None, "rouge2": None, "rougeL": None,
           "bleu": None, "meteor": None}
    if _ROUGE is not None:
        try:
            sc = _ROUGE.score(str(gt or ""), str(pred or ""))
            out["rouge1"] = sc["rouge1"].fmeasure
            out["rouge2"] = sc["rouge2"].fmeasure
            out["rougeL"] = sc["rougeL"].fmeasure
        except Exception:
            pass
    if _NLTK_OK:
        ref, hyp = _tok_for_overlap(gt), _tok_for_overlap(pred)
        if hyp and ref:
            try:
                out["bleu"] = _sentence_bleu([ref], hyp, smoothing_function=_SMOOTH)
            except Exception:
                pass
            try:
                out["meteor"] = _meteor_score([ref], hyp)
            except Exception:
                pass
    return out


def bertscore_batch(preds, refs):
    """Batched at the end of the run - loading the model once for all
    questions instead of per question."""
    global _BERT_SCORER
    try:
        if _BERT_SCORER is None:
            from bert_score import BERTScorer
            _BERT_SCORER = BERTScorer(lang="en", rescale_with_baseline=False)
        _, _, f1 = _BERT_SCORER.score([str(p) for p in preds], [str(r) for r in refs])
        return [float(x) for x in f1]
    except Exception as exc:
        print(f"  (BERTScore unavailable: {exc})")
        return [None] * len(preds)


# Financial-domain vocabulary for FinTermPrec. Deliberately domain terms
# only (not generic English), so the metric measures whether the answer
# speaks in the right financial register - naming the metric, the
# statement basis and the unit - rather than rewarding word overlap in
# general, which ROUGE already covers.
_FIN_TERMS = {
    "revenue", "operations", "profit", "loss", "ebitda", "eps", "earnings",
    "margin", "income", "expense", "expenses", "tax", "interest", "crore",
    "lakh", "million", "billion", "consolidated", "standalone", "quarter",
    "quarterly", "annual", "fiscal", "year", "yoy", "qoq", "sequential",
    "segment", "guidance", "outlook", "dividend", "assets", "liabilities",
    "equity", "cash", "flow", "npa", "aum", "disbursement", "borrower",
    "growth", "decline", "increase", "decrease", "net", "gross", "total",
}


def financial_term_precision(pred, gt):
    """Of the financial terms the ANSWER uses, what fraction also appear in
    the ground truth? Low precision means the answer is talking about the
    wrong financial concepts (e.g. answering about segment revenue when
    asked for total revenue) even when the numbers happen to overlap."""
    def terms(t):
        return {w for w in re.findall(r"[a-z]+", str(t or "").lower())
                if w in _FIN_TERMS}
    p, g = terms(pred), terms(gt)
    if not p:
        return None
    return len(p & g) / len(p)


def factual_consistency(pred, gt):
    """FCD / factual consistency: of the numeric facts the answer asserts,
    what fraction are corroborated by the ground truth? This is the
    complement of numeric recall - recall asks "did it find everything
    required", this asks "is everything it stated actually true". An
    answer inventing figures scores low here even with perfect recall."""
    p = _extract_answer_numbers(pred)
    g = _extract_answer_numbers(gt)
    if not p:
        return None
    # Same value-based matching as numeric_metrics (see _matched_pairs):
    # these metrics must not disagree about whether 21.20 equals 21.2.
    return len(_matched_pairs(g, p)) / len(p)


def index_size():
    """Number of retrievable (embedded) nodes - the corpus the system is
    searching. Reported because retrieval scores are only comparable
    between systems searching comparable index sizes."""
    try:
        import retrieval
        chunks = retrieval._load_chunks()
        return len(chunks) if chunks else None
    except Exception:
        return None


def numeric_match_partial(pred, gt):
    """Fraction of required numbers present - PARTIAL credit. Getting 3 of
    4 figures scores 0.75 here and 0.0 under strict exact match."""
    g = _extract_answer_numbers(gt)
    p = _extract_answer_numbers(pred)
    if not g:
        return None
    # Same value-based matching as numeric_metrics (see _matched_pairs).
    return len(_matched_pairs(g, p)) / len(g)


def evidence_sufficiency(result, qa):
    """Fraction of the ground-truth answer's numbers that appear ANYWHERE in
    the retrieved evidence text.

    WHY THIS EXISTS ALONGSIDE hit@k: hit@k asks "was the specific chunk id
    the annotator picked among those retrieved?". That is not the same
    question as "did retrieval surface enough to answer?", because the same
    figure often appears in several chunks (a summary page, a detail table,
    a later filing's comparative column). Confirmed on real data: Info Edge
    revenue Q2 FY25 scored hit@k=0 because the annotated chunk
    886decb0894e4a08 wasn't retrieved — yet the retrieved evidence DID
    contain 7,008.24 and the system answered it exactly right. Five rows in
    one 40-question run were exactly correct while scoring hit@k=0.

    So this reports retrieval SUFFICIENCY (was the needed fact present?)
    next to hit@k's PRECISION-of-annotation (was the exact expected chunk
    found?). Report both: hit@k understates retrieval on corpora where
    facts repeat, and this metric can't detect retrieving the right number
    from a wrong/coincidental context. Neither alone is the whole picture.
    """
    wanted = _extract_answer_numbers(qa.get("answer"))
    if not wanted:
        return None  # qualitative question - no numbers to look for
    evidence = str(result.get("evidence_text") or "")
    # Compare with separators stripped so "7,008.24" matches "7008.24".
    haystack = evidence.replace(",", "")
    found = sum(1 for n in wanted if str(n).replace(",", "") in haystack)
    return found / len(wanted)


_TOLERANCE_ABSTAIN_PATTERNS = [
    "does not contain", "doesn't contain", "not contain this information",
    "cannot determine", "can't determine", "no information",
    "not available in the", "unable to determine", "not found in the",
    "insufficient evidence", "does not provide",
    # P1 fix: confirmed live case - "The management did not provide any
    # forward-looking commentary" wasn't caught by "does not provide"
    # (present tense only), so graphrag_abstained showed 0.0 despite a
    # genuine refusal being present - undercounting exactly the failure
    # mode this metric exists to measure. Added past-tense variants for
    # every "does not X" pattern already above, not just this one case.
    "did not provide", "did not contain", "did not mention",
    "does not mention", "was not provided", "was not discussed",
    "were not discussed", "was not mentioned", "were not mentioned",
]


def tolerance_match(pred: str, gt: str, tol: float = 0.01):
    """True when every ground-truth number is matched by some number in
    the prediction within `tol` relative error — the fairer companion to
    exact match, not a replacement for it (report both). Exact match
    calls 3.6% wrong when ground truth says 3.7% — a rounding difference
    on an otherwise-correct answer, not a wrong figure. Reuses
    _extract_answer_numbers so this stays consistent with every other
    numeric metric here (same period-label stripping, same FINAL ANSWER
    handling) rather than drifting from it with a second number-parser.
    """
    g = _extract_answer_numbers(gt)
    p = _extract_answer_numbers(pred)
    if not g:
        return None

    def _close(a, b):
        if a == b:
            return True
        try:
            af, bf = float(str(a).rstrip("%")), float(str(b).rstrip("%"))
        except ValueError:
            return False
        if af == bf:
            return True
        denom = max(abs(af), abs(bf))
        return denom > 0 and abs(af - bf) / denom <= tol

    return all(any(_close(gv, pv) for pv in p) for gv in g)


def abstained(answer: str) -> bool:
    """True when the system declined to answer rather than guessing.
    Reported as its own signal because it is invisible in every other
    metric here otherwise — a system that correctly says "the evidence
    doesn't contain this" scores 0 on numeric_exact identically to one
    that confidently states a wrong number, and those are very different
    failure modes to be conflating into one number."""
    low = str(answer or "").lower()
    return any(p in low for p in _TOLERANCE_ABSTAIN_PATTERNS)


_QUARTER_YEAR_RE = re.compile(r"Q([1-4])[\s\-]*(?:FY)?[\s\-]*(\d{2,4})", re.I)


def _quarter_year_pairs(text: str) -> set:
    """Extracts (quarter, last-2-digits-of-year) pairs from text,
    normalizing "Q4 FY26", "Q4-FY26", "Q4-2026", "Q4 2026" all to the
    same (4, 26) representation — the model and ground truth don't
    consistently agree on which of these notations to use, so this
    lets a ranking answer be recognized regardless of which one it
    picked."""
    pairs = set()
    for m in _QUARTER_YEAR_RE.finditer(text or ""):
        pairs.add((m.group(1), m.group(2)[-2:]))
    return pairs


def is_multivalue(gt: str) -> bool:
    """Ground truths with 3+ distinct numbers are comparison-style (two
    periods plus a change/percentage), where partial credit matters more
    than for a single-figure lookup."""
    return len(_extract_answer_numbers(gt)) >= 3


def multivalue_complete(pred: str, gt: str, expected_answer: str = None):
    """For a comparison-style ground truth, whether the prediction
    supplied every value it contains — not just some of them. Partial
    answers (one period right, one missing) were a recurring failure
    mode this measures directly instead of via a depressed recall
    average that doesn't distinguish "missed one of four" from "missed
    all four".

    P1 fix: RANKING questions ("which quarter had the highest revenue")
    were scoring 0 here even when correctly answered, because their
    ground truth `answer` text restates every candidate period's figure
    as supporting reasoning (e.g. "Q4 FY26 ... (Q4 FY24: X, Q4 FY25: Y,
    Q4 FY26: Z)") — this function was checking whether the prediction
    restated ALL of those numbers, but the prompt's own ranking rule
    correctly tells the model to state only the winner, not every
    candidate. ground_truth.json has a separate `expected_answer` field
    ("Q4 FY26") holding just the actual answer, used EXCLUSIVELY for
    these ranking-style questions (confirmed: populated for exactly the
    2 cross_quarter_reasoning questions in this project's ground truth,
    0 others) — when present, completeness means "did the answer name
    the right period", not "did it restate every comparandum". Falls
    back to the original all-numbers check for every other question,
    where that check is the right one."""
    if expected_answer:
        expected_pairs = _quarter_year_pairs(expected_answer)
        if expected_pairs:
            pred_pairs = _quarter_year_pairs(pred)
            return int(bool(expected_pairs & pred_pairs))
    if not is_multivalue(gt):
        return None
    g, p = _extract_answer_numbers(gt), _extract_answer_numbers(pred)
    matched = _numeric_intersection(g, p)
    return int(len(matched) == len(g))


def source_hit_recall_at_k(result: dict, qa: dict) -> dict:
    """Loose companion to retrieval_hit_recall_at_k's strict chunk_id
    match: also credits a retrieved chunk from the SAME (document, page)
    as an annotated supporting chunk, even when the chunk_id itself
    differs. Confirmed real gap this closes: the same table can be
    chunked into an overlapping Table node and a narrative Chunk node
    covering the same page, or the annotator's chunk boundary can simply
    differ from the pipeline's — both retrieve the right source location
    with a different id, which strict hit@k scores as a total miss even
    when the answer that came out of it was exactly correct (evidence_
    sufficiency's docstring documents a live case of exactly this: Info
    Edge Q2 FY25, hit@k=0, numeric_exact=1.0). Reported ALONGSIDE the
    strict metric, never replacing it — this is a broader, still
    ground-truth-anchored definition of "retrieved the right place", not
    a loosening of what counts as correct. Returns None (not 0) when the
    ground truth has neither chunk ids nor documents annotated, same
    convention as the strict version, so an unannotated question isn't
    silently counted as a retrieval failure.
    """
    chunks = _retrieved_chunks(result)
    retrieved_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id") is not None}
    retrieved_locations = {
        (str(c.get("document_name") or "").lower(), c.get("page"))
        for c in chunks
        if c.get("document_name") and c.get("page") is not None
    }

    supporting_ids = set(qa.get("supporting_chunk_ids") or [])
    # P1 fix: ground_truth.json's supporting_documents are consistently
    # named after the ingestion .json artifact ("...statement.json"),
    # while a retrieved chunk's document_name is the original source
    # file ("...statement.pdf"/".mp3"). A plain substring check between
    # these never matches — different trailing extensions defeat it even
    # when the base filename is identical - confirmed this made the
    # location-match branch never fire at all in testing. Strip the
    # extension from both sides before comparing.
    _ext_re = re.compile(r"\.[a-zA-Z0-9]+$")
    supporting_docs = {
        _ext_re.sub("", str(d or "").lower())
        for d in (qa.get("supporting_documents") or [])
    }
    supporting_pages = set(qa.get("supporting_pages") or [])
    # Cross product: any annotated (doc, page) combination counts as a
    # target location, since the ground truth doesn't pair them 1:1.
    supporting_locations = {
        (doc, page) for doc in supporting_docs for page in supporting_pages
    }

    if not supporting_ids and not supporting_locations:
        return {"source_hit_at_k": None, "source_recall_at_k": None}

    id_hits = retrieved_ids & supporting_ids
    location_hits = {
        (doc, page)
        for doc, page in supporting_locations
        for rdoc, rpage in retrieved_locations
        if page == rpage and (doc in _ext_re.sub("", rdoc) or _ext_re.sub("", rdoc) in doc)
    }

    total_hits = len(id_hits) + len(location_hits)
    total_targets = len(supporting_ids) + len(supporting_locations)
    if total_targets == 0:
        return {"source_hit_at_k": None, "source_recall_at_k": None}

    return {
        "source_hit_at_k": 1.0 if total_hits else 0.0,
        "source_recall_at_k": min(1.0, total_hits / total_targets),
    }


def rank_retrieval_metrics(result, qa):
    """MRR, context precision and source-type accuracy from the retrieved
    chunk ids. hit@k is computed separately (retrieval_hit_recall_at_k)."""
    out = {"mrr": None, "context_precision": None, "source_type_acc": None}
    chunks = _retrieved_chunks(result)
    if not chunks:
        return out

    retrieved = [c.get("chunk_id") for c in chunks]
    supporting = set(qa.get("supporting_chunk_ids") or [])
    if supporting and retrieved:
        hits = [i for i, c in enumerate(retrieved) if c in supporting]
        out["mrr"] = 1.0 / (hits[0] + 1) if hits else 0.0
        out["context_precision"] = len(set(retrieved) & supporting) / len(retrieved)

    want = str(qa.get("modality") or "")
    types = {str(c.get("source_type") or "").lower() for c in chunks}
    if want and types:
        if want == "pdf":
            out["source_type_acc"] = 1.0 if "pdf" in types else 0.0
        elif want == "audio":
            out["source_type_acc"] = 1.0 if "audio" in types else 0.0
        elif want == "pdf+audio":
            out["source_type_acc"] = 1.0 if {"pdf", "audio"} <= types else 0.0
    return out


_JUDGE_PROMPT = """You are evaluating a financial question-answering system.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

SYSTEM'S GENERATED ANSWER:
{answer}

RETRIEVED EVIDENCE GIVEN TO THE SYSTEM:
{evidence}

Judge two things INDEPENDENTLY:

1. FAITHFUL - is every claim in the generated answer supported by the
   retrieved evidence above? Correctly stating that the evidence lacks
   the information IS faithful. Stating figures absent from the evidence
   is NOT faithful.

2. CORRECT - is the answer consistent with the ground truth in substance?
   Ignore wording, unit phrasing and rounding (3.6% vs 3.7% is correct).
   A genuinely different number is not correct.

Respond with ONLY a JSON object:
{{"faithful": true/false, "correct": true/false, "failure_type": "none|retrieval|generation|evaluation|data_quality|abstained", "reason": "one sentence"}}"""


def llm_judge(question, ground_truth, answer, evidence):
    """LLM-as-judge. Separating FAITHFULNESS (grounded in the retrieved
    evidence) from CORRECTNESS (matches ground truth) is what makes this
    worth the runtime: faithful-but-incorrect means the evidence was bad,
    while correct-but-unfaithful means the model guessed right without
    grounding. Neither is visible in a string-overlap metric."""
    import llm_client

    prompt = _JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer,
        evidence=str(evidence or "")[:6000],
    )
    try:
        raw = llm_client.generate(
            prompt, model=config.ANSWER_MODEL, temperature=0.0, num_predict=300
        ).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"faithful": None, "correct": None,
                    "failure_type": "judge_unparseable", "reason": raw[:150]}
        return json.loads(m.group(0))
    except Exception as exc:
        return {"faithful": None, "correct": None,
                "failure_type": "judge_error", "reason": str(exc)[:150]}


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")


def _extract_numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text or "")}


_PERIOD_LABEL_RE = re.compile(
    # P1 fix: this alternative MUST come first. graph_rag_pipeline.py's
    # own fact strings are formatted "(Q2-2025)" — hyphenated, no space —
    # and the LLM frequently echoes that exact substring back in its
    # answer (confirmed: 28/75 rows in the post-P0 benchmark carried a
    # spurious "-20YY" extra number purely from this). The old pattern
    # never matched this shape at all, so "-2025" fell through to
    # NUMBER_RE and counted as a hallucinated/extra number even on
    # fully correct answers. Python's re alternation is leftmost-first
    # (not leftmost-longest), so this has to be tried before the bare
    # "\bQ[1-4]\b" alternative below, or that shorter match wins first
    # and leaves the "-YYYY" tail for NUMBER_RE to still pick up.
    r"\bQ[1-4]\s*-\s*\d{2,4}\b"
    # P2 fix: a smaller/chattier model (observed on a run using
    # qwen3.5:4b) tends to spell out the quarter-end calendar date in its
    # explanation — "for the quarter ended September 30, 2025 (Q2 FY26)".
    # The existing month+year alternative below only matches "Month YYYY"
    # (no day), so on "September 30, 2025" it was matching just
    # "September 30" (treating the DAY as if it were a 2-digit year) and
    # leaving the real YYYY stranded afterward as a spurious extra
    # number — confirmed on 15/49 rows in that run, each contributing
    # exactly one bogus "extra number" despite the answer being fully
    # correct. This alternative (day + comma optional, then year) must
    # come before the bare month+year one for the same leftmost-first
    # reason as the Qn-YYYY fix above — otherwise the shorter
    # "Month <day-as-year>" match wins first.
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b"
    r"|\bQ[1-4]\s*(?:FY\s*)?['’]?\d{2,4}\b"
    r"|\bFY\s*['’]?\d{2,4}\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+['’]?\d{2,4}\b"
    r"|\bQ[1-4]\b",
    re.IGNORECASE,
)


_FINAL_ANSWER_MARKER_RE = re.compile(
    re.escape(graph_rag_pipeline.FINAL_ANSWER_MARKER), re.IGNORECASE
)


def _extract_answer_numbers(text: str) -> set[str]:
    """Extract numeric answer values without counting period/date labels.

    P2 fix: both pipelines' prompts now ask the model to end with a
    "FINAL ANSWER:" line, precisely so scoring doesn't depend on the
    model reliably following "don't restate numbers in your reasoning"
    instructions — that alone wasn't robust across models (confirmed: a
    smaller model produced fully correct, well-reasoned answers that
    explicitly explained and cited a REJECTED candidate value, e.g.
    "the narrative text lists a standalone figure of ₹14.62 crore... but
    the correct figure is ₹19.05 crore" — every one of those zeroed
    exact-match despite being a better, more transparent answer than a
    bare number). When the marker is present, extract only from what
    follows it — the model's own reasoning above the line is ignored for
    scoring purposes, same as a human reading only the final line. Falls
    back to the whole text when the marker isn't present, so this is
    safe against a model that doesn't follow the instruction and against
    any historical answers being re-scored.
    """
    text = text or ""
    m = _FINAL_ANSWER_MARKER_RE.search(text)
    if m:
        text = text[m.end():]
    return _extract_numbers(_PERIOD_LABEL_RE.sub(" ", text))


def _as_number(token):
    """Parse a numeric token to a float for VALUE comparison. Returns None
    for anything unparseable (which then falls back to string equality)."""
    t = str(token).strip().replace(",", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def _matched_pairs(gt, pred):
    """Pair up ground-truth and predicted numbers that are the SAME VALUE.

    Previously matching was exact STRING set intersection, which scored
    numerically-identical answers as completely wrong: ground truth
    "21.20" against a predicted "21.2" gave F1 = 0.00, and "3.7%" vs
    "3.70%" likewise. Commas were already handled, trailing zeros and
    decimal formatting were not. Percentages compare on their numeric
    part, so "3.7%" matches "3.7%" but not the bare quantity 3.7 in a
    different role — the '%' is normalised off both sides only when both
    sides carry it.

    A tiny relative tolerance (1e-9) is used purely to absorb float
    representation error, NOT to accept genuinely different figures:
    4,778.91 and 4,778.92 still do not match.
    """
    pairs, used = [], set()
    for g in gt:
        gv = _as_number(g)
        for p in pred:
            if p in used:
                continue
            if g == p:  # identical strings always match
                pairs.append((g, p)); used.add(p); break
            pv = _as_number(p)
            if gv is None or pv is None:
                continue
            # both-or-neither percent, so a rate isn't matched to a quantity
            if (str(g).strip().endswith("%")) != (str(p).strip().endswith("%")):
                continue
            if gv == pv or abs(gv - pv) <= 1e-9 * max(abs(gv), abs(pv), 1.0):
                pairs.append((g, p)); used.add(p); break
    return pairs


def _numeric_intersection(gt, pred):
    return [g for g, _ in _matched_pairs(gt, pred)]


def numeric_metrics(
    predicted: str,
    ground_truth: str,
    expected_numbers=None,
) -> dict:
    """
    Old numeric_match was GT-recall only:
        |pred ∩ GT| / |GT|

    That can score 1.0 even if the model adds several unrelated numbers.

    We now report:
      recall    = required GT numbers recovered
      precision = predicted numbers that are actually required
      F1        = harmonic mean
      exact     = predicted numeric set == expected numeric set
    """
    # Ground-truth JSON previously contained a polluted expected_numbers field
    # where period labels such as Q2 FY25 contributed "2" and "25".  The
    # authoritative numeric set is the actual ground-truth answer text.
    #
    # Applied to BOTH sides, not just ground truth: a correct answer that
    # naturally restates the period in its own phrasing ("...in Q2 FY26")
    # was having "2" and "26" counted as wrong extra numbers against it,
    # dragging down precision/F1 for answers that were actually fully
    # correct — the period mention isn't a financial figure on either side.
    gt = _extract_answer_numbers(ground_truth)
    pred = _extract_answer_numbers(predicted)

    if not gt:
        return {
            "recall": None,
            "precision": None,
            "f1": None,
            "exact": None,
            "missing": [],
            "extra": sorted(pred),
        }

    matched = _numeric_intersection(gt, pred)
    recall = len(matched) / len(gt)
    precision = len(matched) / len(pred) if pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        # exact = every required number found AND nothing extra, compared
        # NUMERICALLY rather than as strings (see _numeric_intersection).
        "exact": int(len(matched) == len(gt) and len(matched) == len(pred)),
        "missing": sorted(gt - {m[0] for m in _matched_pairs(gt, pred)}),
        "extra": sorted(pred - {m[1] for m in _matched_pairs(gt, pred)}),
    }


def numeric_match(predicted: str, ground_truth: str, expected_numbers=None):
    # Preserve the old metric name for compatibility with existing scripts.
    return numeric_metrics(predicted, ground_truth, expected_numbers)["recall"]


_embed_model = None


def semantic_similarity(predicted: str, ground_truth: str) -> float:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)

    vecs = _embed_model.encode(
        [predicted, ground_truth],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return float(vecs[0] @ vecs[1])


def hallucinated_number_count(
    predicted: str,
    evidence_text: str,
    ground_truth: str,
) -> int:
    """Heuristic: numbers in the answer found in neither evidence nor GT.
    Period-label numbers (the "2"/"26" in "...in Q2 FY26") are stripped
    from the predicted text first — a correct answer restating its own
    period isn't a hallucinated financial figure, and evidence_text won't
    reliably contain every incidental date fragment to allowlist it
    through the union below."""
    pred_numbers = _extract_answer_numbers(predicted)
    allowed = _extract_numbers(evidence_text or "") | _extract_numbers(ground_truth)
    return len(pred_numbers - allowed)


def retrieval_overlap(graph_sources, baseline_sources) -> float:
    graph_ids = {
        s["chunk_id"] if isinstance(s, dict) else s for s in (graph_sources or [])
    }
    baseline_ids = {
        s["chunk_id"] if isinstance(s, dict) else s for s in (baseline_sources or [])
    }
    union = graph_ids | baseline_ids

    if not union:
        return None

    return len(graph_ids & baseline_ids) / len(union)


def _retrieved_chunks(result: dict) -> list:
    chunks = result.get("retrieved_chunks")
    if chunks:
        return chunks

    sources = result.get("sources") or []
    return [s for s in sources if isinstance(s, dict)]


def avg_retrieval_scores(result: dict) -> dict:
    chunks = _retrieved_chunks(result)

    out = {
        "avg_retrieval_score": None,
        "avg_embedding_score": None,
        "avg_bm25_score": None,
        # FastRP structural-similarity score, present only when
        # USE_FASTRP is on and fast_rp.py has been run. Reported so an
        # ablation can show what the graph-structure signal actually
        # contributed, rather than inferring it from F1 alone.
        "avg_fastrp_score": None,
    }

    if not chunks:
        return out

    for key, out_key in (
        ("score", "avg_retrieval_score"),
        ("embedding_score", "avg_embedding_score"),
        ("bm25_score", "avg_bm25_score"),
        ("fastrp_score", "avg_fastrp_score"),
    ):
        vals = [c.get(key) for c in chunks if c.get(key) is not None]
        if vals:
            out[out_key] = float(np.mean(vals))

    return out


def retrieval_hit_recall_at_k(result: dict, qa: dict) -> dict:
    """
    Uses annotated supporting_chunk_ids first, then supporting_pages.

    If the GT has no supporting evidence annotations, returns None rather than
    falsely reporting a retrieval failure.
    """
    chunks = _retrieved_chunks(result)

    retrieved_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id") is not None}
    retrieved_pages = {c.get("page") for c in chunks if c.get("page") is not None}

    supporting_ids = set(qa.get("supporting_chunk_ids") or [])
    supporting_pages = set(qa.get("supporting_pages") or [])

    if supporting_ids:
        hits = retrieved_ids & supporting_ids
        gt_set = supporting_ids
    elif supporting_pages:
        hits = retrieved_pages & supporting_pages
        gt_set = supporting_pages
    else:
        return {"hit_at_k": None, "recall_at_k": None}

    return {
        "hit_at_k": 1.0 if hits else 0.0,
        "recall_at_k": len(hits) / len(gt_set) if gt_set else None,
    }


def _normalize_key_field(value) -> str:
    """Normalize a value for the resume-matching key so a field read back
    from CSV — which pandas coerces to int64/float64 for anything that
    looks purely numeric, e.g. year "2026" -> 2026 or 2026.0 — still
    matches the same field's original string form from ground_truth.json.
    Without this, (question, company, quarter, year) tuples never
    compared equal across the JSON/CSV boundary, so every question looked
    "not yet done" even when it was: resume mode silently reprocessed and
    duplicated every row instead of skipping completed ones."""
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    return str(value).strip()


def _question_key(qa: dict) -> tuple:
    return tuple(
        _normalize_key_field(qa.get(field))
        for field in ("question", "company", "quarter", "year")
    )


def _save_progress(rows: list):
    df = pd.DataFrame(rows)

    detailed_path = config.RESULTS_DIR / "benchmark_detailed.csv"
    df.to_csv(detailed_path, index=False)

    detailed_json_path = config.RESULTS_DIR / "benchmark_detailed.json"
    detailed_json_path.write_text(
        df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    return df, detailed_path, detailed_json_path


def _load_ground_truth(gt_path):
    """
    Normal format:
        [ {...}, {...} ]

    Also accepts the old broken format:
        [ {...} ][ {...} ][ {...} ]

    This prevents JSONDecodeError: Extra data from stopping the benchmark.
    """
    raw = gt_path.read_text(encoding="utf-8")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    pos = 0
    flattened = []

    while pos < len(raw):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1

        if pos >= len(raw):
            break

        obj, end = decoder.raw_decode(raw, pos)

        if not isinstance(obj, list):
            raise ValueError(
                f"ground_truth.json contains a non-array JSON value near offset {pos}"
            )

        flattened.extend(obj)
        pos = end

    if not flattened:
        raise ValueError("No QA pairs found in ground_truth.json")

    return flattened


def run_benchmark(modality_filter: set = None, question_id_filter: set = None):
    """Runs the full benchmark, or a subset filtered by ground_truth.json's
    "modality" field ("pdf", "audio", "pdf+audio") and/or specific
    question "id" values. Both filters can be combined (AND logic) - e.g.
    modality_filter={"audio"} + question_id_filter={"..."} to check one
    specific question and confirm it's actually in the audio set.

    Two ways to set EACH filter, in priority order:
      1. Pass it directly - e.g. run_benchmark(question_id_filter={
         "Jindal_Stainless_Limited_2025_Q2_risk_guidance"}) - this is
         what a future ablation script should do, calling this function
         directly rather than shelling out, so a whole sweep doesn't
         reload the embedding model / Neo4j connection once per
         subprocess.
      2. Leave it as None and set the matching env var instead:
           BENCHMARK_MODALITY=audio python benchmark.py
           BENCHMARK_QUESTION_IDS=id_one,id_two python benchmark.py
         - for a quick one-off run from the command line without editing
         any code. Comma-separated for more than one value. Combine both
         env vars in the same run to intersect them, same as the
         parameters do.

    Neither set for a given filter: that filter doesn't apply - runs on
    every question that passes whichever filter(s) ARE set, or every
    question if none are set at all (unchanged from before either
    parameter existed - existing callers/scripts see no behavior change).

    Output files are unaffected by filtering (benchmark_detailed.csv/
    .json, benchmark_summary.json still write to the same
    config.RESULTS_DIR paths) - a filtered run overwrites those with
    just the filtered subset's rows, so if you want to keep a full run's
    results around, copy them out before running a filtered one, the
    same as you'd need to for any other benchmark.py run.
    """
    if modality_filter is None:
        env_filter = os.getenv("BENCHMARK_MODALITY", "").strip()
        if env_filter:
            modality_filter = {m.strip() for m in env_filter.split(",") if m.strip()}

    if question_id_filter is None:
        env_ids = os.getenv("BENCHMARK_QUESTION_IDS", "").strip()
        if env_ids:
            question_id_filter = {i.strip() for i in env_ids.split(",") if i.strip()}

    gt_path = config.RESULTS_DIR / "ground_truth.json"
    qa_pairs = _load_ground_truth(gt_path)

    if modality_filter:
        before = len(qa_pairs)
        qa_pairs = [qa for qa in qa_pairs if qa.get("modality") in modality_filter]
        print(
            f"Modality filter {sorted(modality_filter)}: "
            f"{len(qa_pairs)}/{before} questions match."
        )
        if not qa_pairs:
            print(
                "No questions matched this filter - check the modality "
                "values in ground_truth.json (expected: 'pdf', 'audio', "
                "'pdf+audio')."
            )
            return

    if question_id_filter:
        before = len(qa_pairs)
        qa_pairs = [qa for qa in qa_pairs if qa.get("id") in question_id_filter]
        print(
            f"Question ID filter ({len(question_id_filter)} id(s) given): "
            f"{len(qa_pairs)}/{before} questions match."
        )
        matched_ids = {qa.get("id") for qa in qa_pairs}
        unmatched = question_id_filter - matched_ids
        if unmatched:
            print(
                f"  Warning: {len(unmatched)} given id(s) not found "
                f"(after any modality filter) - check spelling/id "
                f"values: {sorted(unmatched)}"
            )
        if not qa_pairs:
            print("No questions matched this filter.")
            return

    print(f"Running benchmark on {len(qa_pairs)} ground-truth questions.")

    review_count = sum(
        1
        for qa in qa_pairs
        if qa.get("modality") in ("pdf", "pdf+audio")
        and (qa.get("statement_type") or "unspecified") == "unspecified"
    )

    if review_count:
        print(
            f"\nWARNING: {review_count} PDF/BOTH questions do not have an "
            "explicit standalone/consolidated statement_type."
        )
        print("Those questions are NOT silently reinterpreted by the benchmark.")

    detailed_path = config.RESULTS_DIR / "benchmark_detailed.csv"

    rows = []
    completed = set()

    # Never reuse stale benchmark rows after code/config changes unless the
    # user explicitly opts into resume mode.
    resume = os.getenv("BENCHMARK_RESUME", "false").lower() == "true"
    if resume and detailed_path.exists():
        try:
            df_existing = pd.read_csv(detailed_path)
            rows = df_existing.to_dict("records")
            completed = {_question_key(r) for r in rows}
            print(
                f"Found existing {detailed_path.name} with {len(rows)} "
                "completed question(s) — resuming because "
                "BENCHMARK_RESUME=true."
            )
        except Exception as exc:
            print(f"Warning: could not resume existing detailed results: {exc}")
            rows = []
            completed = set()
    elif detailed_path.exists():
        print(
            f"Existing {detailed_path.name} will be overwritten with a fresh "
            "run (set BENCHMARK_RESUME=true to resume)."
        )

    failures = 0

    for i, qa in enumerate(qa_pairs, start=1):
        if _question_key(qa) in completed:
            continue

        print(
            f"[{i}/{len(qa_pairs)}] {qa['question'][:70]}... "
            f"(ok={len(rows)}, failed={failures})"
        )

        common_args = {
            "company": qa["company"],
            "year": qa["year"],
            "quarter": qa["quarter"],
        }
        if qa.get("statement_type") in ("standalone", "consolidated"):
            # This is evaluation metadata, not answer leakage: it records which
            # statement the ground-truth answer was verified against. Passing
            # the same metadata to both pipelines makes the comparison fair.
            common_args["statement_type"] = qa["statement_type"]

        try:
            t0 = time.time()
            graph_result = graph_rag_pipeline.answer(
                qa["question"],
                **common_args,
            )
            graph_latency = time.time() - t0

            t0 = time.time()
            baseline_result = baseline_pipeline.answer(
                qa["question"],
                **common_args,
            )
            baseline_latency = time.time() - t0

            print(f"GraphRAG: {graph_latency:.2f}s | Baseline: {baseline_latency:.2f}s")

        except Exception as e:
            failures += 1
            print(f"  !! skipped due to error: {e}")
            continue

        graph_retrieval = avg_retrieval_scores(graph_result)
        baseline_retrieval = avg_retrieval_scores(baseline_result)

        graph_hit_recall = retrieval_hit_recall_at_k(graph_result, qa)
        baseline_hit_recall = retrieval_hit_recall_at_k(
            baseline_result,
            qa,
        )

        graph_numeric = numeric_metrics(
            graph_result["answer"],
            qa["answer"],
            qa.get("expected_numbers"),
        )
        baseline_numeric = numeric_metrics(
            baseline_result["answer"],
            qa["answer"],
            qa.get("expected_numbers"),
        )

        rows.append(
            {
                "question": qa["question"],
                # ground_truth.json entries use "modality" (pdf/audio/
                # pdf+audio), not "type" — qa.get("type", ...) always
                # missed and silently fell back to "unknown" for every row.
                "type": qa.get("modality", "unknown"),
                "company": qa["company"],
                "quarter": qa["quarter"],
                "year": qa["year"],
                "metric": qa.get("metric"),
                # `or "unspecified"` (not just a .get default) because some
                # entries have statement_type explicitly set to null/None
                # rather than the key being absent — .get()'s default only
                # covers a missing key, not a present-but-None value.
                "statement_type": qa.get("statement_type") or "unspecified",
                "unit": qa.get("unit") or "unspecified",
                "ground_truth": qa["answer"],
                "expected_numbers": json.dumps(
                    sorted(str(x) for x in (_extract_answer_numbers(qa["answer"])))
                ),
                "graphrag_answer": graph_result["answer"],
                "graphrag_evidence": graph_result.get("evidence_text"),
                # Old metric retained.
                "graphrag_numeric_match": graph_numeric["recall"],
                # New stricter metrics.
                "graphrag_numeric_recall": graph_numeric["recall"],
                "graphrag_numeric_precision": graph_numeric["precision"],
                "graphrag_numeric_f1": graph_numeric["f1"],
                "graphrag_numeric_exact": graph_numeric["exact"],
                "graphrag_missing_numbers": json.dumps(graph_numeric["missing"]),
                "graphrag_extra_numbers": json.dumps(graph_numeric["extra"]),
                "graphrag_semantic_sim": semantic_similarity(
                    graph_result["answer"],
                    qa["answer"],
                ),
                "graphrag_latency_sec": round(
                    graph_latency,
                    2,
                ),
                "graphrag_num_facts_used": graph_result.get("num_facts_used"),
                "graphrag_hallucinated_numbers": hallucinated_number_count(
                    graph_result["answer"],
                    graph_result.get("evidence_text", ""),
                    qa["answer"],
                ),
                "baseline_answer": baseline_result["answer"],
                "baseline_evidence": baseline_result.get("evidence_text"),
                # Old metric retained.
                "baseline_numeric_match": baseline_numeric["recall"],
                # New stricter metrics.
                "baseline_numeric_recall": baseline_numeric["recall"],
                "baseline_numeric_precision": baseline_numeric["precision"],
                "baseline_numeric_f1": baseline_numeric["f1"],
                "baseline_numeric_exact": baseline_numeric["exact"],
                "baseline_missing_numbers": json.dumps(baseline_numeric["missing"]),
                "baseline_extra_numbers": json.dumps(baseline_numeric["extra"]),
                "baseline_semantic_sim": semantic_similarity(
                    baseline_result["answer"],
                    qa["answer"],
                ),
                "baseline_latency_sec": round(
                    baseline_latency,
                    2,
                ),
                "baseline_hallucinated_numbers": hallucinated_number_count(
                    baseline_result["answer"],
                    baseline_result.get("evidence_text", ""),
                    qa["answer"],
                ),
                "retrieval_overlap": retrieval_overlap(
                    graph_result.get("sources", []),
                    baseline_result.get("sources", []),
                ),
                "graphrag_avg_retrieval_score": graph_retrieval["avg_retrieval_score"],
                "graphrag_avg_embedding_score": graph_retrieval["avg_embedding_score"],
                "graphrag_avg_bm25_score": graph_retrieval["avg_bm25_score"],
                "graphrag_avg_fastrp_score": graph_retrieval["avg_fastrp_score"],
                "baseline_avg_retrieval_score": baseline_retrieval[
                    "avg_retrieval_score"
                ],
                "baseline_avg_embedding_score": baseline_retrieval[
                    "avg_embedding_score"
                ],
                "baseline_avg_bm25_score": baseline_retrieval["avg_bm25_score"],
                "baseline_avg_fastrp_score": baseline_retrieval[
                    "avg_fastrp_score"
                ],
                "graphrag_hit_at_k": graph_hit_recall["hit_at_k"],
                "graphrag_recall_at_k": graph_hit_recall["recall_at_k"],
                # Per-question retrieved chunk ids and their modalities.
                # Stored so downstream evaluation (full_metrics.py) can
                # compute MRR, context precision and source-type accuracy
                # without re-running the pipeline - previously only the
                # aggregate hit@k survived, so those three metrics were
                # impossible to reconstruct from archived results.
                "graphrag_retrieved_chunk_ids": json.dumps(
                    [c.get("chunk_id") for c in _retrieved_chunks(graph_result)]
                ),
                "graphrag_retrieved_source_types": json.dumps(
                    [c.get("source_type") for c in _retrieved_chunks(graph_result)]
                ),
                "baseline_retrieved_chunk_ids": json.dumps(
                    [c.get("chunk_id") for c in _retrieved_chunks(baseline_result)]
                ),
                "baseline_hit_at_k": baseline_hit_recall["hit_at_k"],
                "baseline_recall_at_k": baseline_hit_recall["recall_at_k"],
            }
        )

        # --- extended metrics, computed inline so a single run produces
        # the full report instead of needing a second offline pass ---
        gr_text = text_overlap_metrics(graph_result["answer"], qa["answer"])
        bl_text = text_overlap_metrics(baseline_result["answer"], qa["answer"])
        rows[-1].update({f"graphrag_{k}": v for k, v in gr_text.items()})
        rows[-1].update({f"baseline_{k}": v for k, v in bl_text.items()})

        rows[-1]["graphrag_numeric_match_partial"] = numeric_match_partial(
            graph_result["answer"], qa["answer"]
        )
        rows[-1]["baseline_numeric_match_partial"] = numeric_match_partial(
            baseline_result["answer"], qa["answer"]
        )

        # FinTermPrec / FCD / retrieval latency - see each function above.
        rows[-1]["graphrag_fin_term_precision"] = financial_term_precision(
            graph_result["answer"], qa["answer"]
        )
        rows[-1]["baseline_fin_term_precision"] = financial_term_precision(
            baseline_result["answer"], qa["answer"]
        )
        rows[-1]["graphrag_factual_consistency"] = factual_consistency(
            graph_result["answer"], qa["answer"]
        )
        rows[-1]["baseline_factual_consistency"] = factual_consistency(
            baseline_result["answer"], qa["answer"]
        )
        # graph_rag_pipeline.answer() reports retrieval_sec nested under
        # "timing" (alongside graph_fact_fetch_sec/llm_generation_sec),
        # unlike baseline_pipeline.answer() which returns a flat
        # "retrieval_sec" key. Reading graph_result.get("retrieval_sec")
        # directly always returns None - confirmed 0/N non-null rows for
        # graphrag vs N/N for baseline, which silently zeroed out
        # graphrag_retrieval_latency in every summary bucket. This exact
        # fix was made once before and was lost when this file got
        # refreshed from an upload later in the project without
        # re-checking it was still present - re-applying it now.
        rows[-1]["graphrag_retrieval_latency"] = graph_result.get(
            "timing", {}
        ).get("retrieval_sec")
        rows[-1]["baseline_retrieval_latency"] = baseline_result.get("retrieval_sec")

        rows[-1]["graphrag_evidence_sufficiency"] = evidence_sufficiency(graph_result, qa)
        rows[-1]["baseline_evidence_sufficiency"] = evidence_sufficiency(baseline_result, qa)

        # Tolerance match / abstained / multi-value completeness - the
        # "not too blunt" companions to strict numeric_exact. All three
        # report ALONGSIDE the strict metrics already above, never in
        # place of them.
        rows[-1]["graphrag_tolerance_match"] = tolerance_match(
            graph_result["answer"], qa["answer"]
        )
        rows[-1]["baseline_tolerance_match"] = tolerance_match(
            baseline_result["answer"], qa["answer"]
        )
        rows[-1]["graphrag_abstained"] = abstained(graph_result["answer"])
        rows[-1]["baseline_abstained"] = abstained(baseline_result["answer"])
        rows[-1]["graphrag_multivalue_complete"] = multivalue_complete(
            graph_result["answer"], qa["answer"], qa.get("expected_answer")
        )
        rows[-1]["baseline_multivalue_complete"] = multivalue_complete(
            baseline_result["answer"], qa["answer"], qa.get("expected_answer")
        )

        # Loose (document+page) retrieval hit — see source_hit_recall_at_k's
        # docstring for why this exists alongside the strict chunk_id one.
        gr_source = source_hit_recall_at_k(graph_result, qa)
        bl_source = source_hit_recall_at_k(baseline_result, qa)
        rows[-1]["graphrag_source_hit_at_k"] = gr_source["source_hit_at_k"]
        rows[-1]["graphrag_source_recall_at_k"] = gr_source["source_recall_at_k"]
        rows[-1]["baseline_source_hit_at_k"] = bl_source["source_hit_at_k"]
        rows[-1]["baseline_source_recall_at_k"] = bl_source["source_recall_at_k"]

        gr_rank = rank_retrieval_metrics(graph_result, qa)
        bl_rank = rank_retrieval_metrics(baseline_result, qa)
        rows[-1].update({f"graphrag_{k}": v for k, v in gr_rank.items()})
        rows[-1].update({f"baseline_{k}": v for k, v in bl_rank.items()})

        if getattr(config, "BENCHMARK_LLM_JUDGE", False):
            verdict = llm_judge(
                qa["question"], qa["answer"],
                graph_result["answer"], graph_result.get("evidence_text", ""),
            )
            rows[-1]["graphrag_judge_correct"] = verdict.get("correct")
            rows[-1]["graphrag_judge_faithful"] = verdict.get("faithful")
            rows[-1]["graphrag_judge_failure_type"] = verdict.get("failure_type")
            rows[-1]["graphrag_judge_reason"] = verdict.get("reason")

            # P1 fix: only graphrag was ever judged, so
            # graphrag_judge_correct had nothing to compare against for
            # whether the graph structure actually helps beyond what an
            # LLM judge (not just string-overlap metrics) would credit
            # baseline for too. Judging both sides doubles the judge
            # LLM-call cost per question when this flag is on, but
            # without it the "does the graph help" question this whole
            # benchmark exists to answer can't be asked at the judge
            # level, only at the string-metric level.
            baseline_verdict = llm_judge(
                qa["question"], qa["answer"],
                baseline_result["answer"], baseline_result.get("evidence_text", ""),
            )
            rows[-1]["baseline_judge_correct"] = baseline_verdict.get("correct")
            rows[-1]["baseline_judge_faithful"] = baseline_verdict.get("faithful")
            rows[-1]["baseline_judge_failure_type"] = baseline_verdict.get("failure_type")
            rows[-1]["baseline_judge_reason"] = baseline_verdict.get("reason")

        completed.add(_question_key(qa))
        _save_progress(rows)

    if failures:
        print(f"\n{failures} question(s) failed and were skipped.")

    # BERTScore in one batch - loads the model once rather than per question.
    if rows:
        print("\nComputing BERTScore...")
        gr = bertscore_batch([r.get("graphrag_answer") for r in rows],
                             [r.get("ground_truth") for r in rows])
        bl = bertscore_batch([r.get("baseline_answer") for r in rows],
                             [r.get("ground_truth") for r in rows])
        for r, a, b in zip(rows, gr, bl):
            r["graphrag_bertscore_f1"] = a
            r["baseline_bertscore_f1"] = b

    global _INDEX_SIZE
    _INDEX_SIZE = index_size()

    df, detailed_path, detailed_json_path = _save_progress(rows)

    print(
        f"\nDetailed per-question results saved to "
        f"{detailed_path} and {detailed_json_path}"
    )

    _print_summary(df)


def _print_summary(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SUMMARY — GraphRAG vs Baseline")
    print("=" * 80)

    retrieval_cols = [
        "graphrag_avg_retrieval_score",
        "graphrag_avg_embedding_score",
        "graphrag_avg_bm25_score",
        "graphrag_avg_fastrp_score",
        "baseline_avg_retrieval_score",
        "baseline_avg_embedding_score",
        "baseline_avg_bm25_score",
        "baseline_avg_fastrp_score",
        "graphrag_hit_at_k",
        "graphrag_recall_at_k",
        "baseline_hit_at_k",
        "baseline_recall_at_k",
    ]

    def _safe_mean(subset, col):
        if col not in subset.columns:
            return None

        numeric = pd.to_numeric(
            subset[col],
            errors="coerce",
        )

        return numeric.mean() if numeric.notna().any() else None

    summary_rows = []

    for qtype in ["pdf", "audio", "pdf+audio", "ALL"]:
        subset = df if qtype == "ALL" else df[df["type"] == qtype]

        if subset.empty:
            continue

        row = {
            "question_type": qtype,
            "n_questions": len(subset),
            "graphrag_numeric_match": subset["graphrag_numeric_match"].mean(),
            "baseline_numeric_match": subset["baseline_numeric_match"].mean(),
            "graphrag_numeric_precision": _safe_mean(
                subset,
                "graphrag_numeric_precision",
            ),
            "baseline_numeric_precision": _safe_mean(
                subset,
                "baseline_numeric_precision",
            ),
            "graphrag_numeric_f1": _safe_mean(
                subset,
                "graphrag_numeric_f1",
            ),
            "baseline_numeric_f1": _safe_mean(
                subset,
                "baseline_numeric_f1",
            ),
            "graphrag_numeric_exact": _safe_mean(
                subset,
                "graphrag_numeric_exact",
            ),
            "baseline_numeric_exact": _safe_mean(
                subset,
                "baseline_numeric_exact",
            ),
            "graphrag_semantic_sim": subset["graphrag_semantic_sim"].mean(),
            "baseline_semantic_sim": subset["baseline_semantic_sim"].mean(),
            "graphrag_avg_latency": subset["graphrag_latency_sec"].mean(),
            "baseline_avg_latency": subset["baseline_latency_sec"].mean(),
            "graphrag_avg_facts_used": _safe_mean(
                subset,
                "graphrag_num_facts_used",
            ),
            "graphrag_avg_hallucinated_numbers": subset[
                "graphrag_hallucinated_numbers"
            ].mean(),
            "baseline_avg_hallucinated_numbers": subset[
                "baseline_hallucinated_numbers"
            ].mean(),
            "avg_retrieval_overlap": subset["retrieval_overlap"].mean(),
        }

        for col in retrieval_cols:
            row[col] = _safe_mean(subset, col)

        # Extended metrics. Grouped in the printed report as PRIMARY
        # (paraphrase-robust) vs SECONDARY (token overlap) so the harsher
        # overlap scores are not mistaken for the headline result.
        for col in (
            "graphrag_numeric_match_partial", "baseline_numeric_match_partial",
            "graphrag_bertscore_f1", "baseline_bertscore_f1",
            "graphrag_rouge1", "graphrag_rouge2", "graphrag_rougeL",
            "graphrag_bleu", "graphrag_meteor",
            "baseline_rouge1", "baseline_rouge2", "baseline_rougeL",
            "baseline_bleu", "baseline_meteor",
            "graphrag_evidence_sufficiency", "baseline_evidence_sufficiency",
            "graphrag_mrr", "graphrag_context_precision", "graphrag_source_type_acc",
            "baseline_mrr", "baseline_context_precision", "baseline_source_type_acc",
            "graphrag_judge_correct", "graphrag_judge_faithful",
            "baseline_judge_correct", "baseline_judge_faithful",
            "graphrag_fin_term_precision", "baseline_fin_term_precision",
            "graphrag_factual_consistency", "baseline_factual_consistency",
            "graphrag_retrieval_latency", "baseline_retrieval_latency",
            # "Not too blunt" companions to the strict metrics above -
            # see tolerance_match/abstained/multivalue_complete/
            # source_hit_recall_at_k docstrings for what each one adds.
            "graphrag_tolerance_match", "baseline_tolerance_match",
            "graphrag_abstained", "baseline_abstained",
            "graphrag_multivalue_complete", "baseline_multivalue_complete",
            "graphrag_source_hit_at_k", "graphrag_source_recall_at_k",
            "baseline_source_hit_at_k", "baseline_source_recall_at_k",
        ):
            row[col] = _safe_mean(subset, col)

        # Index size is a property of the corpus, identical for every row.
        row["index_size"] = _INDEX_SIZE

        lat = row.get("graphrag_avg_latency")
        row["graphrag_throughput_qps"] = (1.0 / lat) if lat else None
        # P1 fix: confirmed missing entirely - only graphrag ever got a
        # throughput number, baseline_throughput_qps didn't exist as a
        # field at all, showing up as 100% blank across every row of
        # generate_report.py's output (28/28 rows, not a partial gap).
        # Same computation, mirrored for baseline.
        baseline_lat = row.get("baseline_avg_latency")
        row["baseline_throughput_qps"] = (1.0 / baseline_lat) if baseline_lat else None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df.to_string(index=False))

    summary_path = config.RESULTS_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_json_path = config.RESULTS_DIR / "benchmark_summary.json"
    summary_json_path.write_text(
        summary_df.to_json(
            orient="records",
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSummary saved to {summary_path} and {summary_json_path}")

    excel_path = config.RESULTS_DIR / "benchmark_results.xlsx"

    try:
        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        ) as writer:
            df.to_excel(
                writer,
                sheet_name="detailed",
                index=False,
            )
            summary_df.to_excel(
                writer,
                sheet_name="summary",
                index=False,
            )

        print(f"Excel workbook saved to {excel_path}")

    except ImportError:
        print(
            "[warn] Excel export skipped — install openpyxl "
            "if you want benchmark_results.xlsx"
        )


if __name__ == "__main__":
    run_benchmark()
