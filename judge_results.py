"""
judge_results.py
=================
Adds an LLM-as-judge evaluation and several cheap-but-meaningful metrics
to ALREADY-COMPLETED benchmark runs. Nothing is re-run: every input the
judge needs (question, ground truth, generated answer, retrieved
evidence) is already stored in benchmark_detailed.csv.

Usage (from your project directory):
    python judge_results.py                       # judges the best config
    python judge_results.py --config full_graphfacts5
    python judge_results.py --all                 # every config (slow)
    python judge_results.py --no-llm              # deterministic metrics only, instant
    python judge_results.py --limit 10            # judge first N rows (smoke test)

Output, written next to the config's existing results:
    ablation/<config>/judged.csv        per-question judge verdicts + metrics
    ablation/<config>/judged_summary.json
    ablation/JUDGE_COMPARISON.csv       one row per judged config

WHY AN LLM JUDGE HERE
---------------------
The existing numeric metrics are string-set comparisons. They punish
answers that are substantively right but formatted differently, and they
cannot see WHY an answer failed. Three failure modes look identical to
exact-match but are completely different engineering problems:

  1. the evidence never contained the answer      -> retrieval problem
  2. the evidence had it, the model missed it     -> generation problem
  3. the model was right, the scorer was too rigid-> measurement problem

The judge separates these by scoring FAITHFULNESS (is the answer
supported by the retrieved evidence?) independently from CORRECTNESS (is
it consistent with the ground truth?). The cross-tab of those two is the
useful artifact — see the summary this prints at the end.

DETERMINISTIC METRICS ADDED (no LLM, always computed)
-----------------------------------------------------
  tolerance_match   Numeric match within a relative tolerance (default
                    1%). Exact-match calls 3.6% wrong when ground truth
                    says 3.7% — a rounding difference on a correctly
                    computed percentage, not a wrong answer. This is the
                    fairer companion metric to exact-match, not a
                    replacement for it; report both.
  abstained         The system said the evidence doesn't contain the
                    answer. Currently invisible in every metric, yet
                    abstaining is very different from answering wrongly —
                    a system that declines when it lacks evidence is
                    behaving correctly, and you cannot show that without
                    measuring it.
  multivalue_complete
                    For comparison questions (which need both periods'
                    figures plus a change), whether the answer supplied
                    every value the ground truth contains. Partial
                    answers were a recurring failure mode; this measures
                    it directly instead of via depressed recall.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

import config

ABLATION_DIR = Path(__file__).resolve().parent / "ablation"
DEFAULT_CONFIG = "dense_bm25_fastrp_expand"  # best F1 in the ablation

NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

ABSTAIN_PATTERNS = [
    "does not contain", "doesn't contain", "not contain this information",
    "cannot determine", "can't determine", "no information",
    "not available in the", "unable to determine", "not found in the",
    "insufficient evidence", "does not provide",
]

JUDGE_PROMPT = """You are evaluating a financial question-answering system.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

SYSTEM'S GENERATED ANSWER:
{answer}

EVIDENCE THAT WAS RETRIEVED AND GIVEN TO THE SYSTEM:
{evidence}

Assess two things INDEPENDENTLY.

1. FAITHFUL — is every factual claim in the generated answer actually
   supported by the retrieved evidence above? An answer can be faithful
   but still wrong (if the evidence itself was wrong or incomplete). An
   answer that states figures not present in the evidence is NOT
   faithful. An answer that correctly says the evidence lacks the
   information IS faithful.

2. CORRECT — is the generated answer consistent with the ground truth?
   Judge the substance, not the wording or formatting. A different unit
   phrasing ("2,345.98 Crore INR" vs "Rs2,345.98 crore") is still
   correct. A rounding difference in a percentage (3.6% vs 3.7%) is
   still correct. A different NUMBER is not correct.

Also give a one-sentence reason, and classify the failure if any:
  none              - answer is correct
  retrieval         - the evidence did not contain what was needed
  generation        - the evidence had it but the answer got it wrong
  evaluation        - the answer is substantively right; only formatting
                      or rounding differs from the ground truth
  data_quality      - the evidence itself contains garbled or wrong
                      source values
  abstained         - the system declined to answer

Respond with ONLY a JSON object, no other text:
{{"faithful": true/false, "correct": true/false, "failure_type": "...", "reason": "..."}}"""


def _numbers(text):
    """Extract answer numbers using benchmark.py's OWN extractor, so these
    metrics stay consistent with the official ones (it strips period
    labels like "Q2 FY26" and honours the FINAL ANSWER marker). Falls
    back to a plain regex only if benchmark.py can't be imported, which
    would otherwise count the "2" and "26" in "Q2 FY26" as figures and
    make tolerance_match come out LOWER than exact match."""
    try:
        import benchmark
        return benchmark._extract_answer_numbers(str(text or ""))
    except Exception:
        return {n.replace(",", "") for n in NUMBER_RE.findall(str(text or ""))}


def tolerance_match(answer, ground_truth, tol=0.01):
    """True when every ground-truth number is matched by some number in
    the answer within `tol` relative error. Catches correct-but-rounded
    values that exact string matching rejects."""
    gt = _numbers(ground_truth)
    got = _numbers(answer)
    if not gt:
        return None
    def close(a, b):
        # Identical strings match regardless of parseability — this also
        # covers percentage tokens like "3.7%", which float() rejects and
        # which would otherwise make tolerance_match score BELOW exact
        # match on rows whose number sets are literally identical.
        if a == b:
            return True
        try:
            a, b = float(str(a).rstrip("%")), float(str(b).rstrip("%"))
        except ValueError:
            return False
        if a == b:
            return True
        denom = max(abs(a), abs(b))
        return denom > 0 and abs(a - b) / denom <= tol
    return all(any(close(g, o) for o in got) for g in gt)


def abstained(answer):
    low = str(answer or "").lower()
    return any(p in low for p in ABSTAIN_PATTERNS)


def is_multivalue(ground_truth):
    """Ground truths with 3+ distinct numbers are comparison-style
    (two periods plus a change/percentage)."""
    return len(_numbers(ground_truth)) >= 3


def multivalue_complete(answer, ground_truth):
    if not is_multivalue(ground_truth):
        return None
    gt, got = _numbers(ground_truth), _numbers(answer)
    return len(gt & got) == len(gt)


def judge_row(row, llm):
    evidence = str(row.get("graphrag_evidence") or "")
    # Keep the prompt within the model's context; the metrics block at the
    # top of the evidence is the part that matters most for judging.
    if len(evidence) > 6000:
        evidence = evidence[:6000] + "\n[...evidence truncated...]"

    prompt = JUDGE_PROMPT.format(
        question=row["question"],
        ground_truth=row["ground_truth"],
        answer=row["graphrag_answer"],
        evidence=evidence,
    )
    try:
        raw = llm.generate(prompt, model=config.ANSWER_MODEL,
                           temperature=0.0, num_predict=300)
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"faithful": None, "correct": None,
                    "failure_type": "judge_unparseable", "reason": raw[:200]}
        return json.loads(m.group(0))
    except Exception as e:
        return {"faithful": None, "correct": None,
                "failure_type": "judge_error", "reason": str(e)[:200]}


def process(config_name, use_llm=True, limit=None):
    cfg_dir = ABLATION_DIR / config_name
    detailed = cfg_dir / "benchmark_detailed.csv"
    if not detailed.exists():
        print(f"  no results at {detailed} — skipping")
        return None

    df = pd.read_csv(detailed)
    if limit:
        df = df.head(limit)
    print(f"\n=== {config_name} ({len(df)} questions) ===")

    df["tolerance_match"] = df.apply(
        lambda r: tolerance_match(r["graphrag_answer"], r["ground_truth"]), axis=1)
    df["abstained"] = df["graphrag_answer"].apply(abstained)
    df["multivalue_complete"] = df.apply(
        lambda r: multivalue_complete(r["graphrag_answer"], r["ground_truth"]), axis=1)

    if use_llm:
        import llm_client
        verdicts = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            v = judge_row(row, llm_client)
            verdicts.append(v)
            mark = "OK " if v.get("correct") else "-- "
            print(f"  [{i}/{len(df)}] {mark} {str(row['question'])[:58]}")
        df["judge_faithful"] = [v.get("faithful") for v in verdicts]
        df["judge_correct"] = [v.get("correct") for v in verdicts]
        df["judge_failure_type"] = [v.get("failure_type") for v in verdicts]
        df["judge_reason"] = [v.get("reason") for v in verdicts]

    out_csv = cfg_dir / "judged.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "config": config_name,
        "n_questions": len(df),
        "exact_match": float(df["graphrag_numeric_exact"].mean(skipna=True))
        if "graphrag_numeric_exact" in df else None,
        "tolerance_match_1pct": float(df["tolerance_match"].dropna().mean())
        if df["tolerance_match"].notna().any() else None,
        "abstain_rate": float(df["abstained"].mean()),
        "multivalue_complete": float(df["multivalue_complete"].dropna().mean())
        if df["multivalue_complete"].notna().any() else None,
    }
    if use_llm:
        summary.update({
            "judge_faithful": float(pd.Series(df["judge_faithful"]).dropna().mean()),
            "judge_correct": float(pd.Series(df["judge_correct"]).dropna().mean()),
            "failure_breakdown": df["judge_failure_type"].value_counts().to_dict(),
            # The diagnostic cross-tab: faithful-but-incorrect means the
            # evidence was bad; unfaithful-but-correct means the model got
            # lucky rather than grounded.
            "faithful_and_correct": int(
                ((df["judge_faithful"] == True) & (df["judge_correct"] == True)).sum()),
            "faithful_not_correct": int(
                ((df["judge_faithful"] == True) & (df["judge_correct"] == False)).sum()),
            "unfaithful_but_correct": int(
                ((df["judge_faithful"] == False) & (df["judge_correct"] == True)).sum()),
        })

    (cfg_dir / "judged_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  -> {out_csv.name}, judged_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--all", action="store_true", help="judge every config")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic metrics only (instant)")
    ap.add_argument("--limit", type=int, help="only first N questions")
    args = ap.parse_args()

    if not ABLATION_DIR.exists():
        print(f"{ABLATION_DIR} not found — run this from your project directory.")
        sys.exit(1)

    if args.all:
        names = sorted(p.name for p in ABLATION_DIR.iterdir()
                       if p.is_dir() and (p / "benchmark_detailed.csv").exists())
    else:
        names = [args.config]

    summaries = [s for s in
                 (process(n, use_llm=not args.no_llm, limit=args.limit) for n in names)
                 if s]

    if summaries:
        out = ABLATION_DIR / "JUDGE_COMPARISON.csv"
        pd.DataFrame(summaries).to_csv(out, index=False)
        print(f"\nWrote {out}")
        print()
        for s in summaries:
            print(f"--- {s['config']} ---")
            print(f"  exact match (strict)      : {s.get('exact_match')}")
            print(f"  tolerance match (1%)      : {s.get('tolerance_match_1pct')}")
            print(f"  abstain rate              : {s.get('abstain_rate')}")
            print(f"  multi-value completeness  : {s.get('multivalue_complete')}")
            if "judge_correct" in s:
                print(f"  JUDGE correct             : {s['judge_correct']}")
                print(f"  JUDGE faithful            : {s['judge_faithful']}")
                print(f"  faithful AND correct      : {s['faithful_and_correct']}")
                print(f"  faithful but NOT correct  : {s['faithful_not_correct']}"
                      f"   <- evidence was wrong/incomplete")
                print(f"  correct but NOT faithful  : {s['unfaithful_but_correct']}"
                      f"   <- answered right without grounding")
                print(f"  failures: {s.get('failure_breakdown')}")


if __name__ == "__main__":
    main()
