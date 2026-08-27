"""
generate_report_markdown.py
=============================
WHAT THIS FILE DOES:
Reads REPORT_MASTER.csv (already produced by generate_report.py - this
doesn't recompute or re-derive anything, just reformats the same,
already-verified data) and writes a Markdown file with real tables
instead of the plain-text aligned columns in REPORT.txt.

Table layout: experiments as ROWS, metrics as COLUMNS - this is the
orientation that actually matters for an ablation report, since the
question you're answering is "how does each metric change as I add each
component", which means comparing down a column across EXP1->EXP7, not
across a row within one experiment. One table per (bucket, metric
category, pipeline) - GraphRAG and baseline get separate tables so each
stays a readable width instead of one 17-column table.

USAGE:
    python generate_report_markdown.py
    python generate_report_markdown.py --csv ablation/REPORT_MASTER.csv --out ablation/REPORT.md

Output:
    ablation/REPORT.md
"""

import argparse
import csv
from pathlib import Path

BUCKET_ORDER = ["pdf", "audio", "pdf+audio", "ALL"]
BUCKET_LABELS = {"pdf": "PDF", "audio": "Audio", "pdf+audio": "PDF+Audio", "ALL": "Combined (ALL)"}

EXP_ORDER = [
    "exp1_dense", "exp2_dense_bm25", "exp3_dense_bm25_fastrp",
    "exp4_plus_graph_facts", "exp5_plus_temporal_graph",
    "exp6_plus_routing", "exp7_plus_hyde_final",
]
EXP_SHORT_LABELS = {
    "exp1_dense": "EXP1 Dense",
    "exp2_dense_bm25": "EXP2 +BM25",
    "exp3_dense_bm25_fastrp": "EXP3 +FastRP",
    "exp4_plus_graph_facts": "EXP4 +GraphFacts",
    "exp5_plus_temporal_graph": "EXP5 +Temporal",
    "exp6_plus_routing": "EXP6 +Routing",
    "exp7_plus_hyde_final": "EXP7 +HyDE (final)",
}

# Same metric groupings as generate_report.py, kept identical so the
# Markdown version and the plain-text version show exactly the same
# numbers organized the same way - just rendered differently.
PRIMARY_METRICS = [
    ("LLM Judge Correct", "judge_correct", "pct"),
    ("LLM Judge Faithful", "judge_faithful", "pct"),
    ("Numeric Exact", "numeric_exact", "pct"),
    ("Numeric Match (partial)", "numeric_match", "pct"),
    ("Numeric F1", "numeric_f1", "pct"),
    ("Numeric Precision", "numeric_precision", "pct"),
    ("Tolerance Match (1%)", "tolerance_match", "pct"),
    ("BERTScore F1", "bertscore_f1", "pct"),
]
SECONDARY_METRICS = [
    ("METEOR", "meteor", "num"),
    ("ROUGE-1", "rouge1", "num"),
    ("ROUGE-2", "rouge2", "num"),
    ("ROUGE-L", "rougeL", "num"),
    ("BLEU", "bleu", "num"),
]
RETRIEVAL_METRICS = [
    ("Hit@K", "hit_at_k", "pct"),
    ("Recall@K", "recall_at_k", "pct"),
    ("Source Hit@K", "source_hit_at_k", "pct"),
    ("Source Recall@K", "source_recall_at_k", "pct"),
    ("MRR", "mrr", "num"),
    ("Context Precision", "context_precision", "num"),
    ("Source Type Acc", "source_type_acc", "pct"),
    ("Evidence Sufficiency", "evidence_sufficiency", "pct"),
    ("Factual Consistency", "factual_consistency", "pct"),
    ("Fin. Term Precision", "fin_term_precision", "pct"),
    ("Avg Cosine Sim", "semantic_sim", "pct"),
    ("Avg FastRP Score", "avg_fastrp_score", "num"),
    ("Avg BM25 Score", "avg_bm25_score", "num"),
]
BEHAVIOR_METRICS = [
    ("Abstain Rate", "abstained", "pct"),
    ("Multi-value Complete", "multivalue_complete", "pct"),
    ("Hallucinated Numbers", "avg_hallucinated_numbers", "num"),
]
EFFICIENCY_METRICS = [
    ("Retrieval Latency", "retrieval_latency", "sec"),
    ("E2E Latency", "avg_latency", "sec"),
    ("Throughput", "throughput_qps", "qps"),
]
CATEGORIES = [
    ("Primary Metrics (paraphrase-robust)", PRIMARY_METRICS),
    ("Secondary Metrics (token overlap)", SECONDARY_METRICS),
    ("Retrieval Quality", RETRIEVAL_METRICS),
    ("Answer Behavior", BEHAVIOR_METRICS),
    ("System Efficiency", EFFICIENCY_METRICS),
]


def _fmt(value, kind):
    if value in (None, ""):
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "sec":
        return f"{v:.3f}s"
    if kind == "qps":
        return f"{v:.4f}"
    return f"{v:.4f}"


def build_markdown(rows: list) -> str:
    by_key = {(r["experiment"], r["bucket"]): r for r in rows}
    lines = ["# Ablation Report\n"]
    lines.append(
        "Experiments as rows, metrics as columns — read down a column to "
        "see how that metric changes as each component gets added "
        "(EXP1 -> EXP7). GraphRAG and baseline are separate tables.\n"
    )

    # Experiment legend once at the top, not repeated in every table.
    lines.append("## Experiment key\n")
    lines.append("| Experiment | Description |")
    lines.append("|---|---|")
    seen_desc = {}
    for r in rows:
        seen_desc[r["experiment"]] = r.get("description", "")
    for exp in EXP_ORDER:
        if exp in seen_desc:
            lines.append(f"| {EXP_SHORT_LABELS.get(exp, exp)} | {seen_desc[exp]} |")
    lines.append("")

    for bucket in BUCKET_ORDER:
        n = None
        for exp in EXP_ORDER:
            r = by_key.get((exp, bucket))
            if r:
                n = r.get("n_questions")
                break
        if n is None:
            continue

        lines.append(f"## {BUCKET_LABELS[bucket]} (n={n})\n")

        for category_title, metrics in CATEGORIES:
            for pipeline in ("graphrag", "baseline"):
                pipeline_label = "GraphRAG" if pipeline == "graphrag" else "Baseline"
                lines.append(f"### {category_title} — {pipeline_label}\n")
                header = "| Experiment | " + " | ".join(label for label, _, _ in metrics) + " |"
                sep = "|---|" + "|".join("---" for _ in metrics) + "|"
                lines.append(header)
                lines.append(sep)
                for exp in EXP_ORDER:
                    r = by_key.get((exp, bucket))
                    if not r:
                        continue
                    cells = [EXP_SHORT_LABELS.get(exp, exp)]
                    for _, field, kind in metrics:
                        cells.append(_fmt(r.get(f"{pipeline}_{field}"), kind))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ablation/REPORT_MASTER.csv")
    parser.add_argument("--out", default="ablation/REPORT.md")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    md = build_markdown(rows)

    out_path = Path(args.out)
    out_path.write_text(md, encoding="utf-8")
    print(f"Saved: {out_path}  ({len(rows)} source rows, {md.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
