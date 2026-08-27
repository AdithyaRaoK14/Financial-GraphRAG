"""
generate_final_report.py
===========================
WHAT THIS FILE DOES:
The one combined report: the EXP1-7 ablation ladder (ablation/*/
benchmark_summary.json) AND the GraphRAG vs baseline vs vector-DB
3-way comparison (vectordb_comparison/summary.json), in one markdown
file. Reads both sources directly, computes nothing new - this is a
presentation layer over data that's already been generated and
verified separately by run_ablation.py and benchmark_vectordb.py.

Two sections:
  1. FINAL 3-WAY COMPARISON - GraphRAG (your best config, EXP7) vs
     baseline vs the new vector-DB baseline, one table per bucket per
     metric category, pipelines as rows (there's only 3 systems here,
     not a ladder to read down a column of - a straight comparison
     table is the right shape for this section).
  2. ABLATION LADDER (EXP1-7) - same table shape as generate_report_
     markdown.py already produces (experiments as rows, metrics as
     columns) - read down a column to see what each component added.

USAGE:
    python generate_final_report.py
    python generate_final_report.py --ablation-dir ablation --vectordb-dir vectordb_comparison --out FINAL_REPORT.md

Output:
    FINAL_REPORT.md
"""

import argparse
import json
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
FINAL_CONFIG_NAME = "exp7_plus_hyde_final"

PIPELINE_LABELS = {"graphrag": "GraphRAG", "baseline": "Baseline (Neo4j)", "vectordb": "Vector-DB (FAISS)"}

# Same metric groupings as generate_report_markdown.py, kept identical
# so both reports show the same numbers organized the same way.
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

# vectordb_comparison/summary.json uses different field-name suffixes
# than benchmark_summary.json for latency/hallucination (confirmed from
# real output: "latency_sec" not "avg_latency", "hallucinated_numbers"
# not "avg_hallucinated_numbers") - this maps the metric-definition
# suffix used above to the actual key for each source file's schema,
# so one shared CATEGORIES list works against both without silently
# reading a field that isn't there.
VECTORDB_FIELD_ALIASES = {
    "avg_latency": "latency_sec",
    "avg_hallucinated_numbers": "hallucinated_numbers",
}


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


def _row_for_bucket(summary_rows: list, bucket: str) -> dict:
    for row in summary_rows:
        if row.get("question_type") == bucket:
            return row
    return {}


def build_three_way_section(ablation_dir: Path, vectordb_dir: Path) -> list:
    lines = ["## Final 3-Way Comparison\n"]
    lines.append(
        f"GraphRAG (from `{FINAL_CONFIG_NAME}`, your final ablation config) vs "
        f"the existing Neo4j-hosted baseline vs the new FAISS vector-DB baseline. "
        f"Baseline's numbers here come from `vectordb_comparison/summary.json` "
        f"(reused from the same {FINAL_CONFIG_NAME} run via --baseline-from, so "
        f"they should already agree with the ablation section below).\n"
    )

    exp7_path = ablation_dir / FINAL_CONFIG_NAME / "benchmark_summary.json"
    vdb_path = vectordb_dir / "summary.json"
    if not exp7_path.exists() or not vdb_path.exists():
        lines.append(
            f"*(Skipped - missing {exp7_path if not exp7_path.exists() else vdb_path})*\n"
        )
        return lines

    exp7_rows = json.loads(exp7_path.read_text(encoding="utf-8"))
    vdb_rows = json.loads(vdb_path.read_text(encoding="utf-8"))

    for bucket in BUCKET_ORDER:
        exp7_row = _row_for_bucket(exp7_rows, bucket)
        vdb_row = _row_for_bucket(vdb_rows, bucket)
        if not exp7_row and not vdb_row:
            continue
        n = exp7_row.get("n_questions") or vdb_row.get("n_questions")
        lines.append(f"### {BUCKET_LABELS[bucket]} (n={n})\n")

        for category_title, metrics in CATEGORIES:
            lines.append(f"#### {category_title}\n")
            header = "| Pipeline | " + " | ".join(label for label, _, _ in metrics) + " |"
            sep = "|---|" + "|".join("---" for _ in metrics) + "|"
            lines.append(header)
            lines.append(sep)

            # GraphRAG only exists in the ablation source.
            cells = [PIPELINE_LABELS["graphrag"]]
            for _, field, kind in metrics:
                cells.append(_fmt(exp7_row.get(f"graphrag_{field}"), kind))
            lines.append("| " + " | ".join(cells) + " |")

            # Baseline and vectordb both come from the vectordb_comparison
            # source, which uses its own field-name schema for a couple
            # of metrics (see VECTORDB_FIELD_ALIASES).
            for pipeline in ("baseline", "vectordb"):
                cells = [PIPELINE_LABELS[pipeline]]
                for _, field, kind in metrics:
                    actual_field = VECTORDB_FIELD_ALIASES.get(field, field)
                    cells.append(_fmt(vdb_row.get(f"{pipeline}_{actual_field}"), kind))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    return lines


def build_ablation_section(ablation_dir: Path) -> list:
    lines = ["## Ablation Ladder (EXP1 -> EXP7)\n"]
    lines.append(
        "Experiments as rows, metrics as columns - read down a column to see "
        "how each metric changes as each component gets added.\n"
    )

    all_rows = {}
    descriptions = {}
    for exp in EXP_ORDER:
        summary_path = ablation_dir / exp / "benchmark_summary.json"
        if not summary_path.exists():
            continue
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in rows:
            all_rows[(exp, row.get("question_type"))] = row
        cfg_path = ablation_dir / exp / "run_config.json"
        if cfg_path.exists():
            descriptions[exp] = json.loads(cfg_path.read_text(encoding="utf-8")).get("description", "")

    lines.append("### Experiment key\n")
    lines.append("| Experiment | Description |")
    lines.append("|---|---|")
    for exp in EXP_ORDER:
        if exp in descriptions:
            lines.append(f"| {EXP_SHORT_LABELS.get(exp, exp)} | {descriptions[exp]} |")
    lines.append("")

    for bucket in BUCKET_ORDER:
        n = None
        for exp in EXP_ORDER:
            row = all_rows.get((exp, bucket))
            if row:
                n = row.get("n_questions")
                break
        if n is None:
            continue
        lines.append(f"### {BUCKET_LABELS[bucket]} (n={n})\n")

        for category_title, metrics in CATEGORIES:
            for pipeline in ("graphrag", "baseline"):
                pipeline_label = "GraphRAG" if pipeline == "graphrag" else "Baseline"
                lines.append(f"#### {category_title} — {pipeline_label}\n")
                header = "| Experiment | " + " | ".join(label for label, _, _ in metrics) + " |"
                sep = "|---|" + "|".join("---" for _ in metrics) + "|"
                lines.append(header)
                lines.append(sep)
                for exp in EXP_ORDER:
                    row = all_rows.get((exp, bucket))
                    if not row:
                        continue
                    cells = [EXP_SHORT_LABELS.get(exp, exp)]
                    for _, field, kind in metrics:
                        cells.append(_fmt(row.get(f"{pipeline}_{field}"), kind))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default="ablation")
    parser.add_argument("--vectordb-dir", default="vectordb_comparison")
    parser.add_argument("--out", default="FINAL_REPORT.md")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    vectordb_dir = Path(args.vectordb_dir)

    lines = ["# Final Combined Report\n"]
    lines.append(
        "Two sections: the final 3-way comparison (GraphRAG vs baseline vs "
        "the new vector-DB baseline), then the full EXP1-7 ablation ladder "
        "that produced the GraphRAG config used in that comparison.\n"
    )
    lines.extend(build_three_way_section(ablation_dir, vectordb_dir))
    lines.extend(build_ablation_section(ablation_dir))

    out_path = Path(args.out)
    content = "\n".join(lines)
    out_path.write_text(content, encoding="utf-8")
    print(f"Saved: {out_path}  ({content.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
