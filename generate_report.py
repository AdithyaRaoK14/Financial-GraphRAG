"""
generate_report.py
====================
WHAT THIS FILE DOES:
Reads every completed ablation experiment's benchmark_summary.json (each
already has separate pdf/audio/pdf+audio/ALL rows — nothing new to
compute, this just presents what's already there properly) and prints a
report per experiment, in the pdf/audio/pdf+audio/ALL breakdown kept
genuinely SEPARATE rather than collapsed into one blended row, with the
full metric set (BERTScore, ROUGE, BLEU, METEOR, LLM judge, tolerance
match, abstain rate, multivalue completeness — everything, not the
smaller subset run_ablation.py's own COMPARISON.csv shows) for both
GraphRAG and baseline side by side.

Also writes one long-format master CSV across every experiment x bucket
x metric, so you can pivot/filter however you want in Excel without
re-running anything — one row per (experiment, bucket), all metrics as
columns.

USAGE:
    python generate_report.py                       # all experiments in ablation/
    python generate_report.py --only exp7_plus_hyde_final
    python generate_report.py --ablation-dir ablation

Output:
    ablation/REPORT.txt         the full formatted report (also printed)
    ablation/REPORT_MASTER.csv  long-format, every experiment x bucket x metric
"""

import argparse
import csv
import json
from pathlib import Path

ABLATION_DIR_DEFAULT = "ablation"

BUCKET_ORDER = ["pdf", "audio", "pdf+audio", "ALL"]

# (report label, graphrag field, baseline field, format) - format is
# "pct" (0-1 -> XX.XX%), "num" (plain number, 4 decimals), "sec"
# (seconds, 2 decimals), "int" (plain integer), or "qps" (4 decimals).
# Grouped exactly the way the reference report's sections are, with our
# actual field names substituted in - nothing here is invented, every
# one of these exists in benchmark_summary.json as confirmed above.
PRIMARY_METRICS = [
    ("LLM Judge (factual correctness)", "judge_correct", "pct"),
    ("LLM Judge (faithfulness)", "judge_faithful", "pct"),
    ("Numeric Exact Match", "numeric_exact", "pct"),
    ("Numeric Match (partial credit)", "numeric_match", "pct"),
    ("Numeric F1", "numeric_f1", "pct"),
    ("Numeric Precision", "numeric_precision", "pct"),
    ("Tolerance Match (1% rel.)", "tolerance_match", "pct"),
    ("BERTScore F1 (semantic sim)", "bertscore_f1", "pct"),
]
SECONDARY_METRICS = [
    ("METEOR", "meteor", "num"),
    ("ROUGE-1", "rouge1", "num"),
    ("ROUGE-2", "rouge2", "num"),
    ("ROUGE-L", "rougeL", "num"),
    ("BLEU", "bleu", "num"),
]
RETRIEVAL_METRICS = [
    ("Hit@K (strict chunk-id)", "hit_at_k", "pct"),
    ("Recall@K (strict chunk-id)", "recall_at_k", "pct"),
    ("Source Hit@K (doc+page/timestamp)", "source_hit_at_k", "pct"),
    ("Source Recall@K", "source_recall_at_k", "pct"),
    ("MRR", "mrr", "num"),
    ("Context Precision", "context_precision", "num"),
    ("Source Type Accuracy", "source_type_acc", "pct"),
    ("Evidence Sufficiency", "evidence_sufficiency", "pct"),
    ("Factual Consistency", "factual_consistency", "pct"),
    ("Fin. Term Precision", "fin_term_precision", "pct"),
    ("Avg Cosine Sim (embedding)", "semantic_sim", "pct"),
    ("Avg FastRP Score", "avg_fastrp_score", "num"),
    ("Avg BM25 Score", "avg_bm25_score", "num"),
]
BEHAVIOR_METRICS = [
    ("Abstain Rate", "abstained", "pct"),
    ("Multi-value Completeness", "multivalue_complete", "pct"),
    ("Hallucinated Numbers (avg)", "avg_hallucinated_numbers", "num"),
]
EFFICIENCY_METRICS = [
    ("Avg Retrieval Latency", "retrieval_latency", "sec"),
    ("Avg E2E Latency", "avg_latency", "sec"),
    ("Throughput", "throughput_qps", "qps"),
]


def _fmt(value, kind):
    if value is None:
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
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.4f}"


def _row_for_bucket(summary_rows: list, bucket: str) -> dict:
    for row in summary_rows:
        if row.get("question_type") == bucket:
            return row
    return {}


def print_section(title: str, metrics: list, row: dict, indent: str = "  "):
    print(f"{indent}── {title} " + "─" * max(1, 58 - len(title)))
    for label, field, kind in metrics:
        gv = row.get(f"graphrag_{field}")
        bv = row.get(f"baseline_{field}")
        print(
            f"{indent}  {label:<34} graphrag: {_fmt(gv, kind):>10}   "
            f"baseline: {_fmt(bv, kind):>10}"
        )


def _backfill_baseline_throughput(summary_rows: list) -> list:
    """P1 fix: benchmark_summary.json files from runs before benchmark.py's
    matching fix never computed baseline_throughput_qps at all - it's
    simply absent, not zero. Confirmed: 0/28 rows populated across a real
    completed ablation. This backfills it here from data already present
    (baseline_avg_latency), the same formula benchmark.py's own fix uses,
    so an already-completed run doesn't need to be re-run just for one
    cheap, fully-derivable number. Only fills it when genuinely missing -
    a value already present (from a run using the fixed benchmark.py)
    is left untouched, not recomputed and potentially overwritten."""
    for row in summary_rows:
        if row.get("baseline_throughput_qps") in (None, ""):
            lat = row.get("baseline_avg_latency")
            row["baseline_throughput_qps"] = (1.0 / lat) if lat else None
    return summary_rows


def print_experiment_report(exp_name: str, description: str, summary_rows: list):
    print("=" * 78)
    print(f"  {exp_name}")
    if description:
        print(f"  {description}")
    print("=" * 78)

    for bucket in BUCKET_ORDER:
        row = _row_for_bucket(summary_rows, bucket)
        if not row:
            continue
        label = "COMBINED (ALL)" if bucket == "ALL" else bucket.upper()
        n = row.get("n_questions", "?")
        index_size = row.get("index_size")
        print()
        print(f"  ── SECTION: {label}  (n={n} questions) " + "─" * 20)
        print_section("PRIMARY METRICS (paraphrase-robust)", PRIMARY_METRICS, row)
        print_section("SECONDARY METRICS (token overlap)", SECONDARY_METRICS, row)
        print_section("RETRIEVAL QUALITY", RETRIEVAL_METRICS, row)
        print_section("ANSWER BEHAVIOR", BEHAVIOR_METRICS, row)
        print_section("SYSTEM EFFICIENCY", EFFICIENCY_METRICS, row)
        if index_size is not None:
            print(f"  Index Size: {index_size}")
    print()


def build_master_csv(all_experiments: list, out_path: Path):
    """One row per (experiment, bucket), every metric as a column - the
    long format Excel/pandas pivots easiest from, without needing to
    re-run anything or re-derive numbers already computed."""
    all_metric_defs = (
        PRIMARY_METRICS + SECONDARY_METRICS + RETRIEVAL_METRICS
        + BEHAVIOR_METRICS + EFFICIENCY_METRICS
    )
    fieldnames = ["experiment", "description", "bucket", "n_questions", "index_size"]
    for label, field, _ in all_metric_defs:
        fieldnames.append(f"graphrag_{field}")
        fieldnames.append(f"baseline_{field}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp_name, description, summary_rows in all_experiments:
            for bucket in BUCKET_ORDER:
                row = _row_for_bucket(summary_rows, bucket)
                if not row:
                    continue
                out = {
                    "experiment": exp_name,
                    "description": description,
                    "bucket": bucket,
                    "n_questions": row.get("n_questions"),
                    "index_size": row.get("index_size"),
                }
                for label, field, _ in all_metric_defs:
                    out[f"graphrag_{field}"] = row.get(f"graphrag_{field}")
                    out[f"baseline_{field}"] = row.get(f"baseline_{field}")
                writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default=ABLATION_DIR_DEFAULT)
    parser.add_argument("--only", nargs="+", help="Only these experiment names")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    exp_dirs = sorted(
        p for p in ablation_dir.iterdir()
        if p.is_dir() and (p / "benchmark_summary.json").exists()
    )
    if args.only:
        exp_dirs = [p for p in exp_dirs if p.name in args.only]

    if not exp_dirs:
        print(f"No completed experiments found under {ablation_dir}/")
        return

    all_experiments = []
    report_lines = []
    import io, sys

    for exp_dir in exp_dirs:
        summary_rows = json.loads((exp_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
        summary_rows = _backfill_baseline_throughput(summary_rows)
        description = ""
        cfg_path = exp_dir / "run_config.json"
        if cfg_path.exists():
            description = json.loads(cfg_path.read_text(encoding="utf-8")).get("description", "")

        # Capture this experiment's printed report into the saved .txt too,
        # not just stdout, without changing what gets printed live.
        buf = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, buf)
        print_experiment_report(exp_dir.name, description, summary_rows)
        sys.stdout = real_stdout
        report_lines.append(buf.getvalue())

        all_experiments.append((exp_dir.name, description, summary_rows))

    report_path = ablation_dir / "REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    csv_path = ablation_dir / "REPORT_MASTER.csv"
    build_master_csv(all_experiments, csv_path)

    print("=" * 78)
    print(f"Saved: {report_path}")
    print(f"Saved: {csv_path}  (one row per experiment x bucket, every metric as a column)")


class _Tee:
    """Writes to two streams at once - lets print_experiment_report's
    existing print() calls go to both the live terminal and the saved
    report file without duplicating the function or buffering silently."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    main()
