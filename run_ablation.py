"""
run_ablation.py
================
Runs a defined set of retrieval-configuration ablations back to back,
archiving every run's full output into its own folder, then builds one
comparison table across all of them.

Usage (from your project directory):
    python run_ablation.py                 # run everything not yet done
    python run_ablation.py --list          # show configs, run nothing
    python run_ablation.py --only exp1_dense  # run one config by name
    python run_ablation.py --force         # re-run configs already done
    python run_ablation.py --compare-only  # just rebuild COMPARISON from
                                           # whatever is already archived

Output layout:
    ablation/
        dense/
            benchmark_detailed.csv
            benchmark_detailed.json
            benchmark_summary.csv
            benchmark_summary.json
            benchmark_results.xlsx
            run_config.json      <- exact env overrides used
            run_log.txt          <- full stdout/stderr of the run
        dense_bm25/
            ...
        COMPARISON.csv           <- one row per config, all key metrics
        COMPARISON.md            <- same, readable table for your writeup

WHY SUBPROCESSES: each config runs as a fresh `python benchmark.py`
process. retrieval.py caches chunks/embeddings/BM25 in module-level
globals, so running configs in-process would leak state between them and
silently invalidate the comparison. A new process per config guarantees
each one is measured cleanly.

NOTE ON FASTRP: this comment used to say nothing reads fastrp_embedding
back out of Neo4j and FastRP configs would be meaningless. That's stale
- retrieval.py's hybrid scoring (rescore/expand modes) actively reads
and blends FastRP scores now, confirmed by graphrag_avg_fastrp_score
populating with real, varying values across every benchmark run in this
project. EXP3 below (+ FastRP Graph Expansion) is a real, meaningful
experiment on the current codebase - the _preflight_fastrp() check in
main() still verifies live Neo4j data before running it, in case the
graph hasn't had fast_rp.py run against it in this specific environment.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
ABLATION_DIR = PROJECT_ROOT / "ablation"

# Files benchmark.py produces that we archive per run.
RESULT_FILES = [
    "benchmark_detailed.csv",
    "benchmark_detailed.json",
    "benchmark_summary.csv",
    "benchmark_summary.json",
    "benchmark_results.xlsx",
]

# Each config: env overrides applied on top of your normal .env.
# "description" is carried into COMPARISON so the table is self-explaining.
# Each config: env overrides applied on top of your normal .env.
# "description" is carried into COMPARISON so the table is self-explaining.
#
# EXP1-7: the final incremental-component ablation ladder. Each stage
# adds exactly one capability on top of the previous one, isolating that
# capability's own contribution:
#   EXP1 Dense                                          -> EXP7 is the
#   EXP2 + BM25                                             actual
#   EXP3 + FastRP Graph Expansion                           production
#   EXP4 + Graph Facts                                      pipeline -
#   EXP5 + Temporal Graph                                   EXP1-6 are
#   EXP6 + Question-Type Routing                            deliberate
#   EXP7 + HyDE (audio-routed retrieval)                    strip-downs
#                                                            of it.
#
# EXP1-5 force FORCE_SOURCE_FILTER=both and FORCE_TEMPORAL_FACTS=false so
# neither Question-Type Routing's nor Temporal Graph's effect leaks into
# an earlier stage — EXP5 forces temporal facts ON for every question
# (not routing-gated) for the same reason, so "what does Temporal Graph
# add" is measured cleanly rather than only on whichever subset routing
# happened to already flag as a comparison. EXP6 removes both forces,
# switching to router.route_question()'s real per-question decision for
# both source_filter AND whether a question needs temporal facts. USE_
# RERANKER=false is set explicitly in every config — it's already off by
# default (measured net-harmful, see config.py's USE_RERANKER comment),
# but pinning it here guards against a leftover USE_RERANKER=true in
# your shell environment from earlier testing silently changing results
# (the exact class of bug BENCHMARK_MODALITY stuck around from earlier
# in this project) - explicit is safer than reordering "reranker" as
# one more implicit conclusion.
#
# BENCHMARK_LLM_JUDGE=true is set explicitly in every config too - real
# per-question LLM-judge scoring (both faithful AND correct, both
# pipelines) is on for the whole ablation. This roughly doubles each
# experiment's LLM-call cost on top of everything else (one extra judge
# call per pipeline per question), so expect noticeably longer per-
# experiment runtime than the string-metric-only numbers you've seen in
# earlier rounds of this project.
CONFIGS = [
    {
        "name": "exp1_dense",
        "description": "EXP1: Dense embeddings only",
        "env": {
            "BM25_WEIGHT": "0.0", "USE_FASTRP": "false",
            "USE_GRAPH_FACTS": "false", "FORCE_TEMPORAL_FACTS": "false",
            "FORCE_SOURCE_FILTER": "both", "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
    {
        "name": "exp2_dense_bm25",
        "description": "EXP2: + BM25 (lexical/keyword retrieval)",
        "env": {
            "USE_FASTRP": "false",
            "USE_GRAPH_FACTS": "false", "FORCE_TEMPORAL_FACTS": "false",
            "FORCE_SOURCE_FILTER": "both", "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
    {
        "name": "exp3_dense_bm25_fastrp",
        "description": "EXP3: + FastRP Graph Expansion (graph-based candidate expansion)",
        "env": {
            "USE_FASTRP": "true",
            "USE_GRAPH_FACTS": "false", "FORCE_TEMPORAL_FACTS": "false",
            "FORCE_SOURCE_FILTER": "both", "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
    {
        "name": "exp4_plus_graph_facts",
        "description": "EXP4: + Graph Facts (structured graph-derived evidence to the LLM)",
        "env": {
            "USE_FASTRP": "true",
            "USE_GRAPH_FACTS": "true", "FORCE_TEMPORAL_FACTS": "false",
            "FORCE_SOURCE_FILTER": "both", "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
    {
        "name": "exp5_plus_temporal_graph",
        "description": "EXP5: + Temporal Graph (NEXT_VALUE/PREVIOUS_VALUE/UPDATED_TO facts, forced on for every question)",
        "env": {
            "USE_FASTRP": "true",
            "USE_GRAPH_FACTS": "true", "FORCE_TEMPORAL_FACTS": "true",
            "FORCE_SOURCE_FILTER": "both", "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
    {
        "name": "exp6_plus_routing",
        "description": "EXP6: + Question-Type Routing (real per-question pdf/audio/both + temporal-comparison detection)",
        "env": {
            "USE_FASTRP": "true",
            "USE_GRAPH_FACTS": "true",
            "USE_HYDE_FOR_AUDIO": "false",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
            # FORCE_TEMPORAL_FACTS and FORCE_SOURCE_FILTER intentionally
            # absent here - routing now makes both decisions for real.
        },
    },
    {
        "name": "exp7_plus_hyde_final",
        "description": "EXP7 (FINAL/production config): + HyDE for audio-routed retrieval",
        "env": {
            "USE_FASTRP": "true",
            "USE_GRAPH_FACTS": "true",
            "USE_HYDE_FOR_AUDIO": "true",
            "USE_RERANKER": "false", "BENCHMARK_LLM_JUDGE": "true",
        },
    },
]

# Metrics pulled from each run's ALL-questions summary row into COMPARISON.
COMPARE_METRICS = [
    "n_questions",
    "graphrag_numeric_match",
    "graphrag_numeric_precision",
    "graphrag_numeric_f1",
    "graphrag_numeric_exact",
    "graphrag_semantic_sim",
    "graphrag_hit_at_k",
    "graphrag_recall_at_k",
    "graphrag_avg_fastrp_score",
    "graphrag_avg_hallucinated_numbers",
    "graphrag_avg_facts_used",
    "graphrag_avg_latency",
    "graphrag_throughput_qps",
    # extended metrics now produced inline by benchmark.py
    "graphrag_numeric_match_partial",
    "graphrag_bertscore_f1",
    "graphrag_rougeL",
    "graphrag_bleu",
    "graphrag_meteor",
    "graphrag_mrr",
    "graphrag_context_precision",
    "graphrag_source_type_acc",
    "graphrag_judge_correct",
    "graphrag_judge_faithful",
    "graphrag_fin_term_precision",
    "graphrag_factual_consistency",
    "graphrag_retrieval_latency",
    "index_size",
    # P1 fix: newer metrics from benchmark.py's fixes this project - not
    # previously in this comparison, worth having for a final ablation
    # writeup since they've each been shown to catch something numeric_
    # exact alone misses (see tolerance_match/abstained/multivalue_
    # complete/source_hit_at_k's docstrings in benchmark.py for what
    # each one adds).
    "graphrag_tolerance_match", "baseline_tolerance_match",
    "graphrag_abstained", "baseline_abstained",
    "graphrag_multivalue_complete", "baseline_multivalue_complete",
    "graphrag_source_hit_at_k", "baseline_source_hit_at_k",
    "graphrag_source_recall_at_k", "baseline_source_recall_at_k",
    "baseline_numeric_match_partial",
    "baseline_fin_term_precision",
    "baseline_factual_consistency",
    "baseline_retrieval_latency",
    "baseline_bertscore_f1",
    "baseline_rougeL",
    "baseline_numeric_f1",
    "baseline_numeric_exact",
    "baseline_avg_latency",
]


def config_dir(name):
    return ABLATION_DIR / name


def is_done(name):
    """A config counts as done if its summary landed — that's the file
    COMPARISON is built from."""
    return (config_dir(name) / "benchmark_summary.json").exists()


def run_one(cfg, force=False):
    name = cfg["name"]
    out_dir = config_dir(name)

    if is_done(name) and not force:
        print(f"[skip] {name} — already has results (use --force to re-run)")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(cfg["env"])
    # Unbuffered child stdout — otherwise benchmark.py's per-question
    # prints arrive in large chunks (or only at process exit) when piped,
    # making a live run look frozen for minutes at a time.
    env["PYTHONUNBUFFERED"] = "1"

    # Mid-config resume. benchmark.py saves results/benchmark_detailed.csv
    # after EVERY question, so an interrupted config can pick up where it
    # stopped instead of redoing hours of work. The catch is that
    # results/ is shared by every config, so resuming blindly would let
    # one config inherit another's rows and silently corrupt the
    # comparison. This restores THIS config's own partial rows into
    # results/ first, then enables resume — so a config only ever resumes
    # from itself.
    partial = out_dir / "benchmark_detailed.csv"
    resuming = False
    if partial.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(partial, RESULTS_DIR / "benchmark_detailed.csv")
        partial_json = out_dir / "benchmark_detailed.json"
        if partial_json.exists():
            shutil.copy2(partial_json, RESULTS_DIR / "benchmark_detailed.json")
        env["BENCHMARK_RESUME"] = "true"
        resuming = True
        try:
            import pandas as _pd
            n_done = len(_pd.read_csv(partial))
            print(f"  [resume] {n_done} question(s) already completed for "
                  f"this config — continuing from there.")
        except Exception:
            print("  [resume] partial results found — continuing from there.")
    else:
        # Fresh config: make sure it can't inherit the PREVIOUS config's
        # rows out of the shared results/ directory.
        env["BENCHMARK_RESUME"] = "false"
        stale = RESULTS_DIR / "benchmark_detailed.csv"
        if stale.exists():
            stale.unlink()

    print(f"\n{'=' * 70}")
    print(f"RUNNING: {name}")
    print(f"  {cfg['description']}")
    print(f"  env overrides: {cfg['env']}")
    print(f"{'=' * 70}\n")

    started = datetime.now(timezone.utc)
    t0 = time.time()

    log_path = out_dir / "run_log.txt"
    mode = "a" if resuming else "w"
    interrupted = False
    proc = None
    try:
        with open(log_path, mode, encoding="utf-8") as log:
            log.write(f"config: {name}\n{cfg['description']}\n")
            log.write(f"env overrides: {json.dumps(cfg['env'])}\n")
            log.write(f"started: {started.isoformat()}"
                      f"{' (resumed)' if resuming else ''}\n{'=' * 60}\n\n")
            log.flush()

            proc = subprocess.Popen(
                [sys.executable, "benchmark.py"],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                # Mirror to console so you can watch progress, and to the
                # log so the full run is preserved per config. Both are
                # flushed per line: without the flush the log sits in
                # Python's buffer until the config ends, which makes
                # tailing it live show nothing but the header.
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        # Ctrl+C reaches the whole process group, so benchmark.py may die
        # before the parent's handler runs — in which case the loop ends
        # normally with a non-zero returncode rather than raising. Either
        # way, archive whatever benchmark.py managed to save (it writes
        # after every question) so the next invocation resumes instead of
        # redoing hours of work. Doing this in `finally` rather than in
        # the except branch covers both shapes of interruption.
        killed = proc is not None and proc.returncode not in (0, None)
        if interrupted or killed:
            interrupted = True
            saved = 0
            for fname in RESULT_FILES:
                src = RESULTS_DIR / fname
                if src.exists():
                    shutil.copy2(src, out_dir / fname)
                    saved += 1
            # The summary file is what marks a config "done" — remove it
            # so an interrupted config isn't mistaken for a complete one.
            (out_dir / "benchmark_summary.json").unlink(missing_ok=True)
            print(f"\n[interrupted] {name} — partial progress archived "
                  f"({saved} file(s)). Re-run to continue from here.")

    if interrupted:
        raise KeyboardInterrupt

    elapsed = time.time() - t0
    ok = proc.returncode == 0

    if not ok:
        print(f"\n[FAILED] {name} exited with code {proc.returncode} "
              f"— see {log_path}")
        # Leave the folder in place with its log so the failure is
        # inspectable, but don't archive partial result files as if the
        # run succeeded.
        return False

    copied = []
    for fname in RESULT_FILES:
        src = RESULTS_DIR / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            copied.append(fname)

    meta = {
        "name": name,
        "description": cfg["description"],
        "env_overrides": cfg["env"],
        "started_utc": started.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_human": f"{elapsed / 60:.1f} min",
        "returncode": proc.returncode,
        "files_archived": copied,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[done] {name} in {elapsed / 60:.1f} min "
          f"— archived {len(copied)} files to {out_dir}")
    return True


def _all_row(summary):
    """benchmark.py's summary is a list of per-question_type rows plus an
    'ALL' row; the ALL row is what COMPARISON reports."""
    for row in summary:
        if row.get("question_type") == "ALL":
            return row
    return summary[-1] if summary else {}


def build_comparison():
    rows = []
    for cfg in CONFIGS:
        name = cfg["name"]
        summary_path = config_dir(name) / "benchmark_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        all_row = _all_row(summary)

        meta_path = config_dir(name) / "run_config.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )

        row = {
            "config": name,
            "description": cfg["description"],
            "env_overrides": json.dumps(cfg["env"]),
            "run_minutes": meta.get("elapsed_human", ""),
        }
        for m in COMPARE_METRICS:
            row[m] = all_row.get(m)
        rows.append(row)

    if not rows:
        print("No completed runs found under ablation/ — nothing to compare.")
        return

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = ABLATION_DIR / "COMPARISON.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}")
    except ImportError:
        csv_path = ABLATION_DIR / "COMPARISON.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            headers = list(rows[0].keys())
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(f'"{r.get(h, "")}"' for h in headers) + "\n")
        print(f"\nWrote {csv_path} (pandas unavailable — plain CSV)")

    # Markdown table — the headline numbers only, for pasting into a report.
    headline = [
        ("config", "Config"),
        ("graphrag_numeric_f1", "GraphRAG F1"),
        ("graphrag_numeric_exact", "Exact"),
        ("graphrag_numeric_precision", "Precision"),
        ("graphrag_numeric_match", "Recall"),
        ("graphrag_hit_at_k", "Hit@K"),
        ("graphrag_avg_hallucinated_numbers", "Halluc."),
        ("graphrag_avg_latency", "Latency(s)"),
    ]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        "# Ablation comparison",
        "",
        "All figures are the ALL-questions row from each run's summary.",
        "",
        "| " + " | ".join(label for _, label in headline) + " |",
        "|" + "|".join(["---"] * len(headline)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k)) for k, _ in headline) + " |")

    lines += ["", "## Configurations", ""]
    for cfg in CONFIGS:
        if any(r["config"] == cfg["name"] for r in rows):
            lines.append(f"- **{cfg['name']}** — {cfg['description']} "
                         f"(`{json.dumps(cfg['env'])}`)")

    md_path = ABLATION_DIR / "COMPARISON.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(lines[4 : 6 + len(rows)]))


def _preflight_env_knobs():
    """Verify every env var the configs rely on actually reaches config.py.

    A knob that config.py hardcodes instead of reading from the
    environment produces ablation rows that are byte-identical to the
    default — which reads as "this component does nothing" when it really
    means "this switch was never connected". That happened for real:
    BM25_WEIGHT was a hardcoded literal, so a whole 10-config run
    contained two duplicated pairs. This catches that class of mistake in
    seconds instead of after hours of runtime.
    """
    import subprocess as sp

    knobs = {}
    for cfg in CONFIGS:
        for k, v in cfg["env"].items():
            knobs.setdefault(k, set()).add(v)

    dead = []
    for key, values in sorted(knobs.items()):
        probe = sorted(values)[0]
        code = (
            "import config, os, sys;"
            f"v=getattr(config,{key!r},'<MISSING>');"
            "sys.stdout.write(str(v))"
        )
        env = os.environ.copy()
        env[key] = probe
        try:
            got = sp.run(
                [sys.executable, "-c", code],
                cwd=PROJECT_ROOT, env=env, capture_output=True,
                text=True, timeout=120,
            ).stdout.strip()
        except Exception as e:
            print(f"[preflight] could not probe {key}: {e}")
            continue

        # Compare loosely: "0.0"/"0.0", "false"/"False", "5"/"5".
        norm_probe = probe.strip().lower()
        norm_got = got.strip().lower()
        matched = (
            norm_got == norm_probe
            or (norm_probe in ("true", "false") and norm_got == norm_probe)
        )
        if not matched:
            try:
                matched = float(norm_got) == float(norm_probe)
            except (TypeError, ValueError):
                pass
        if not matched:
            dead.append((key, probe, got))

    if dead:
        print("\n" + "!" * 70)
        print("WARNING: these env vars do NOT reach config.py —")
        print("configs that only differ by them will produce IDENTICAL results:")
        for key, probe, got in dead:
            print(f"  {key}={probe}  ->  config.{key} == {got}")
        print("\nFix config.py to read them via os.getenv() before running,")
        print("or those ablation rows will be meaningless duplicates.")
        print("!" * 70 + "\n")
        try:
            if input("Continue anyway? [y/N] ").strip().lower() != "y":
                sys.exit(0)
        except EOFError:
            print("(non-interactive; continuing)")
    else:
        print(f"[preflight] all {len(knobs)} env knobs reach config.py correctly.")


def _preflight_fastrp():
    """Warn loudly BEFORE a multi-hour run if the FastRP configs can't
    actually differ from the non-FastRP ones. fast_rp.py writes
    fastrp_embedding onto Neo4j nodes; if it was never run, retrieval.py
    degrades FastRP to a no-op and the four FastRP rows below would come
    out identical to their non-FastRP counterparts — a table that looks
    like 'FastRP does nothing' when it actually means 'FastRP has no
    data'. Non-fatal: you may legitimately want the non-FastRP rows
    anyway."""
    try:
        import config as _cfg
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            _cfg.NEO4J_URI, auth=(_cfg.NEO4J_USER, _cfg.NEO4J_PASSWORD)
        )
        with driver.session(database=_cfg.NEO4J_DATABASE) as s:
            n = s.run(
                "MATCH (n) WHERE n.fastrp_embedding IS NOT NULL "
                "RETURN count(n) AS c"
            ).single()["c"]
        driver.close()
    except Exception as e:
        print(f"[preflight] Could not check FastRP data ({e}). Continuing.")
        return

    if n == 0:
        print("\n" + "!" * 70)
        print("WARNING: no node in Neo4j has a fastrp_embedding property.")
        print("The four FastRP configs will produce results IDENTICAL to")
        print("their non-FastRP counterparts, because retrieval.py treats")
        print("missing FastRP data as a no-op.")
        print("")
        print("Run this first, then re-run the ablation:")
        print("    python fast_rp.py")
        print("!" * 70 + "\n")
        try:
            if input("Continue anyway? [y/N] ").strip().lower() != "y":
                sys.exit(0)
        except EOFError:
            print("(non-interactive; continuing)")
    else:
        print(f"[preflight] FastRP data found on {n} nodes — FastRP configs "
              f"will be meaningful.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list configs, run nothing")
    ap.add_argument("--only", metavar="NAME", help="run just this config")
    ap.add_argument("--force", action="store_true", help="re-run completed configs")
    ap.add_argument("--compare-only", action="store_true",
                    help="rebuild COMPARISON from existing results")
    args = ap.parse_args()

    if args.list:
        print(f"{len(CONFIGS)} configs:\n")
        for c in CONFIGS:
            status = "DONE" if is_done(c["name"]) else "pending"
            print(f"  [{status:7}] {c['name']:26} {c['description']}")
            print(f"{'':12}env: {c['env']}")
        return

    if args.compare_only:
        build_comparison()
        return

    if not (PROJECT_ROOT / "benchmark.py").exists():
        print("benchmark.py not found — run this from your project directory.")
        sys.exit(1)

    gt = RESULTS_DIR / "ground_truth.json"
    if not gt.exists():
        print(f"{gt} not found — benchmark.py reads its questions from there.")
        sys.exit(1)

    todo = CONFIGS
    if args.only:
        todo = [c for c in CONFIGS if c["name"] == args.only]
        if not todo:
            print(f"No config named {args.only!r}. Use --list to see names.")
            sys.exit(1)

    if any(c["env"].get("USE_FASTRP") == "true" for c in todo):
        _preflight_fastrp()

    _preflight_env_knobs()

    pending = [c for c in todo if args.force or not is_done(c["name"])]
    print(f"{len(pending)} config(s) to run "
          f"({len(todo) - len(pending)} already done).")
    if pending:
        print("This takes roughly as long as one benchmark run per config.\n")

    failures = []
    try:
        for cfg in todo:
            if not run_one(cfg, force=args.force):
                failures.append(cfg["name"])
    except KeyboardInterrupt:
        print("\nStopped. Partial progress was archived — re-run the same "
              "command to continue from where it left off.")
        build_comparison()
        sys.exit(130)

    build_comparison()

    if failures:
        print(f"\n{len(failures)} config(s) FAILED: {', '.join(failures)}")
        print("Their run_log.txt has the details. Re-run with --only <name>.")
        sys.exit(1)


if __name__ == "__main__":
    main()
