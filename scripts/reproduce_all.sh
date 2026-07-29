#!/usr/bin/env bash
# Rebuild reproducibility artifacts from existing runs/.
#
# This script does not launch training. It only reads run artifacts and writes
# aggregate tables, audit reports, and figures.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-./.venv/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
AGG_DIR="${AGG_DIR:-results/aggregates}"
PLOTS_DIR="${PLOTS_DIR:-results/thesis/plots_v2}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-12345}"
CANONICAL_SEEDS="${CANONICAL_SEEDS:-0 1 2 3 4}"
read -r -a CANONICAL_SEED_ARGS <<< "$CANONICAL_SEEDS"

mkdir -p "$AGG_DIR" "$PLOTS_DIR"

echo "==> Auditing seed completeness"
"$PY" tools/audit_seed_completeness.py \
  --root "$RUNS_ROOT" \
  --out-csv "$AGG_DIR/seed_completeness_report.csv" \
  --out-md "$AGG_DIR/seed_completeness_report.md"

echo "==> Aggregating per-run metrics"
"$PY" tools/aggregate_results.py \
  --root "$RUNS_ROOT" \
  --require-metrics \
  --seed-policy latest \
  --seed "${CANONICAL_SEED_ARGS[@]}" \
  --out "$AGG_DIR/server_results.csv"

cp "$AGG_DIR/server_results.csv" "$AGG_DIR/results_summary.csv"

echo "==> Building group-by-config thesis tables"
"$PY" tools/make_thesis_table.py \
  --root "$RUNS_ROOT" \
  --group-by-config \
  --seed-policy latest \
  --seed "${CANONICAL_SEED_ARGS[@]}" \
  --out-csv "$AGG_DIR/server_thesis_table.csv" \
  --out-md "$AGG_DIR/server_thesis_table.md"

echo "==> Building compact report tables"
"$PY" tools/make_report_table.py \
  --input "$AGG_DIR/server_thesis_table.csv" \
  --out-csv "$AGG_DIR/server_report_table.csv" \
  --out-md "$AGG_DIR/server_report_table.md"

echo "==> Building legacy comparison tables"
"$PY" tools/make_comparison_table.py \
  --in "$AGG_DIR/results_summary.csv" \
  --out-csv "$AGG_DIR/comparison_table.csv" \
  --out-md "$AGG_DIR/comparison_table.md"

echo "==> Building legacy ablation tables"
"$PY" tools/make_ablation_table.py \
  --in "$AGG_DIR/results_summary.csv" \
  --out-csv "$AGG_DIR/ablation_table.csv" \
  --out-md "$AGG_DIR/ablation_table.md"

echo "==> Building reviewer-grade PDF figures"
"$PY" tools/plot_report_results.py \
  --paper-figures \
  --input "$AGG_DIR/server_thesis_table.csv" \
  --runs-root "$RUNS_ROOT" \
  --outdir "$PLOTS_DIR" \
  --dpi 300 \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED"

echo "==> Building the compact AAAI figures directly from canonical aggregates"
"$PY" tools/make_aaai_figures.py \
  --outdir paper/AuthorKit27/Figures

echo "==> Building the generated main standard-comparison LaTeX table"
"$PY" tools/generate_main_standard_table.py

echo "==> Done"
echo "Tables:  $AGG_DIR"
echo "Figures: $PLOTS_DIR"
