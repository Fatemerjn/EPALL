#!/usr/bin/env bash
# Queue every remaining GPU group back to back, in dependency order.
#
# There is exactly one hard dependency in this queue: g30 must not run
# concurrently with anything that writes pall_modified runs under the standard
# schedules, because the resume check matches on config and would see a partial
# sibling. Everything else is independent and the order below is by value per
# GPU-hour, cheapest and most load-bearing first.
#
#   1. g30  reinit MIA               (already running; the queue waits for it)
#   2. g18  linear probe   -> 5 seeds  cheapest, feeds the honesty claim
#   3. g32  overlap-by-sparsity        turns RQ3 from a proxy trend into a
#                                      direct manipulation; ~2.5h for the single
#                                      most attackable claim in the thesis
#   4. g17  overlap curves -> 5 seeds  strengthens the proxy benchmark too
#   5. g31  conflict-gamma sweep       new evidence for the adapter path; by far
#                                      the most expensive group, and the adapter
#                                      path is already reported as a negative
#                                      result, so trim CONFLICT_GAMMA_VALUES
#                                      first if GPU time runs short
#
# Seed extensions land under the existing experiment tags, which is the same
# convention group 28 used: added seeds append rows, they do not form a new arm.
# The gamma sweep gets its own per-gamma tags and contains its own gamma=0
# control, so it never touches a reported table.
#
# Usage:
#   bash tools/run_final_queue.sh --dry-run   # validate every command, run none
#   bash tools/run_final_queue.sh             # run the whole queue
#
# Resumable: every group below is resume-safe, so re-running the queue after an
# interruption skips whatever already completed.

set -u

cd "$(dirname "$0")/.." || exit 1

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

RUNNER="tools/run_server_experiments.sh"
LOG_DIR="logs/final_queue"
mkdir -p "$LOG_DIR"

# Seed sets for the two extensions. Seeds 0 and 1 already exist for both groups,
# so the queue asks only for the new ones. The groups are resume-safe, but the
# 2026-07 runs predate several config keys and therefore do not always match the
# exact-resume check on a fresh checkout; naming only the new seeds makes the
# queue correct regardless of whether that check fires.
export PROBE_SEEDS="${PROBE_SEEDS:-2 3 4}"
export OVERLAP_CURVE_SEEDS="${OVERLAP_CURVE_SEEDS:-2 3 4}"
export OVERLAP_SPARSITY_SEEDS="${OVERLAP_SPARSITY_SEEDS:-0 1 2}"
export OVERLAP_SPARSITY_VALUES="${OVERLAP_SPARSITY_VALUES:-0.5 0.6 0.7 0.8 0.9}"
export CONFLICT_GAMMA_SEEDS="${CONFLICT_GAMMA_SEEDS:-0 1 2}"
export CONFLICT_GAMMA_VALUES="${CONFLICT_GAMMA_VALUES:-0 0.25 0.5 1 2}"

QUEUE=(
    g18_probe
    g32_overlap_sparsity
    g17_overlap_curve
    g31_conflict_gamma_sweep
)

echo "=============================================================="
echo "FINAL QUEUE${DRY_RUN:+ (dry run)}"
echo "  probe seeds          : $PROBE_SEEDS"
echo "  overlap curve seeds  : $OVERLAP_CURVE_SEEDS"
echo "  sparsity levels      : $OVERLAP_SPARSITY_VALUES"
echo "  gamma sweep seeds    : $CONFLICT_GAMMA_SEEDS"
echo "  gamma values         : $CONFLICT_GAMMA_VALUES"
echo "=============================================================="

failed=()
for group in "${QUEUE[@]}"; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="${LOG_DIR}/${group}_${stamp}.log"
    echo
    echo ">>> ${group}  ->  ${log}"
    if bash "$RUNNER" "$group" $DRY_RUN 2>&1 | tee "$log"; then
        echo "<<< ${group} finished"
    else
        echo "<<< ${group} FAILED (see ${log})" >&2
        failed+=("$group")
    fi
done

echo
echo "=============================================================="
if ((${#failed[@]})); then
    echo "FAILED GROUPS: ${failed[*]}"
    exit 1
fi
echo "Queue complete. Next, on the laptop after syncing runs/:"
cat <<'NEXT'
  python tools/aggregate_results.py --root runs --require-metrics \
      --seed-policy latest --out results/aggregates/server_results_final.csv
  python tools/analyze_anchor_paired.py --input results/aggregates/server_results_final.csv
  python tools/analyze_forgetting_persistence.py
  python tools/check_thesis_numbers.py
NEXT
echo "=============================================================="
