#!/usr/bin/env bash
# Server experiment runner for the reviewer-response experiments.
# Covers three groups, then aggregates tables.
#   GROUP 1: extra baselines (ewc, lwf, clpu) on fixed schedules, CIFAR-10 + CIFAR-100, seeds 0,1
#   GROUP 2: rerun PALL-Modified (gradient importance, with protection) to replace stale numbers
#   GROUP 3: PALL-Adapter bottleneck ablation on CIFAR-10 (seed 0)
#
# Usage (run on a GPU node, inside tmux):
#   bash tools/run_server_experiments.sh            # all groups
#   bash tools/run_server_experiments.sh g1         # only group 1
#   SEEDS="0 1 2" bash tools/run_server_experiments.sh
#
# main.py auto-selects CUDA when available (--device auto is the default).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1}"
WHICH="${1:-all}"
mkdir -p logs results/aggregates

# Data directory: override with DATA_DIR=/path/to/data to keep CIFAR off the SSD.
DATA_DIR="${DATA_DIR:-./data}"

# Shared hyperparameters, mirroring tools/run_paper_experiments.sh
COMMON="--data_dir ${DATA_DIR} --k_shot 50 --alpha 0.5 --beta 1.0 --mem_budget 500 --optim sgd \
        --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --batch_size 32 --n_epochs 3 --deterministic"
C10="--dataset cifar10  --class_per_task 2 --n_tasks 5  --n_forget 3 --arch resnet18 --sparsity 0.8"
C100="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9"

launch () {  # launch <logname> <main.py args...>
    local name="$1"; shift
    echo ">>> ${name}"
    if $PY -u main.py "$@" > "logs/${name}.log" 2>&1; then
        echo "    PASS ${name}"
    else
        echo "    FAIL ${name}  (see logs/${name}.log)"
    fi
}

# -------------------------------------------------------------------- GROUP 1
group1 () {
    echo "===== GROUP 1: baselines ewc / lwf / clpu ====="
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for m in ewc lwf clpu; do
            launch "c10_${m}_s${s}"  $C10  $COMMON --method $m --seed $s \
                   --request_schedule_file $c10  --experiment_tag cifar10_baselines_v2
            launch "c100_${m}_s${s}" $C100 $COMMON --method $m --seed $s \
                   --request_schedule_file $c100 --experiment_tag cifar100_baselines_v2
        done
    done
}

# -------------------------------------------------------------------- GROUP 2
group2 () {
    echo "===== GROUP 2: rerun PALL-Modified (gradient, protected) ====="
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        launch "c10_pall_modified_grad_s${s}"  $C10  $COMMON --seed $s \
               --request_schedule_file $c10  --method pall_modified \
               --protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 \
               --retrain_steps 50 --experiment_tag cifar10_main
        launch "c100_pall_modified_grad_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_modified \
               --protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 \
               --retrain_steps 50 --experiment_tag cifar100_main
    done
}

# -------------------------------------------------------------------- GROUP 3
group3 () {
    echo "===== GROUP 3: PALL-Adapter bottleneck ablation (CIFAR-10, seed 0) ====="
    local sch="schedules/cifar10_t5_f3_fixed_seed0.json"
    for b in 4 8 16 32 64 128; do
        launch "c10_adapter_bottleneck_${b}" $C10 $COMMON --seed 0 \
               --request_schedule_file $sch --method pall_adapter \
               --adapter_bottleneck $b --adapter_shared_bottleneck 16 \
               --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
               --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10 \
               --experiment_tag adapter_bottleneck_ablation_v1
    done
}

case "$WHICH" in
    all) group1; group2; group3 ;;
    g1)  group1 ;;
    g2)  group2 ;;
    g3)  group3 ;;
    *)   echo "unknown arg: $WHICH (use: all | g1 | g2 | g3)"; exit 1 ;;
esac

echo "===================================================================="
echo "AGGREGATING TABLES ..."
$PY tools/aggregate_results.py --root runs --out results/aggregates/server_results.csv || true
$PY tools/make_thesis_table.py --root runs --group-by-config \
    --out-csv results/aggregates/server_thesis_table.csv \
    --out-md  results/aggregates/server_thesis_table.md || true
$PY tools/make_report_table.py \
    --input results/aggregates/server_thesis_table.csv \
    --out-csv results/aggregates/server_report_table.csv \
    --out-md  results/aggregates/server_report_table.md || true

echo "DONE."
echo "Tables written under results/aggregates/ :"
echo "  server_thesis_table.md   (full)"
echo "  server_report_table.md   (compact)"
echo "Per-run logs are under logs/."
