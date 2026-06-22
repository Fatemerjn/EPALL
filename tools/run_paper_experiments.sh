#!/usr/bin/env bash
# Main paper comparisons (PALL-Original vs PALL-Modified{grad,conflict} and the
# PALL-Adapter family) on fixed, reproducible schedules. Launch under tmux on the
# GPU server. CUDA is selected automatically by main.py when available.
#
# Usage:
#   bash tools/run_paper_experiments.sh            # both datasets
#   bash tools/run_paper_experiments.sh cifar10    # one dataset
#   SEEDS="0 1 2" bash tools/run_paper_experiments.sh cifar100
#
# Results land in runs/<dataset>/...; aggregate at the end (printed command).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1}"
WHICH="${1:-all}"
mkdir -p logs

launch () {  # launch <logname> <main.py args...>
    local name="$1"; shift
    echo ">>> ${name}"
    if $PY -u main.py "$@" > "logs/${name}.log" 2>&1; then
        echo "    PASS ${name}"
    else
        echo "    FAIL ${name} (see logs/${name}.log)"
    fi
}

run_dataset () {
    local ds="$1" sch_prefix="$2"
    local BASE AD s sch tag
    if [ "$ds" = "cifar10" ]; then
        BASE="--dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 3 --arch resnet18 --sparsity 0.8"
    else
        BASE="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9"
    fi
    BASE="$BASE --k_shot 50 --alpha 0.5 --beta 1.0 --mem_budget 500 --optim sgd \
        --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --batch_size 32 --n_epochs 3 --deterministic"
    AD="--method pall_adapter --adapter_bottleneck 16 --adapter_train_classifier \
        --retrain_steps 50 --adapter_forget_steps 10"
    tag="${ds}_main"
    for s in $SEEDS; do
        sch="schedules/${sch_prefix}_seed${s}.json"
        # ---- PALL-Modified family ----
        launch "${ds}_pall_original_s${s}" $BASE --seed $s --request_schedule_file $sch \
            --method pall_original --retrain_steps 50 --experiment_tag $tag
        launch "${ds}_pall_modified_grad_s${s}" $BASE --seed $s --request_schedule_file $sch \
            --method pall_modified --protect_importance gradient --protect_ratio 0.2 \
            --lambda_protect 1.0 --retrain_steps 50 --experiment_tag $tag
        launch "${ds}_pall_modified_conflict_s${s}" $BASE --seed $s --request_schedule_file $sch \
            --method pall_modified --protect_importance conflict --adaptive_protect \
            --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50 --experiment_tag $tag
        # ---- PALL-Adapter family ----
        launch "${ds}_adapter_reset_s${s}" $BASE --seed $s --request_schedule_file $sch $AD \
            --adapter_shared_bottleneck 0 --experiment_tag $tag
        launch "${ds}_adapter_shared_s${s}" $BASE --seed $s --request_schedule_file $sch $AD \
            --adapter_shared_bottleneck 16 --adapter_shared_forget_ratio 0.3 \
            --adapter_shared_protect_ratio 0.0 --experiment_tag $tag
        launch "${ds}_adapter_protected_s${s}" $BASE --seed $s --request_schedule_file $sch $AD \
            --adapter_shared_bottleneck 16 --adapter_shared_forget_ratio 0.3 \
            --adapter_shared_protect_ratio 0.2 --protect_importance gradient --experiment_tag $tag
        launch "${ds}_adapter_protected_conflict_s${s}" $BASE --seed $s --request_schedule_file $sch $AD \
            --adapter_shared_bottleneck 16 --adapter_shared_forget_ratio 0.3 \
            --adapter_shared_protect_ratio 0.2 --protect_importance conflict --experiment_tag $tag
    done
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "cifar10" ];  then run_dataset cifar10  cifar10_t5_f3_fixed; fi
if [ "$WHICH" = "all" ] || [ "$WHICH" = "cifar100" ]; then run_dataset cifar100 cifar100_t10_f3;    fi

echo "===================================================================="
echo "DONE. Aggregate the results with:"
echo "  $PY tools/aggregate_results.py --root runs --out results/aggregates/paper_results.csv"
