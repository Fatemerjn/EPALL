#!/usr/bin/env bash
# Fast end-to-end smoke tests. NOT a correctness benchmark -- just confirms every
# core method pipeline runs without crashing on a tiny CIFAR-10 configuration.
# Runs land in runs/cifar10/T3_F1/<method>/seed_0/<timestamp>/ and are tagged
# experiment_tag=smoke, so aggregation excludes them from thesis tables by default.
# Usage:  bash tools/smoke_test.sh
# Requires CIFAR-10 under ./data (already present in this repo).

cd "$(dirname "$0")/.."
PY="${PYTHON:-./.venv/bin/python}"
COMMON="--dataset cifar10 --class_per_task 2 --n_tasks 3 --n_forget 1 \
        --n_epochs 1 --batch_size 32 --mem_budget 60 --k_shot 10 \
        --optim sgd --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --seed 0 \
        --arch resnet18 --sparsity 0.8 --experiment_tag smoke"

fails=0
run () {  # run <label> <extra args...>
    local label="$1"; shift
    echo "==================== ${label} ===================="
    if $PY -u main.py $COMMON "$@"; then
        echo "PASS: ${label}"
    else
        echo "FAIL: ${label}"; fails=$((fails + 1))
    fi
}

run "[1/6] baseline ER"        --method er --forget_iters 5
run "[2/6] pall_original"      --method pall_original --alpha 0.5 --beta 1.0 --retrain_steps 5
run "[3/6] pall_modified grad" --method pall_modified --alpha 0.5 --beta 1.0 \
    --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 5 --protect_importance gradient
run "[4/6] pall_modified conflict (high-overlap contribution)" --method pall_modified \
    --alpha 0.5 --beta 1.0 --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 5 \
    --protect_importance conflict --adaptive_protect
run "[5/6] pall_adapter overlap" --method pall_adapter \
    --adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
    --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
    --adapter_train_classifier --retrain_steps 5

echo "==================== [6/6] aggregation (smoke) ===================="
if $PY -u tools/aggregate_results.py --root runs --out /tmp/smoke_summary.csv --include-smoke \
       --experiment-tag smoke; then
    echo "PASS: aggregation -> /tmp/smoke_summary.csv"
else
    echo "FAIL: aggregation"; fails=$((fails + 1))
fi

echo "===================================================================="
if [ "$fails" -eq 0 ]; then
    echo "ALL SMOKE TESTS PASSED"
else
    echo "SMOKE TESTS FAILED: ${fails} step(s)"; exit 1
fi
