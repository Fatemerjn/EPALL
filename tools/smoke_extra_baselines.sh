#!/usr/bin/env bash
# Fast smoke test for the extra unlearning baselines. This is not a benchmark:
# it checks that SSD and SalUn each complete train -> forget -> metrics on CIFAR-10.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-./.venv/bin/python}"
COMMON="--dataset cifar10 --class_per_task 2 --n_tasks 1 --n_forget 1 \
        --n_epochs 1 --batch_size 1000 --mem_budget 20 --k_shot 1 \
        --optim sgd --momentum 0.9 --lr 1e-2 --weight_decay 5e-4 --seed 0 \
        --arch resnet18 --sparsity 0.8 --device cpu --deterministic \
        --experiment_tag smoke_extra_baselines"

fails=0
run () {
    local label="$1"; shift
    echo "==================== ${label} ===================="
    if $PY -u main.py $COMMON "$@"; then
        echo "PASS: ${label}"
    else
        echo "FAIL: ${label}"
        fails=$((fails + 1))
    fi
}

run "[1/2] SSD" --method ssd --ssd_alpha 1.0 --ssd_lambda 1.0
run "[2/2] SalUn" --method salun --salun_mask_ratio 0.1 --salun_target uniform --forget_iters 1

echo "===================================================================="
if [ "$fails" -eq 0 ]; then
    echo "EXTRA BASELINE SMOKE TESTS PASSED"
else
    echo "EXTRA BASELINE SMOKE TESTS FAILED: ${fails} step(s)"
    exit 1
fi
