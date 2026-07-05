#!/usr/bin/env bash
# Server experiment runner for the reviewer-response experiments.
# Covers three groups, then aggregates tables.
#   GROUP 1: extra baselines (ewc, lwf, clpu) on fixed schedules, CIFAR-10 + CIFAR-100, seeds 0,1
#   GROUP 2: proposed methods (pall_original, pall_modified, pall_adapter, lora), CIFAR-10 + CIFAR-100, seeds 0,1
#   GROUP 3: PALL-Adapter bottleneck ablation on CIFAR-10 (seed 0)
#   GROUP 4: pretrained-backbone PEFT (pall_adapter, lora on a frozen ImageNet ResNet-18) -- ON DEMAND, not in `all`
#   GROUP 5: pretrained PALL-Adapter hyperparameter sweep on CIFAR-100 -- ON DEMAND, not in `all`
#   GROUP 6 (g6_standard): literature-comparable STANDARD Split-CIFAR, all 9 methods -- ON DEMAND, not in `all`
#   GROUP 7 (g7_tiny): TinyImageNet main + pretrained PEFT runs -- ON DEMAND, not in `all`
#   GROUP 8 (g8_mia): MIA-enabled proposed/pretrained PEFT + CLPU runs -- ON DEMAND, not in `all`
#
# Usage (run on a GPU node, inside tmux):
#   bash tools/run_server_experiments.sh            # all = g1 + g2 + g3 (g4/g5/g6_standard/g7_tiny/g8_mia are NOT included)
#   bash tools/run_server_experiments.sh g1         # only group 1
#   bash tools/run_server_experiments.sh g4         # only the pretrained-backbone PEFT group
#   bash tools/run_server_experiments.sh g5         # only the pretrained PALL-Adapter tuning sweep
#   bash tools/run_server_experiments.sh g6_standard # standard Split-CIFAR benchmark (see docs/standard_vs_overlap.md)
#   bash tools/run_server_experiments.sh g7_tiny    # TinyImageNet main + pretrained PEFT group
#   bash tools/run_server_experiments.sh g8_mia     # MIA-enabled proposed/pretrained PEFT + CLPU group
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
# Proposed methods on both datasets (seeds 0,1). Flags copied from
# tools/run_paper_experiments.sh. main.py auto-prefixes the arch per method
# (subnet_/adapter_/lora_), so the same --arch in $C10/$C100 serves every method.
group2 () {
    echo "===== GROUP 2: proposed methods (pall_original / pall_modified / pall_adapter / lora) ====="
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        # ---- CIFAR-10 (resnet18) ----
        launch "c10_pall_original_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_original \
               --retrain_steps 50 --experiment_tag cifar10_main
        launch "c10_pall_modified_grad_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_modified $PALL_MOD \
               --experiment_tag cifar10_main
        launch "c10_pall_adapter_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_adapter $ADAPTER \
               --experiment_tag cifar10_main
        launch "c10_lora_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method lora $LORA \
               --experiment_tag cifar10_main
        # ---- CIFAR-100 (resnet34) ----
        launch "c100_pall_original_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_original \
               --retrain_steps 50 --experiment_tag cifar100_main
        launch "c100_pall_modified_grad_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_modified $PALL_MOD \
               --experiment_tag cifar100_main
        launch "c100_pall_adapter_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_adapter $ADAPTER \
               --experiment_tag cifar100_main
        launch "c100_lora_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method lora $LORA \
               --experiment_tag cifar100_main
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

# -------------------------------------------------------------------- GROUP 4
# Pretrained-backbone PEFT variant: pall_adapter + lora on a FROZEN ImageNet
# ResNet-18 feature extractor. Run ON DEMAND (g4) -- intentionally NOT part of
# `all`. Both datasets use --arch resnet18 (the frozen backbone replaces the
# from-scratch one anyway). Same per-method flags as GROUP 2 + the shared BASE.
group4 () {
    echo "===== GROUP 4: pretrained-backbone PEFT (pall_adapter / lora, frozen ImageNet ResNet-18) ====="
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    local C100_R18="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet18 --sparsity 0.9"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        # ---- CIFAR-10 (resnet18) ----
        launch "c10_pall_adapter_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
               --request_schedule_file $c10 --method pall_adapter $ADAPTER \
               --experiment_tag cifar10_pretrained
        launch "c10_lora_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
               --request_schedule_file $c10 --method lora $LORA \
               --experiment_tag cifar10_pretrained
        # ---- CIFAR-100 (resnet18) ----
        launch "c100_pall_adapter_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
               --request_schedule_file $c100 --method pall_adapter $ADAPTER \
               --experiment_tag cifar100_pretrained
        launch "c100_lora_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
               --request_schedule_file $c100 --method lora $LORA \
               --experiment_tag cifar100_pretrained
    done
}

# -------------------------------------------------------------------- GROUP 5
# Focused pretrained PALL-Adapter tuning sweep. Run ON DEMAND (g5) -- intentionally
# NOT part of `all`. Fixed to CIFAR-100 seeds 0 and 1 so the grid is always
# 8 configs x 2 seeds = 16 runs, independent of the SEEDS override.
group5 () {
    echo "===== GROUP 5: pretrained PALL-Adapter tuning sweep (CIFAR-100, frozen ImageNet ResNet-18) ====="
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth"
    local COMMON_E5="${COMMON/--n_epochs 3/--n_epochs 5}"
    local C100_R18="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet18 --sparsity 0.9"
    for s in 0 1; do
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for forget_ratio in 0.3 0.5; do
            for protect_ratio in 0.2 0.4; do
                for forget_steps in 10 30; do
                    launch "c100_pall_adapter_pretrained_fr${forget_ratio}_pr${protect_ratio}_fs${forget_steps}_s${s}" \
                           $C100_R18 $COMMON_E5 $PRE --seed $s \
                           --request_schedule_file $c100 --method pall_adapter \
                           --adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
                           --adapter_shared_forget_ratio $forget_ratio \
                           --adapter_shared_protect_ratio $protect_ratio \
                           --adapter_train_classifier --retrain_steps 50 \
                           --adapter_forget_steps $forget_steps \
                           --experiment_tag adapter_tune_pretrained_v1
                done
            done
        done
    done
}

# ----------------------------------------------------------- GROUP 6 (standard)
# Literature-comparable STANDARD Split-CIFAR benchmark (NOT our overlap-heavy
# setting). Run ON DEMAND (g6_standard) -- intentionally NOT part of `all`.
# Differences vs the default groups (see docs/standard_vs_overlap.md):
#   * CIFAR-10  : standard 5 tasks x 2 classes (already disjoint/random).
#   * CIFAR-100 : standard Split-CIFAR-100 = 10 tasks x 10 RANDOM disjoint classes
#                 (--cifar100_split standard), instead of the 20 semantic
#                 superclasses x 5 fine classes used elsewhere.
#   * n_epochs 20 (reference training length) instead of the overlap runs' 3.
# All 9 methods, both datasets, seeds 0/1. er/derpp need --forget_iters.
group6_standard () {
    echo "===== GROUP 6 (standard): literature-comparable Split-CIFAR (all methods) ====="
    local COMMON_E20="${COMMON/--n_epochs 3/--n_epochs 20}"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    local FI="--forget_iters 50"
    local C100_STD="--dataset cifar100 --class_per_task 10 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9 --cifar100_split standard"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        # ---- CIFAR-10 (standard 5x2) ----
        launch "std_c10_pall_original_s${s}" $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_original --retrain_steps 50 --experiment_tag cifar10_standard
        launch "std_c10_pall_modified_s${s}" $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_modified $PALL_MOD --experiment_tag cifar10_standard
        launch "std_c10_pall_adapter_s${s}"  $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_adapter $ADAPTER --experiment_tag cifar10_standard
        launch "std_c10_lora_s${s}"          $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method lora $LORA --experiment_tag cifar10_standard
        launch "std_c10_er_s${s}"            $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method er $FI --experiment_tag cifar10_standard
        launch "std_c10_derpp_s${s}"         $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method derpp $FI --experiment_tag cifar10_standard
        launch "std_c10_ewc_s${s}"           $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method ewc --experiment_tag cifar10_standard
        launch "std_c10_lwf_s${s}"           $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method lwf --experiment_tag cifar10_standard
        launch "std_c10_clpu_s${s}"          $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method clpu --experiment_tag cifar10_standard
        # ---- CIFAR-100 (standard Split-CIFAR-100, 10x10 random disjoint) ----
        launch "std_c100_pall_original_s${s}" $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_original --retrain_steps 50 --experiment_tag cifar100_standard
        launch "std_c100_pall_modified_s${s}" $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_modified $PALL_MOD --experiment_tag cifar100_standard
        launch "std_c100_pall_adapter_s${s}"  $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_adapter $ADAPTER --experiment_tag cifar100_standard
        launch "std_c100_lora_s${s}"          $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method lora $LORA --experiment_tag cifar100_standard
        launch "std_c100_er_s${s}"            $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method er $FI --experiment_tag cifar100_standard
        launch "std_c100_derpp_s${s}"         $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method derpp $FI --experiment_tag cifar100_standard
        launch "std_c100_ewc_s${s}"           $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method ewc --experiment_tag cifar100_standard
        launch "std_c100_lwf_s${s}"           $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method lwf --experiment_tag cifar100_standard
        launch "std_c100_clpu_s${s}"          $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method clpu --experiment_tag cifar100_standard
    done
}

# --------------------------------------------------------------- GROUP 7 tiny
# TinyImageNet main + pretrained PEFT runs. Run ON DEMAND (g7_tiny) --
# intentionally NOT part of `all`. Uses the fixed TinyImageNet schedule for
# seed 0, and seed 1 too if schedules/tinyimagenet_t20_f3_seed1.json exists.
group7_tiny () {
    echo "===== GROUP 7: TinyImageNet main + pretrained PEFT ====="
    local TINY="--dataset tinyimagenet --class_per_task 10 --n_tasks 20 --n_forget 3 --arch resnet18"
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth --cache_features"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    for s in 0 1; do
        local sch="schedules/tinyimagenet_t20_f3_seed${s}.json"
        if [[ ! -f "$sch" ]]; then
            echo "    SKIP tinyimagenet seed ${s}: missing ${sch}"
            continue
        fi
        launch "tiny_pall_original_s${s}" $TINY $COMMON --seed $s \
               --request_schedule_file $sch --method pall_original \
               --retrain_steps 50 --experiment_tag tiny_main
        launch "tiny_pall_modified_grad_s${s}" $TINY $COMMON --seed $s \
               --request_schedule_file $sch --method pall_modified $PALL_MOD \
               --experiment_tag tiny_main
        launch "tiny_clpu_s${s}" $TINY $COMMON --seed $s \
               --request_schedule_file $sch --method clpu \
               --experiment_tag tiny_main
        launch "tiny_pall_adapter_pretrained_s${s}" $TINY $COMMON $PRE --seed $s \
               --request_schedule_file $sch --method pall_adapter $ADAPTER \
               --experiment_tag tiny_pretrained
        launch "tiny_lora_pretrained_s${s}" $TINY $COMMON $PRE --seed $s \
               --request_schedule_file $sch --method lora $LORA \
               --experiment_tag tiny_pretrained
    done
}

# --------------------------------------------------------------- GROUP 8 MIA
# MIA-enabled reruns for the proposed/full-network and pretrained-PEFT configs.
# Run ON DEMAND (g8_mia) -- intentionally NOT part of `all`.
group8_mia () {
    echo "===== GROUP 8: MIA-enabled proposed/pretrained PEFT + CLPU ====="
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    local C100_R18="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet18 --sparsity 0.9"
    for s in 0 1; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        # ---- CIFAR-10 ----
        launch "mia_c10_pall_modified_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_modified $PALL_MOD \
               --eval_mia --experiment_tag cifar10_mia
        launch "mia_c10_pall_adapter_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
               --request_schedule_file $c10 --method pall_adapter $ADAPTER \
               --eval_mia --experiment_tag cifar10_pretrained_mia
        launch "mia_c10_lora_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
               --request_schedule_file $c10 --method lora $LORA \
               --eval_mia --experiment_tag cifar10_pretrained_mia
        launch "mia_c10_clpu_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method clpu \
               --eval_mia --experiment_tag cifar10_mia
        # ---- CIFAR-100 ----
        launch "mia_c100_pall_modified_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_modified $PALL_MOD \
               --eval_mia --experiment_tag cifar100_mia
        launch "mia_c100_pall_adapter_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
               --request_schedule_file $c100 --method pall_adapter $ADAPTER \
               --eval_mia --experiment_tag cifar100_pretrained_mia
        launch "mia_c100_lora_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
               --request_schedule_file $c100 --method lora $LORA \
               --eval_mia --experiment_tag cifar100_pretrained_mia
        launch "mia_c100_clpu_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method clpu \
               --eval_mia --experiment_tag cifar100_mia
    done
}

case "$WHICH" in
    all) group1; group2; group3 ;;
    g1)  group1 ;;
    g2)  group2 ;;
    g3)  group3 ;;
    g4)  group4 ;;
    g5)  group5 ;;
    g6_standard) group6_standard ;;
    g7_tiny) group7_tiny ;;
    g8_mia) group8_mia ;;
    *)   echo "unknown arg: $WHICH (use: all | g1 | g2 | g3 | g4 | g5 | g6_standard | g7_tiny | g8_mia)"; exit 1 ;;
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
