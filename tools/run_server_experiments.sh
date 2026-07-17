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
#   GROUP 9 (g9_extra_baselines): SSD + SalUn baselines -- ON DEMAND, not in `all`
#   GROUP 9b (g9b_ssd_tune): SSD alpha/lambda tuning sweep (CIFAR-10) -- ON DEMAND, not in `all`
#   GROUP 10 (g10_anchor): PALL-Modified anchor ablation, protect_anchor old vs reinit -- ON DEMAND, not in `all`
#   GROUP 14 (g14_conflict): conflict-vs-gradient protect_importance ablation (+adaptive_protect for pall_modified) -- ON DEMAND, not in `all`
#   GROUP 11 (g11_vit): ViT-T/8 cross-architecture, pall_original + pall_modified -- ON DEMAND, not in `all`
#   GROUP 15 (g15_seed2): seed-2 reruns for paper main-table rows only -- ON DEMAND, not in `all`
#   GROUP 17 (g17_overlap_curve): five-grade overlap-response curves, six methods -- ON DEMAND, not in `all`
#   GROUP 18 (g18_probe): linear-probe audit on MAIN configs, five methods -- ON DEMAND, not in `all`
#   GROUP 20 (g20_paper_completion): only missing paper-completion reruns -- ON DEMAND, not in `all`
#   GROUP 21 (g21_standard_unlearning): standard SSD + SalUn, both CIFAR datasets, seeds 0/1/2 -- ON DEMAND, resume-safe
#
# Usage (run on a GPU node, inside tmux):
#   bash tools/run_server_experiments.sh            # all = g1 + g2 + g3 (g4/g5/g6_standard/g7_tiny/g8_mia/g9_extra_baselines are NOT included)
#   bash tools/run_server_experiments.sh g1         # only group 1
#   bash tools/run_server_experiments.sh g4         # only the pretrained-backbone PEFT group
#   bash tools/run_server_experiments.sh g5         # only the pretrained PALL-Adapter tuning sweep
#   bash tools/run_server_experiments.sh g6_standard # standard Split-CIFAR benchmark (see docs/standard_vs_overlap.md)
#   bash tools/run_server_experiments.sh g7_tiny    # TinyImageNet main + pretrained PEFT group
#   bash tools/run_server_experiments.sh g8_mia     # MIA-enabled proposed/pretrained PEFT + CLPU group
#   bash tools/run_server_experiments.sh g9_extra_baselines # SSD + SalUn baseline group
#   bash tools/run_server_experiments.sh g9b_ssd_tune # SSD alpha/lambda tuning sweep (9 configs, CIFAR-10, seed 0)
#   bash tools/run_server_experiments.sh g10_anchor # PALL-Modified anchor ablation (old vs reinit, MIA on), both datasets/seeds
#   bash tools/run_server_experiments.sh g11_vit # ViT-T/8 cross-architecture (pall_original + pall_modified), both datasets/seeds
#   bash tools/run_server_experiments.sh g12_agreement # model agreement rate (pall_modified + pall_adapter, T5_F1, seeds 0/1)
#   bash tools/run_server_experiments.sh g13_bottleneck # task-bottleneck seed1 + shared-bottleneck sweep (also: g3b_bottleneck_seed1, g13_shared_bottleneck)
#   bash tools/run_server_experiments.sh g14_conflict # conflict-vs-gradient protect_importance ablation (+adaptive_protect for pall_modified)
#   bash tools/run_server_experiments.sh g15_seed2 # third seed for selected main, standard, and pretrained paper rows
#   bash tools/run_server_experiments.sh g17_overlap_curve # five-grade overlap-response curves, both CIFAR datasets/seeds
#   bash tools/run_server_experiments.sh g18_probe # linear-probe audit on MAIN configs, both CIFAR datasets/seeds
#   bash tools/run_server_experiments.sh g20_paper_completion # 15 missing paper-completion runs only
#   bash tools/run_server_experiments.sh g21_standard_unlearning --dry-run # inspect the exact 12-run standard SSD/SalUn matrix
#   SEEDS="0 1 2" bash tools/run_server_experiments.sh
#
# main.py auto-selects CUDA when available (--device auto is the default).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1}"
WHICH="${1:-all}"
DRY_RUN=0

usage () {
    printf '%s\n' \
        "Usage: bash tools/run_server_experiments.sh [GROUP] [--dry-run]" \
        "" \
        "Default GROUP is 'all' (g1 + g2 + g3)." \
        "The --dry-run option is supported by g15_seed2, g20_paper_completion," \
        "and g21_standard_unlearning." \
        "" \
        "Groups: all g1 g2 g3 g4 g5 g6_standard g7_tiny g8_mia" \
        "        g9_extra_baselines g9b_ssd_tune g10_anchor g11_vit" \
        "        g12_agreement g3b_bottleneck_seed1 g13_shared_bottleneck" \
        "        g13_bottleneck g14_conflict g15_seed2 g17_overlap_curve" \
        "        g18_probe g20_paper_completion g21_standard_unlearning"
}

if [[ "$WHICH" == "-h" || "$WHICH" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ -n "${2:-}" ]]; then
    echo "unknown option: ${2}" >&2
    usage >&2
    exit 1
fi
if [[ -n "${3:-}" ]]; then
    echo "too many arguments" >&2
    usage >&2
    exit 1
fi
if (( DRY_RUN )); then
    case "$WHICH" in
        g15_seed2|g20_paper_completion|g21_standard_unlearning) ;;
        *)
            echo "--dry-run is supported only for g15_seed2, g20_paper_completion, and g21_standard_unlearning" >&2
            exit 1
            ;;
    esac
fi
mkdir -p logs results/aggregates
FAILED_RUNS=0
FAILED_AGGREGATIONS=0

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
        ((FAILED_RUNS += 1))
        echo "    FAIL ${name}  (see logs/${name}.log)"
    fi
}

launch_resume_safe () {  # launch_resume_safe <logname> <main.py args...>
    local name="$1"; shift
    local completed_path=""
    local resume_status=0
    if completed_path="$($PY tools/find_exact_completed_run.py --root runs -- "$@")"; then
        echo ">>> ${name}"
        echo "    SKIP exact completed config/seed match: ${completed_path}"
        return 0
    else
        resume_status=$?
    fi
    if (( resume_status != 1 )); then
        ((FAILED_RUNS += 1))
        echo "    FAIL ${name}: exact-resume check exited ${resume_status}" >&2
        return 0
    fi
    if (( DRY_RUN )); then
        local dataset="" method="" seed="" n_tasks="" n_forget=""
        local -a argv=("$@")
        local i
        for ((i = 0; i < ${#argv[@]}; i++)); do
            case "${argv[$i]}" in
                --dataset) dataset="${argv[$((i + 1))]}" ;;
                --method) method="${argv[$((i + 1))]}" ;;
                --seed) seed="${argv[$((i + 1))]}" ;;
                --n_tasks) n_tasks="${argv[$((i + 1))]}" ;;
                --n_forget) n_forget="${argv[$((i + 1))]}" ;;
            esac
        done
        echo ">>> ${name}"
        echo "    DRY-RUN output: runs/${dataset}/T${n_tasks}_F${n_forget}/${method}/seed_${seed}/<timestamp>"
        printf '    COMMAND %q -u main.py' "$PY"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    launch "$name" "$@"
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
        # LoRA on standard Split-CIFAR-100 diverges to NaN at the default lr=1e-2 over
        # 20 epochs (frozen RANDOM backbone + 10-way tasks -> logits blow up on task 0,
        # collapsing every task to chance=0.1000 for both seeds). It needs lr=1e-3; the
        # trailing --lr overrides COMMON_E20's 1e-2 for this one command only (argparse
        # keeps the last value). See docs/standard_vs_overlap.md. Paper footnote required.
        launch "std_c100_lora_s${s}"          $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method lora $LORA --lr 1e-3 --experiment_tag cifar100_standard
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

# ---------------------------------------------------- GROUP 9 extra baselines
# Standard machine-unlearning baselines. Run ON DEMAND (g9_extra_baselines) --
# intentionally NOT part of `all`.
group9_extra_baselines () {
    echo "===== GROUP 9: SSD + SalUn extra unlearning baselines ====="
    local SSD_ARGS="--ssd_alpha 1.0 --ssd_lambda 1.0"
    local SALUN_ARGS="--salun_mask_ratio 0.1 --salun_target uniform --forget_iters 50"
    for s in 0 1; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for m in ssd salun; do
            local extra="$SSD_ARGS"
            if [[ "$m" == "salun" ]]; then
                extra="$SALUN_ARGS"
            fi
            launch "c10_${m}_extra_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method $m $extra \
                   --experiment_tag cifar10_extra_baselines
            launch "c100_${m}_extra_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method $m $extra \
                   --experiment_tag cifar100_extra_baselines
        done
    done
}

# ------------------------------------------------------- GROUP 9b SSD tuning
# SSD alpha/lambda tuning sweep. Run ON DEMAND (g9b_ssd_tune) -- intentionally
# NOT part of `all`. Motivation: at the default --ssd_alpha 1.0 / --ssd_lambda 1.0
# the SSD baseline barely forgets (Au ~0.93 on CIFAR-10, chance 0.5). A 1-epoch
# CIFAR-10 diagnostic (T5_F1, seed 0) shows selection coverage
# (updated_params/ssd_total_params) is actually substantial and monotonic in
# alpha: alpha=1.0 -> 0.485, 0.5 -> 0.653, 0.2 -> 0.817, 0.05 -> 0.937. So the
# weak forgetting is NOT from tiny coverage; it is the dampening STRENGTH: the
# factor clamp(lambda/ratio, 0, 1) sits near 1.0 for the bulk of selected params
# (ratio just above 1.0) when lambda=1.0, so weights are barely reduced. Lowering
# alpha widens coverage AND lowering lambda deepens the dampening. Sweep both on
# CIFAR-10 (T5_F3, seed 0) to pick a config whose Au approaches chance (0.5) with
# the smallest WorstDrop. Same schedule/epochs as cifar10_extra_baselines so the
# tuned cell is directly comparable to the default SSD run.
group9b_ssd_tune () {
    echo "===== GROUP 9b: SSD alpha/lambda tuning sweep (CIFAR-10, seed 0) ====="
    local s=0
    local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
    for a in 0.05 0.2 0.5; do
        for l in 0.1 0.5 1.0; do
            launch "c10_ssd_tune_a${a}_l${l}_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method ssd \
                   --ssd_alpha $a --ssd_lambda $l \
                   --experiment_tag ssd_tune_v1
        done
    done
}

# ---------------------------------------------------- GROUP 10 anchor ablation
# Directly answers the advisor's main theoretical objection (docs/review.tex
# Issue 1): PALL-Modified's L2 protection anchors critical shared weights to
# their pre-forget values w_old, which STILL encode the forgotten task. The
# corrected alternative anchors instead to a fresh reinitialization
# (--protect_anchor reinit, methods/pall_base._reinit_anchor_values). No full
# runs with reinit exist yet. This group re-runs pall_modified with the SAME
# base config as cifar10_main/cifar100_main, sweeping protect_anchor in
# {old, reinit} with MIA enabled for both so the leakage difference is
# measurable. Run ON DEMAND (g10_anchor) -- intentionally NOT part of `all`.
group10_anchor () {
    echo "===== GROUP 10: anchor ablation (pall_modified old vs reinit, MIA on) ====="
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for anchor in old reinit; do
            launch "anchor_c10_pall_modified_${anchor}_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method pall_modified $PALL_MOD \
                   --protect_anchor $anchor --eval_mia --experiment_tag anchor_ablation_v1
            launch "anchor_c100_pall_modified_${anchor}_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method pall_modified $PALL_MOD \
                   --protect_anchor $anchor --eval_mia --experiment_tag anchor_ablation_v1
        done
    done
}

# --------------------------------------------------------- GROUP 11 ViT arch
# Cross-architecture evaluation (mandatory reviewer item): run the subnet PALL
# methods on a ViT-T/8 backbone (models.subnet_vit_t8, sized for 32x32 CIFAR:
# 16 patches + cls) instead of ResNet. --arch is passed as vit_t8 and main.py
# prefixes 'subnet_' for PALL methods -> subnet_vit_t8 (passing subnet_vit_t8
# directly would DOUBLE-prefix to subnet_subnet_vit_t8 and fail). Only
# pall_original + pall_modified are in scope: pall_adapter is OUT (there is no
# adapter_vit model / adapter insertion into ViT blocks is a separate effort).
# n_epochs 5. Run ON DEMAND (g11_vit) -- intentionally NOT part of `all`.
group11_vit () {
    echo "===== GROUP 11: ViT-T/8 cross-architecture (pall_original / pall_modified) ====="
    local COMMON_E5="${COMMON/--n_epochs 3/--n_epochs 5}"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local VIT_C10="--dataset cifar10  --class_per_task 2 --n_tasks 5  --n_forget 3 --arch vit_t8 --sparsity 0.8"
    local VIT_C100="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch vit_t8 --sparsity 0.9"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        # ---- CIFAR-10 (ViT-T/8) ----
        launch "vit_c10_pall_original_s${s}" $VIT_C10 $COMMON_E5 --seed $s \
               --request_schedule_file $c10 --method pall_original --retrain_steps 50 \
               --experiment_tag cifar10_vit_v1
        launch "vit_c10_pall_modified_s${s}" $VIT_C10 $COMMON_E5 --seed $s \
               --request_schedule_file $c10 --method pall_modified $PALL_MOD \
               --experiment_tag cifar10_vit_v1
        # ---- CIFAR-100 (ViT-T/8) ----
        launch "vit_c100_pall_original_s${s}" $VIT_C100 $COMMON_E5 --seed $s \
               --request_schedule_file $c100 --method pall_original --retrain_steps 50 \
               --experiment_tag cifar100_vit_v1
        launch "vit_c100_pall_modified_s${s}" $VIT_C100 $COMMON_E5 --seed $s \
               --request_schedule_file $c100 --method pall_modified $PALL_MOD \
               --experiment_tag cifar100_vit_v1
    done
}

# ---------------------------------------------------- GROUP 12 agreement rate
# Model Agreement Rate (reviewer metric). EXPENSIVE: --eval_agreement trains a
# from-scratch Sequential REFERENCE per forget event on the same schedule with
# the forgotten task never trained, then reports the fraction of test predictions
# that match the unlearned model. Gated to single-forget runs (n_forget==1), so
# this group uses the T5_F1 candidate-forget schedule (forget task 0). Run ON
# DEMAND (g12_agreement) -- intentionally NOT part of `all`.
group12_agreement () {
    echo "===== GROUP 12: model agreement rate (pall_modified + pall_adapter, n_forget=1) ====="
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local C10_F1="--dataset cifar10 --class_per_task 2 --n_tasks 5 --n_forget 1 --arch resnet18 --sparsity 0.8"
    for s in $SEEDS; do
        local c10="schedules/cifar10_candidate_forget_task0_seed${s}.json"
        launch "agree_c10_pall_modified_s${s}" $C10_F1 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_modified $PALL_MOD \
               --eval_agreement --experiment_tag agreement_v1
        launch "agree_c10_pall_adapter_s${s}" $C10_F1 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_adapter $ADAPTER \
               --eval_agreement --experiment_tag agreement_v1
    done
}

# ------------------------------------------------- GROUP 3b task bottleneck s1
# Seed-1 half of the PALL-Adapter TASK-bottleneck ablation (group3 is seed 0).
# Same {4,8,16,32,64,128} sweep, same experiment_tag adapter_bottleneck_ablation_v1
# so plot_bottleneck_ablation merges both seeds into one bootstrap-CI figure with
# NO plot code change. Run ON DEMAND -- intentionally NOT part of `all`.
group3b_bottleneck_seed1 () {
    echo "===== GROUP 3b: PALL-Adapter task-bottleneck ablation (CIFAR-10, seed 1) ====="
    local sch="schedules/cifar10_t5_f3_fixed_seed1.json"
    for b in 4 8 16 32 64 128; do
        launch "c10_adapter_bottleneck_${b}_s1" $C10 $COMMON --seed 1 \
               --request_schedule_file $sch --method pall_adapter \
               --adapter_bottleneck $b --adapter_shared_bottleneck 16 \
               --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
               --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10 \
               --experiment_tag adapter_bottleneck_ablation_v1
    done
}

# ------------------------------------------------- GROUP 13 shared bottleneck
# SHARED-adapter bottleneck ablation -- the sweep that matters for cross-task
# overlap (docs/review.tex Lemma: the shared adapter bottleneck bounds critical
# overlap). Vary --adapter_shared_bottleneck in {4,8,16,32} with the TASK
# bottleneck FIXED at 16 (all other adapter knobs at the main-group defaults),
# CIFAR-10, seeds 0 1, tag shared_bottleneck_ablation_v1. Consumed by
# plot_shared_bottleneck_ablation(). Run ON DEMAND -- intentionally NOT in `all`.
group13_shared_bottleneck () {
    echo "===== GROUP 13: PALL-Adapter shared-bottleneck ablation (CIFAR-10, seeds 0 1) ====="
    for s in $SEEDS; do
        local sch="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        for sb in 4 8 16 32; do
            launch "c10_adapter_shared_bottleneck_${sb}_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $sch --method pall_adapter \
                   --adapter_bottleneck 16 --adapter_shared_bottleneck $sb \
                   --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
                   --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10 \
                   --experiment_tag shared_bottleneck_ablation_v1
        done
    done
}

# ------------------------------------------------- GROUP 14 conflict ablation
# Decides the fate of the gradient-CONFLICT protection criterion empirically.
# --protect_importance conflict (relu(-g_forget*g_retain): methods/pall_base.
# _compute_conflict_importance and methods/pall_adapter._compute_shared_conflict)
# is wired for BOTH pall_modified and pall_adapter but has only ever run as a
# 1-epoch smoke; every recorded full run used protect_importance=gradient. This
# group re-runs the SAME base config as cifar10_main/cifar100_main, sweeping
# protect_importance in {gradient, conflict}. For pall_modified it additionally
# sweeps --adaptive_protect {off, on} (that flag scales lambda_protect by the
# measured critical-overlap ratio and is a pall_modified-only knob -- it is NOT
# wired for pall_adapter). The gradient cells reproduce the main config as a
# same-tag paired baseline for a clean conflict-vs-gradient comparison. Run
# ON DEMAND (g14_conflict) -- intentionally NOT part of `all`.
group14_conflict () {
    echo "===== GROUP 14: conflict-importance ablation (pall_modified + pall_adapter) ====="
    local PALL_MOD_BASE="--protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    for s in $SEEDS; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for imp in gradient conflict; do
            # ---- pall_modified: protect_importance x adaptive_protect {off,on} ----
            for adapt in off on; do
                local aflag=""
                if [[ "$adapt" == "on" ]]; then aflag="--adaptive_protect"; fi
                launch "conflict_c10_pall_modified_${imp}_adapt${adapt}_s${s}" $C10 $COMMON --seed $s \
                       --request_schedule_file $c10 --method pall_modified $PALL_MOD_BASE \
                       --protect_importance $imp $aflag --experiment_tag conflict_ablation_v1
                launch "conflict_c100_pall_modified_${imp}_adapt${adapt}_s${s}" $C100 $COMMON --seed $s \
                       --request_schedule_file $c100 --method pall_modified $PALL_MOD_BASE \
                       --protect_importance $imp $aflag --experiment_tag conflict_ablation_v1
            done
            # ---- pall_adapter: protect_importance only (no adaptive_protect flag) ----
            launch "conflict_c10_pall_adapter_${imp}_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method pall_adapter $ADAPTER \
                   --protect_importance $imp --experiment_tag conflict_ablation_v1
            launch "conflict_c100_pall_adapter_${imp}_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method pall_adapter $ADAPTER \
                   --protect_importance $imp --experiment_tag conflict_ablation_v1
        done
    done
}

# ---------------------------------------------------- GROUP 15 paper seed 2
# Third-seed reruns for rows consumed by the paper's main tables. This is a
# deliberately curated subset: all existing cifar10_main/cifar100_main rows;
# the three PALL methods, LoRA, and CLPU from each standard table; and the
# PALL-Adapter/LoRA pretrained rows. Fixed to seed 2 and intentionally NOT in
# `all`, independent of the SEEDS override. Total: 22 runs.
group15_seed2 () {
    echo "===== GROUP 15: seed-2 paper main-table rows only ====="
    local s=2
    local c10="schedules/cifar10_t5_f3_fixed_seed2.json"
    local c100="schedules/cifar100_t10_f3_seed2.json"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"

    # ---- overlap-heavy main rows (three PALL methods + LoRA) ----
    launch_resume_safe "c10_pall_original_s${s}" $C10 $COMMON --seed $s \
           --request_schedule_file $c10 --method pall_original \
           --retrain_steps 50 --experiment_tag cifar10_main
    launch_resume_safe "c10_pall_modified_grad_s${s}" $C10 $COMMON --seed $s \
           --request_schedule_file $c10 --method pall_modified $PALL_MOD \
           --experiment_tag cifar10_main
    launch_resume_safe "c10_pall_adapter_s${s}" $C10 $COMMON --seed $s \
           --request_schedule_file $c10 --method pall_adapter $ADAPTER \
           --experiment_tag cifar10_main
    launch_resume_safe "c10_lora_s${s}" $C10 $COMMON --seed $s \
           --request_schedule_file $c10 --method lora $LORA \
           --experiment_tag cifar10_main

    launch_resume_safe "c100_pall_original_s${s}" $C100 $COMMON --seed $s \
           --request_schedule_file $c100 --method pall_original \
           --retrain_steps 50 --experiment_tag cifar100_main
    launch_resume_safe "c100_pall_modified_grad_s${s}" $C100 $COMMON --seed $s \
           --request_schedule_file $c100 --method pall_modified $PALL_MOD \
           --experiment_tag cifar100_main
    launch_resume_safe "c100_pall_adapter_s${s}" $C100 $COMMON --seed $s \
           --request_schedule_file $c100 --method pall_adapter $ADAPTER \
           --experiment_tag cifar100_main
    launch_resume_safe "c100_lora_s${s}" $C100 $COMMON --seed $s \
           --request_schedule_file $c100 --method lora $LORA \
           --experiment_tag cifar100_main

    # ---- selected standard rows (three PALL methods + LoRA + CLPU) ----
    local COMMON_E20="${COMMON/--n_epochs 3/--n_epochs 20}"
    local C100_STD="--dataset cifar100 --class_per_task 10 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9 --cifar100_split standard"
    launch_resume_safe "std_c10_pall_original_s${s}" $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_original --retrain_steps 50 --experiment_tag cifar10_standard
    launch_resume_safe "std_c10_pall_modified_s${s}" $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_modified $PALL_MOD --experiment_tag cifar10_standard
    launch_resume_safe "std_c10_pall_adapter_s${s}"  $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method pall_adapter $ADAPTER --experiment_tag cifar10_standard
    launch_resume_safe "std_c10_lora_s${s}"          $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method lora $LORA --experiment_tag cifar10_standard
    launch_resume_safe "std_c10_clpu_s${s}"          $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method clpu --experiment_tag cifar10_standard

    launch_resume_safe "std_c100_pall_original_s${s}" $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_original --retrain_steps 50 --experiment_tag cifar100_standard
    launch_resume_safe "std_c100_pall_modified_s${s}" $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_modified $PALL_MOD --experiment_tag cifar100_standard
    launch_resume_safe "std_c100_pall_adapter_s${s}"  $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method pall_adapter $ADAPTER --experiment_tag cifar100_standard
    launch_resume_safe "std_c100_lora_s${s}"          $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method lora $LORA --lr 1e-3 --experiment_tag cifar100_standard
    launch_resume_safe "std_c100_clpu_s${s}"          $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method clpu --experiment_tag cifar100_standard

    # ---- pretrained PEFT rows (PALL-Adapter + LoRA) ----
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth"
    local C100_R18="--dataset cifar100 --class_per_task 5 --n_tasks 10 --n_forget 3 --arch resnet18 --sparsity 0.9"
    launch_resume_safe "c10_pall_adapter_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
           --request_schedule_file $c10 --method pall_adapter $ADAPTER \
           --experiment_tag cifar10_pretrained
    launch_resume_safe "c10_lora_pretrained_s${s}" $C10 $COMMON $PRE --seed $s \
           --request_schedule_file $c10 --method lora $LORA \
           --experiment_tag cifar10_pretrained
    launch_resume_safe "c100_pall_adapter_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
           --request_schedule_file $c100 --method pall_adapter $ADAPTER \
           --experiment_tag cifar100_pretrained
    launch_resume_safe "c100_lora_pretrained_s${s}" $C100_R18 $COMMON $PRE --seed $s \
           --request_schedule_file $c100 --method lora $LORA \
           --experiment_tag cifar100_pretrained
}

# ---------------------------------------------------- GROUP 17 overlap curve
# Controlled request-position overlap at five grades. The `laterK` filename
# knob is the number of training requests after the task that is eventually
# forgotten. This group uses the same three-epoch configs as
# cifar10_main/cifar100_main, plus the existing CLPU and SalUn baseline configs.
# `--dump_overlap` is diagnostic-only on the mask-based PALL methods and supplies
# their measured mask-IoU fallback for the response-curve x axis.
# It is fixed to seeds 0/1 and intentionally NOT part of `all`. Total: 120 runs.
group17_overlap_curve () {
    echo "===== GROUP 17: five-grade overlap-response curves (six methods) ====="
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"
    local SALUN="--salun_mask_ratio 0.1 --salun_target uniform --forget_iters 50"

    for s in 0 1; do
        for grade in very_low low medium high very_high; do
            local c10_k c100_k
            case "$grade" in
                very_low)  c10_k=0; c100_k=0 ;;
                low)       c10_k=1; c100_k=2 ;;
                medium)    c10_k=2; c100_k=4 ;;
                high)      c10_k=3; c100_k=7 ;;
                very_high) c10_k=4; c100_k=9 ;;
            esac
            local c10="schedules/cifar10_controlled_${grade}_later${c10_k}_seed${s}.json"
            local c100="schedules/cifar100_controlled_${grade}_later${c100_k}_seed${s}.json"
            local tag="overlap_curve_v1_${grade}"

            launch "overlap_curve_c10_${grade}_pall_original_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method pall_original \
                   --retrain_steps 50 --dump_overlap --experiment_tag $tag
            launch "overlap_curve_c10_${grade}_pall_modified_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method pall_modified $PALL_MOD \
                   --dump_overlap --experiment_tag $tag
            launch "overlap_curve_c10_${grade}_pall_adapter_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method pall_adapter $ADAPTER \
                   --experiment_tag $tag
            launch "overlap_curve_c10_${grade}_lora_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method lora $LORA \
                   --experiment_tag $tag
            launch "overlap_curve_c10_${grade}_clpu_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method clpu \
                   --experiment_tag $tag
            launch "overlap_curve_c10_${grade}_salun_s${s}" $C10 $COMMON --seed $s \
                   --request_schedule_file $c10 --method salun $SALUN \
                   --experiment_tag $tag

            launch "overlap_curve_c100_${grade}_pall_original_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method pall_original \
                   --retrain_steps 50 --dump_overlap --experiment_tag $tag
            launch "overlap_curve_c100_${grade}_pall_modified_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method pall_modified $PALL_MOD \
                   --dump_overlap --experiment_tag $tag
            launch "overlap_curve_c100_${grade}_pall_adapter_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method pall_adapter $ADAPTER \
                   --experiment_tag $tag
            launch "overlap_curve_c100_${grade}_lora_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method lora $LORA \
                   --experiment_tag $tag
            launch "overlap_curve_c100_${grade}_clpu_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method clpu \
                   --experiment_tag $tag
            launch "overlap_curve_c100_${grade}_salun_s${s}" $C100 $COMMON --seed $s \
                   --request_schedule_file $c100 --method salun $SALUN \
                   --experiment_tag $tag
        done
    done
}

# ---------------------------------------------------- GROUP 18 linear probe
# Re-run the exact three-epoch MAIN configurations with the existing linear-probe
# leakage audit enabled. CLPU inherits its baseline MAIN configuration from group1;
# the four proposed/PEFT methods inherit their configurations from group2. Fixed to
# seeds 0/1 and intentionally NOT part of `all`. Total: 20 runs.
group18_probe () {
    echo "===== GROUP 18: MAIN-config linear-probe audit (five methods) ====="
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    local LORA="--lora_rank 8 --lora_alpha 16"

    for s in 0 1; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"

        launch "probe_c10_pall_original_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_original \
               --retrain_steps 50 --eval_probe --experiment_tag probe_v1
        launch "probe_c10_pall_modified_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_modified $PALL_MOD \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c10_pall_adapter_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method pall_adapter $ADAPTER \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c10_lora_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method lora $LORA \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c10_clpu_s${s}" $C10 $COMMON --seed $s \
               --request_schedule_file $c10 --method clpu \
               --eval_probe --experiment_tag probe_v1

        launch "probe_c100_pall_original_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_original \
               --retrain_steps 50 --eval_probe --experiment_tag probe_v1
        launch "probe_c100_pall_modified_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_modified $PALL_MOD \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c100_pall_adapter_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method pall_adapter $ADAPTER \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c100_lora_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method lora $LORA \
               --eval_probe --experiment_tag probe_v1
        launch "probe_c100_clpu_s${s}" $C100 $COMMON --seed $s \
               --request_schedule_file $c100 --method clpu \
               --eval_probe --experiment_tag probe_v1
    done
}

# ----------------------------------------------- GROUP 20 paper completion
# Only the remaining paper-completion runs: corrected standard CIFAR-100 LoRA
# seeds 0/1, TinyImageNet seed 1, and the standard ER/DER++/EWC/LwF seed-2 rows
# omitted from g15_seed2. Fixed to these seeds and intentionally NOT in `all`.
# Total: 2 + 5 + 8 = 15 runs.
group20_paper_completion () {
    echo "===== GROUP 20: missing paper-completion runs only (15 runs) ====="

    # ---- corrected standard Split-CIFAR-100 LoRA, seeds 0/1 ----
    local COMMON_E20="${COMMON/--n_epochs 3/--n_epochs 20}"
    local C100_STD="--dataset cifar100 --class_per_task 10 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9 --cifar100_split standard"
    local LORA="--lora_rank 8 --lora_alpha 16"
    for s in 0 1; do
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        launch_resume_safe "paper_completion_std_c100_lora_s${s}" $C100_STD $COMMON_E20 \
               --seed $s --request_schedule_file $c100 --method lora $LORA \
               --lr 1e-3 --experiment_tag cifar100_standard
    done

    # ---- TinyImageNet seed 1, exact g7_tiny configurations ----
    local s=1
    local sch="schedules/tinyimagenet_t20_f3_seed1.json"
    local TINY="--dataset tinyimagenet --class_per_task 10 --n_tasks 20 --n_forget 3 --arch resnet18"
    local PRE="--pretrained_backbone imagenet_resnet18 --pretrained_weights pretrained/resnet18_imagenet.pth --cache_features"
    local PALL_MOD="--protect_importance gradient --protect_ratio 0.2 --lambda_protect 1.0 --retrain_steps 50"
    local ADAPTER="--adapter_bottleneck 16 --adapter_shared_bottleneck 16 \
        --adapter_shared_forget_ratio 0.3 --adapter_shared_protect_ratio 0.2 \
        --adapter_train_classifier --retrain_steps 50 --adapter_forget_steps 10"
    launch_resume_safe "paper_completion_tiny_pall_original_s${s}" $TINY $COMMON --seed $s \
           --request_schedule_file $sch --method pall_original \
           --retrain_steps 50 --experiment_tag tiny_main
    launch_resume_safe "paper_completion_tiny_pall_modified_grad_s${s}" $TINY $COMMON --seed $s \
           --request_schedule_file $sch --method pall_modified $PALL_MOD \
           --experiment_tag tiny_main
    launch_resume_safe "paper_completion_tiny_clpu_s${s}" $TINY $COMMON --seed $s \
           --request_schedule_file $sch --method clpu \
           --experiment_tag tiny_main
    launch_resume_safe "paper_completion_tiny_pall_adapter_pretrained_s${s}" $TINY $COMMON $PRE --seed $s \
           --request_schedule_file $sch --method pall_adapter $ADAPTER \
           --experiment_tag tiny_pretrained
    launch_resume_safe "paper_completion_tiny_lora_pretrained_s${s}" $TINY $COMMON $PRE --seed $s \
           --request_schedule_file $sch --method lora $LORA \
           --experiment_tag tiny_pretrained

    # ---- standard Split-CIFAR seed 2 baselines omitted from g15_seed2 ----
    local s=2
    local c10="schedules/cifar10_t5_f3_fixed_seed2.json"
    local c100="schedules/cifar100_t10_f3_seed2.json"
    local FI="--forget_iters 50"
    launch_resume_safe "paper_completion_std_c10_er_s${s}"    $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method er $FI --experiment_tag cifar10_standard
    launch_resume_safe "paper_completion_std_c10_derpp_s${s}" $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method derpp $FI --experiment_tag cifar10_standard
    launch_resume_safe "paper_completion_std_c10_ewc_s${s}"   $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method ewc --experiment_tag cifar10_standard
    launch_resume_safe "paper_completion_std_c10_lwf_s${s}"   $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 --method lwf --experiment_tag cifar10_standard
    launch_resume_safe "paper_completion_std_c100_er_s${s}"    $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method er $FI --experiment_tag cifar100_standard
    launch_resume_safe "paper_completion_std_c100_derpp_s${s}" $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method derpp $FI --experiment_tag cifar100_standard
    launch_resume_safe "paper_completion_std_c100_ewc_s${s}"   $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method ewc --experiment_tag cifar100_standard
    launch_resume_safe "paper_completion_std_c100_lwf_s${s}"   $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 --method lwf --experiment_tag cifar100_standard
}

# ----------------------------------------- GROUP 21 standard unlearning only
# Literature-comparable SSD + SalUn completion group. This deliberately reuses
# the valid g6_standard protocol (20 epochs, schedules, architectures, optimizer,
# evaluation flow, and CIFAR-100 10x10 random-disjoint split), not the
# overlap-heavy three-epoch g9/g17 setup. Fixed to seeds 0/1/2, independent of
# SEEDS, so expansion is always 2 datasets x 2 methods x 3 seeds = 12 runs.
# launch_resume_safe skips only a fully completed, exact effective config+seed
# match; partial runs and differently tagged/configured runs are preserved and
# do not suppress a launch.
group21_standard_unlearning () {
    echo "===== GROUP 21: standard SSD + SalUn (12 resume-safe runs) ====="
    local COMMON_E20="${COMMON/--n_epochs 3/--n_epochs 20}"
    local C100_STD="--dataset cifar100 --class_per_task 10 --n_tasks 10 --n_forget 3 --arch resnet34 --sparsity 0.9 --cifar100_split standard"
    local SSD_ARGS="--ssd_alpha 1.0 --ssd_lambda 1.0"
    local SALUN_ARGS="--salun_mask_ratio 0.1 --salun_target uniform --forget_iters 50"
    local TAG="standard_unlearning_ssd_salun_v1"

    for s in 0 1 2; do
        local c10="schedules/cifar10_t5_f3_fixed_seed${s}.json"
        local c100="schedules/cifar100_t10_f3_seed${s}.json"
        for m in ssd salun; do
            local extra="$SSD_ARGS"
            if [[ "$m" == "salun" ]]; then
                extra="$SALUN_ARGS"
            fi
            launch_resume_safe "std_unlearning_c10_${m}_s${s}" \
                $C10 $COMMON_E20 --seed $s --request_schedule_file $c10 \
                --method $m $extra --experiment_tag $TAG
            launch_resume_safe "std_unlearning_c100_${m}_s${s}" \
                $C100_STD $COMMON_E20 --seed $s --request_schedule_file $c100 \
                --method $m $extra --experiment_tag $TAG
        done
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
    g9_extra_baselines) group9_extra_baselines ;;
    g9b_ssd_tune) group9b_ssd_tune ;;
    g10_anchor) group10_anchor ;;
    g11_vit) group11_vit ;;
    g12_agreement) group12_agreement ;;
    g3b_bottleneck_seed1) group3b_bottleneck_seed1 ;;
    g13_shared_bottleneck) group13_shared_bottleneck ;;
    g13_bottleneck) group3b_bottleneck_seed1; group13_shared_bottleneck ;;
    g14_conflict) group14_conflict ;;
    g15_seed2) group15_seed2 ;;
    g17_overlap_curve) group17_overlap_curve ;;
    g18_probe) group18_probe ;;
    g20_paper_completion) group20_paper_completion ;;
    g21_standard_unlearning) group21_standard_unlearning ;;
    *)   echo "unknown arg: $WHICH" >&2; usage >&2; exit 1 ;;
esac

if (( DRY_RUN )); then
    if (( FAILED_RUNS > 0 )); then
        echo "[FAILED] ${FAILED_RUNS} dry-run validation check(s) failed." >&2
        exit 1
    fi
    echo "DRY-RUN complete. No training or aggregation was executed."
    exit 0
fi

echo "===================================================================="
echo "AGGREGATING TABLES ..."
if ! $PY tools/aggregate_results.py --root runs --require-metrics --seed-policy latest \
    --out results/aggregates/server_results.csv; then
    ((FAILED_AGGREGATIONS += 1))
    echo "[FAIL] aggregate_results.py" >&2
fi
if ! $PY tools/make_thesis_table.py --root runs --group-by-config \
    --seed-policy latest \
    --out-csv results/aggregates/server_thesis_table.csv \
    --out-md  results/aggregates/server_thesis_table.md; then
    ((FAILED_AGGREGATIONS += 1))
    echo "[FAIL] make_thesis_table.py" >&2
fi
if ! $PY tools/make_report_table.py \
    --input results/aggregates/server_thesis_table.csv \
    --out-csv results/aggregates/server_report_table.csv \
    --out-md  results/aggregates/server_report_table.md; then
    ((FAILED_AGGREGATIONS += 1))
    echo "[FAIL] make_report_table.py" >&2
fi

if (( FAILED_RUNS > 0 || FAILED_AGGREGATIONS > 0 )); then
    echo "[FAILED] ${FAILED_RUNS} run(s) and ${FAILED_AGGREGATIONS} aggregation step(s) failed." >&2
    echo "         Review logs/ and rerun only the affected group(s)." >&2
    exit 1
fi

echo "DONE. All runs and aggregation steps passed."
echo "Tables written under results/aggregates/ :"
echo "  server_thesis_table.md   (full)"
echo "  server_report_table.md   (compact)"
echo "Per-run logs are under logs/."
