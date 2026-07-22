import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import *
from methods import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Privacy-Aware Lifelong Learning')
parser.add_argument('--data_dir', default='./data', type=str, help='data directory')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'])
parser.add_argument('--class_per_task', default=2, type=int, help='number of classes per task in CL')
parser.add_argument('--cifar100_split', default='superclass', choices=['superclass', 'standard'],
                    help="CIFAR-100 task construction: 'superclass' (default; the 20 semantic "
                         "superclasses, 5 fine classes/task -- our overlap-heavy setting) or 'standard' "
                         "(random disjoint Split-CIFAR-100, comparable to the literature/PALL reference)")
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks in CL')
parser.add_argument('--n_forget', default=3, type=int, help='number of forget requests by the user to simulate')
parser.add_argument('--request_schedule_file', default=None, type=str,
                    help='optional JSON file with fixed request schedule')
parser.add_argument('--experiment_tag', default=None, type=str, help='optional tag for grouping experiment runs')
parser.add_argument('--arch', default='resnet18', type=str, help='neural network architecture')
parser.add_argument('--norm_params', default=False, action='store_true', help='use batch-norm params in dense models')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', default='0', type=str, help='CUDA device id')
parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                    help='compute device preference')

parser.add_argument('--n_epochs', default=20, type=int, help='number of iterations per task')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer choice')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_workers', default=None, type=int, help='DataLoader workers; defaults to 0 on macOS, 2 elsewhere')
parser.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='enable DataLoader pin_memory')
parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false', help='disable DataLoader pin_memory')

parser.add_argument(
    '--method',
    default='pall_modified',
    choices=[
        'sequential',
        'ewc',
        'lwf',
        'er',
        'derpp',
        'lsf',
        'clpu',
        'ssd',
        'salun',
        'pall',            # deprecated alias for pall_modified (warns)
        'pall_original',
        'pall_modified',   # main method (default)
        'pall_adapter',
        'lora',            # parameter-efficient LoRA baseline
    ],
    help='method for CL with unlearning (default: pall_modified, the main method)',
)
parser.add_argument('--sparsity', default=0.8, type=float, help="layer-wise sparsity for PALL")
parser.add_argument('--mem_budget', default=500, type=int, help='rehearsal memory capacity')
parser.add_argument('--mem_type', default='random', choices=['random'])
parser.add_argument('--ewc_lmbd', default=100., type=float, help='EWC lambda parameter')
parser.add_argument('--lsf_gamma', default=10.0, type=float, help='LSF gamma parameter')
parser.add_argument('--lwf_alpha', default=1.0, type=float, help='LWF alpha parameter')
parser.add_argument('--lwf_temp', default=2.0, type=float, help='LWF temp parameter')
parser.add_argument('--alpha', default=0.5, type=float, help='DERPP alpha parameter')
parser.add_argument('--beta', default=1.0, type=float, help='DERPP beta parameter')
parser.add_argument('--k_shot', default=1, type=int, help='k-shot finetuning for PALL')
parser.add_argument('--forget_iters', default=None, type=int, help='forgetting iterations for ER methods')
parser.add_argument('--ssd_alpha', default=1.0, type=float, help='SSD Fisher ratio threshold for dampening')
parser.add_argument('--ssd_lambda', default=1.0, type=float, help='SSD dampening strength multiplier')
parser.add_argument('--salun_mask_ratio', default=0.1, type=float, help='fraction of trainable weights updated by SalUn')
parser.add_argument(
    '--salun_target',
    default='uniform',
    choices=['uniform', 'random'],
    help='SalUn forget target for masked updates',
)
parser.add_argument('--deterministic', default=False, action='store_true', help='enable deterministic runs')
# PALL modified-unlearning knobs (all defaults are explicit and serialized in config.json):
# - Protection selection: protect_ratio takes precedence over protect_threshold when both are provided.
# - Retrain-step resolution: retrain_epochs (alias) > retrain_steps > k_shot.
# - If adaptive_retrain is enabled, resolved steps are scaled by overlap ratio.
parser.add_argument('--protect_ratio', default=None, type=float, help='fraction of shared params to protect')
parser.add_argument('--protect_threshold', default=None, type=float, help='abs weight threshold for protection')
parser.add_argument(
    '--protect_importance', default='gradient', choices=['gradient', 'weight', 'conflict'],
    help="pall_modified criterion for ranking critical shared params: 'conflict' "
         "(gradient-conflict energy relu(-g_forget*g_retain) on the rehearsal buffer; "
         "best under HIGH overlap), 'gradient' (|grad L_retain|; default main method), "
         "or 'weight' (legacy absolute weight magnitude, ablation)")
parser.add_argument(
    '--modified_component_mode',
    default='full',
    choices=['full', 'no_anchor', 'overlap_only', 'random_budget', 'ranking_no_overlap'],
    help="direct PALL-Modified mechanism control: full; no_anchor (run with "
         "lambda_protect=0); overlap_only (protect all structural overlap without "
         "ranking); random_budget (random structural-overlap coordinates with the same "
         "count as full); or ranking_no_overlap (retained-gradient ranking over the "
         "target subnet without structural-overlap restriction)",
)
parser.add_argument('--lambda_protect', default=0.0, type=float, help='regularization weight for protected params')
parser.add_argument(
    '--protect_anchor', default='old', choices=['old', 'reinit'],
    help="pall_modified anchor target for the critical-shared L2 penalty: 'old' "
         "anchors to pre-forget weights w_old (default); 'reinit' anchors to a "
         "fresh data-independent reinitialization sample (w_old still encodes "
         "forget-task information, so 'reinit' removes that leakage)")
parser.add_argument(
    '--adaptive_protect', default=False, action='store_true',
    help="scale lambda_protect by the measured critical-overlap ratio "
         "(stronger protection when forget/retain overlap is higher)")
parser.add_argument('--retrain_steps', default=None, type=int, help='override retrain steps for PALL unlearning')
parser.add_argument('--retrain_epochs', default=None, type=int, help='alias for retrain steps (PALL)')
parser.add_argument('--allow_zero_retrain', default=False, action='store_true',
                    help='allow retrain_steps=0 to skip finetune without fallback')
parser.add_argument('--adaptive_retrain', default=False, action='store_true', help='adapt retrain steps to overlap')
parser.add_argument('--debug_unlearning', default=False, action='store_true', help='dump unlearning artifacts')
parser.add_argument('--dump_overlap', default=False, action='store_true', help='dump overlap matrix CSV')
parser.add_argument('--eval_mia', default=False, action='store_true',
                    help='run a simple membership-inference attack before and after each '
                         'forget event (members=forget-task train samples, non-members=its '
                         'test split); writes AUC and balanced accuracy under metrics "mia"')
parser.add_argument('--eval_agreement', default=False, action='store_true',
                    help='EXPENSIVE retraining-reference audit: after a forget event, train a fresh '
                         'instance of the SAME method/architecture on the schedule with the forgotten '
                         'task never trained. Reports task-local argmax agreement, Jensen-Shannon '
                         'divergence, logit L2 distance, and feature cosine similarity under metrics '
                         '"agreement". Requires n_forget==1 (one retrain per run).')
parser.add_argument('--eval_probe', default=False, action='store_true',
                    help='linear-probe leakage audit: before and after each forget event, freeze the '
                         'model, extract the penultimate representation the deployed model exposes for '
                         'the forgotten task, train a fresh logistic-regression probe on that task\'s '
                         'TRAIN-split features, and report probe_acc_before/after on its held-out TEST '
                         'split (vs the 1/class_per_task chance level) under metrics "probe".')
parser.add_argument('--eval_bound', default=False, action='store_true',
                    help='shared-subphase first-order diagnostic (pall_adapter only): record the '
                         'fixed-gradient loss/energy quantity and, separately, measured accuracy '
                         'change. Their units are not calibrated; no satisfaction claim is made.')
parser.add_argument('--cache_features', default=False, action='store_true',
                    help='precompute the frozen backbone 512-d features once and train/eval the PEFT '
                         'head on the cached features (big speedup + memory). Only active for '
                         'pall_adapter/lora with --pretrained_backbone; features are augmentation-free.')
parser.add_argument('--adapter_bottleneck', default=16, type=int, help='adapter bottleneck size for pall_adapter')
parser.add_argument(
    '--adapter_shared_bottleneck',
    default=0,
    type=int,
    help='optional shared adapter bottleneck size for pall_adapter; 0 disables shared overlap module',
)
parser.add_argument(
    '--adapter_shared_forget_ratio',
    default=0.0,
    type=float,
    help='fraction of shared_adapter params selected as forget candidates for pall_adapter forgetting',
)
parser.add_argument(
    '--adapter_shared_protect_ratio',
    default=0.0,
    type=float,
    help='fraction of shared_adapter params protected for active tasks during pall_adapter forgetting',
)
parser.add_argument(
    '--adapter_shared_forget_lr',
    default=None,
    type=float,
    help='optional lr override for shared_adapter forgetting update in pall_adapter',
)
parser.add_argument(
    '--adapter_shared_protect_strength',
    default=None,
    type=float,
    help='optional soft protection strength for shared critical shared-adapter params in pall_adapter forgetting',
)
parser.add_argument(
    '--adapter_forget_steps',
    default=10,
    type=int,
    help='number of Phase-3 iterations of the shared-adapter uniform-target soft-masked '
         'forgetting loop in pall_adapter (1 reproduces the old single-step behaviour)',
)
parser.add_argument(
    '--adapter_forget_mode',
    default='uniform_loop',
    choices=['ascent_step', 'uniform_loop'],
    help="pall_adapter shared-adapter forgetting rule. 'uniform_loop' (default, "
         "current behaviour): iterate --adapter_forget_steps gradient-DESCENT steps "
         "toward a uniform target over the forget task's classes. 'ascent_step': a "
         "single gradient-ASCENT step on the true-label loss. Both use the same "
         "soft mask (full on S_forget_only, scaled by 1-protect_strength on "
         "S_share_crit, frozen elsewhere).",
)
parser.add_argument(
    '--adapter_component_mode',
    default='full',
    choices=['full', 'reset_only', 'reset_repair', 'uniform_unprotected', 'mask_no_ascent'],
    help="request-time PALL-Adapter component ablation. 'full' preserves the current "
         "ordered method; 'reset_only' resets only the target adapter and classifier "
         "slice; 'reset_repair' adds retained-task repair; 'uniform_unprotected' uses "
         "the same S_forget support with multiplier one instead of overlap protection; "
         "'mask_no_ascent' keeps the soft-masked shared update and repair but omits "
         "cached-gradient classifier ascent.",
)
parser.add_argument(
    '--eval_component_stages',
    default=False,
    action='store_true',
    help='evaluate and serialize PALL-Adapter accuracy after target reset, shared update, '
         'classifier ascent, and retained-task repair (intended for component ablations)',
)
parser.add_argument(
    '--adapter_mask_mode',
    default='discrete',
    choices=['discrete', 'continuous'],
    help="pall_adapter Phase-3 soft-mask type. 'discrete' (default, unchanged): the "
         "binary full/soft/frozen partition (1 on S_forget_only, 1-p on S_share_crit, "
         "0 on hard-protected/outside). 'continuous': a per-coordinate multiplier "
         "m_i = clamp(1 - gamma * c_hat_i, 0, 1) on S_forget coordinates only (frozen "
         "outside), where c_hat is the normalized gradient-conflict energy "
         "relu(-g_forget*g_retain)/max. Masks stay fixed across the Phase-3 iterations.",
)
parser.add_argument(
    '--adapter_conflict_gamma',
    default=1.0,
    type=float,
    help='strength of the continuous conflict-weighted soft mask (--adapter_mask_mode '
         'continuous): m_i = clamp(1 - gamma * c_hat_i, 0, 1). gamma=0 recovers a full '
         'update on S_forget; larger gamma suppresses high-conflict coordinates more.',
)
parser.add_argument(
    '--adapter_train_classifier',
    default=False,
    action='store_true',
    help='allow classifier/head training for pall_adapter',
)
parser.add_argument(
    '--adapter_location',
    default='residual',
    choices=['residual'],
    help='adapter insertion location for pall_adapter',
)
parser.add_argument('--lora_rank', default=8, type=int, help='LoRA rank r for the lora baseline')
parser.add_argument('--lora_alpha', default=16, type=float, help='LoRA scaling alpha for the lora baseline (scale = alpha/r)')
parser.add_argument('--pretrained_backbone', default='none', choices=['none', 'imagenet_resnet18'],
                    help="feature extractor for the PEFT methods (pall_adapter, lora): 'none' = the "
                         "from-scratch backbone (default, unchanged); 'imagenet_resnet18' = a frozen "
                         "ImageNet ResNet-18 loaded from --pretrained_weights (adapters/LoRA run on its 512-d feature)")
parser.add_argument('--pretrained_weights', default='pretrained/resnet18_imagenet.pth', type=str,
                    help='local .pth weights for --pretrained_backbone imagenet_resnet18 (offline-safe)')
parser.add_argument(
    '--pretrained_input_norm',
    default='imagenet',
    choices=['imagenet', 'legacy_dataset_stats'],
    help='input normalization seen by the ImageNet-pretrained backbone. "imagenet" '
         'undoes dataset normalization and applies ImageNet statistics; '
         '"legacy_dataset_stats" reproduces earlier CIFAR pretrained runs.',
)
parser.set_defaults(pin_memory=None)
args = parser.parse_args()

PALL_METHODS = {"pall", "pall_original", "pall_modified"}
ADAPTER_METHODS = {"pall_adapter"}
LORA_METHODS = {"lora"}
DENSE_UNLEARNING_METHODS = {"ssd", "salun"}


def normalize_method(arg_namespace):
    # `pall` is a DEPRECATED alias for `pall_modified` (the main method), kept so
    # old scripts/configs do not break. Prefer the explicit name.
    if arg_namespace.method == "pall":
        print(
            "[WARN] --method pall is a deprecated alias for pall_modified "
            "(gradient-based, the main method). Use --method pall_modified explicitly.",
            flush=True,
        )
        arg_namespace.method = "pall_modified"
    if arg_namespace.method == "pall_modified":
        arg_namespace.method_variant = "modified"
    elif arg_namespace.method == "pall_original":
        arg_namespace.method_variant = "original"
    elif arg_namespace.method == "pall_adapter":
        arg_namespace.method_variant = "adapter"
    elif arg_namespace.method in DENSE_UNLEARNING_METHODS:
        arg_namespace.method_variant = arg_namespace.method
    else:
        arg_namespace.method_variant = None
    arg_namespace.variant = derive_variant(arg_namespace)


def derive_variant(arg_namespace):
    """Human-readable label that captures method + the flag combination.

    Stored in config.json (and surfaced by tools/aggregate_results.py) so every
    run self-describes and result tables can separate variants without parsing
    flags. Paper-name mapping lives in README ("Method taxonomy").

      pall_original                -> 'pall_original'        (PALL-Original)
      pall_modified (grad, prot.)  -> 'pall_modified_grad'   (PALL-Modified, MAIN)
      pall_modified (weight, prot.)-> 'pall_modified_weight' (PALL-Modified-W, ablation)
      pall_modified (no protection)-> 'pall_modified_noprotect'
      pall_adapter (no shared)     -> 'adapter_reset'        (PALL-Adapter reset baseline)
      pall_adapter (shared, p=0)   -> 'adapter_shared'       (shared, no protection)
      pall_adapter (shared, p>0)   -> 'adapter_protected'    (shared critical-protection)
      <baseline>                   -> '<method>'             (er, derpp, ...)
    """
    method = arg_namespace.method
    if method == "pall_modified":
        component_mode = getattr(arg_namespace, "modified_component_mode", "full")
        if component_mode == "no_anchor":
            return "pall_modified_no_anchor"
        has_target = (
            arg_namespace.protect_ratio is not None
            or arg_namespace.protect_threshold is not None
        )
        protecting = has_target and (arg_namespace.lambda_protect or 0.0) > 0.0
        if not protecting:
            return "pall_modified_noprotect"
        importance = getattr(arg_namespace, "protect_importance", "gradient")
        label = {
            "gradient": "pall_modified_grad",
            "weight": "pall_modified_weight",
            "conflict": "pall_modified_conflict",
        }.get(importance, "pall_modified_grad")
        if getattr(arg_namespace, "adaptive_protect", False):
            label += "_adapt"
        if component_mode != "full":
            label += f"_{component_mode}"
        return label
    if method == "pall_adapter":
        if (arg_namespace.adapter_shared_bottleneck or 0) <= 0 or (
            arg_namespace.adapter_shared_forget_ratio or 0.0
        ) <= 0.0:
            return "adapter_reset"
        if (arg_namespace.adapter_shared_protect_ratio or 0.0) <= 0.0:
            return "adapter_shared"
        label = "adapter_protected"
        if getattr(arg_namespace, "protect_importance", "gradient") == "conflict":
            label += "_conflict"
        return label
    if method == "lora":
        return f"lora_r{getattr(arg_namespace, 'lora_rank', 8)}"
    if method == "ssd":
        return (
            f"ssd_a{getattr(arg_namespace, 'ssd_alpha', 1.0):g}"
            f"_l{getattr(arg_namespace, 'ssd_lambda', 1.0):g}"
        )
    if method == "salun":
        return (
            f"salun_{getattr(arg_namespace, 'salun_target', 'uniform')}"
            f"_m{getattr(arg_namespace, 'salun_mask_ratio', 0.1):g}"
        )
    return method


def validate_experiment_args(arg_namespace):
    modified_component_mode = getattr(arg_namespace, "modified_component_mode", "full")
    if modified_component_mode != "full" and arg_namespace.method != "pall_modified":
        parser.error("--modified_component_mode controls are only valid for pall_modified.")
    if modified_component_mode == "no_anchor" and (arg_namespace.lambda_protect or 0.0) != 0.0:
        parser.error("--modified_component_mode no_anchor requires --lambda_protect 0.")
    if arg_namespace.dataset == "cifar100":
        if getattr(arg_namespace, "cifar100_split", "superclass") == "superclass":
            if arg_namespace.class_per_task != 5:
                parser.error("CIFAR-100 superclass tasks require --class_per_task 5 "
                             "(use --cifar100_split standard for arbitrary class_per_task).")
            if not (1 <= arg_namespace.n_tasks <= 20):
                parser.error("CIFAR-100 superclass tasks require --n_tasks in [1, 20].")
        else:  # standard Split-CIFAR-100: random disjoint class splits
            if arg_namespace.class_per_task * arg_namespace.n_tasks > 100:
                parser.error("Standard Split-CIFAR-100 requires --class_per_task * --n_tasks <= 100.")
    if arg_namespace.dataset == "tinyimagenet":
        if arg_namespace.class_per_task * arg_namespace.n_tasks > 200:
            parser.error("TinyImageNet requires --class_per_task * --n_tasks <= 200.")
    if arg_namespace.num_workers is not None and arg_namespace.num_workers < 0:
        parser.error("--num_workers must be >= 0.")
    if arg_namespace.method == "pall_adapter" and arg_namespace.adapter_bottleneck <= 0:
        parser.error("--adapter_bottleneck must be > 0 for pall_adapter.")
    if arg_namespace.method == "pall_adapter" and arg_namespace.adapter_shared_bottleneck < 0:
        parser.error("--adapter_shared_bottleneck must be >= 0 for pall_adapter.")
    if arg_namespace.method == "pall_adapter":
        if not (0.0 <= arg_namespace.adapter_shared_forget_ratio <= 1.0):
            parser.error("--adapter_shared_forget_ratio must be in [0, 1] for pall_adapter.")
        if not (0.0 <= arg_namespace.adapter_shared_protect_ratio <= 1.0):
            parser.error("--adapter_shared_protect_ratio must be in [0, 1] for pall_adapter.")
        if (
            arg_namespace.adapter_shared_forget_lr is not None
            and arg_namespace.adapter_shared_forget_lr <= 0.0
        ):
            parser.error("--adapter_shared_forget_lr must be > 0 when provided for pall_adapter.")
    if arg_namespace.method == "ssd":
        if arg_namespace.ssd_alpha < 0.0:
            parser.error("--ssd_alpha must be >= 0.")
        if arg_namespace.ssd_lambda < 0.0:
            parser.error("--ssd_lambda must be >= 0.")
    if arg_namespace.method == "salun":
        if not (0.0 <= arg_namespace.salun_mask_ratio <= 1.0):
            parser.error("--salun_mask_ratio must be in [0, 1].")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def init_run_dir(arg_namespace):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_seq = f"T{arg_namespace.n_tasks}_F{arg_namespace.n_forget}"
    run_dir = Path("runs") / arg_namespace.dataset / task_seq / arg_namespace.method
    run_dir = run_dir / f"seed_{arg_namespace.seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir, timestamp


def init_logger(run_dir):
    logger = logging.getLogger("pall")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "events.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_event(logger, message):
    print(message)
    if logger is not None:
        logger.info(message)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def serialize_config(arg_namespace, run_dir, timestamp):
    config = vars(arg_namespace).copy()
    config["device"] = str(arg_namespace.device)
    config["run_dir"] = str(run_dir)
    config["timestamp"] = timestamp
    return config


def _coerce_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_model_param_stats(model):
    stats = getattr(model, "param_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    net = getattr(model, "net", None)
    if net is None:
        return None
    total_params = _coerce_int_or_none(stats.get("total_params"))
    if total_params is None:
        total_params = sum(param.numel() for param in net.parameters())

    trainable_params = _coerce_int_or_none(stats.get("num_trainable_params"))
    if trainable_params is None:
        trainable_params = sum(param.numel() for param in net.parameters() if param.requires_grad)

    model_stats = dict(stats)
    model_stats["total_params"] = int(total_params)
    model_stats["num_trainable_params"] = int(trainable_params)
    model_stats["trainable_param_ratio"] = float(trainable_params / total_params) if total_params else 0.0

    has_adapters = hasattr(net, "count_adapter_params") or any(
        key in model_stats
        for key in ("num_adapter_params", "task_adapter_params", "shared_adapter_params", "adapter_params")
    )
    adapter_stats = None
    if has_adapters:
        task_adapter_params = _coerce_int_or_none(model_stats.get("task_adapter_params"))
        if task_adapter_params is None:
            task_adapter_params = _coerce_int_or_none(model_stats.get("num_adapter_params"))
        if task_adapter_params is None and hasattr(net, "count_adapter_params"):
            task_adapter_params = int(net.count_adapter_params())
        if task_adapter_params is None:
            task_adapter_params = 0

        shared_adapter_params = _coerce_int_or_none(model_stats.get("shared_adapter_params"))
        if shared_adapter_params is None and hasattr(net, "count_shared_adapter_params"):
            shared_adapter_counts = net.count_shared_adapter_params()
            if isinstance(shared_adapter_counts, tuple):
                shared_adapter_params = int(shared_adapter_counts[0])
            else:
                shared_adapter_params = int(shared_adapter_counts)
        if shared_adapter_params is None:
            shared_adapter_params = 0

        adapter_params = _coerce_int_or_none(model_stats.get("adapter_params"))
        if adapter_params is None:
            adapter_params = int(task_adapter_params + shared_adapter_params)

        adapter_param_ratio = float(adapter_params / total_params) if total_params else 0.0
        adapter_stats = {
            "total_model_params": int(total_params),
            "trainable_params": int(trainable_params),
            "adapter_params": int(adapter_params),
            "shared_adapter_params": int(shared_adapter_params),
            "task_adapter_params": int(task_adapter_params),
            "adapter_param_ratio": adapter_param_ratio,
            "shared_adapter_ratio": float(shared_adapter_params / adapter_params) if adapter_params else 0.0,
            "updated_param_ratio": None,
        }
        model_stats["num_adapter_params"] = int(task_adapter_params)
        model_stats["shared_adapter_params"] = int(shared_adapter_params)
        model_stats["task_adapter_params"] = int(task_adapter_params)
        model_stats["adapter_params"] = int(adapter_params)
        model_stats["adapter_param_ratio"] = adapter_param_ratio
        model_stats["shared_adapter_ratio"] = float(shared_adapter_params / adapter_params) if adapter_params else 0.0
    else:
        model_stats["num_adapter_params"] = int(_coerce_int_or_none(model_stats.get("num_adapter_params")) or 0)
        model_stats["shared_adapter_params"] = int(_coerce_int_or_none(model_stats.get("shared_adapter_params")) or 0)

    return {
        "model_stats": json_safe(model_stats),
        "adapter_stats": json_safe(adapter_stats) if adapter_stats is not None else None,
    }


normalize_method(args)
validate_experiment_args(args)


def resolve_device(arg_namespace):
    """Select an execution device with macOS MPS support."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if arg_namespace.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = arg_namespace.gpu
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but no CUDA devices are available.")

    if arg_namespace.device == "mps":
        if mps_available:
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available on this system.")

    if arg_namespace.device == "cpu":
        return torch.device("cpu")

    # Auto mode: prefer CUDA, then MPS, then CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = arg_namespace.gpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def resolve_num_workers(arg_namespace):
    if arg_namespace.num_workers is not None:
        return int(arg_namespace.num_workers)
    if sys.platform == "darwin":
        return 0
    return 2


def resolve_pin_memory(arg_namespace):
    if arg_namespace.pin_memory is not None:
        return bool(arg_namespace.pin_memory)
    if sys.platform == "darwin":
        return False
    return arg_namespace.device.type == "cuda"


args.device = resolve_device(args)
args.is_macos = sys.platform == "darwin"
args.resolved_num_workers = resolve_num_workers(args)
args.resolved_pin_memory = resolve_pin_memory(args)
args.resolved_persistent_workers = False if args.resolved_num_workers > 0 else False
dataset_metadata = get_dataset_metadata(args.dataset)
args.num_classes = dataset_metadata["num_classes"]
args.image_size = dataset_metadata["image_size"]
args.channels = dataset_metadata["channels"]
if args.method in PALL_METHODS:
    args.arch = 'subnet_' + args.arch.lower()
elif args.method in ADAPTER_METHODS:
    args.arch = 'adapter_' + args.arch.lower()
elif args.method in LORA_METHODS:
    args.arch = 'lora_' + args.arch.lower()
elif args.method in DENSE_UNLEARNING_METHODS:
    args.arch = args.arch.lower()
else:
    args.arch = args.arch.lower()
args.dim_input = (args.channels, args.image_size, args.image_size)


def evaluate(test_datasets, args, model, return_logits=True, verbose=True):
    model.eval_mode()
    L, A = torch.zeros(args.n_tasks), torch.zeros(args.n_tasks)
    logits = [] if return_logits else None
    cpt = args.class_per_task
    with torch.no_grad():
        for task, dataset in enumerate(test_datasets):
            bsize = args.batch_size
            if hasattr(model, "build_dataloader"):
                loader = model.build_dataloader(dataset, batch_size=bsize, shuffle=False, context=f"eval_task_{task}")
            else:
                loader = DataLoader(dataset, batch_size=bsize, shuffle=False)
            l = a = n = 0.0
            logit_ = torch.zeros(len(dataset), cpt) if return_logits else None
            for i, (x, y) in enumerate(loader):
                x_tensor, y_tensor = x.to(args.device), y.to(args.device)
                y_ = model.evaluate(x_tensor, task)
                l += F.cross_entropy(y_, y_tensor, reduction='sum').item()
                a += y_.argmax(-1).eq(y_tensor).float().sum().item()
                if return_logits:
                    logit_[i * bsize:i * bsize + y_tensor.shape[0]].copy_(
                        y_[..., cpt * task:cpt * (task + 1)].cpu()
                    )
                n += y_tensor.shape[0]

            L[task], A[task] = l / n, a / n
            if return_logits:
                logits.append(logit_)

    model.train_mode()
    if verbose:
        print("[INFO] loss: ", L)
        print("[INFO] acc.: ", A)

    return {
        'loss': L,
        'accuracy': A,
        'logits': logits,
    }


def to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def avg_for_tasks(acc_list, task_ids):
    if not task_ids:
        return 0.0
    return float(sum(acc_list[t] for t in task_ids) / len(task_ids))


def acc_list_to_dict(acc_list):
    return {str(idx): float(acc) for idx, acc in enumerate(acc_list)}


def to_optional_float(value):
    if value is None:
        return None
    return float(value)


def to_optional_int(value):
    if value is None:
        return None
    return int(value)


def format_optional_float(value, precision=4):
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def format_optional_int(value):
    if value is None:
        return "NA"
    return str(int(value))


def format_optional_text(value):
    if value is None:
        return "NA"
    if isinstance(value, str) and value == "":
        return "NA"
    return str(value)


def first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def compute_average_forgetting(accuracy_history, requests, n_tasks):
    """
    Standard CL forgetting at step r for task t:
      F_t(r) = max_{k<r} a_t(k) - a_t(r)
    where a_t(k) is task-t accuracy at request k.

    We average over tasks that have been trained at least once and have at least
    one past accuracy point before the current request.
    """
    if torch.is_tensor(accuracy_history):
        acc_history = accuracy_history.detach().cpu().tolist()
    else:
        acc_history = to_list(accuracy_history)

    if not acc_history:
        return {
            "definition": "avg_t(max_{k<r} a_t(k) - a_t(r)) over trained tasks with at least one past point",
            "per_request": [],
            "per_task": [],
            "final": 0.0,
        }

    best_past = [None] * n_tasks
    trained_tasks = set()
    per_request = []
    per_task = []

    for request_id, (task_id, learn_type, _) in enumerate(requests):
        curr_acc = acc_history[request_id]
        if learn_type == "T":
            trained_tasks.add(int(task_id))

        task_forgetting = {}
        forgetting_vals = []
        for task_id in sorted(trained_tasks):
            best = best_past[task_id]
            curr = float(curr_acc[task_id])
            if best is not None:
                f_val = float(best - curr)
                task_forgetting[str(task_id)] = f_val
                forgetting_vals.append(f_val)

        per_task.append(task_forgetting)
        per_request.append(float(sum(forgetting_vals) / len(forgetting_vals)) if forgetting_vals else 0.0)

        for task_id in trained_tasks:
            curr = float(curr_acc[task_id])
            best = best_past[task_id]
            if best is None or curr > best:
                best_past[task_id] = curr

    final_forgetting = float(per_request[-1]) if per_request else 0.0
    return {
        "definition": "avg_t(max_{k<r} a_t(k) - a_t(r)) over trained tasks with at least one past point",
        "per_request": per_request,
        "per_task": per_task,
        "final": final_forgetting,
    }


def normalize_unlearning_event(event, availability=None):
    availability = availability or {}
    overlap = event.get("overlap", {}) or {}
    mia = event.get("mia")
    agreement = event.get("agreement")
    probe = event.get("probe")
    bound_check = event.get("bound_check")
    conflict_mask = event.get("conflict_mask")

    t_reset = to_optional_float(event.get("t_reset")) if availability.get("t_reset", True) else None
    t_retrain = to_optional_float(event.get("t_retrain")) if availability.get("t_retrain", True) else None
    t_forget_total = to_optional_float(event.get("t_forget_total"))
    if t_forget_total is None and (t_reset is not None or t_retrain is not None):
        t_forget_total = (t_reset or 0.0) + (t_retrain or 0.0)

    return {
        "unlearning_step": to_optional_int(event.get("unlearning_step")),
        "request_id": to_optional_int(event.get("request_id")),
        "task_id": to_optional_int(event.get("task_id")),
        "Fu": to_optional_float(event.get("Fu")),
        "WorstDrop": to_optional_float(event.get("WorstDrop")),
        "Au": to_optional_float(event.get("Au")),
        "grad_norm_ratio": to_optional_float(event.get("grad_norm_ratio")),
        "avg_before": to_optional_float(event.get("avg_before")),
        "avg_after_reset": to_optional_float(event.get("avg_after_reset")),
        "avg_after_retrain": to_optional_float(event.get("avg_after_retrain")),
        "t_reset": t_reset,
        "t_retrain": t_retrain,
        "t_forget_total": t_forget_total,
        "num_updated_params": (
            to_optional_int(event.get("num_updated_params")) if availability.get("num_updated_params", True) else None
        ),
        "overlap": {
            "s_t": to_optional_int(overlap.get("s_t")) if availability.get("overlap", True) else None,
            "s_share": to_optional_int(overlap.get("s_share")) if availability.get("overlap", True) else None,
            "s_share_crit": to_optional_int(overlap.get("s_share_crit")) if availability.get("overlap", True) else None,
            "s_share_ratio": (
                to_optional_float(overlap.get("s_share_ratio")) if availability.get("overlap", True) else None
            ),
            "s_share_crit_ratio": (
                to_optional_float(overlap.get("s_share_crit_ratio")) if availability.get("overlap", True) else None
            ),
        },
        "shared_adapter": {
            "adapter_shared_forget_ratio": (
                to_optional_float(event.get("adapter_shared_forget_ratio"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "adapter_shared_protect_ratio": (
                to_optional_float(event.get("adapter_shared_protect_ratio"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_protect_strength": (
                to_optional_float(event.get("shared_protect_strength"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_adapter_params": (
                to_optional_int(event.get("shared_adapter_params"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "classifier_param_count": (
                to_optional_int(event.get("classifier_param_count"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "classifier_forget_param_count": (
                to_optional_int(event.get("classifier_forget_param_count"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_forget_candidates": (
                to_optional_int(event.get("shared_forget_candidates"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_protected_params": (
                to_optional_int(event.get("shared_protected_params"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_active_critical": (
                to_optional_int(event.get("shared_active_critical"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_overlap_critical": (
                to_optional_int(event.get("shared_overlap_critical"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_effective_forget_params": (
                to_optional_int(event.get("shared_effective_forget_params"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_full_update_params": (
                to_optional_int(event.get("shared_full_update_params"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_soft_update_params": (
                to_optional_int(event.get("shared_soft_update_params"))
                if availability.get("shared_adapter", True)
                else None
            ),
            "shared_s_share_crit": (
                to_optional_int(event.get("shared_s_share_crit"))
                if availability.get("shared_adapter", True)
                else None
            ),
        },
        "mia": json_safe(mia) if isinstance(mia, dict) else None,
        "agreement": json_safe(agreement) if isinstance(agreement, dict) else None,
        "probe": json_safe(probe) if isinstance(probe, dict) else None,
        "bound_check": json_safe(bound_check) if isinstance(bound_check, dict) else None,
        "conflict_mask": json_safe(conflict_mask) if isinstance(conflict_mask, dict) else None,
    }


def _mia_auc(member_scores, nonmember_scores):
    """AUC of a membership score (higher == more member-like), computed from the
    rank statistic (= probability a random member outscores a random non-member,
    the Mann-Whitney U form). Returns None if either group is empty."""
    # Keep optional audit helpers lazy: ordinary training/component runs must not
    # fail at process startup merely because privacy-only support was not copied.
    from privacy_metrics import roc_auc_from_scores
    return roc_auc_from_scores(member_scores, nonmember_scores)


def _mia_balanced_accuracy(member_scores, nonmember_scores):
    """Best balanced accuracy over score thresholds (predict member when
    score >= threshold). Returns None if either group is empty."""
    member = np.asarray(member_scores, dtype=np.float64)
    nonmember = np.asarray(nonmember_scores, dtype=np.float64)
    if member.size == 0 or nonmember.size == 0:
        return None
    best = 0.0
    for thr in np.unique(np.concatenate([member, nonmember])):
        tpr = float((member >= thr).mean())
        tnr = float((nonmember < thr).mean())
        best = max(best, 0.5 * (tpr + tnr))
    return float(best)


def _mia_per_sample_scores(model, dataset, task_id, args, max_samples=512):
    """Per-sample (loss, max-softmax confidence) for `dataset` under `model`,
    via the model-agnostic ``evaluate(x, task)`` head. Capped at `max_samples`."""
    model.eval_mode()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    losses, confidences, seen = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model.evaluate(x, task_id)
            losses.append(F.cross_entropy(logits, y, reduction="none").detach().cpu())
            confidences.append(torch.softmax(logits, dim=1).max(dim=1).values.detach().cpu())
            seen += int(x.size(0))
            if seen >= max_samples:
                break
    if not losses:
        return np.zeros(0), np.zeros(0)
    return torch.cat(losses).numpy(), torch.cat(confidences).numpy()


def _mia_loss_over_confidence_scores(losses, confidences):
    raw_scores = np.asarray(losses, dtype=np.float64) / np.maximum(
        np.asarray(confidences, dtype=np.float64),
        1e-12,
    )
    # Lower loss/confidence means a sample looks more member-like, so negate for
    # the shared AUC/threshold helpers where higher scores mean "member".
    return -raw_scores


def _dump_mia_scores(args, task_id, phase, unlearning_step, member_scores, nonmember_scores):
    """Persist the raw per-sample membership scores for an empirical privacy audit.

    Writes ``{run_dir}/mia_scores/step{unlearning_step}_task{task_id}_{phase}.json``
    (phase is ``before``/``after``). Gated by the caller behind ``--eval_mia``; the
    aggregated auc/acc "mia" block in metrics.json is unaffected. Scores are the
    (higher == more member-like) arrays from ``_mia_loss_over_confidence_scores``.
    """
    run_dir = getattr(args, "run_dir", None)
    if not run_dir:
        return None
    out_dir = os.path.join(run_dir, "mia_scores")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"step{unlearning_step}_task{task_id}_{phase}.json")
    payload = {
        "unlearning_step": int(unlearning_step) if unlearning_step is not None else None,
        "task_id": int(task_id),
        "phase": phase,
        "score": "loss_over_confidence",
        "score_definition": "cross_entropy_loss / max_softmax_confidence; higher stored score is more member-like",
        "n_members": int(np.asarray(member_scores).size),
        "n_nonmembers": int(np.asarray(nonmember_scores).size),
        "member_scores": [float(v) for v in np.asarray(member_scores, dtype=np.float64).ravel()],
        "nonmember_scores": [float(v) for v in np.asarray(nonmember_scores, dtype=np.float64).ravel()],
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle)
    return out_path


def compute_mia(model, member_dataset, nonmember_dataset, task_id, args, max_samples=512,
                dump_phase=None, unlearning_step=None):
    """Simple membership-inference attack for one forget target.

    Members = the forgotten task's own (training) samples the model was exposed to;
    non-members = that task's held-out test split. Successful unlearning drives the
    attack toward chance (AUC ~ 0.5). The raw score is per-sample loss divided
    by max-softmax confidence; lower raw scores are more member-like. Model-
    agnostic: works through ``model.evaluate`` for all methods.

    When ``dump_phase`` is set ("before"/"after") the raw per-sample member/
    non-member scores are also written to ``{run_dir}/mia_scores/`` for the
    empirical privacy audit (tools/audit_privacy.py); the returned aggregate dict
    is unchanged."""
    member_dataset = make_augmentation_free_evaluation_view(member_dataset, args.dataset)
    m_loss, m_conf = _mia_per_sample_scores(model, member_dataset, task_id, args, max_samples)
    n_loss, n_conf = _mia_per_sample_scores(model, nonmember_dataset, task_id, args, max_samples)
    member_scores = _mia_loss_over_confidence_scores(m_loss, m_conf)
    nonmember_scores = _mia_loss_over_confidence_scores(n_loss, n_conf)
    if dump_phase is not None:
        _dump_mia_scores(args, task_id, dump_phase, unlearning_step, member_scores, nonmember_scores)
    return {
        "auc": _mia_auc(member_scores, nonmember_scores),
        "acc": _mia_balanced_accuracy(member_scores, nonmember_scores),
        "score": "loss_over_confidence",
        "score_definition": "cross_entropy_loss / max_softmax_confidence; lower raw score is more member-like",
        "n_members": int(m_loss.size),
        "n_nonmembers": int(n_loss.size),
    }


def build_mia_event(before, after):
    if before is None or after is None:
        return None
    return {
        "auc_before": before.get("auc"),
        "auc_after": after.get("auc"),
        "acc_before": before.get("acc"),
        "acc_after": after.get("acc"),
        "score": "loss_over_confidence",
        "score_definition": "cross_entropy_loss / max_softmax_confidence; lower raw score is more member-like",
        "before": before,
        "after": after,
    }


def _probe_extract_features(model, dataset, task_id, args, max_samples):
    """Frozen penultimate features (flattened) + labels for ``dataset`` under the
    model's DEPLOYED forward path (``model.evaluate_features``). No gradients."""
    model.eval_mode()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    feats, labels, seen = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            f = model.evaluate_features(x, task_id)
            feats.append(f.detach().float().reshape(f.shape[0], -1).cpu())
            labels.append(y.detach().cpu())
            seen += int(x.size(0))
            if seen >= max_samples:
                break
    if not feats:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.int64)
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(labels).numpy()


def _sgd_logreg_predict(x_tr, y_tr, x_te, n_classes, epochs=200, lr=0.1):
    """Fallback multinomial logistic regression: 200 epochs of full-batch SGD."""
    x_train = torch.tensor(x_tr, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    x_test = torch.tensor(x_te, dtype=torch.float32)
    weight = torch.zeros(x_train.shape[1], n_classes, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.SGD([weight, bias], lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(x_train @ weight + bias, y_train).backward()
        opt.step()
    with torch.no_grad():
        return (x_test @ weight + bias).argmax(dim=1).numpy()


def _train_linear_probe(x_train, y_train, x_test, y_test):
    """Fit a fresh logistic-regression probe on frozen (standardized) train
    features; return its accuracy on held-out test features. sklearn if available,
    else a 200-epoch SGD logistic regression."""
    classes = sorted(int(v) for v in set(int(v) for v in y_train))
    remap = {c: i for i, c in enumerate(classes)}
    y_tr = np.asarray([remap[int(v)] for v in y_train], dtype=np.int64)
    y_te = np.asarray([remap.get(int(v), -1) for v in y_test], dtype=np.int64)  # unseen -> never matches
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    # standardize, then sanitize any non-finite feature values so the probe solver
    # never sees inf/nan (real penultimate features can occasionally be extreme).
    x_tr = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    x_te = np.nan_to_num((x_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        backend = "sklearn"
    except Exception:
        pred = _sgd_logreg_predict(x_tr, y_tr, x_te, len(classes))
        backend = "sgd200"
    acc = float((pred == y_te).mean()) if len(y_te) else None
    return acc, backend


def compute_probe(model, train_dataset, test_dataset, task_id, args,
                  max_train=2000, max_test=1000):
    """Linear-probe leakage audit for one forget target: how linearly-decodable the
    forgotten task's labels remain in the frozen penultimate representation. A fresh
    probe is trained on the task's TRAIN-split features and scored on its held-out
    TEST split (so this measures generalizable leakage, not probe memorization).
    Chance = 1/class_per_task; successful unlearning drives probe_acc toward it."""
    train_dataset = make_augmentation_free_evaluation_view(train_dataset, args.dataset)
    x_tr, y_tr = _probe_extract_features(model, train_dataset, task_id, args, max_train)
    x_te, y_te = _probe_extract_features(model, test_dataset, task_id, args, max_test)
    if x_tr.shape[0] == 0 or x_te.shape[0] == 0:
        return None
    acc, backend = _train_linear_probe(x_tr, y_tr, x_te, y_te)
    n_classes = len(set(int(v) for v in y_tr))
    return {
        "probe_acc": acc,
        "chance": 1.0 / max(1, n_classes),
        "n_classes": int(n_classes),
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "feature_dim": int(x_tr.shape[1]),
        "backend": backend,
        "eval_split": "task_test",
        "representation": (
            "deployed penultimate features via model.evaluate_features: "
            "pall_adapter = backbone + shared/task adapter output; subnet methods = "
            "task-masked subnet before forgetting, mask-free (no_mask) full-network after"
        ),
    }


def build_probe_event(before, after):
    if before is None or after is None:
        return None
    return {
        "acc_before": before.get("probe_acc"),
        "acc_after": after.get("probe_acc"),
        "chance": first_non_none(before.get("chance"), after.get("chance")),
        "eval_split": before.get("eval_split"),
        "representation": before.get("representation"),
        "before": before,
        "after": after,
    }


def _compare_models_on_task(unlearned_model, reference, dataset, task, args):
    """Compare two models on identical samples using the task-local output slice."""
    # This dependency is needed only by --eval_agreement / g24, not by every
    # training run launched through main.py.
    from reference_metrics import paired_reference_batch_sums
    unlearned_model.eval_mode()
    reference.eval_mode()
    loader = unlearned_model.build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        context=f"retraining_reference_task_{task}",
    )
    totals = {
        "n": 0,
        "agreement_sum": 0.0,
        "js_sum": 0.0,
        "logit_l2_sum": 0.0,
        "feature_n": 0,
        "feature_cosine_sum": 0.0,
    }
    start = int(task * args.class_per_task)
    end = int((task + 1) * args.class_per_task)
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(args.device)
            logits_unlearned = unlearned_model.evaluate(x, task)[:, start:end]
            logits_reference = reference.evaluate(x, task)[:, start:end]
            features_unlearned = unlearned_model.evaluate_features(x, task)
            features_reference = reference.evaluate_features(x, task)
            batch = paired_reference_batch_sums(
                logits_unlearned,
                logits_reference,
                features_unlearned,
                features_reference,
            )
            for key in totals:
                totals[key] += batch[key]

    n = int(totals["n"])
    feature_n = int(totals["feature_n"])
    if n == 0:
        return None
    return {
        "n_samples": n,
        "agreement": float(totals["agreement_sum"] / n),
        "js_divergence": float(totals["js_sum"] / n),
        "logit_l2": float(totals["logit_l2_sum"] / n),
        "feature_cosine": (
            float(totals["feature_cosine_sum"] / feature_n) if feature_n else None
        ),
        "feature_n_samples": feature_n,
    }


def compute_model_agreement(unlearned_model, args, requests, train_datasets, test_datasets,
                            forgotten_task_id, active_tasks, logger):
    """Compare unlearning with a same-method retrain that never sees the target task.

    The caller gates this expensive audit to ``n_forget==1``. Matching the method
    and architecture avoids attributing ordinary cross-method differences to the
    forgetting operation.
    """
    # Reference: same seed, method, architecture and training hyperparameters.
    set_seed(args.seed, getattr(args, "deterministic", False))
    reference = type(unlearned_model)(args).to(args.device)
    if hasattr(unlearned_model.net, "features_are_precomputed"):
        reference.net.features_are_precomputed = bool(
            getattr(unlearned_model.net, "features_are_precomputed", False)
        )

    # Train on every "T"-request task in schedule order, except the forgotten one.
    train_task_ids = []
    for task_id, learn_type, _active in requests:
        if learn_type == "T" and task_id != forgotten_task_id and task_id not in train_task_ids:
            train_task_ids.append(task_id)
    log_event(
        logger,
        f"[INFO] retraining-reference audit: training fresh "
        f"{type(unlearned_model).__name__} on tasks={train_task_ids} "
        f"(forgotten task {forgotten_task_id} never trained)",
    )
    for task_id in train_task_ids:
        reference.learn(task_id, train_datasets[task_id])

    task_ids = [forgotten_task_id] + [t for t in active_tasks if t != forgotten_task_id]
    per_task = {
        str(task): _compare_models_on_task(
            unlearned_model, reference, test_datasets[task], task, args
        )
        for task in task_ids
    }
    forget_metrics = per_task.get(str(forgotten_task_id))
    retained_metrics = [
        per_task[str(task)] for task in active_tasks
        if per_task.get(str(task)) is not None
    ]

    def _retained_mean(key):
        values = [metrics[key] for metrics in retained_metrics if metrics.get(key) is not None]
        return float(np.mean(values)) if values else None

    unlearned_model.train_mode()
    del reference
    return {
        # Keep the original agreement keys for backward-compatible aggregation.
        "agreement_forget": forget_metrics.get("agreement") if forget_metrics else None,
        "agreement_retained_mean": _retained_mean("agreement"),
        "js_forget": forget_metrics.get("js_divergence") if forget_metrics else None,
        "js_retained_mean": _retained_mean("js_divergence"),
        "logit_l2_forget": forget_metrics.get("logit_l2") if forget_metrics else None,
        "logit_l2_retained_mean": _retained_mean("logit_l2"),
        "feature_cosine_forget": forget_metrics.get("feature_cosine") if forget_metrics else None,
        "feature_cosine_retained_mean": _retained_mean("feature_cosine"),
        "reference": "same_method_retrain_without_forget",
        "reference_method_class": type(unlearned_model).__name__,
        "reference_trained_tasks": train_task_ids,
        "forgotten_task_id": int(forgotten_task_id),
        "n_retained": len(retained_metrics),
        "per_task": per_task,
        "metric_notes": {
            "task_local_logits": True,
            "js_divergence": "symmetric Jensen-Shannon divergence; lower is closer",
            "logit_l2": "mean per-sample raw-logit L2; scale-sensitive; lower is closer",
            "feature_cosine": "mean per-sample penultimate-feature cosine; higher is closer",
        },
    }


def process_requests(args, model, train_datasets, test_datasets, requests, run_context):
    forgotten_tasks = []
    loss = torch.zeros(len(requests), args.n_tasks)
    accuracy = torch.zeros(len(requests), args.n_tasks)
    times = torch.zeros(len(requests))
    forgotten_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    active_tasks_mask = torch.zeros(len(requests), args.n_tasks)
    logits = [torch.zeros(len(requests), len(ds), args.class_per_task) for ds in test_datasets]

    logger = run_context.get("logger")
    metrics_state = run_context.get("metrics_state")
    metrics_path = run_context.get("metrics_path")
    debug_dir = run_context.get("debug_dir")

    unlearning_step = 0

    for request_id, (task_id, learn_type, active_tasks) in enumerate(requests):
        log_event(logger, "============================================================")
        learn_type_str = {"T": "Training", "F": "Forgetting"}[learn_type]
        log_event(logger, f"[INFO] {learn_type_str} Task {task_id} ...")

        if learn_type == "F":
            forgotten_tasks.append(task_id)

        if learn_type == "F":
            pre_eval = evaluate(test_datasets, args, model, return_logits=False, verbose=False)
            pre_acc = to_list(pre_eval["accuracy"])
            remaining_tasks = list(active_tasks)
            avg_before = avg_for_tasks(pre_acc, remaining_tasks)
            forget_phase_start = time.perf_counter()
            log_event(logger, f"[INFO] forgetting start: task={task_id} remaining_tasks={remaining_tasks}")

            mia_before = None
            if getattr(args, "eval_mia", False):
                mia_before = compute_mia(model, train_datasets[task_id], test_datasets[task_id], task_id, args,
                                         dump_phase="before", unlearning_step=unlearning_step)
                log_event(logger, f"[INFO] MIA before forget: task={task_id} {mia_before}")

            probe_before = None
            if getattr(args, "eval_probe", False):
                probe_before = compute_probe(model, train_datasets[task_id], test_datasets[task_id], task_id, args)
                log_event(logger, f"[INFO] linear probe before forget: task={task_id} {probe_before}")

            def eval_callback(stage):
                return evaluate(test_datasets, args, model, return_logits=False, verbose=False)

            if hasattr(model, "forget_with_diagnostics"):
                if task_id not in model.task_status:
                    raise AssertionError(f"[ERROR] {task_id} was not learned")
                model.task_status[task_id] = "F"
                debug_context = None
                if args.debug_unlearning:
                    debug_context = {
                        "debug_dir": str(debug_dir),
                        "request_id": request_id,
                        "task_id": task_id,
                        "unlearning_step": unlearning_step,
                    }
                info = model.forget_with_diagnostics(
                    task_id,
                    eval_fn=eval_callback,
                    debug_context=debug_context,
                    remaining_tasks=remaining_tasks,
                )
            else:
                t0 = time.perf_counter()
                model.privacy_aware_lifelong_learning(task_id, train_datasets[task_id], learn_type)
                t1 = time.perf_counter()
                info = {
                    "t_reset": None,
                    "t_retrain": None,
                    "t_forget_total": t1 - t0,
                    "num_updated_params": None,
                }
            log_event(
                logger,
                f"[INFO] forgetting end: task={task_id} elapsed={time.perf_counter() - forget_phase_start:.2f}s",
            )

            after_reset_eval = info.get("after_reset_eval")
            after_reset_acc = to_list(after_reset_eval["accuracy"]) if after_reset_eval else pre_acc

            stat = evaluate(test_datasets, args, model, return_logits=True, verbose=True)
            post_acc = to_list(stat["accuracy"])

            mia_after = None
            if getattr(args, "eval_mia", False):
                mia_after = compute_mia(model, train_datasets[task_id], test_datasets[task_id], task_id, args,
                                        dump_phase="after", unlearning_step=unlearning_step)
                log_event(logger, f"[INFO] MIA after forget: task={task_id} {mia_after}")
            mia_event = build_mia_event(mia_before, mia_after) if getattr(args, "eval_mia", False) else None

            probe_after = None
            if getattr(args, "eval_probe", False):
                probe_after = compute_probe(model, train_datasets[task_id], test_datasets[task_id], task_id, args)
                log_event(logger, f"[INFO] linear probe after forget: task={task_id} {probe_after}")
            probe_event = build_probe_event(probe_before, probe_after) if getattr(args, "eval_probe", False) else None

            agreement_block = None
            if getattr(args, "eval_agreement", False):
                if int(getattr(args, "n_forget", 0)) == 1:
                    agreement_block = compute_model_agreement(
                        model, args, requests, train_datasets, test_datasets,
                        task_id, remaining_tasks, logger,
                    )
                    log_event(logger, f"[INFO] agreement after forget: task={task_id} {agreement_block}")
                else:
                    log_event(
                        logger,
                        f"[INFO] agreement skipped: --eval_agreement requires n_forget==1 "
                        f"(got n_forget={getattr(args, 'n_forget', None)}); a reference retrain would "
                        f"repeat for every forget event.",
                    )

            avg_after_reset = avg_for_tasks(after_reset_acc, remaining_tasks)
            avg_after_retrain = avg_for_tasks(post_acc, remaining_tasks)
            fu = avg_before - avg_after_retrain
            worst_drop = 0.0
            if remaining_tasks:
                worst_drop = max(pre_acc[t] - post_acc[t] for t in remaining_tasks)
            au = post_acc[task_id] if task_id < len(post_acc) else 0.0
            finetune_diag = json_safe(info.get("finetune_diag", None))
            storage = model.storage_accounting() if hasattr(model, "storage_accounting") else None

            # Store measured accuracy changes only as a separate diagnostic. The
            # fixed-gradient quantity uses loss/energy units, so comparing the two
            # magnitudes cannot verify a bound.
            bound_check = info.get("bound_check")
            if isinstance(bound_check, dict):
                bound_check = json_safe(bound_check)
                per_task = bound_check.get("per_task", {}) or {}
                for t in remaining_tasks:
                    measured = float(pre_acc[t] - post_acc[t])
                    entry = per_task.setdefault(str(int(t)), {})
                    entry["measured_accuracy_drop_diagnostic"] = measured
                bound_check["per_task"] = per_task
                bound_check["measured_worstdrop_diagnostic"] = float(worst_drop)
                bound_check["comparison_note"] = (
                    "predicted_bound is in fixed-gradient loss/energy units; measured values "
                    "are accuracy changes and are not used for a satisfaction test"
                )

            event = {
                "unlearning_step": unlearning_step,
                "request_id": request_id,
                "task_id": task_id,
                "adapter_component_mode": info.get("adapter_component_mode"),
                "component_stages": json_safe(info.get("component_stages", {})),
                "stage_evals": json_safe(info.get("stage_evals", {})),
                "remaining_tasks": remaining_tasks,
                "per_task_acc_before": acc_list_to_dict(pre_acc),
                "per_task_acc_after_reset": acc_list_to_dict(after_reset_acc),
                "per_task_acc_after_retrain": acc_list_to_dict(post_acc),
                "avg_before": avg_before,
                "avg_after_reset": avg_after_reset,
                "avg_after_retrain": avg_after_retrain,
                "Fu": fu,
                "WorstDrop": worst_drop,
                "Au": au,
                "grad_norm_ratio": info.get("grad_norm_ratio"),
                "mia": mia_event,
                "agreement": agreement_block,
                "probe": probe_event,
                "bound_check": bound_check,
                "conflict_mask": json_safe(info.get("conflict_mask_stats")) if isinstance(info.get("conflict_mask_stats"), dict) else None,
                "t_reset": info.get("t_reset", 0.0) if info.get("t_reset") is not None else 0.0,
                "t_retrain": info.get("t_retrain", 0.0) if info.get("t_retrain") is not None else 0.0,
                "t_target_reset": info.get("t_target_reset"),
                "t_shared_update": info.get("t_shared_update"),
                "t_classifier_ascent": info.get("t_classifier_ascent"),
                "t_retained_repair": info.get("t_retained_repair"),
                "t_component_eval": info.get("t_component_eval"),
                "t_forget_total_raw": info.get("t_forget_total_raw"),
                "t_forget_total": info.get("t_forget_total"),
                "num_updated_params": (
                    info.get("num_updated_params")
                    if info.get("num_updated_params") is not None
                    else 0
                ),
                "overlap": {
                    "s_t": info.get("s_t", 0) if info.get("s_t") is not None else 0,
                    "s_share": info.get("s_share", 0) if info.get("s_share") is not None else 0,
                    "s_share_crit": info.get("s_share_crit", 0) if info.get("s_share_crit") is not None else 0,
                    "s_share_ratio": (
                        info.get("s_share_ratio", 0.0) if info.get("s_share_ratio") is not None else 0.0
                    ),
                    "s_share_crit_ratio": (
                        info.get("s_share_crit_ratio", 0.0) if info.get("s_share_crit_ratio") is not None else 0.0
                    ),
                },
                "protection": info.get("protection", {}),
                "finetune_diag": finetune_diag,
                "storage": json_safe(storage),
                "adapter_shared_forget_ratio": info.get("adapter_shared_forget_ratio"),
                "adapter_shared_protect_ratio": info.get("adapter_shared_protect_ratio"),
                "shared_protect_strength": info.get("shared_protect_strength"),
                "shared_adapter_params": info.get("shared_adapter_params"),
                "classifier_param_count": info.get("classifier_param_count"),
                "classifier_forget_param_count": info.get("classifier_forget_param_count"),
                "shared_forget_candidates": info.get("shared_forget_candidates"),
                "shared_protected_params": info.get("shared_protected_params"),
                "shared_active_critical": info.get("shared_active_critical"),
                "shared_overlap_critical": info.get("shared_overlap_critical"),
                "shared_effective_forget_params": info.get("shared_effective_forget_params"),
                "shared_full_update_params": info.get("shared_full_update_params"),
                "shared_soft_update_params": info.get("shared_soft_update_params"),
                "shared_s_share_crit": info.get("shared_s_share_crit"),
            }
            normalized_event = normalize_unlearning_event(
                event,
                availability={
                    "t_reset": info.get("t_reset") is not None,
                    "t_retrain": info.get("t_retrain") is not None,
                    "num_updated_params": info.get("num_updated_params") is not None,
                    "overlap": any(
                        key in info
                        for key in ("s_t", "s_share", "s_share_crit", "s_share_ratio", "s_share_crit_ratio")
                    ),
                    "shared_adapter": any(
                        key in info
                        for key in (
                            "adapter_shared_forget_ratio",
                            "adapter_shared_protect_ratio",
                            "shared_protect_strength",
                            "shared_adapter_params",
                            "classifier_param_count",
                            "classifier_forget_param_count",
                            "shared_forget_candidates",
                            "shared_protected_params",
                            "shared_active_critical",
                            "shared_overlap_critical",
                            "shared_effective_forget_params",
                            "shared_full_update_params",
                            "shared_soft_update_params",
                            "shared_s_share_crit",
                        )
                    ),
                },
            )
            log_event(
                logger,
                "[INFO] overlap: |S_t|={s_t} |S_share|={s_share} |S_share_crit|={s_share_crit} "
                "ratios: share={share_ratio} crit={crit_ratio}".format(
                    s_t=format_optional_int(normalized_event["overlap"].get("s_t")),
                    s_share=format_optional_int(normalized_event["overlap"].get("s_share")),
                    s_share_crit=format_optional_int(normalized_event["overlap"].get("s_share_crit")),
                    share_ratio=format_optional_float(normalized_event["overlap"].get("s_share_ratio")),
                    crit_ratio=format_optional_float(normalized_event["overlap"].get("s_share_crit_ratio")),
                )
            )
            log_event(
                logger,
                "[INFO] unlearning timing: t_reset={t_reset}s t_retrain={t_retrain}s "
                "t_forget_total={t_total}s updated_params={updated}".format(
                    t_reset=format_optional_float(normalized_event.get("t_reset")),
                    t_retrain=format_optional_float(normalized_event.get("t_retrain")),
                    t_total=format_optional_float(normalized_event.get("t_forget_total")),
                    updated=format_optional_int(normalized_event.get("num_updated_params")),
                )
            )
            if finetune_diag is not None:
                log_event(logger, f"[INFO] finetune_diag: {json.dumps(finetune_diag)}")
            if metrics_state is not None:
                metrics_state.setdefault("unlearning_events", []).append(event)
                metrics_state.setdefault("normalized_results", {}).setdefault("unlearning_events", []).append(
                    normalized_event
                )
                write_json(metrics_path, metrics_state)

            chance_acc = 1.0 / args.class_per_task
            sanity_msgs = []
            if pre_acc[task_id] - au < 0.05:
                sanity_msgs.append(
                    f"[WARN] Unlearned task {task_id} accuracy drop is small: "
                    f"{pre_acc[task_id]:.4f} -> {au:.4f}"
                )
            if au > chance_acc + 0.05:
                sanity_msgs.append(
                    f"[WARN] Unlearned task {task_id} accuracy above chance: {au:.4f} (chance {chance_acc:.4f})"
                )
            share_ratio = info.get("s_share_ratio")
            if args.method_variant == "modified" and share_ratio is not None and share_ratio < 0.05:
                sanity_msgs.append(
                    f"[INFO] Overlap ratio is low ({share_ratio:.4f}); modified method should be close to baseline."
                )
            if sanity_msgs and metrics_state is not None:
                metrics_state.setdefault("sanity_checks", []).extend(sanity_msgs)
                write_json(metrics_path, metrics_state)
            for msg in sanity_msgs:
                log_event(logger, msg)

            unlearning_step += 1
            t_reset = info.get("t_reset")
            t_retrain = info.get("t_retrain")
            t_forget_total = info.get("t_forget_total")
            if t_forget_total is None:
                t_forget_total = (t_reset or 0.0) + (t_retrain or 0.0)
            times[request_id] = float(t_forget_total)
        else:
            t0 = time.perf_counter()
            log_event(logger, f"[INFO] task training start: task={task_id}")
            model.privacy_aware_lifelong_learning(task_id, train_datasets[task_id], learn_type)
            t1 = time.perf_counter()
            log_event(logger, f"[INFO] task training end: task={task_id} elapsed={t1 - t0:.2f}s")
            stat = evaluate(test_datasets, args, model, return_logits=True, verbose=True)
            times[request_id] = t1 - t0

        # evaluate bookkeeping
        for forget_task in forgotten_tasks:
            forgotten_tasks_mask[request_id][forget_task] = 1.0
        for active_task in active_tasks:
            active_tasks_mask[request_id][active_task] = 1.0

        loss[request_id] = stat["loss"]
        accuracy[request_id] = stat["accuracy"]
        if stat["logits"] is not None:
            for t in range(args.n_tasks):
                logits[t][request_id] = stat["logits"][t]

        # Resident tensor-state accounting after every request makes task-growth
        # comparisons possible without using peak CUDA allocator statistics.
        # It is read-only and excludes transient optimizer/activation memory.
        if metrics_state is not None and hasattr(model, "storage_accounting"):
            metrics_state.setdefault("storage_history", []).append({
                "request_id": int(request_id),
                "task_id": int(task_id),
                "request_type": learn_type,
                "active_tasks": [int(task) for task in active_tasks],
                "storage": json_safe(model.storage_accounting()),
            })
            write_json(metrics_path, metrics_state)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "times": times,
        "forgotten_tasks_mask": forgotten_tasks_mask,
        "active_tasks_mask": active_tasks_mask,
        "logits": logits,
    }


def write_overlap_csv(path, task_ids, matrix):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id"] + [str(t) for t in task_ids])
        for task_id, row in zip(task_ids, matrix):
            writer.writerow([str(task_id)] + [f"{val:.6f}" for val in row])


def write_summary(path, summary, normalized_results):
    normalized_results = normalized_results or {}
    final_block = normalized_results.get("final", {})
    final_unlearning = final_block.get("final_unlearning", {})
    final_overlap = final_unlearning.get("overlap", {}) if isinstance(final_unlearning, dict) else {}
    final_mia = final_unlearning.get("mia", {}) if isinstance(final_unlearning.get("mia"), dict) else {}

    lines = [
        f"run_dir: {summary['run_dir']}",
        f"dataset: {summary['dataset']}",
        f"method: {summary['method']} ({summary['method_variant']})",
        f"seed: {summary['seed']} (deterministic={summary['deterministic']})",
        f"tasks: {summary['n_tasks']} | forget_requests: {summary['n_forget']}",
        (
            "model_params: total {total} trainable {trainable} adapter {adapter} shared_adapter {shared} trainable_ratio {ratio}".format(
                total=format_optional_int(summary.get("total_params")),
                trainable=format_optional_int(summary.get("num_trainable_params")),
                adapter=format_optional_int(summary.get("num_adapter_params")),
                shared=format_optional_int(summary.get("shared_adapter_params")),
                ratio=format_optional_float(summary.get("trainable_param_ratio")),
            )
        ),
        f"final_avg_accuracy: {summary['final_avg_accuracy']:.4f}",
        f"final_avg_forgetting: {summary['final_avg_forgetting']:.4f}",
        (
            "final_unlearning: Fu {fu} WorstDrop {worst_drop} Au {au} "
            "MIA_AUC {mia_before}->{mia_after} "
            "t_reset {t_reset}s t_retrain {t_retrain}s t_forget_total {t_total}s "
            "updated_params {updated} share_ratio {share_ratio} crit_ratio {crit_ratio}".format(
                fu=format_optional_float(final_unlearning.get("Fu")),
                worst_drop=format_optional_float(final_unlearning.get("WorstDrop")),
                au=format_optional_float(final_unlearning.get("Au")),
                mia_before=format_optional_float(final_mia.get("auc_before")),
                mia_after=format_optional_float(final_mia.get("auc_after")),
                t_reset=format_optional_float(final_unlearning.get("t_reset")),
                t_retrain=format_optional_float(final_unlearning.get("t_retrain")),
                t_total=format_optional_float(final_unlearning.get("t_forget_total")),
                updated=format_optional_int(final_unlearning.get("num_updated_params")),
                share_ratio=format_optional_float(final_overlap.get("s_share_ratio")),
                crit_ratio=format_optional_float(final_overlap.get("s_share_crit_ratio")),
            )
        ),
        "",
        "normalized_unlearning_events:",
    ]
    unlearning_events = normalized_results.get("unlearning_events", [])
    if not unlearning_events:
        lines.append("none")
    for event in unlearning_events:
        overlap = event.get("overlap", {})
        mia = event.get("mia", {}) if isinstance(event.get("mia"), dict) else {}
        lines.append(
            "step {step} task {task} avg_before {avg_before} "
            "avg_after_reset {avg_after_reset} avg_after_retrain {avg_after_retrain} "
            "Fu {fu} WorstDrop {worst_drop} Au {au} MIA_AUC {mia_before}->{mia_after} "
            "t_reset {t_reset}s t_retrain {t_retrain}s t_forget_total {t_total}s "
            "updated_params {updated} share_ratio {share_ratio} crit_ratio {crit_ratio}".format(
                step=format_optional_int(event.get("unlearning_step")),
                task=format_optional_int(event.get("task_id")),
                avg_before=format_optional_float(event.get("avg_before")),
                avg_after_reset=format_optional_float(event.get("avg_after_reset")),
                avg_after_retrain=format_optional_float(event.get("avg_after_retrain")),
                fu=format_optional_float(event.get("Fu")),
                worst_drop=format_optional_float(event.get("WorstDrop")),
                au=format_optional_float(event.get("Au")),
                mia_before=format_optional_float(mia.get("auc_before")),
                mia_after=format_optional_float(mia.get("auc_after")),
                t_reset=format_optional_float(event.get("t_reset")),
                t_retrain=format_optional_float(event.get("t_retrain")),
                t_total=format_optional_float(event.get("t_forget_total")),
                updated=format_optional_int(event.get("num_updated_params")),
                share_ratio=format_optional_float(overlap.get("s_share_ratio")),
                crit_ratio=format_optional_float(overlap.get("s_share_crit_ratio")),
            ),
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def summarize_overlap_csv(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2 or len(rows[0]) < 2:
        return None

    n = len(rows[0]) - 1
    matrix = []
    for row in rows[1:]:
        if len(row) < n + 1:
            return None
        parsed = []
        for cell in row[1:n + 1]:
            try:
                parsed.append(float(cell))
            except ValueError:
                return None
        matrix.append(parsed)
    if len(matrix) != n:
        return None

    diag_vals = []
    offdiag_vals = []
    all_vals = []
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            all_vals.append(val)
            if i == j:
                diag_vals.append(val)
            else:
                offdiag_vals.append(val)

    def mean_or_none(values):
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "n_tasks_in_overlap": n,
        "num_task_pairs": (n * (n - 1)) // 2,
        "avg_overlap_offdiag": mean_or_none(offdiag_vals),
        "max_overlap_offdiag": max(offdiag_vals) if offdiag_vals else None,
        "min_overlap_offdiag": min(offdiag_vals) if offdiag_vals else None,
        "avg_overlap_all": mean_or_none(all_vals),
        "diag_mean": mean_or_none(diag_vals),
    }


def compute_unlearning_score(fu, worst_drop, updated_param_ratio, lambda_1=0.5, lambda_2=0.5):
    fu_val = to_optional_float(fu) if fu is not None else None
    worst_drop_val = to_optional_float(worst_drop) if worst_drop is not None else None
    updated_ratio_val = to_optional_float(updated_param_ratio) if updated_param_ratio is not None else None
    if fu_val is None or worst_drop_val is None or updated_ratio_val is None:
        return None
    return float(fu_val - lambda_1 * worst_drop_val - lambda_2 * updated_ratio_val)


def extend_markdown_table(lines, title, rows):
    lines.extend(["", title, "| Metric | Value |", "| --- | --- |"])
    if not rows:
        lines.append("| status | NA |")
        return
    for key, value in rows:
        lines.append(f"| {key} | {value} |")


def describe_forgetting_success(fu, au, chance_acc):
    fu_val = to_optional_float(fu) if fu is not None else None
    au_val = to_optional_float(au) if au is not None else None
    if fu_val is None and au_val is None:
        return "Insufficient data: Fu and deleted-task accuracy are unavailable."
    if chance_acc is None:
        low_deleted_acc = au_val is not None and au_val <= 0.10
        chance_text = "NA"
    else:
        low_deleted_acc = au_val is not None and au_val <= chance_acc + 0.05
        chance_text = f"{chance_acc:.4f}"

    if fu_val is not None and fu_val >= 0.01 and low_deleted_acc:
        return (
            "Likely yes: Fu is positive and deleted-task accuracy is low "
            f"(Au={format_optional_float(au_val)}, chance≈{chance_text})."
        )
    if low_deleted_acc:
        return (
            "Mixed evidence: deleted-task accuracy is low "
            f"(Au={format_optional_float(au_val)}, chance≈{chance_text}), but Fu is not clearly high."
        )
    if fu_val is not None and fu_val >= 0.01:
        return (
            "Mixed evidence: Fu is positive, but deleted-task accuracy remains elevated "
            f"(Au={format_optional_float(au_val)}, chance≈{chance_text})."
        )
    return (
        "Unclear or weak: neither high Fu nor low deleted-task accuracy is present "
        f"(Fu={format_optional_float(fu_val)}, Au={format_optional_float(au_val)}, chance≈{chance_text})."
    )


def describe_preservation(worst_drop):
    worst_drop_val = to_optional_float(worst_drop) if worst_drop is not None else None
    if worst_drop_val is None:
        return "Insufficient data: WorstDrop is unavailable."
    if worst_drop_val <= 0.05:
        return f"Likely yes: WorstDrop is low at {worst_drop_val:.4f}."
    if worst_drop_val <= 0.10:
        return f"Mixed evidence: WorstDrop is moderate at {worst_drop_val:.4f}."
    return f"Likely no: WorstDrop is high at {worst_drop_val:.4f}, suggesting notable collateral damage."


def describe_efficiency(updated_param_ratio, t_forget_total):
    ratio_val = to_optional_float(updated_param_ratio) if updated_param_ratio is not None else None
    time_val = to_optional_float(t_forget_total) if t_forget_total is not None else None
    if ratio_val is None and time_val is None:
        return "Insufficient data: updated_param_ratio and forget time are unavailable."
    if ratio_val is None:
        return f"Partial evidence: forget time is {format_optional_float(time_val)}s, but updated_param_ratio is unavailable."
    if ratio_val <= 0.01:
        return (
            "Likely yes: only a small fraction of parameters were updated "
            f"(updated_param_ratio={ratio_val:.4f}, t_forget_total={format_optional_float(time_val)}s)."
        )
    if ratio_val <= 0.05:
        return (
            "Moderate efficiency: parameter updates remain limited "
            f"(updated_param_ratio={ratio_val:.4f}, t_forget_total={format_optional_float(time_val)}s)."
        )
    return (
        "Likely no: the method updates a relatively large fraction of parameters "
        f"(updated_param_ratio={ratio_val:.4f}, t_forget_total={format_optional_float(time_val)}s)."
    )


def write_run_report(path, run_dir, config, metrics_state):
    normalized_results = metrics_state.get("normalized_results", {})
    normalized_final = normalized_results.get("final", {}) if isinstance(normalized_results, dict) else {}
    summary = metrics_state.get("summary", {})
    forgetting = metrics_state.get("forgetting", {})
    model_stats = metrics_state.get("model", {}) if isinstance(metrics_state.get("model"), dict) else {}
    adapter_stats = metrics_state.get("adapter_stats", {}) if isinstance(metrics_state.get("adapter_stats"), dict) else {}

    final_avg_acc = first_non_none(
        normalized_final.get("final_avg_accuracy"),
        summary.get("final_avg_accuracy"),
    )
    avg_forgetting = first_non_none(
        normalized_final.get("average_forgetting"),
        forgetting.get("final") if isinstance(forgetting, dict) else None,
        summary.get("final_avg_forgetting"),
    )

    final_unlearning = normalized_final.get("final_unlearning", {})
    if not isinstance(final_unlearning, dict) or not final_unlearning:
        events = metrics_state.get("unlearning_events", [])
        if isinstance(events, list) and events:
            last = events[-1] if isinstance(events[-1], dict) else {}
        else:
            last = {}
        overlap = last.get("overlap", {}) if isinstance(last.get("overlap"), dict) else {}
        mia = last.get("mia") if isinstance(last.get("mia"), dict) else None
        final_unlearning = {
            "Fu": last.get("Fu"),
            "WorstDrop": last.get("WorstDrop"),
            "Au": last.get("Au"),
            "mia": mia,
            "t_reset": last.get("t_reset"),
            "t_retrain": last.get("t_retrain"),
            "t_forget_total": first_non_none(
                last.get("t_forget_total"),
                (last.get("t_reset", 0.0) + last.get("t_retrain", 0.0)) if (last.get("t_reset") is not None or last.get("t_retrain") is not None) else None,
            ),
            "num_updated_params": last.get("num_updated_params"),
            "overlap": {
                "s_t": overlap.get("s_t"),
                "s_share": overlap.get("s_share"),
                "s_share_crit": overlap.get("s_share_crit"),
                "s_share_ratio": overlap.get("s_share_ratio"),
                "s_share_crit_ratio": overlap.get("s_share_crit_ratio"),
            },
        }

    overlap_block = final_unlearning.get("overlap", {}) if isinstance(final_unlearning, dict) else {}
    mia_block = final_unlearning.get("mia", {}) if isinstance(final_unlearning.get("mia"), dict) else {}
    overlap_csv_path = run_dir / "overlap.csv"
    overlap_csv_summary = summarize_overlap_csv(overlap_csv_path)
    updated_param_ratio = first_non_none(
        metrics_state.get("updated_param_ratio"),
        normalized_final.get("updated_param_ratio"),
        summary.get("updated_param_ratio"),
        adapter_stats.get("updated_param_ratio"),
    )
    unlearning_score = first_non_none(
        metrics_state.get("unlearning_score"),
        normalized_final.get("unlearning_score"),
        summary.get("unlearning_score"),
    )
    adapter_param_ratio = first_non_none(
        metrics_state.get("adapter_param_ratio"),
        normalized_final.get("adapter_param_ratio"),
        summary.get("adapter_param_ratio"),
        adapter_stats.get("adapter_param_ratio"),
        model_stats.get("adapter_param_ratio"),
    )
    chance_acc = None
    if config.get("class_per_task") not in (None, 0):
        chance_acc = 1.0 / float(config.get("class_per_task"))

    config_rows = [
        ("dataset", config.get("dataset")),
        ("method", config.get("method")),
        ("method_variant", config.get("method_variant")),
        ("seed", config.get("seed")),
        ("arch", config.get("arch")),
        ("class_per_task", config.get("class_per_task")),
        ("n_tasks", config.get("n_tasks")),
        ("n_forget", config.get("n_forget")),
        ("n_epochs", config.get("n_epochs")),
        ("batch_size", config.get("batch_size")),
        ("optim", config.get("optim")),
        ("lr", config.get("lr")),
        ("deterministic", config.get("deterministic")),
    ]
    if config.get("method") == "pall_adapter":
        config_rows.extend(
            [
                ("adapter_bottleneck", config.get("adapter_bottleneck")),
                ("adapter_shared_bottleneck", config.get("adapter_shared_bottleneck")),
                ("adapter_shared_forget_ratio", config.get("adapter_shared_forget_ratio")),
                ("adapter_shared_protect_ratio", config.get("adapter_shared_protect_ratio")),
                ("adapter_shared_forget_lr", config.get("adapter_shared_forget_lr")),
                ("adapter_shared_protect_strength", config.get("adapter_shared_protect_strength")),
                ("adapter_location", config.get("adapter_location")),
                ("adapter_train_classifier", config.get("adapter_train_classifier")),
            ]
        )

    lines = [
        "# Run Report",
    ]
    extend_markdown_table(
        lines,
        "## Config",
        [(key, format_optional_text(value)) for key, value in config_rows],
    )
    extend_markdown_table(
        lines,
        "## Final Performance",
        [
            ("final_avg_accuracy", format_optional_float(final_avg_acc)),
            ("average_forgetting", format_optional_float(avg_forgetting)),
            ("num_unlearning_events", format_optional_int(normalized_final.get("num_unlearning_events"))),
            ("chance_accuracy", format_optional_float(chance_acc)),
        ],
    )
    extend_markdown_table(
        lines,
        "## Unlearning Metrics",
        [
            ("Fu", format_optional_float(final_unlearning.get("Fu"))),
            ("WorstDrop", format_optional_float(final_unlearning.get("WorstDrop"))),
            ("Au", format_optional_float(final_unlearning.get("Au"))),
            ("mia_auc_before", format_optional_float(mia_block.get("auc_before"))),
            ("mia_auc_after", format_optional_float(mia_block.get("auc_after"))),
            ("mia_acc_before", format_optional_float(mia_block.get("acc_before"))),
            ("mia_acc_after", format_optional_float(mia_block.get("acc_after"))),
            ("unlearning_score", format_optional_float(unlearning_score)),
            ("t_reset", format_optional_float(final_unlearning.get("t_reset"))),
            ("s_t", format_optional_int(overlap_block.get("s_t"))),
            ("s_share", format_optional_int(overlap_block.get("s_share"))),
            ("s_share_crit", format_optional_int(overlap_block.get("s_share_crit"))),
            ("s_share_ratio", format_optional_float(overlap_block.get("s_share_ratio"))),
            ("s_share_crit_ratio", format_optional_float(overlap_block.get("s_share_crit_ratio"))),
        ],
    )
    extend_markdown_table(
        lines,
        "## Efficiency Metrics",
        [
            ("total_params", format_optional_int(model_stats.get("total_params"))),
            ("num_trainable_params", format_optional_int(model_stats.get("num_trainable_params"))),
            ("t_retrain", format_optional_float(final_unlearning.get("t_retrain"))),
            ("t_forget_total", format_optional_float(final_unlearning.get("t_forget_total"))),
            ("num_updated_params", format_optional_int(final_unlearning.get("num_updated_params"))),
            ("updated_param_ratio", format_optional_float(updated_param_ratio)),
            ("trainable_param_ratio", format_optional_float(model_stats.get("trainable_param_ratio"))),
        ],
    )
    if adapter_stats:
        extend_markdown_table(
            lines,
            "## Adapter Stats",
            [
                ("total_model_params", format_optional_int(adapter_stats.get("total_model_params"))),
                ("trainable_params", format_optional_int(adapter_stats.get("trainable_params"))),
                ("adapter_params", format_optional_int(adapter_stats.get("adapter_params"))),
                ("shared_adapter_params", format_optional_int(adapter_stats.get("shared_adapter_params"))),
                ("task_adapter_params", format_optional_int(adapter_stats.get("task_adapter_params"))),
                ("adapter_param_ratio", format_optional_float(adapter_param_ratio)),
                ("updated_param_ratio", format_optional_float(updated_param_ratio)),
            ],
        )

    lines.extend(["", "## Automatic Interpretation"])
    lines.append(
        f"- Forgetting succeeded? {describe_forgetting_success(final_unlearning.get('Fu'), final_unlearning.get('Au'), chance_acc)}"
    )
    lines.append(
        f"- Preserved other tasks? {describe_preservation(final_unlearning.get('WorstDrop'))}"
    )
    lines.append(
        f"- Efficient method? {describe_efficiency(updated_param_ratio, final_unlearning.get('t_forget_total'))}"
    )

    extend_markdown_table(lines, "## Overlap CSV Summary", [])
    if overlap_csv_summary is None:
        lines[-1] = "| overlap_csv | NA |"
    else:
        lines.pop()
        lines.append(f"| n_tasks_in_overlap | {format_optional_int(overlap_csv_summary.get('n_tasks_in_overlap'))} |")
        lines.append(f"| num_task_pairs | {format_optional_int(overlap_csv_summary.get('num_task_pairs'))} |")
        lines.append(f"| avg_overlap_offdiag | {format_optional_float(overlap_csv_summary.get('avg_overlap_offdiag'))} |")
        lines.append(f"| max_overlap_offdiag | {format_optional_float(overlap_csv_summary.get('max_overlap_offdiag'))} |")
        lines.append(f"| min_overlap_offdiag | {format_optional_float(overlap_csv_summary.get('min_overlap_offdiag'))} |")
        lines.append(f"| avg_overlap_all | {format_optional_float(overlap_csv_summary.get('avg_overlap_all'))} |")
        lines.append(f"| diag_mean | {format_optional_float(overlap_csv_summary.get('diag_mean'))} |")

    artifact_files = [
        "config.json",
        "metrics.json",
        "results.pth",
        "summary.txt",
        "overlap.csv",
    ]
    lines.extend(["", "## Artifacts", "| File | Location |", "| --- | --- |"])
    for name in artifact_files:
        artifact_path = run_dir / name
        lines.append(f"| {name} | {artifact_path if artifact_path.exists() else 'NA'} |")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_user_requests(num_tasks, sequence_length):
    if sequence_length < num_tasks:
        raise ValueError("Sequence length must be at least the number of tasks.")

    user_requests = [(i, "T") for i in range(num_tasks)]
    trained_tasks = list(range(num_tasks))

    remaining_slots = sequence_length - num_tasks
    f_requests = []
    while remaining_slots > 0 and trained_tasks:
        task = random.choice(trained_tasks)
        f_requests.append((task, "F"))
        trained_tasks.pop(trained_tasks.index(task))
        remaining_slots -= 1

    for f_request in f_requests:
        t_index = user_requests.index((f_request[0], "T"))
        valid_positions = list(range(t_index + 1, len(user_requests) + 1))
        insert_position = random.choice(valid_positions)
        user_requests.insert(insert_position, f_request)

    return user_requests


def parse_schedule_entries(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("requests", "schedule", "user_requests", "user_requests_with_active_tasks"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise ValueError(
            "Schedule JSON object must contain one of: "
            "'requests', 'schedule', 'user_requests', 'user_requests_with_active_tasks'."
        )
    raise ValueError("Schedule JSON root must be a list or object.")


def parse_schedule_request(entry, request_id):
    if isinstance(entry, dict):
        task_id = entry.get("task_id")
        learn_type = entry.get("request_type", entry.get("learn_type", entry.get("type")))
        active_tasks = entry.get("active_tasks")
    elif isinstance(entry, (list, tuple)):
        if len(entry) < 2:
            raise ValueError(f"Schedule request {request_id} must include at least [task_id, request_type].")
        task_id = entry[0]
        learn_type = entry[1]
        active_tasks = entry[2] if len(entry) >= 3 else None
    else:
        raise ValueError(f"Schedule request {request_id} must be an object or list.")

    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        raise ValueError(f"Schedule request {request_id} has invalid task_id={task_id!r}.")

    learn_type = str(learn_type).strip().upper()
    if learn_type not in {"T", "F"}:
        raise ValueError(f"Schedule request {request_id} has invalid request type={learn_type!r}; expected 'T' or 'F'.")

    parsed_active_tasks = None
    if active_tasks is not None:
        if not isinstance(active_tasks, list):
            raise ValueError(f"Schedule request {request_id} has non-list active_tasks={active_tasks!r}.")
        parsed_active_tasks = []
        for idx, task in enumerate(active_tasks):
            try:
                parsed_active_tasks.append(int(task))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Schedule request {request_id} has invalid active_tasks[{idx}]={task!r}; expected int task IDs."
                )

    return task_id, learn_type, parsed_active_tasks


def build_requests_with_active_tasks(user_requests, n_tasks):
    learned_tasks = set()
    active_tasks = []
    user_requests_with_active_tasks = []
    normalized_schedule = []

    for request_id, entry in enumerate(user_requests):
        task_id, learn_type, provided_active_tasks = parse_schedule_request(entry, request_id)

        if not (0 <= task_id < n_tasks):
            raise ValueError(
                f"Schedule request {request_id} has task_id={task_id} out of range [0, {n_tasks - 1}]."
            )

        if learn_type == "T":
            if task_id in learned_tasks:
                raise ValueError(
                    f"Schedule request {request_id} tries to learn task {task_id} more than once."
                )
            learned_tasks.add(task_id)
            active_tasks.append(task_id)
        else:
            if task_id not in learned_tasks:
                raise ValueError(
                    f"Schedule request {request_id} tries to forget task {task_id} before learning it."
                )
            if task_id not in active_tasks:
                raise ValueError(
                    f"Schedule request {request_id} tries to forget task {task_id} which is already forgotten."
                )
            active_tasks.remove(task_id)

        computed_active = list(active_tasks)
        if provided_active_tasks is not None and provided_active_tasks != computed_active:
            raise ValueError(
                f"Schedule request {request_id} has inconsistent active_tasks={provided_active_tasks}; "
                f"expected {computed_active}."
            )

        user_requests_with_active_tasks.append((task_id, learn_type, computed_active))
        normalized_schedule.append(
            {
                "task_id": task_id,
                "request_type": learn_type,
                "active_tasks": computed_active,
            }
        )

    return user_requests_with_active_tasks, normalized_schedule


def load_request_schedule(schedule_file, n_tasks):
    path = Path(schedule_file).expanduser()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(f"Failed to read request schedule file {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse request schedule JSON {path}: {exc}")

    entries = parse_schedule_entries(payload)
    user_requests_with_active_tasks, normalized_schedule = build_requests_with_active_tasks(entries, n_tasks)
    return {
        "source": "file",
        "request_schedule_file": str(path.resolve()),
        "loaded_request_schedule": payload,
        "request_schedule": normalized_schedule,
        "requests_with_active_tasks": user_requests_with_active_tasks,
    }


def get_request_datasets():
    def clear_all_forget_requests(li):
        to_be_removed = []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if learn_type == "F":
                to_be_removed.append(request_id)
                for j in range(request_id):
                    if li[j][0] == task_id and li[j][1] == "T":
                        to_be_removed.append(j)
                        break
        new_list, new_active_tasks = [], []
        for request_id, (task_id, learn_type, active_tasks) in enumerate(li):
            if request_id not in to_be_removed:
                if learn_type == "T" and (task_id not in new_active_tasks):
                    new_active_tasks.append(task_id)
                new_list.append((task_id, learn_type, list(new_active_tasks)))
        return new_list

    # Loading the datasets
    train_datasets, test_datasets = get_task_datasets(args)

    if args.request_schedule_file:
        schedule_info = load_request_schedule(args.request_schedule_file, args.n_tasks)
        user_requests_with_active_tasks = schedule_info["requests_with_active_tasks"]
    else:
        user_requests = generate_user_requests(num_tasks=args.n_tasks, sequence_length=int(args.n_tasks + args.n_forget))
        user_requests_with_active_tasks, normalized_schedule = build_requests_with_active_tasks(
            user_requests,
            args.n_tasks,
        )
        schedule_info = {
            "source": "generated",
            "request_schedule_file": None,
            "loaded_request_schedule": None,
            "request_schedule": normalized_schedule,
            "requests_with_active_tasks": user_requests_with_active_tasks,
        }

    print('user_requests_with_active_tasks: ', user_requests_with_active_tasks)

    user_requests_without_forgotten = []
    for request_id, (task_id, learn_type, active_tasks) in enumerate(user_requests_with_active_tasks):
        if learn_type == "F":
            list_up_to = list(user_requests_with_active_tasks[:request_id + 1])
            user_requests_without_forgotten.append(clear_all_forget_requests(list_up_to))
    print('user_requests_without_forgotten: ', user_requests_without_forgotten)

    return (
        train_datasets,
        test_datasets,
        user_requests_with_active_tasks,
        user_requests_without_forgotten,
        schedule_info,
    )


def main():
    global args
    run_start = time.perf_counter()
    set_seed(args.seed, args.deterministic)

    run_dir, timestamp = init_run_dir(args)
    args.run_dir = str(run_dir)
    logger = init_logger(run_dir)
    config = serialize_config(args, run_dir, timestamp)
    config["request_schedule_source"] = None
    config["request_schedule_file"] = args.request_schedule_file
    config["request_schedule"] = None
    config["loaded_request_schedule"] = None
    write_json(run_dir / "config.json", config)

    methods_dict = {
        "sequential": Sequential,
        "ewc": EWC,
        "lwf": LwF,
        "er": ER,
        "derpp": Derpp,
        "lsf": LSF,
        "clpu": CLPU,
        "ssd": SSD,
        "salun": SalUn,
        "pall": PALLModified,        # deprecated alias -> main method
        "pall_original": PALLOriginal,
        "pall_modified": PALLModified,
        "pall_adapter": PALLAdapter,
        "lora": LoRA,
    }

    log_event(logger, "============================================================")
    log_event(logger, "[INFO] -- Experiment Configs --")
    log_event(logger, "       1. data & task")
    log_event(logger, f"          dataset:      {args.dataset}")
    log_event(
        logger,
        f"          shape:        {args.channels}x{args.image_size}x{args.image_size} | classes={args.num_classes}",
    )
    log_event(logger, f"          n_tasks:      {args.n_tasks}")
    log_event(logger, f"          # class/task: {args.class_per_task}")
    log_event(logger, "       2. training")
    log_event(logger, f"          lr:           {args.lr:5.4f}")
    log_event(logger, "       3. model")
    log_event(logger, f"          method:       {args.method} ({args.method_variant})")
    log_event(logger, f"          architecture: {args.arch}")
    log_event(logger, f"          norm params:  {args.norm_params}")
    log_event(logger, f"          device:       {args.device}")
    log_event(logger, f"          host_os:      {'macOS' if args.is_macos else sys.platform}")
    log_event(
        logger,
        "          dataloader:   num_workers={num_workers} pin_memory={pin_memory} persistent_workers={persistent}".format(
            num_workers=args.resolved_num_workers,
            pin_memory=args.resolved_pin_memory,
            persistent=args.resolved_persistent_workers,
        ),
    )
    log_event(
        logger,
        f"          placement:    batches move in-loop via x.to({args.device}), y.to({args.device})",
    )
    log_event(logger, f"          deterministic:{args.deterministic}")
    log_event(logger, f"          run_dir:      {run_dir}")
    log_event(logger, "============================================================")

    data_start = time.perf_counter()
    (
        train_datasets,
        test_datasets,
        user_requests_with_active_tasks,
        user_requests_without_forgotten,
        schedule_info,
    ) = get_request_datasets()
    log_event(logger, f"[INFO] finish processing data in {time.perf_counter() - data_start:.2f}s")
    log_event(
        logger,
        "[INFO] request schedule source: {source} file: {path}".format(
            source=schedule_info.get("source"),
            path=schedule_info.get("request_schedule_file") or "NA",
        ),
    )

    config["request_schedule_source"] = schedule_info.get("source")
    config["request_schedule_file"] = schedule_info.get("request_schedule_file")
    config["request_schedule"] = schedule_info.get("request_schedule")
    config["loaded_request_schedule"] = schedule_info.get("loaded_request_schedule")
    write_json(run_dir / "config.json", config)

    metrics_state = {
        "run": {
            "dataset": args.dataset,
            "method": args.method,
            "method_variant": args.method_variant,
            "experiment_tag": args.experiment_tag,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "n_tasks": args.n_tasks,
            "n_forget": args.n_forget,
            "request_schedule_source": schedule_info.get("source"),
            "request_schedule_file": schedule_info.get("request_schedule_file"),
            "timestamp": timestamp,
            "run_dir": str(run_dir),
        },
        "unlearning_events": [],
        "sanity_checks": [],
        "request_schedule_source": schedule_info.get("source"),
        "request_schedule_file": schedule_info.get("request_schedule_file"),
        "request_schedule": schedule_info.get("request_schedule"),
        "loaded_request_schedule": schedule_info.get("loaded_request_schedule"),
        "requests": user_requests_with_active_tasks,
        "requests_without_forgotten": user_requests_without_forgotten,
        "final_avg_accuracy": None,
        "average_forgetting": None,
        "Fu": None,
        "WorstDrop": None,
        "Au": None,
        "t_retrain": None,
        "t_forget_total": None,
        "num_updated_params": None,
        "updated_param_ratio": None,
        "adapter_param_ratio": None,
        "unlearning_score": None,
        "normalized_results": {
            "schema_version": "v1",
            "definition": {
                "average_forgetting": "avg_t(max_{k<r} a_t(k) - a_t(r)) over trained tasks with at least one past point"
            },
            "final": {},
            "unlearning_events": [],
        },
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics_state)

    debug_dir = run_dir / "debug" if args.debug_unlearning else None
    if debug_dir is not None:
        debug_dir.mkdir(exist_ok=True)

    run_context = {
        "run_dir": run_dir,
        "logger": logger,
        "metrics_state": metrics_state,
        "metrics_path": metrics_path,
        "debug_dir": debug_dir,
    }

    log_event(logger, f"[INFO] processing user requests: {user_requests_with_active_tasks}")
    model_start = time.perf_counter()
    model = methods_dict[args.method](args).to(args.device)
    init_model = model.state_dict()
    model.load_state_dict(init_model)
    model_param_blocks = extract_model_param_stats(model)
    if model_param_blocks is not None:
        model_stats = model_param_blocks.get("model_stats", {})
        adapter_stats = model_param_blocks.get("adapter_stats")
        metrics_state["model"] = model_stats
        if isinstance(adapter_stats, dict):
            metrics_state["adapter_stats"] = adapter_stats
            metrics_state["adapter_param_ratio"] = adapter_stats.get("adapter_param_ratio")
        write_json(metrics_path, metrics_state)
        log_event(
            logger,
            "[INFO] model params: total={total} trainable={trainable} task_adapter={task_adapter} "
            "shared_adapter={shared} shared_adapter_ratio={shared_ratio} trainable_ratio={ratio}".format(
                total=format_optional_int(model_stats.get("total_params")),
                trainable=format_optional_int(model_stats.get("num_trainable_params")),
                task_adapter=format_optional_int(model_stats.get("task_adapter_params", model_stats.get("num_adapter_params"))),
                shared=format_optional_int(model_stats.get("shared_adapter_params")),
                shared_ratio=format_optional_float(model_stats.get("shared_adapter_ratio")),
                ratio=format_optional_float(model_stats.get("trainable_param_ratio")),
            ),
        )
    log_event(logger, f"[INFO] model initialized on {args.device} in {time.perf_counter() - model_start:.2f}s")

    # Frozen-backbone feature caching (pretrained PEFT path only): precompute the
    # 512-d features once, then train/eval the adapters/classifier on them.
    if getattr(args, "cache_features", False):
        peft_pretrained = (
            args.method in ("pall_adapter", "lora")
            and getattr(args, "pretrained_backbone", "none") != "none"
            and getattr(getattr(model, "net", None), "frozen_backbone", None) is not None
        )
        if peft_pretrained:
            import feature_cache
            cache_start = time.perf_counter()
            train_datasets, test_datasets = feature_cache.apply_feature_cache(
                args, model, train_datasets, test_datasets
            )
            log_event(logger, f"[INFO] feature caching enabled in {time.perf_counter() - cache_start:.2f}s")
        else:
            log_event(logger, "[WARN] --cache_features ignored: only applies to pall_adapter/lora "
                              "with --pretrained_backbone imagenet_resnet18.")

    request_start = time.perf_counter()
    current_stat = process_requests(
        args,
        model,
        train_datasets,
        test_datasets,
        user_requests_with_active_tasks,
        run_context,
    )
    log_event(logger, f"[INFO] finished processing requests in {time.perf_counter() - request_start:.2f}s")
    forgetting_stats = compute_average_forgetting(
        current_stat["accuracy"],
        user_requests_with_active_tasks,
        args.n_tasks,
    )
    current_stat["avg_forgetting"] = torch.tensor(forgetting_stats["per_request"], dtype=torch.float32)

    if not metrics_state.get("unlearning_events"):
        msg = (
            "[INFO] No unlearning events in this run; compare against pall_original to "
            "confirm CL training does not regress."
        )
        metrics_state.setdefault("sanity_checks", []).append(msg)
        write_json(metrics_path, metrics_state)
        log_event(logger, msg)

    result = {
        'stats': current_stat,
        'user_requests_with_active_tasks': user_requests_with_active_tasks,
        'user_requests_without_forgotten': user_requests_without_forgotten,
    }

    torch.save(result, run_dir / "results.pth")

    if args.dump_overlap and hasattr(model, "compute_overlap_matrix"):
        overlap = model.compute_overlap_matrix(include_forgotten=True)
        task_ids = overlap.get("task_ids", [])
        matrix = overlap.get("matrix", [])
        if task_ids and matrix:
            write_overlap_csv(run_dir / "overlap.csv", task_ids, matrix)
            log_event(logger, f"[INFO] wrote overlap matrix to {run_dir / 'overlap.csv'}")

    final_acc = []
    if len(current_stat["accuracy"]) > 0:
        final_acc = to_list(current_stat["accuracy"][-1])
    final_active_tasks = user_requests_with_active_tasks[-1][2] if user_requests_with_active_tasks else []
    final_avg = avg_for_tasks(final_acc, final_active_tasks)

    summary = {
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "method": args.method,
        "method_variant": args.method_variant,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "n_tasks": args.n_tasks,
        "n_forget": args.n_forget,
        "final_avg_accuracy": final_avg,
        "average_forgetting": forgetting_stats["final"],
        "final_avg_forgetting": forgetting_stats["final"],
    }
    model_stats = metrics_state.get("model", {})
    if isinstance(model_stats, dict):
        summary["total_params"] = model_stats.get("total_params")
        summary["num_trainable_params"] = model_stats.get("num_trainable_params")
        summary["num_adapter_params"] = model_stats.get("num_adapter_params")
        summary["shared_adapter_params"] = model_stats.get("shared_adapter_params")
        summary["trainable_param_ratio"] = model_stats.get("trainable_param_ratio")
    normalized_results = metrics_state.get("normalized_results", {})
    normalized_events = normalized_results.get("unlearning_events", [])
    final_unlearning = {
        "Fu": None,
        "WorstDrop": None,
        "Au": None,
        "mia": None,
        "probe": None,
        "t_reset": None,
        "t_retrain": None,
        "t_forget_total": None,
        "num_updated_params": None,
        "overlap": {
            "s_t": None,
            "s_share": None,
            "s_share_crit": None,
            "s_share_ratio": None,
            "s_share_crit_ratio": None,
        },
    }
    if normalized_events:
        last_event = normalized_events[-1]
        final_unlearning = {
            "Fu": last_event.get("Fu"),
            "WorstDrop": last_event.get("WorstDrop"),
            "Au": last_event.get("Au"),
            "mia": last_event.get("mia"),
            "probe": last_event.get("probe"),
            "t_reset": last_event.get("t_reset"),
            "t_retrain": last_event.get("t_retrain"),
            "t_forget_total": last_event.get("t_forget_total"),
            "num_updated_params": last_event.get("num_updated_params"),
            "overlap": last_event.get("overlap", final_unlearning["overlap"]),
        }
    adapter_stats = metrics_state.get("adapter_stats", {})
    adapter_param_ratio = None
    updated_param_ratio = None
    if isinstance(adapter_stats, dict) and adapter_stats:
        adapter_param_ratio = _coerce_float_or_none(adapter_stats.get("adapter_param_ratio"))
        total_model_params = _coerce_int_or_none(adapter_stats.get("total_model_params"))
        num_updated_params = _coerce_int_or_none(final_unlearning.get("num_updated_params"))
        if total_model_params and num_updated_params is not None:
            updated_param_ratio = float(num_updated_params / total_model_params)
        adapter_stats["updated_param_ratio"] = updated_param_ratio
        metrics_state["adapter_stats"] = adapter_stats
        summary["task_adapter_params"] = adapter_stats.get("task_adapter_params")
        summary["adapter_params"] = adapter_stats.get("adapter_params")
        summary["adapter_param_ratio"] = adapter_param_ratio
        summary["updated_param_ratio"] = updated_param_ratio
    unlearning_score = compute_unlearning_score(
        final_unlearning.get("Fu"),
        final_unlearning.get("WorstDrop"),
        updated_param_ratio,
        lambda_1=0.5,
        lambda_2=0.5,
    )
    summary["Fu"] = final_unlearning.get("Fu")
    summary["WorstDrop"] = final_unlearning.get("WorstDrop")
    summary["Au"] = final_unlearning.get("Au")
    final_mia = final_unlearning.get("mia") if isinstance(final_unlearning.get("mia"), dict) else {}
    summary["mia_auc_before"] = final_mia.get("auc_before")
    summary["mia_auc_after"] = final_mia.get("auc_after")
    summary["mia_acc_before"] = final_mia.get("acc_before")
    summary["mia_acc_after"] = final_mia.get("acc_after")
    final_probe = final_unlearning.get("probe") if isinstance(final_unlearning.get("probe"), dict) else {}
    summary["probe_acc_before"] = final_probe.get("acc_before")
    summary["probe_acc_after"] = final_probe.get("acc_after")
    summary["t_retrain"] = final_unlearning.get("t_retrain")
    summary["t_forget_total"] = final_unlearning.get("t_forget_total")
    summary["num_updated_params"] = final_unlearning.get("num_updated_params")
    summary["unlearning_score"] = unlearning_score
    normalized_results["final"] = {
        "final_avg_accuracy": final_avg,
        "average_forgetting": forgetting_stats["final"],
        "final_avg_forgetting": forgetting_stats["final"],
        "Fu": final_unlearning.get("Fu"),
        "WorstDrop": final_unlearning.get("WorstDrop"),
        "Au": final_unlearning.get("Au"),
        "mia_auc_before": final_mia.get("auc_before"),
        "mia_auc_after": final_mia.get("auc_after"),
        "mia_acc_before": final_mia.get("acc_before"),
        "mia_acc_after": final_mia.get("acc_after"),
        "probe_acc_before": final_probe.get("acc_before"),
        "probe_acc_after": final_probe.get("acc_after"),
        "unlearning_score": unlearning_score,
        "t_retrain": final_unlearning.get("t_retrain"),
        "t_forget_total": final_unlearning.get("t_forget_total"),
        "num_updated_params": final_unlearning.get("num_updated_params"),
        "updated_param_ratio": updated_param_ratio,
        "adapter_param_ratio": adapter_param_ratio,
        "num_unlearning_events": len(normalized_events),
        "final_unlearning": final_unlearning,
    }
    metrics_state["normalized_results"] = normalized_results
    metrics_state["forgetting"] = forgetting_stats
    metrics_state["final_avg_accuracy"] = final_avg
    metrics_state["average_forgetting"] = forgetting_stats["final"]
    metrics_state["Fu"] = final_unlearning.get("Fu")
    metrics_state["WorstDrop"] = final_unlearning.get("WorstDrop")
    metrics_state["Au"] = final_unlearning.get("Au")
    metrics_state["mia_auc_before"] = final_mia.get("auc_before")
    metrics_state["mia_auc_after"] = final_mia.get("auc_after")
    metrics_state["mia_acc_before"] = final_mia.get("acc_before")
    metrics_state["mia_acc_after"] = final_mia.get("acc_after")
    metrics_state["probe_acc_before"] = final_probe.get("acc_before")
    metrics_state["probe_acc_after"] = final_probe.get("acc_after")
    metrics_state["t_retrain"] = final_unlearning.get("t_retrain")
    metrics_state["t_forget_total"] = final_unlearning.get("t_forget_total")
    metrics_state["num_updated_params"] = final_unlearning.get("num_updated_params")
    metrics_state["updated_param_ratio"] = updated_param_ratio
    metrics_state["adapter_param_ratio"] = adapter_param_ratio
    metrics_state["unlearning_score"] = unlearning_score
    metrics_state["summary"] = summary
    write_json(metrics_path, metrics_state)
    write_summary(run_dir / "summary.txt", summary, metrics_state.get("normalized_results"))
    try:
        write_run_report(run_dir / "report.md", run_dir, config, metrics_state)
        log_event(logger, f"[INFO] wrote run report to {run_dir / 'report.md'}")
    except Exception as exc:
        log_event(logger, f"[WARN] failed to write run report: {exc}")
    log_event(logger, f"[UNLEARNING_SCORE] value={format_optional_float(unlearning_score)}")
    log_event(logger, f"[INFO] total run time: {time.perf_counter() - run_start:.2f}s")


if __name__ == "__main__":
    main()
