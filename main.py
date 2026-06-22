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
        'pall',            # deprecated alias for pall_modified (warns)
        'pall_original',
        'pall_modified',   # main method (default)
        'pall_adapter',
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
parser.add_argument('--lambda_protect', default=0.0, type=float, help='regularization weight for protected params')
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
parser.set_defaults(pin_memory=None)
args = parser.parse_args()

PALL_METHODS = {"pall", "pall_original", "pall_modified"}
ADAPTER_METHODS = {"pall_adapter"}


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
    return method


def validate_experiment_args(arg_namespace):
    if arg_namespace.dataset == "cifar100":
        if arg_namespace.class_per_task != 5:
            parser.error("CIFAR-100 superclass tasks require --class_per_task 5.")
        if not (1 <= arg_namespace.n_tasks <= 20):
            parser.error("CIFAR-100 superclass tasks require --n_tasks in [1, 20].")
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

            avg_after_reset = avg_for_tasks(after_reset_acc, remaining_tasks)
            avg_after_retrain = avg_for_tasks(post_acc, remaining_tasks)
            fu = avg_before - avg_after_retrain
            worst_drop = 0.0
            if remaining_tasks:
                worst_drop = max(pre_acc[t] - post_acc[t] for t in remaining_tasks)
            au = post_acc[task_id] if task_id < len(post_acc) else 0.0
            finetune_diag = json_safe(info.get("finetune_diag", None))

            event = {
                "unlearning_step": unlearning_step,
                "request_id": request_id,
                "task_id": task_id,
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
                "t_reset": info.get("t_reset", 0.0) if info.get("t_reset") is not None else 0.0,
                "t_retrain": info.get("t_retrain", 0.0) if info.get("t_retrain") is not None else 0.0,
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
            "t_reset {t_reset}s t_retrain {t_retrain}s t_forget_total {t_total}s "
            "updated_params {updated} share_ratio {share_ratio} crit_ratio {crit_ratio}".format(
                fu=format_optional_float(final_unlearning.get("Fu")),
                worst_drop=format_optional_float(final_unlearning.get("WorstDrop")),
                au=format_optional_float(final_unlearning.get("Au")),
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
        lines.append(
            "step {step} task {task} avg_before {avg_before} "
            "avg_after_reset {avg_after_reset} avg_after_retrain {avg_after_retrain} "
            "Fu {fu} WorstDrop {worst_drop} Au {au} "
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
        final_unlearning = {
            "Fu": last.get("Fu"),
            "WorstDrop": last.get("WorstDrop"),
            "Au": last.get("Au"),
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
        "pall": PALLModified,        # deprecated alias -> main method
        "pall_original": PALLOriginal,
        "pall_modified": PALLModified,
        "pall_adapter": PALLAdapter,
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
