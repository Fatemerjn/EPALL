#!/usr/bin/env python3
"""Read-only sequential-forgetting damage analysis from completed metrics.json files.

The selector mirrors the explicit MAIN configurations in
``tools/run_server_experiments.sh``.  It accepts only the canonical MAIN tag (or
the matching ``probe_v1`` audit rerun), requires exactly three raw
``unlearning_events``, and de-duplicates by keeping the newest completed run per
dataset/method/seed.  No training code is invoked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "font-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_selection import select_latest_seed_rows


DATASETS = ("cifar100", "cifar10")
METHODS = ("pall_original", "pall_modified", "pall_adapter", "lora", "clpu")
SEEDS = (0, 1)
METHOD_LABELS = {
    "pall_original": "PALL Original",
    "pall_modified": "PALL Modified",
    "pall_adapter": "PALL Adapter",
    "lora": "LoRA",
    "clpu": "CLPU",
}
METHOD_COLORS = {
    "pall_original": "#D55E00",
    "pall_modified": "#0072B2",
    "pall_adapter": "#009E73",
    "lora": "#CC79A7",
    "clpu": "#E69F00",
}

DATASET_MAIN = {
    "cifar10": {
        "class_per_task": 2,
        "n_tasks": 5,
        "sparsity": 0.8,
        "base_arch": "resnet18",
        "schedule_prefix": "cifar10_t5_f3_fixed_seed",
    },
    "cifar100": {
        "class_per_task": 5,
        "n_tasks": 10,
        "sparsity": 0.9,
        "base_arch": "resnet34",
        "schedule_prefix": "cifar100_t10_f3_seed",
    },
}

COMMON_MAIN = {
    "n_forget": 3,
    "n_epochs": 3,
    "k_shot": 50,
    "alpha": 0.5,
    "beta": 1.0,
    "mem_budget": 500,
    "optim": "sgd",
    "momentum": 0.9,
    "lr": 1e-2,
    "weight_decay": 5e-4,
    "batch_size": 32,
    "deterministic": True,
}

METHOD_MAIN = {
    "pall_original": {
        "retrain_steps": 50,
        "protect_ratio": None,
        "lambda_protect": 0.0,
    },
    "pall_modified": {
        "protect_importance": "gradient",
        "protect_ratio": 0.2,
        "lambda_protect": 1.0,
        "retrain_steps": 50,
    },
    "pall_adapter": {
        "adapter_bottleneck": 16,
        "adapter_shared_bottleneck": 16,
        "adapter_shared_forget_ratio": 0.3,
        "adapter_shared_protect_ratio": 0.2,
        "adapter_train_classifier": True,
        "retrain_steps": 50,
        "adapter_forget_steps": 10,
    },
    "lora": {
        "lora_rank": 8,
        "lora_alpha": 16,
    },
    "clpu": {},
}

RAW_COLUMNS = (
    "dataset",
    "role",
    "method",
    "seed",
    "event_index",
    "event_local_worstdrop",
    "cumulative_survivor_damage",
    "signed_mean_retained_accuracy_change",
    "survivor_tasks",
    "survivors_not_active_after_event1",
    "experiment_tag",
    "run_path",
)
SUMMARY_COLUMNS = (
    "dataset",
    "role",
    "method",
    "event_index",
    "n_seeds",
    "event_local_worstdrop_mean",
    "event_local_worstdrop_std",
    "cumulative_survivor_damage_mean",
    "cumulative_survivor_damage_std",
    "signed_mean_retained_accuracy_change_mean",
    "signed_mean_retained_accuracy_change_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cumulative survivor damage from existing three-event MAIN-equivalent runs."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/thesis/sequential_damage"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def values_equal(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def expected_arch(dataset: str, method: str) -> str:
    base = str(DATASET_MAIN[dataset]["base_arch"])
    if method in {"pall_original", "pall_modified"}:
        return f"subnet_{base}"
    if method == "pall_adapter":
        return f"adapter_{base}"
    if method == "lora":
        return f"lora_{base}"
    return base


def expected_schedule(dataset: str, seed: int) -> str:
    return f"{DATASET_MAIN[dataset]['schedule_prefix']}{seed}.json"


def allowed_tags(dataset: str, method: str) -> Tuple[str, str]:
    canonical = f"{dataset}_baselines_v2" if method == "clpu" else f"{dataset}_main"
    return canonical, "probe_v1"


def expected_config(dataset: str, method: str, seed: int) -> Dict[str, Any]:
    dataset_main = DATASET_MAIN[dataset]
    expected: Dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "class_per_task": dataset_main["class_per_task"],
        "n_tasks": dataset_main["n_tasks"],
        "sparsity": dataset_main["sparsity"],
        "arch": expected_arch(dataset, method),
        **COMMON_MAIN,
        **METHOD_MAIN[method],
    }
    return expected


def config_mismatches(config: Mapping[str, Any], dataset: str, method: str, seed: int) -> List[str]:
    mismatches = []
    for key, expected in expected_config(dataset, method, seed).items():
        if not values_equal(config.get(key), expected):
            mismatches.append(f"{key}={config.get(key)!r} (expected {expected!r})")
    schedule = Path(str(config.get("request_schedule_file") or "")).name
    wanted_schedule = expected_schedule(dataset, seed)
    if schedule != wanted_schedule:
        mismatches.append(f"schedule={schedule!r} (expected {wanted_schedule!r})")
    return mismatches


def main_signature(config: Mapping[str, Any], dataset: str, method: str, seed: int) -> Tuple[Any, ...]:
    """Signature of every explicit training argument in the corresponding MAIN command."""

    keys = list(expected_config(dataset, method, seed))
    values: List[Any] = [config.get(key) for key in keys]
    values.append(Path(str(config.get("request_schedule_file") or "")).name)
    return tuple(values)


def raw_events(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    events = metrics.get("unlearning_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def event_payload_is_complete(events: Sequence[Mapping[str, Any]]) -> bool:
    if len(events) != 3:
        return False
    first_baseline = events[0].get("per_task_acc_before")
    final_survivors = events[-1].get("remaining_tasks")
    if not isinstance(first_baseline, dict) or not isinstance(final_survivors, list) or not final_survivors:
        return False
    survivor_keys = {str(int(task)) for task in final_survivors}
    if not survivor_keys.issubset(first_baseline):
        return False
    for event in events:
        if not isinstance(event.get("WorstDrop"), (int, float)):
            return False
        post = event.get("per_task_acc_after_retrain")
        if not isinstance(post, dict) or not survivor_keys.issubset(post):
            return False
    return True


def iter_preliminary_candidates(runs_root: Path) -> Iterable[Dict[str, Any]]:
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                pattern = f"{dataset}/**/{method}/seed_{seed}/*/metrics.json"
                for metrics_path in sorted(runs_root.glob(pattern)):
                    config_path = metrics_path.with_name("config.json")
                    if not config_path.exists():
                        continue
                    try:
                        metrics = load_json(metrics_path)
                        config = load_json(config_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
                    run = metrics.get("run") if isinstance(metrics.get("run"), dict) else {}
                    tag = str(run.get("experiment_tag") or "")
                    if tag not in allowed_tags(dataset, method):
                        continue
                    if str(config.get("experiment_tag") or "") != tag:
                        continue
                    if (
                        run.get("dataset") != dataset
                        or run.get("method") != method
                        or not values_equal(run.get("seed"), seed)
                        or not values_equal(run.get("n_forget"), 3)
                    ):
                        continue
                    events = raw_events(metrics)
                    if len(events) != 3 or not event_payload_is_complete(events):
                        continue
                    if config_mismatches(config, dataset, method, seed):
                        continue
                    yield {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "experiment_tag": tag,
                        "run_path": metrics_path.parent.as_posix(),
                        "metrics_path": metrics_path,
                        "config": config,
                        "metrics": metrics,
                        "events": events,
                    }


def filter_clpu_probe_matches(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["method"] != "clpu" or candidate["experiment_tag"] != "probe_v1":
            filtered.append(candidate)
            continue
        dataset = str(candidate["dataset"])
        seed = int(candidate["seed"])
        baseline_tag = f"{dataset}_baselines_v2"
        references = [
            row
            for row in candidates
            if row["dataset"] == dataset
            and row["method"] == "clpu"
            and int(row["seed"]) == seed
            and row["experiment_tag"] == baseline_tag
        ]
        if not references:
            continue
        reference = max(references, key=lambda row: Path(str(row["run_path"])).name)
        if main_signature(candidate["config"], dataset, "clpu", seed) != main_signature(
            reference["config"], dataset, "clpu", seed
        ):
            continue
        candidate = dict(candidate)
        candidate["clpu_main_reference"] = reference["run_path"]
        filtered.append(candidate)
    return filtered


def select_runs(runs_root: Path) -> List[Dict[str, Any]]:
    candidates = filter_clpu_probe_matches(list(iter_preliminary_candidates(runs_root)))
    selected, _ = select_latest_seed_rows(
        candidates,
        ("dataset", "method"),
        seed_column="seed",
        path_column="run_path",
    )
    by_key = {(row["dataset"], row["method"], int(row["seed"])): row for row in selected}
    missing = [
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in SEEDS
        if (dataset, method, seed) not in by_key
    ]
    if missing:
        formatted = ", ".join(f"{d}/{m}/seed_{s}" for d, m, s in missing)
        raise RuntimeError(f"no completed MAIN-equivalent run for: {formatted}")
    return [by_key[(dataset, method, seed)] for dataset in DATASETS for method in METHODS for seed in SEEDS]


def print_and_verify_selection(selected: Sequence[Mapping[str, Any]]) -> None:
    print("SELECTED RUNS (latest completed, de-duplicated, MAIN-equivalent):")
    for row in selected:
        event_count = len(row["events"])
        if event_count != 3:
            raise RuntimeError(f"selection invariant failed for {row['run_path']}: {event_count} events")
        suffix = ""
        if row.get("clpu_main_reference"):
            suffix = f" | CLPU MAIN signature match: {row['clpu_main_reference']}"
        print(
            f"  {row['dataset']} | {row['method']} | seed {row['seed']} | "
            f"tag={row['experiment_tag']} | events={event_count} | {row['run_path']}{suffix}"
        )
    print("VERIFIED: all 20 selected runs contain exactly 3 raw unlearning_events.\n")


def as_accuracy_map(value: Any, context: str) -> Dict[int, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected per-task accuracy object")
    result: Dict[int, float] = {}
    for key, accuracy in value.items():
        result[int(key)] = float(accuracy)
    return result


def analyze_run(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    events: Sequence[Mapping[str, Any]] = row["events"]
    survivors = tuple(int(task) for task in events[-1]["remaining_tasks"])
    active_after_first = {int(task) for task in events[0]["remaining_tasks"]}
    not_active_after_first = tuple(task for task in survivors if task not in active_after_first)
    baseline = as_accuracy_map(events[0]["per_task_acc_before"], f"{row['run_path']} first event")
    output: List[Dict[str, Any]] = []
    for event_index, event in enumerate(events, start=1):
        post = as_accuracy_map(event["per_task_acc_after_retrain"], f"{row['run_path']} event {event_index}")
        drops = [baseline[task] - post[task] for task in survivors]
        changes = [post[task] - baseline[task] for task in survivors]
        output.append(
            {
                "dataset": row["dataset"],
                "role": "primary" if row["dataset"] == "cifar100" else "secondary",
                "method": row["method"],
                "seed": int(row["seed"]),
                "event_index": event_index,
                "event_local_worstdrop": float(event["WorstDrop"]),
                "cumulative_survivor_damage": max(0.0, max(drops)),
                "signed_mean_retained_accuracy_change": statistics.fmean(changes),
                "survivor_tasks": " ".join(str(task) for task in survivors),
                "survivors_not_active_after_event1": " ".join(str(task) for task in not_active_after_first),
                "experiment_tag": row["experiment_tag"],
                "run_path": row["run_path"],
            }
        )
    return output


def sample_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_rows(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for event_index in (1, 2, 3):
                rows = [
                    row
                    for row in raw_rows
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and int(row["event_index"]) == event_index
                ]
                if len(rows) != len(SEEDS):
                    raise RuntimeError(f"expected two seed rows for {dataset}/{method}/event_{event_index}")
                local_mean, local_std = sample_mean_std([float(row["event_local_worstdrop"]) for row in rows])
                damage_mean, damage_std = sample_mean_std(
                    [float(row["cumulative_survivor_damage"]) for row in rows]
                )
                signed_mean, signed_std = sample_mean_std(
                    [float(row["signed_mean_retained_accuracy_change"]) for row in rows]
                )
                output.append(
                    {
                        "dataset": dataset,
                        "role": "primary" if dataset == "cifar100" else "secondary",
                        "method": method,
                        "event_index": event_index,
                        "n_seeds": len(rows),
                        "event_local_worstdrop_mean": local_mean,
                        "event_local_worstdrop_std": local_std,
                        "cumulative_survivor_damage_mean": damage_mean,
                        "cumulative_survivor_damage_std": damage_std,
                        "signed_mean_retained_accuracy_change_mean": signed_mean,
                        "signed_mean_retained_accuracy_change_std": signed_std,
                    }
                )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8f}" if isinstance(value, float) else value
                    for key, value in row.items()
                    if key in columns
                }
            )


def summary_lookup(summary_rows: Sequence[Mapping[str, Any]], dataset: str, method: str, event: int) -> Mapping[str, Any]:
    return next(
        row
        for row in summary_rows
        if row["dataset"] == dataset and row["method"] == method and int(row["event_index"]) == event
    )


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def damage_trend(summary_rows: Sequence[Mapping[str, Any]], dataset: str, method: str) -> Tuple[str, float]:
    first = float(summary_lookup(summary_rows, dataset, method, 1)["cumulative_survivor_damage_mean"])
    third = float(summary_lookup(summary_rows, dataset, method, 3)["cumulative_survivor_damage_mean"])
    delta = third - first
    if delta > 1e-12:
        return "grew", delta
    if delta < -1e-12:
        return "fell", delta
    return "was unchanged", delta


def write_markdown(
    path: Path,
    selected: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Sequential-forgetting survivor damage",
        "",
        "CIFAR-100 is the primary analysis; CIFAR-10 is secondary. All values are read from existing raw `unlearning_events` in the latest completed, MAIN-equivalent run per dataset/method/seed.",
        "",
        "## Selected runs",
        "",
        "| Dataset | Method | Seed | Tag | Events | Run path |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['dataset']} | {METHOD_LABELS[row['method']]} | {row['seed']} | "
            f"{row['experiment_tag']} | {len(row['events'])} | `{row['run_path']}` |"
        )
    lines.extend(
        [
            "",
            "All selected runs have exactly three events. For each CLPU `probe_v1` run, the explicit training signature matches its dataset/seed `*_baselines_v2` MAIN reference; `eval_probe` and the tag are audit-only differences.",
            "",
            "## Per-seed event values",
            "",
            "| Dataset | Method | Seed | Event | Event-local WorstDrop | Cumulative survivor damage | Signed mean retained-accuracy change | Survivors | Not active after event 1 |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in raw_rows:
        lines.append(
            f"| {row['dataset']} | {METHOD_LABELS[row['method']]} | {row['seed']} | {row['event_index']} | "
            f"{row['event_local_worstdrop']:.4f} | {row['cumulative_survivor_damage']:.4f} | "
            f"{row['signed_mean_retained_accuracy_change']:+.4f} | {row['survivor_tasks']} | "
            f"{row['survivors_not_active_after_event1'] or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Across-seed summary (mean ± sample std)",
            "",
            "| Dataset | Role | Method | Event | Event-local WorstDrop | Cumulative survivor damage | Signed mean retained-accuracy change |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['role']} | {METHOD_LABELS[row['method']]} | {row['event_index']} | "
            f"{format_mean_std(row['event_local_worstdrop_mean'], row['event_local_worstdrop_std'])} | "
            f"{format_mean_std(row['cumulative_survivor_damage_mean'], row['cumulative_survivor_damage_std'])} | "
            f"{format_mean_std(row['signed_mean_retained_accuracy_change_mean'], row['signed_mean_retained_accuracy_change_std'])} |"
        )
    lines.extend(["", "## Descriptive interpretation", ""])
    for dataset in DATASETS:
        role = "primary" if dataset == "cifar100" else "secondary"
        trend_parts = []
        stable = []
        for method in METHODS:
            trend, delta = damage_trend(summary_rows, dataset, method)
            trend_parts.append(f"{METHOD_LABELS[method]} {trend} by {delta:+.4f}")
            method_damage = [
                float(summary_lookup(summary_rows, dataset, method, event)["cumulative_survivor_damage_mean"])
                for event in (1, 2, 3)
            ]
            if max(method_damage) - min(method_damage) <= 0.01:
                stable.append(METHOD_LABELS[method])
        stable_text = ", ".join(stable) if stable else "none"
        lines.append(
            f"- **{dataset} ({role}):** From event 1 to event 3, "
            + "; ".join(trend_parts)
            + f". Under the explicit descriptive stability rule (range across the three event means ≤ 0.01), stable methods are: {stable_text}."
        )
    lines.extend(
        [
            "- Cumulative damage is clipped at zero by definition, while the signed retained-accuracy change is reported alongside it so repair/improvement is not hidden.",
            "- The request stream interleaves learning and forgetting. Some fixed-set survivors were not yet active after event 1, so positive signed changes can include later task learning and must not be interpreted as a causal repair effect.",
            "- Any between-method separation is descriptive only. Two seeds and three sequential events do not support a strong significance or generalization claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_dataset(
    dataset: str,
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = [1, 2, 3]
    for method in METHODS:
        rows = [summary_lookup(summary_rows, dataset, method, event) for event in x]
        mean = [float(row["cumulative_survivor_damage_mean"]) for row in rows]
        std = [float(row["cumulative_survivor_damage_std"]) for row in rows]
        lower = [max(0.0, value - spread) for value, spread in zip(mean, std)]
        upper = [value + spread for value, spread in zip(mean, std)]
        color = METHOD_COLORS[method]
        ax.plot(x, mean, marker="o", linewidth=1.5, markersize=3.5, color=color, label=METHOD_LABELS[method])
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
    role = "Primary" if dataset == "cifar100" else "Secondary"
    ax.set_title(f"Sequential survivor damage — {dataset.upper()} ({role})")
    ax.set_xlabel("Forget event index")
    ax.set_ylabel("Cumulative survivor damage")
    ax.set_xticks(x)
    ax.set_xlim(0.85, 3.15)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_event_values(raw_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    print("PER-EVENT VALUES:")
    for dataset in DATASETS:
        print(f"  {dataset} ({'primary' if dataset == 'cifar100' else 'secondary'}):")
        for method in METHODS:
            print(f"    {METHOD_LABELS[method]}:")
            for event in (1, 2, 3):
                seeds = [
                    row
                    for row in raw_rows
                    if row["dataset"] == dataset and row["method"] == method and row["event_index"] == event
                ]
                summary = summary_lookup(summary_rows, dataset, method, event)
                seed_text = "; ".join(
                    f"s{row['seed']} local={row['event_local_worstdrop']:.4f}, "
                    f"damage={row['cumulative_survivor_damage']:.4f}, "
                    f"signed={row['signed_mean_retained_accuracy_change']:+.4f}"
                    for row in seeds
                )
                print(
                    f"      event {event}: {seed_text} | mean±std damage="
                    f"{format_mean_std(summary['cumulative_survivor_damage_mean'], summary['cumulative_survivor_damage_std'])}"
                )


def main() -> int:
    args = parse_args()
    try:
        selected = select_runs(args.runs_root)
        # Selection and the exact three-event invariant are printed before any
        # metric is computed or output file is written.
        print_and_verify_selection(selected)
        raw_rows = [event_row for row in selected for event_row in analyze_run(row)]
        summary_rows = aggregate_rows(raw_rows)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    raw_csv = args.outdir / "sequential_damage_per_seed.csv"
    summary_csv = args.outdir / "sequential_damage_summary.csv"
    markdown = args.outdir / "sequential_damage.md"
    write_csv(raw_csv, raw_rows, RAW_COLUMNS)
    write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    write_markdown(markdown, selected, raw_rows, summary_rows)

    pdfs = []
    for dataset in DATASETS:
        pdf = args.outdir / f"sequential_damage_{dataset}.pdf"
        plot_dataset(dataset, summary_rows, pdf, args.dpi)
        pdfs.append(pdf)

    print_event_values(raw_rows, summary_rows)
    print("\nOUTPUTS:")
    for path in (raw_csv, summary_csv, markdown, *pdfs):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
