#!/usr/bin/env python3
"""
Build a thesis-ready summary table directly from run metrics.

Usage
-----
python3 tools/make_thesis_table.py \
  --root runs \
  --out-csv results/aggregates/thesis_table.csv \
  --out-md results/aggregates/thesis_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from run_selection import seed_key as canonical_seed_key
    from run_selection import select_latest_seed_rows
except ImportError:  # pragma: no cover - supports package-style imports
    from tools.run_selection import seed_key as canonical_seed_key
    from tools.run_selection import select_latest_seed_rows


CANONICAL_METHOD_VARIANTS = {
    "adapter_hard_critical_mask": "pall_adapter_hard_mask",
    "adapter_explicit_critical_mask": "pall_adapter_soft_mask",
}

_SMOKE_TAG_PREFIXES = ("smoke", "test")


OUTPUT_COLUMNS = [
    "dataset",
    "method",
    "final_avg_acc_mean",
    "final_avg_acc_std",
    "avg_forgetting_mean",
    "avg_forgetting_std",
    "Fu_mean",
    "Fu_std",
    "WorstDrop_mean",
    "WorstDrop_std",
    "Au_mean",
    "Au_std",
    "grad_norm_ratio_mean",
    "grad_norm_ratio_std",
    "mia_auc_before_mean",
    "mia_auc_before_std",
    "mia_auc_after_mean",
    "mia_auc_after_std",
    "unlearning_score_mean",
    "unlearning_score_std",
    "t_retrain_mean",
    "t_forget_total_mean",
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "overlap_shared_critical_ratio",
    "overlap_protected_ratio",
    "overlap_updated_ratio",
    "overlap_shared_critical_count",
    "overlap_protected_params",
    "overlap_updated_params",
]

CONFIG_GROUP_COLUMNS = [
    "experiment_tag",
    "protect_importance",
    "protect_ratio",
    "lambda_protect",
    "protect_anchor",
    "adapter_bottleneck",
    "adapter_shared_bottleneck",
    "adapter_shared_forget_ratio",
    "adapter_shared_protect_ratio",
    "adapter_forget_steps",
    "adapter_shared_forget_lr",
    "adapter_shared_protect_strength",
    "retrain_steps",
    "adapter_train_classifier",
]

SEED_AGG_METRICS = [
    "final_avg_accuracy",
    "average_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "grad_norm_ratio",
    "mia_auc_before",
    "mia_auc_after",
    "unlearning_score",
    "t_retrain",
    "t_forget_total",
    "updated_param_ratio",
    "adapter_param_ratio",
    "overlap_shared_critical_ratio",
    "overlap_protected_ratio",
    "overlap_updated_ratio",
    "overlap_shared_critical_count",
    "overlap_protected_params",
    "overlap_updated_params",
]


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def canonicalize_method_variant(name: Any) -> str:
    if name is None:
        return ""
    text = str(name).strip()
    if text == "" or text.lower() == "none":
        return ""
    return CANONICAL_METHOD_VARIANTS.get(text, text)


def nested_get(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse JSON: {path} ({exc})", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[WARN] Failed to read JSON: {path} ({exc})", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"[WARN] JSON root is not an object: {path}", file=sys.stderr)
        return None
    return data


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def find_metrics_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("metrics.json"))


def config_group_value(config: Dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value is None:
        return ""
    return str(value)


def is_smoke_tag(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower().startswith(_SMOKE_TAG_PREFIXES)


def get_last_unlearning_event(metrics: Dict[str, Any]) -> Dict[str, Any]:
    events = metrics.get("unlearning_events")
    if isinstance(events, list) and events and isinstance(events[-1], dict):
        return events[-1]
    normalized_events = nested_get(metrics, "normalized_results", "unlearning_events")
    if isinstance(normalized_events, list) and normalized_events and isinstance(normalized_events[-1], dict):
        return normalized_events[-1]
    return {}


def get_final_unlearning(metrics: Dict[str, Any]) -> Dict[str, Any]:
    final_unlearning = nested_get(metrics, "normalized_results", "final", "final_unlearning")
    if isinstance(final_unlearning, dict):
        return final_unlearning
    return get_last_unlearning_event(metrics)


def get_overlap_analysis(metrics: Dict[str, Any], final_unlearning: Dict[str, Any], raw_last_event: Dict[str, Any]) -> Dict[str, Any]:
    candidates = (
        nested_get(metrics, "normalized_results", "final", "overlap_analysis"),
        nested_get(metrics, "normalized_results", "final", "protection", "overlap_analysis"),
        final_unlearning.get("overlap_analysis"),
        nested_get(final_unlearning, "protection", "overlap_analysis"),
        raw_last_event.get("overlap_analysis"),
        nested_get(raw_last_event, "protection", "overlap_analysis"),
        metrics.get("overlap_analysis"),
        nested_get(metrics, "protection", "overlap_analysis"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def derive_adapter_param_ratio(metrics: Dict[str, Any]) -> Optional[float]:
    adapter_stats = metrics.get("adapter_stats") if isinstance(metrics.get("adapter_stats"), dict) else {}
    model_stats = metrics.get("model") if isinstance(metrics.get("model"), dict) else {}
    ratio = first_non_none(
        metrics.get("adapter_param_ratio"),
        nested_get(metrics, "normalized_results", "final", "adapter_param_ratio"),
        nested_get(metrics, "summary", "adapter_param_ratio"),
        adapter_stats.get("adapter_param_ratio"),
        model_stats.get("adapter_param_ratio"),
    )
    ratio = to_float(ratio)
    if ratio is not None:
        return ratio

    total_model_params = to_float(
        first_non_none(
            adapter_stats.get("total_model_params"),
            model_stats.get("total_params"),
        )
    )
    task_adapter_params = to_float(
        first_non_none(
            adapter_stats.get("task_adapter_params"),
            model_stats.get("task_adapter_params"),
            model_stats.get("num_adapter_params"),
        )
    )
    shared_adapter_params = to_float(
        first_non_none(
            adapter_stats.get("shared_adapter_params"),
            model_stats.get("shared_adapter_params"),
        )
    )
    adapter_params = to_float(
        first_non_none(
            adapter_stats.get("adapter_params"),
            model_stats.get("adapter_params"),
        )
    )
    if adapter_params is None and (task_adapter_params is not None or shared_adapter_params is not None):
        adapter_params = float((task_adapter_params or 0.0) + (shared_adapter_params or 0.0))
    if total_model_params and adapter_params is not None:
        return float(adapter_params / total_model_params)
    return None


def derive_updated_param_ratio(metrics: Dict[str, Any], final_unlearning: Dict[str, Any]) -> Optional[float]:
    adapter_stats = metrics.get("adapter_stats") if isinstance(metrics.get("adapter_stats"), dict) else {}
    ratio = first_non_none(
        metrics.get("updated_param_ratio"),
        nested_get(metrics, "normalized_results", "final", "updated_param_ratio"),
        nested_get(metrics, "summary", "updated_param_ratio"),
        adapter_stats.get("updated_param_ratio"),
    )
    ratio = to_float(ratio)
    if ratio is not None:
        return ratio

    total_model_params = to_float(
        first_non_none(
            adapter_stats.get("total_model_params"),
            nested_get(metrics, "model", "total_params"),
        )
    )
    num_updated_params = to_float(
        first_non_none(
            metrics.get("num_updated_params"),
            nested_get(metrics, "normalized_results", "final", "num_updated_params"),
            final_unlearning.get("num_updated_params"),
        )
    )
    if total_model_params and num_updated_params is not None:
        return float(num_updated_params / total_model_params)
    return None


def derive_unlearning_score(metrics: Dict[str, Any], final_unlearning: Dict[str, Any]) -> Optional[float]:
    score = first_non_none(
        metrics.get("unlearning_score"),
        nested_get(metrics, "normalized_results", "final", "unlearning_score"),
        nested_get(metrics, "summary", "unlearning_score"),
    )
    score = to_float(score)
    if score is not None:
        return score

    fu = to_float(
        first_non_none(
            metrics.get("Fu"),
            nested_get(metrics, "normalized_results", "final", "Fu"),
            final_unlearning.get("Fu"),
        )
    )
    worst_drop = to_float(
        first_non_none(
            metrics.get("WorstDrop"),
            nested_get(metrics, "normalized_results", "final", "WorstDrop"),
            final_unlearning.get("WorstDrop"),
        )
    )
    updated_param_ratio = derive_updated_param_ratio(metrics, final_unlearning)
    if fu is None or worst_drop is None or updated_param_ratio is None:
        return None
    return float(fu - 0.5 * worst_drop - 0.5 * updated_param_ratio)


def extract_run_row(
    metrics_path: Path,
    group_by_config: bool = False,
    include_tags: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    metrics = load_json(metrics_path)
    if metrics is None:
        return None
    config = load_json(metrics_path.with_name("config.json")) or {}
    experiment_tag = config_group_value(config, "experiment_tag")
    if include_tags is not None and experiment_tag not in include_tags:
        return None

    final_unlearning = get_final_unlearning(metrics)
    raw_last_event = get_last_unlearning_event(metrics)
    overlap_analysis = get_overlap_analysis(metrics, final_unlearning, raw_last_event)

    dataset = first_non_none(
        metrics.get("dataset"),
        nested_get(metrics, "summary", "dataset"),
        nested_get(metrics, "run", "dataset"),
    )
    method = first_non_none(
        metrics.get("method"),
        nested_get(metrics, "summary", "method"),
        nested_get(metrics, "run", "method"),
    )
    if dataset is None or method is None:
        print(f"[WARN] Skipping metrics without dataset/method: {metrics_path}", file=sys.stderr)
        return None

    t_retrain = to_float(
        first_non_none(
            metrics.get("t_retrain"),
            nested_get(metrics, "normalized_results", "final", "t_retrain"),
            final_unlearning.get("t_retrain"),
            raw_last_event.get("t_retrain"),
        )
    )
    t_forget_total = to_float(
        first_non_none(
            metrics.get("t_forget_total"),
            nested_get(metrics, "normalized_results", "final", "t_forget_total"),
            final_unlearning.get("t_forget_total"),
            raw_last_event.get("t_forget_total"),
        )
    )
    if t_forget_total is None:
        t_reset = to_float(first_non_none(final_unlearning.get("t_reset"), raw_last_event.get("t_reset")))
        if t_reset is not None or t_retrain is not None:
            t_forget_total = float((t_reset or 0.0) + (t_retrain or 0.0))
    final_mia = final_unlearning.get("mia", {}) if isinstance(final_unlearning.get("mia"), dict) else {}
    raw_mia = raw_last_event.get("mia", {}) if isinstance(raw_last_event.get("mia"), dict) else {}
    mia_auc_before = to_float(
        first_non_none(
            final_mia.get("auc_before"),
            nested_get(metrics, "normalized_results", "final", "mia_auc_before"),
            metrics.get("mia_auc_before"),
            nested_get(metrics, "summary", "mia_auc_before"),
            raw_mia.get("auc_before"),
            nested_get(raw_mia, "before", "auc"),
            nested_get(raw_mia, "before", "auc_loss"),
        )
    )
    mia_auc_after = to_float(
        first_non_none(
            final_mia.get("auc_after"),
            nested_get(metrics, "normalized_results", "final", "mia_auc_after"),
            metrics.get("mia_auc_after"),
            nested_get(metrics, "summary", "mia_auc_after"),
            raw_mia.get("auc_after"),
            nested_get(raw_mia, "after", "auc"),
            nested_get(raw_mia, "after", "auc_loss"),
        )
    )

    row = {
        "run_path": str(metrics_path.parent),
        "dataset": str(dataset),
        "method": str(method),
        "method_variant": canonicalize_method_variant(
            first_non_none(
                nested_get(metrics, "normalized_results", "final", "protection", "method_variant"),
                final_unlearning.get("method_variant"),
                nested_get(final_unlearning, "protection", "method_variant"),
                raw_last_event.get("method_variant"),
                nested_get(raw_last_event, "protection", "method_variant"),
                metrics.get("method_variant"),
                nested_get(metrics, "protection", "method_variant"),
            )
        ),
        "experiment_tag": experiment_tag,
        "seed": first_non_none(
            metrics.get("seed"),
            nested_get(metrics, "summary", "seed"),
            nested_get(metrics, "run", "seed"),
        ),
        "final_avg_accuracy": to_float(
            first_non_none(
                metrics.get("final_avg_accuracy"),
                nested_get(metrics, "normalized_results", "final", "final_avg_accuracy"),
                nested_get(metrics, "summary", "final_avg_accuracy"),
            )
        ),
        "average_forgetting": to_float(
            first_non_none(
                metrics.get("average_forgetting"),
                nested_get(metrics, "normalized_results", "final", "average_forgetting"),
                nested_get(metrics, "forgetting", "final"),
                nested_get(metrics, "summary", "average_forgetting"),
                nested_get(metrics, "summary", "final_avg_forgetting"),
            )
        ),
        "Fu": to_float(
            first_non_none(
                metrics.get("Fu"),
                nested_get(metrics, "normalized_results", "final", "Fu"),
                final_unlearning.get("Fu"),
                raw_last_event.get("Fu"),
            )
        ),
        "WorstDrop": to_float(
            first_non_none(
                metrics.get("WorstDrop"),
                nested_get(metrics, "normalized_results", "final", "WorstDrop"),
                final_unlearning.get("WorstDrop"),
                raw_last_event.get("WorstDrop"),
            )
        ),
        "Au": to_float(
            first_non_none(
                metrics.get("Au"),
                nested_get(metrics, "normalized_results", "final", "Au"),
                final_unlearning.get("Au"),
                raw_last_event.get("Au"),
            )
        ),
        "grad_norm_ratio": to_float(
            first_non_none(
                nested_get(metrics, "normalized_results", "final", "grad_norm_ratio"),
                final_unlearning.get("grad_norm_ratio"),
                raw_last_event.get("grad_norm_ratio"),
            )
        ),
        "mia_auc_before": mia_auc_before,
        "mia_auc_after": mia_auc_after,
        "unlearning_score": derive_unlearning_score(metrics, final_unlearning),
        "t_retrain": t_retrain,
        "t_forget_total": t_forget_total,
        "updated_param_ratio": derive_updated_param_ratio(metrics, final_unlearning),
        "adapter_param_ratio": derive_adapter_param_ratio(metrics),
        "overlap_shared_critical_ratio": to_float(overlap_analysis.get("critical_ratio")),
        "overlap_protected_ratio": to_float(overlap_analysis.get("protected_ratio")),
        "overlap_updated_ratio": to_float(overlap_analysis.get("updated_ratio")),
        "overlap_shared_critical_count": to_float(overlap_analysis.get("shared_critical")),
        "overlap_protected_params": to_float(overlap_analysis.get("protected_params")),
        "overlap_updated_params": to_float(overlap_analysis.get("updated_params")),
    }
    if group_by_config:
        for key in CONFIG_GROUP_COLUMNS:
            row[key] = config_group_value(config, key)
    return row


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return float(statistics.mean(values)), float(statistics.stdev(values))


def seed_key(row: Dict[str, Any]) -> str:
    return canonical_seed_key(row)


def dedupe_latest_rows(rows: List[Dict[str, Any]], group_by_config: bool) -> Tuple[List[Dict[str, Any]], int]:
    group_columns = ["dataset", "method"]
    if group_by_config:
        group_columns.extend(CONFIG_GROUP_COLUMNS)
    return select_latest_seed_rows(rows, group_columns)


def values_per_seed(rows: List[Dict[str, Any]], metric: str) -> List[float]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        grouped.setdefault(seed_key(row), []).append(float(value))
    return [float(statistics.mean(values)) for values in grouped.values() if values]


def aggregate_group(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    per_metric: Dict[str, List[float]] = {}
    for metric in SEED_AGG_METRICS:
        per_metric[metric] = values_per_seed(rows, metric)

    final_avg_acc_mean, final_avg_acc_std = mean_std(per_metric["final_avg_accuracy"])
    avg_forgetting_mean, avg_forgetting_std = mean_std(per_metric["average_forgetting"])
    fu_mean, fu_std = mean_std(per_metric["Fu"])
    worst_drop_mean, worst_drop_std = mean_std(per_metric["WorstDrop"])
    au_mean, au_std = mean_std(per_metric["Au"])
    grad_norm_ratio_mean, grad_norm_ratio_std = mean_std(per_metric["grad_norm_ratio"])
    mia_auc_before_mean, mia_auc_before_std = mean_std(per_metric["mia_auc_before"])
    mia_auc_after_mean, mia_auc_after_std = mean_std(per_metric["mia_auc_after"])
    unlearning_score_mean, unlearning_score_std = mean_std(per_metric["unlearning_score"])

    return {
        "final_avg_acc_mean": final_avg_acc_mean,
        "final_avg_acc_std": final_avg_acc_std,
        "avg_forgetting_mean": avg_forgetting_mean,
        "avg_forgetting_std": avg_forgetting_std,
        "Fu_mean": fu_mean,
        "Fu_std": fu_std,
        "WorstDrop_mean": worst_drop_mean,
        "WorstDrop_std": worst_drop_std,
        "Au_mean": au_mean,
        "Au_std": au_std,
        "grad_norm_ratio_mean": grad_norm_ratio_mean,
        "grad_norm_ratio_std": grad_norm_ratio_std,
        "mia_auc_before_mean": mia_auc_before_mean,
        "mia_auc_before_std": mia_auc_before_std,
        "mia_auc_after_mean": mia_auc_after_mean,
        "mia_auc_after_std": mia_auc_after_std,
        "unlearning_score_mean": unlearning_score_mean,
        "unlearning_score_std": unlearning_score_std,
        "t_retrain_mean": mean_or_none(per_metric["t_retrain"]),
        "t_forget_total_mean": mean_or_none(per_metric["t_forget_total"]),
        "updated_param_ratio_mean": mean_or_none(per_metric["updated_param_ratio"]),
        "adapter_param_ratio_mean": mean_or_none(per_metric["adapter_param_ratio"]),
        "overlap_shared_critical_ratio": mean_or_none(per_metric["overlap_shared_critical_ratio"]),
        "overlap_protected_ratio": mean_or_none(per_metric["overlap_protected_ratio"]),
        "overlap_updated_ratio": mean_or_none(per_metric["overlap_updated_ratio"]),
        "overlap_shared_critical_count": mean_or_none(per_metric["overlap_shared_critical_count"]),
        "overlap_protected_params": mean_or_none(per_metric["overlap_protected_params"]),
        "overlap_updated_params": mean_or_none(per_metric["overlap_updated_params"]),
    }


def group_key(row: Dict[str, Any], group_by_config: bool) -> Tuple[str, ...]:
    base = [row["dataset"], row["method"]]
    if group_by_config:
        for key in CONFIG_GROUP_COLUMNS:
            base.append(str(row.get(key, "")))
    return tuple(base)


def build_table(rows: List[Dict[str, Any]], group_by_config: bool = False) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = group_key(row, group_by_config)
        grouped.setdefault(key, []).append(row)

    table: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        aggregate = aggregate_group(grouped[key])
        out_row: Dict[str, Any] = {
            "dataset": key[0],
            "method": key[1],
            **aggregate,
        }
        if group_by_config:
            for column, value in zip(CONFIG_GROUP_COLUMNS, key[2:]):
                out_row[column] = value
        table.append(out_row)
    return table


def format_number(value: Any, decimals: int) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:.{decimals}f}"


def output_columns(group_by_config: bool) -> List[str]:
    if not group_by_config:
        return list(OUTPUT_COLUMNS)
    return ["dataset", "method", *CONFIG_GROUP_COLUMNS, *OUTPUT_COLUMNS[2:]]


def write_csv_table(path: Path, rows: List[Dict[str, Any]], decimals: int, group_by_config: bool = False) -> None:
    columns = output_columns(group_by_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column)
                    if column in ("dataset", "method") or column in CONFIG_GROUP_COLUMNS
                    else format_number(row.get(column), decimals)
                    for column in columns
                }
            )


def markdown_cell(column: str, value: Any, decimals: int) -> str:
    if column in ("dataset", "method") or column in CONFIG_GROUP_COLUMNS:
        return str(value)
    formatted = format_number(value, decimals)
    return formatted if formatted else "NA"


def write_markdown_table(path: Path, rows: List[Dict[str, Any]], decimals: int, group_by_config: bool = False) -> None:
    columns = output_columns(group_by_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_cell(column, row.get(column, ""), decimals) for column in columns)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate thesis metrics from runs/**/metrics.json.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Root directory to scan recursively.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/aggregates/thesis_table.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/aggregates/thesis_table.md"),
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--group-by-config",
        action="store_true",
        help="Group by dataset/method plus experiment_tag and key adapter configuration values.",
    )
    parser.add_argument(
        "--include-tags",
        nargs="+",
        default=None,
        help="Only include runs whose config.json experiment_tag matches one of these values.",
    )
    parser.add_argument(
        "--seed-policy",
        choices=("mean", "latest"),
        default="mean",
        help=(
            "How to handle duplicate completed runs for the same group+seed. "
            "'mean' preserves the legacy seed-mean behavior; 'latest' keeps only "
            "the newest timestamped run per group+seed."
        ),
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/test runs (experiment_tag starting with 'smoke'/'test'); excluded by default.",
    )
    parser.add_argument("--decimals", type=int, default=4, help="Decimal precision for numeric outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Root directory does not exist: {args.root}", file=sys.stderr)
        return 1

    run_rows = []
    total_metrics_files = 0
    skipped_by_tag = 0
    include_tags = set(args.include_tags) if args.include_tags else None
    for metrics_path in find_metrics_files(args.root):
        total_metrics_files += 1
        row = extract_run_row(
            metrics_path,
            group_by_config=args.group_by_config,
            include_tags=include_tags,
        )
        if row is not None:
            if not args.include_smoke and is_smoke_tag(row.get("experiment_tag")):
                continue
            run_rows.append(row)
        elif include_tags is not None:
            config = load_json(metrics_path.with_name("config.json")) or {}
            experiment_tag = config_group_value(config, "experiment_tag")
            if experiment_tag not in include_tags:
                skipped_by_tag += 1

    deduped_runs = 0
    if args.seed_policy == "latest":
        run_rows, deduped_runs = dedupe_latest_rows(run_rows, group_by_config=args.group_by_config)

    table = build_table(run_rows, group_by_config=args.group_by_config)
    write_csv_table(args.out_csv, table, args.decimals, group_by_config=args.group_by_config)
    write_markdown_table(args.out_md, table, args.decimals, group_by_config=args.group_by_config)

    print(f"[INFO] Scanned metrics files: {total_metrics_files}")
    print(f"[INFO] Runs included: {len(run_rows)}")
    print(f"[INFO] Duplicate completed runs removed by seed policy: {deduped_runs}")
    print(f"[INFO] Runs skipped by tag filter: {skipped_by_tag}")
    print(f"[INFO] Wrote CSV table: {args.out_csv}")
    print(f"[INFO] Wrote Markdown table: {args.out_md}")
    print(f"[INFO] Groups summarized: {len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
