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
    "unlearning_score_mean",
    "unlearning_score_std",
    "t_retrain_mean",
    "t_forget_total_mean",
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
]

SEED_AGG_METRICS = [
    "final_avg_accuracy",
    "average_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "unlearning_score",
    "t_retrain",
    "t_forget_total",
    "updated_param_ratio",
    "adapter_param_ratio",
]


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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


def extract_run_row(metrics_path: Path) -> Optional[Dict[str, Any]]:
    metrics = load_json(metrics_path)
    if metrics is None:
        return None

    final_unlearning = get_final_unlearning(metrics)
    raw_last_event = get_last_unlearning_event(metrics)

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

    return {
        "run_path": str(metrics_path.parent),
        "dataset": str(dataset),
        "method": str(method),
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
        "unlearning_score": derive_unlearning_score(metrics, final_unlearning),
        "t_retrain": t_retrain,
        "t_forget_total": t_forget_total,
        "updated_param_ratio": derive_updated_param_ratio(metrics, final_unlearning),
        "adapter_param_ratio": derive_adapter_param_ratio(metrics),
    }


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
    seed = row.get("seed")
    if seed is None:
        return f"run:{row['run_path']}"
    seed_text = str(seed).strip()
    if seed_text == "":
        return f"run:{row['run_path']}"
    return seed_text


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
        "unlearning_score_mean": unlearning_score_mean,
        "unlearning_score_std": unlearning_score_std,
        "t_retrain_mean": mean_or_none(per_metric["t_retrain"]),
        "t_forget_total_mean": mean_or_none(per_metric["t_forget_total"]),
        "updated_param_ratio_mean": mean_or_none(per_metric["updated_param_ratio"]),
        "adapter_param_ratio_mean": mean_or_none(per_metric["adapter_param_ratio"]),
    }


def build_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        grouped.setdefault(key, []).append(row)

    table: List[Dict[str, Any]] = []
    for dataset, method in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        aggregate = aggregate_group(grouped[(dataset, method)])
        table.append(
            {
                "dataset": dataset,
                "method": method,
                **aggregate,
            }
        )
    return table


def format_number(value: Any, decimals: int) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:.{decimals}f}"


def write_csv_table(path: Path, rows: List[Dict[str, Any]], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column)
                    if column in ("dataset", "method")
                    else format_number(row.get(column), decimals)
                    for column in OUTPUT_COLUMNS
                }
            )


def markdown_cell(column: str, value: Any, decimals: int) -> str:
    if column in ("dataset", "method"):
        return str(value)
    formatted = format_number(value, decimals)
    return formatted if formatted else "NA"


def write_markdown_table(path: Path, rows: List[Dict[str, Any]], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(OUTPUT_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_cell(column, row.get(column, ""), decimals) for column in OUTPUT_COLUMNS)
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
    parser.add_argument("--decimals", type=int, default=4, help="Decimal precision for numeric outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Root directory does not exist: {args.root}", file=sys.stderr)
        return 1

    run_rows = []
    for metrics_path in find_metrics_files(args.root):
        row = extract_run_row(metrics_path)
        if row is not None:
            run_rows.append(row)

    table = build_table(run_rows)
    write_csv_table(args.out_csv, table, args.decimals)
    write_markdown_table(args.out_md, table, args.decimals)

    print(f"[INFO] Scanned metrics files: {len(run_rows)}")
    print(f"[INFO] Wrote CSV table: {args.out_csv}")
    print(f"[INFO] Wrote Markdown table: {args.out_md}")
    print(f"[INFO] Groups summarized: {len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
