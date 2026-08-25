#!/usr/bin/env python3
"""
Write a concise Markdown experiment summary from aggregated CSV outputs.

Usage:
  python tools/write_experiment_summary.py \
    --results results/aggregates/results_summary.csv \
    --overlap overlap_summary.csv \
    --out results/aggregates/experiment_summary.md

The output is intended as a thesis/report draft: compact tables plus conservative,
editable observations. It does not claim statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRIC_COLUMNS = [
    ("final_avg_accuracy", "Final Avg Acc"),
    ("average_forgetting", "Avg Forgetting"),
    ("Fu", "Fu"),
    ("WorstDrop", "WorstDrop"),
    ("Au", "Au"),
    ("t_retrain", "Retrain Time"),
]

TABLE_COLUMNS = [
    "Method",
    "Dataset",
    "Seeds",
    "Runs",
    "Final Avg Acc",
    "Avg Forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "Retrain Time",
]

OVERLAP_COLUMNS = [
    ("avg_overlap_offdiag", "Avg Offdiag Overlap"),
    ("max_overlap_offdiag", "Max Offdiag Overlap"),
    ("s_share_ratio_last", "Last Shared Ratio"),
    ("s_share_crit_ratio_last", "Last Critical Shared Ratio"),
]


def read_csv(path: Path) -> Optional[List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {path}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[ERROR] Failed to read CSV: {path} ({exc})", file=sys.stderr)
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def unique_values(rows: Iterable[Dict[str, str]], key: str) -> List[str]:
    values = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    return values


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def fmt_number(value: Optional[float], decimals: int) -> str:
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


def fmt_mean_std(values: List[float], decimals: int) -> str:
    mean, std = mean_std(values)
    if mean is None:
        return "NA"
    if std is None:
        return fmt_number(mean, decimals)
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def group_results(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        dataset = (row.get("dataset") or "").strip()
        method = (row.get("method") or "").strip()
        grouped.setdefault((dataset, method), []).append(row)
    return grouped


def build_metric_table(rows: List[Dict[str, str]], decimals: int) -> List[Dict[str, str]]:
    table = []
    grouped = group_results(rows)
    for dataset, method in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        group_rows = grouped[(dataset, method)]
        out = {
            "Method": method,
            "Dataset": dataset,
            "Seeds": ", ".join(unique_values(group_rows, "seed")) or "NA",
            "Runs": str(len(group_rows)),
        }
        for input_col, output_col in METRIC_COLUMNS:
            values = [v for v in (parse_float(row.get(input_col)) for row in group_rows) if v is not None]
            out[output_col] = fmt_mean_std(values, decimals)
        table.append(out)
    return table


def build_overlap_table(rows: List[Dict[str, str]], decimals: int) -> List[Dict[str, str]]:
    table = []
    grouped = group_results(rows)
    for dataset, method in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        group_rows = grouped[(dataset, method)]
        out = {
            "Method": method,
            "Dataset": dataset,
            "Runs": str(len(group_rows)),
        }
        for input_col, output_col in OVERLAP_COLUMNS:
            values = [v for v in (parse_float(row.get(input_col)) for row in group_rows) if v is not None]
            out[output_col] = fmt_mean_std(values, decimals)
        table.append(out)
    return table


def markdown_table(columns: List[str], rows: List[Dict[str, str]]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in columns) + " |")
    return lines


def group_means(rows: List[Dict[str, str]], dataset: str, metric: str) -> Dict[str, Optional[float]]:
    grouped = group_results(rows)
    means: Dict[str, Optional[float]] = {}
    for (group_dataset, method), group_rows in grouped.items():
        if group_dataset != dataset:
            continue
        values = [v for v in (parse_float(row.get(metric)) for row in group_rows) if v is not None]
        means[method] = mean_std(values)[0]
    return means


def describe_difference(
    metric_label: str,
    modified_value: Optional[float],
    original_value: Optional[float],
    higher_is_better: bool,
    decimals: int,
    similar_threshold: float,
) -> Optional[str]:
    if modified_value is None or original_value is None:
        return None
    delta = modified_value - original_value
    if abs(delta) <= similar_threshold:
        relation = "similar to"
    elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        relation = "better than"
    else:
        relation = "worse than"
    return (
        f"{metric_label}: pall_modified is {relation} pall_original "
        f"(mean delta {delta:.{decimals}f})."
    )


def build_observations(rows: List[Dict[str, str]], decimals: int, similar_threshold: float) -> List[str]:
    observations = []
    datasets = unique_values(rows, "dataset")
    for dataset in datasets:
        final_means = group_means(rows, dataset, "final_avg_accuracy")
        fu_means = group_means(rows, dataset, "Fu")
        worst_drop_means = group_means(rows, dataset, "WorstDrop")
        retrain_means = group_means(rows, dataset, "t_retrain")

        pairs = [
            ("Final Avg Acc", final_means, True),
            ("Fu", fu_means, False),
            ("WorstDrop", worst_drop_means, False),
            ("Retrain Time", retrain_means, False),
        ]
        for label, means, higher_is_better in pairs:
            obs = describe_difference(
                label,
                means.get("pall_modified"),
                means.get("pall_original"),
                higher_is_better=higher_is_better,
                decimals=decimals,
                similar_threshold=similar_threshold,
            )
            if obs is not None:
                observations.append(f"{dataset}: {obs}")

    if not observations:
        observations.append("No direct pall_modified vs pall_original comparison was available in the input CSV.")
    observations.append("These observations compare descriptive means only and do not imply statistical significance.")
    observations.append(f"Differences with absolute delta <= {similar_threshold:.{decimals}f} are reported as similar.")
    return observations


def summarize_setup(rows: List[Dict[str, str]]) -> Dict[str, str]:
    keys = ["experiment_tag", "dataset", "method", "arch", "class_per_task", "n_tasks", "n_forget", "seed"]
    setup = {}
    for key in keys:
        values = unique_values(rows, key)
        setup[key] = ", ".join(values) if values else "NA"
    setup["num_runs"] = str(len(rows))
    return setup


def write_summary(
    out_path: Path,
    result_rows: List[Dict[str, str]],
    overlap_rows: Optional[List[Dict[str, str]]],
    decimals: int,
    similar_threshold: float,
) -> None:
    setup = summarize_setup(result_rows)
    metric_rows = build_metric_table(result_rows, decimals)
    observations = build_observations(result_rows, decimals, similar_threshold)

    lines = [
        "# Experiment Summary",
        "",
        "## Setup",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in ["experiment_tag", "dataset", "method", "arch", "class_per_task", "n_tasks", "n_forget", "seed", "num_runs"]:
        lines.append(f"| {key} | {setup.get(key, 'NA')} |")

    lines.extend(["", "## Key Metrics"])
    lines.extend(markdown_table(TABLE_COLUMNS, metric_rows))

    if overlap_rows is not None:
        overlap_table = build_overlap_table(overlap_rows, decimals)
        lines.extend(["", "## Overlap Summary"])
        lines.extend(
            markdown_table(
                ["Method", "Dataset", "Runs"] + [label for _, label in OVERLAP_COLUMNS],
                overlap_table,
            )
        )

    lines.extend(["", "## Automatic Observations"])
    for obs in observations:
        lines.append(f"- {obs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Markdown experiment summary from aggregate CSV files.")
    parser.add_argument("--results", type=Path, required=True, help="Aggregated results CSV.")
    parser.add_argument("--overlap", type=Path, default=None, help="Optional overlap summary CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output Markdown file.")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal precision for tables and deltas.")
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.001,
        help="Absolute mean-delta threshold for reporting two values as similar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_rows = read_csv(args.results)
    if result_rows is None:
        return 1

    overlap_rows = None
    if args.overlap is not None:
        overlap_rows = read_csv(args.overlap)
        if overlap_rows is None:
            return 1

    write_summary(args.out, result_rows, overlap_rows, args.decimals, args.similar_threshold)
    print(f"[INFO] Wrote experiment summary: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
