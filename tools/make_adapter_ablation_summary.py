#!/usr/bin/env python3
"""
Merge adapter ablation CSVs into a compact thesis summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


CSV_OUTPUT_COLUMNS = [
    "dataset",
    "ablation_name",
    "experiment_tag",
    "adapter_shared_protect_ratio",
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
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "t_forget_total_mean",
]

MARKDOWN_OUTPUT_COLUMNS = [
    "dataset",
    "ablation_name",
    "experiment_tag",
    "adapter_shared_protect_ratio",
    "final_avg_acc",
    "avg_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "t_forget_total_mean",
]

SORT_ORDER = {
    "No Shared Adapter": 0,
    "Shared Adapter, No Protection": 1,
    "Critical Protection p=0.05": 2,
    "Critical Protection p=0.10": 3,
    "Critical Protection p=0.20": 4,
}

INPUT_COLUMNS = [
    "dataset",
    "experiment_tag",
    "adapter_shared_protect_ratio",
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
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "t_forget_total_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create adapter ablation summary tables.")
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("results/thesis/adapter_ablation_cifar100_e3.csv"),
        help="Input adapter ablation CSV.",
    )
    parser.add_argument(
        "--protect-sweep-csv",
        type=Path,
        default=Path("results/thesis/adapter_protect_sweep_cifar100_e3.csv"),
        help="Input adapter protection sweep CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/thesis/adapter_ablation_summary.csv"),
        help="Output merged summary CSV.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/thesis/adapter_ablation_summary.md"),
        help="Output merged summary Markdown table.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        print(f"[ERROR] Input CSV not found: {path}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[ERROR] Failed to read CSV: {path} ({exc})", file=sys.stderr)
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def format_number(value: Any, decimals: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if abs(number) < 0.5 * (10 ** (-decimals)):
        number = 0.0
    return f"{number:.{decimals}f}"


def format_mean_std(row: Dict[str, str], mean_key: str, std_key: str, decimals: int = 4) -> str:
    mean_value = parse_float(row.get(mean_key))
    std_value = parse_float(row.get(std_key))
    if mean_value is None:
        return ""
    if abs(mean_value) < 0.5 * (10 ** (-decimals)):
        mean_value = 0.0
    if std_value is None:
        return f"{mean_value:.{decimals}f}"
    if abs(std_value) < 0.5 * (10 ** (-decimals)):
        std_value = 0.0
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def require_columns(rows: Sequence[Dict[str, str]], source_name: str) -> None:
    if not rows:
        raise ValueError(f"{source_name} is empty.")
    missing = [column for column in INPUT_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")


def dedupe_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        experiment_tag = str(row.get("experiment_tag", "")).strip()
        if experiment_tag == "":
            raise ValueError("Encountered row with empty experiment_tag.")
        if experiment_tag not in deduped:
            deduped[experiment_tag] = row
    return list(deduped.values())


def dedupe_ablation_settings(rows: Sequence[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
    deduped: Dict[tuple[str, str, str], Dict[str, str]] = {}
    removed_count = 0

    for row in rows:
        key = (
            str(row.get("dataset", "")).strip(),
            str(row.get("ablation_name", "")).strip(),
            str(row.get("adapter_shared_protect_ratio", "")).strip(),
        )
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
            continue

        removed_count += 1
        current_tag = str(current.get("experiment_tag", "")).strip()
        new_tag = str(row.get("experiment_tag", "")).strip()
        if "_p020_" in new_tag and "_p020_" not in current_tag:
            deduped[key] = row

    return list(deduped.values()), removed_count


def ablation_name_for_tag(experiment_tag: str) -> str:
    tag = experiment_tag.strip()
    if "adapter_no_shared" in tag:
        return "No Shared Adapter"
    if "adapter_shared_no_protection" in tag:
        return "Shared Adapter, No Protection"
    if "adapter_shared_critical_p005" in tag:
        return "Critical Protection p=0.05"
    if "adapter_shared_critical_p010" in tag:
        return "Critical Protection p=0.10"
    if "adapter_shared_critical_p020" in tag or "adapter_shared_critical_e" in tag:
        return "Critical Protection p=0.20"
    raise ValueError(
        "Unsupported experiment_tag for adapter ablation summary: "
        f"{experiment_tag}"
    )


def sort_key(row: Dict[str, str]) -> tuple[int, str, str]:
    ablation_name = str(row.get("ablation_name", "")).strip()
    order = SORT_ORDER.get(ablation_name)
    if order is None:
        raise ValueError(f"Unsupported ablation_name in sort: {ablation_name}")
    dataset = str(row.get("dataset", "")).strip()
    experiment_tag = str(row.get("experiment_tag", "")).strip()
    return order, dataset, experiment_tag


def compact_csv_row(row: Dict[str, str]) -> Dict[str, str]:
    compact = {
        "dataset": str(row.get("dataset", "")).strip(),
        "ablation_name": ablation_name_for_tag(str(row.get("experiment_tag", ""))),
        "experiment_tag": str(row.get("experiment_tag", "")).strip(),
    }
    for column in INPUT_COLUMNS[2:]:
        compact[column] = format_number(row.get(column))
    return compact


def compact_markdown_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "dataset": str(row.get("dataset", "")).strip(),
        "ablation_name": str(row.get("ablation_name", "")).strip(),
        "experiment_tag": str(row.get("experiment_tag", "")).strip(),
        "adapter_shared_protect_ratio": format_number(row.get("adapter_shared_protect_ratio")),
        "final_avg_acc": format_mean_std(row, "final_avg_acc_mean", "final_avg_acc_std"),
        "avg_forgetting": format_mean_std(row, "avg_forgetting_mean", "avg_forgetting_std"),
        "Fu": format_mean_std(row, "Fu_mean", "Fu_std"),
        "WorstDrop": format_mean_std(row, "WorstDrop_mean", "WorstDrop_std"),
        "Au": format_mean_std(row, "Au_mean", "Au_std"),
        "updated_param_ratio_mean": format_number(row.get("updated_param_ratio_mean")),
        "adapter_param_ratio_mean": format_number(row.get("adapter_param_ratio_mean")),
        "t_forget_total_mean": format_number(row.get("t_forget_total_mean")),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_OUTPUT_COLUMNS})


def write_markdown(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(MARKDOWN_OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(MARKDOWN_OUTPUT_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") or "NA" for column in MARKDOWN_OUTPUT_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    ablation_rows = read_rows(args.ablation_csv)
    if ablation_rows is None:
        return 1
    protect_sweep_rows = read_rows(args.protect_sweep_csv)
    if protect_sweep_rows is None:
        return 1

    try:
        require_columns(ablation_rows, str(args.ablation_csv))
        require_columns(protect_sweep_rows, str(args.protect_sweep_csv))
        merged_rows = dedupe_rows([*ablation_rows, *protect_sweep_rows])
        csv_rows, removed_ablation_duplicates = dedupe_ablation_settings(
            [compact_csv_row(row) for row in merged_rows]
        )
        csv_rows.sort(key=sort_key)
        markdown_rows = [compact_markdown_row(row) for row in csv_rows]
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    write_csv(args.out_csv, csv_rows)
    write_markdown(args.out_md, markdown_rows)

    removed_count = len(ablation_rows) + len(protect_sweep_rows) - len(csv_rows)
    removed_experiment_tag_duplicates = len(ablation_rows) + len(protect_sweep_rows) - len(merged_rows)
    print(f"[INFO] Input rows: {len(ablation_rows) + len(protect_sweep_rows)}")
    print(f"[INFO] Deduplicated rows: {len(csv_rows)}")
    print(f"[INFO] Duplicate experiment_tag rows removed: {removed_experiment_tag_duplicates}")
    print(f"[INFO] Duplicate ablation setting rows removed: {removed_ablation_duplicates}")
    print(f"[INFO] Wrote CSV summary: {args.out_csv}")
    print(f"[INFO] Wrote Markdown summary: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
