#!/usr/bin/env python3
"""
Create a compact paper summary table from report and adapter ablation summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


OUTPUT_COLUMNS = [
    "dataset",
    "method",
    "Final Accuracy",
    "Avg. Forgetting",
    "Worst Drop",
    "Au",
    "Updated Params",
    "Adapter Params",
    "Forget Time",
]

METHOD_ORDER = {
    "derpp": 0,
    "er": 1,
    "pall_original": 2,
    "pall_modified": 3,
    "pall_adapter": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the paper summary table.")
    parser.add_argument(
        "--report-table",
        type=Path,
        default=Path("results/thesis/report_table.csv"),
        help="Input compact report table CSV.",
    )
    parser.add_argument(
        "--adapter-summary",
        type=Path,
        default=Path("results/thesis/adapter_ablation_summary.csv"),
        help="Input adapter ablation summary CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/thesis/paper_summary_table.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/thesis/paper_summary_table.md"),
        help="Output Markdown path.",
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


def parse_mean_from_text(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    for token in ("+/-", "±"):
        if token in text:
            return parse_float(text.split(token, 1)[0].strip())
    return parse_float(text)


def format_number(value: Any, decimals: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if abs(number) < 0.5 * (10 ** (-decimals)):
        number = 0.0
    return f"{number:.{decimals}f}"


def format_mean_std(mean_value: Any, std_value: Any, decimals: int = 4) -> str:
    mean_number = parse_float(mean_value)
    std_number = parse_float(std_value)
    if mean_number is None:
        return ""
    if abs(mean_number) < 0.5 * (10 ** (-decimals)):
        mean_number = 0.0
    if std_number is None:
        return f"{mean_number:.{decimals}f}"
    if abs(std_number) < 0.5 * (10 ** (-decimals)):
        std_number = 0.0
    return f"{mean_number:.{decimals}f} ± {std_number:.{decimals}f}"


def normalize_report_value(value: str) -> str:
    text = str(value or "").strip()
    if text == "":
        return ""
    return text.replace("+/-", "±")


def compact_report_row(row: Dict[str, str]) -> Dict[str, str]:
    method = str(row.get("method", "")).strip()
    return {
        "dataset": str(row.get("dataset", "")).strip(),
        "method": method,
        "Final Accuracy": normalize_report_value(row.get("final_avg_acc", "")),
        "Avg. Forgetting": normalize_report_value(row.get("avg_forgetting", "")),
        "Worst Drop": normalize_report_value(row.get("WorstDrop", "")),
        "Au": normalize_report_value(row.get("Au", "")),
        "Updated Params": format_number(row.get("updated_param_ratio_mean")),
        "Adapter Params": format_number(row.get("adapter_param_ratio_mean")),
        "Forget Time": format_number(row.get("t_forget_total_mean")),
    }


def choose_best_adapter_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    best_rows: Dict[str, Dict[str, str]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        if dataset == "":
            continue
        current = best_rows.get(dataset)
        score = parse_float(row.get("final_avg_acc_mean"))
        current_score = parse_float(current.get("final_avg_acc_mean")) if current is not None else None
        if current is None or (score is not None and (current_score is None or score > current_score)):
            best_rows[dataset] = row
    return best_rows


def compact_adapter_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "dataset": str(row.get("dataset", "")).strip(),
        "method": "pall_adapter",
        "Final Accuracy": format_mean_std(row.get("final_avg_acc_mean"), row.get("final_avg_acc_std")),
        "Avg. Forgetting": format_mean_std(row.get("avg_forgetting_mean"), row.get("avg_forgetting_std")),
        "Worst Drop": format_mean_std(row.get("WorstDrop_mean"), row.get("WorstDrop_std")),
        "Au": format_mean_std(row.get("Au_mean"), row.get("Au_std")),
        "Updated Params": format_number(row.get("updated_param_ratio_mean")),
        "Adapter Params": format_number(row.get("adapter_param_ratio_mean")),
        "Forget Time": format_number(row.get("t_forget_total_mean")),
    }


def sort_key(row: Dict[str, str]) -> Tuple[str, int, str]:
    method = str(row.get("method", "")).strip()
    return (
        str(row.get("dataset", "")).strip(),
        METHOD_ORDER.get(method, 999),
        method,
    )


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def write_markdown(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(OUTPUT_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join((str(row.get(column, "")).strip() or "NA") for column in OUTPUT_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_rows = read_rows(args.report_table)
    if report_rows is None:
        return 1
    adapter_rows = read_rows(args.adapter_summary)
    if adapter_rows is None:
        return 1

    best_adapter_by_dataset = choose_best_adapter_rows(adapter_rows)
    output_rows: List[Dict[str, str]] = []

    for row in report_rows:
        method = str(row.get("method", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        if method == "pall_adapter" and dataset in best_adapter_by_dataset:
            continue
        output_rows.append(compact_report_row(row))

    for dataset, row in sorted(best_adapter_by_dataset.items()):
        del dataset
        output_rows.append(compact_adapter_row(row))

    output_rows.sort(key=sort_key)
    write_csv(args.out_csv, output_rows)
    write_markdown(args.out_md, output_rows)

    print(f"[INFO] Report rows read: {len(report_rows)}")
    print(f"[INFO] Adapter summary rows read: {len(adapter_rows)}")
    print(f"[INFO] Best adapter dataset rows used: {len(best_adapter_by_dataset)}")
    print(f"[INFO] Wrote CSV table: {args.out_csv}")
    print(f"[INFO] Wrote Markdown table: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
