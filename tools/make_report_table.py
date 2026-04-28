#!/usr/bin/env python3
"""
Create a compact thesis/report table from the final thesis table CSV.
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
    "final_avg_acc",
    "avg_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "t_forget_total_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact report table from thesis_final_table.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/thesis/thesis_final_table.csv"),
        help="Input final thesis table CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/thesis/report_table.csv"),
        help="Output compact CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/thesis/report_table.md"),
        help="Output compact Markdown path.",
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


def dedupe_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        str(row.get("dataset", "")).strip(),
        str(row.get("method", "")).strip(),
    )


def row_priority(row: Dict[str, str]) -> Tuple[int, str]:
    experiment_tag = str(row.get("experiment_tag", "")).strip()
    thesis_rank = 0 if experiment_tag.startswith("thesis_") else 1
    return thesis_rank, experiment_tag


def dedupe_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    best_rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = dedupe_key(row)
        current = best_rows.get(key)
        if current is None or row_priority(row) < row_priority(current):
            best_rows[key] = row
    deduped_rows = sorted(
        best_rows.values(),
        key=lambda row: (str(row.get("dataset", "")).strip(), str(row.get("method", "")).strip()),
    )
    removed_count = max(0, len(rows) - len(deduped_rows))
    return deduped_rows, removed_count


def format_number(value: Any, decimals: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{decimals}f}"


def format_mean_std(row: Dict[str, str], mean_key: str, std_key: str, decimals: int = 4) -> str:
    mean_value = parse_float(row.get(mean_key))
    std_value = parse_float(row.get(std_key))
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"


def compact_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "dataset": str(row.get("dataset", "")).strip(),
        "method": str(row.get("method", "")).strip(),
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
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def markdown_cell(row: Dict[str, str], column: str) -> str:
    value = str(row.get(column, "")).strip()
    return value if value else "NA"


def write_markdown(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(OUTPUT_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row, column) for column in OUTPUT_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if rows is None:
        return 1

    deduped_rows, removed_count = dedupe_rows(rows)
    compact_rows = [compact_row(row) for row in deduped_rows]

    write_csv(args.out_csv, compact_rows)
    write_markdown(args.out_md, compact_rows)

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Deduplicated rows: {len(deduped_rows)}")
    print(f"[INFO] Duplicate rows removed: {removed_count}")
    print(f"[INFO] Wrote CSV table: {args.out_csv}")
    print(f"[INFO] Wrote Markdown table: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
