#!/usr/bin/env python3
"""
Analyze overlap statistics from experiment outputs.

Usage:
python tools/analyze_overlap.py --root runs --out-csv overlap_summary.csv --out-md overlap_summary.md

Primary input source is overlap.csv found recursively under --root.
Optional metadata/stat enrichment is read from sibling config.json/metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


OUTPUT_COLUMNS = [
    "run_path",
    "dataset",
    "method",
    "seed",
    "arch",
    "n_tasks_in_overlap",
    "num_task_pairs",
    "avg_overlap_offdiag",
    "max_overlap_offdiag",
    "min_overlap_offdiag",
    "avg_overlap_all",
    "diag_mean",
    "s_share_ratio_mean",
    "s_share_ratio_last",
    "s_share_crit_ratio_mean",
    "s_share_crit_ratio_last",
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        return payload
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if math.isnan(number):
        return None
    return number


def first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def parse_overlap_matrix(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except OSError:
        return None

    if len(rows) < 2:
        return None
    header = rows[0]
    if len(header) < 2:
        return None

    task_labels = header[1:]
    matrix: List[List[float]] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        vals = row[1:1 + len(task_labels)]
        if len(vals) != len(task_labels):
            continue
        parsed = []
        for v in vals:
            num = parse_float(v)
            if num is None:
                return None
            parsed.append(num)
        matrix.append(parsed)

    n = len(task_labels)
    if len(matrix) != n:
        return None

    return {
        "task_labels": task_labels,
        "matrix": matrix,
    }


def summarize_overlap(matrix: List[List[float]]) -> Dict[str, Any]:
    n = len(matrix)
    all_values: List[float] = []
    diag_values: List[float] = []
    offdiag_values: List[float] = []

    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            all_values.append(value)
            if i == j:
                diag_values.append(value)
            else:
                offdiag_values.append(value)

    num_pairs = (n * (n - 1)) // 2
    return {
        "n_tasks_in_overlap": n,
        "num_task_pairs": num_pairs,
        "avg_overlap_offdiag": mean_or_none(offdiag_values),
        "max_overlap_offdiag": max(offdiag_values) if offdiag_values else None,
        "min_overlap_offdiag": min(offdiag_values) if offdiag_values else None,
        "avg_overlap_all": mean_or_none(all_values),
        "diag_mean": mean_or_none(diag_values),
    }


def extract_ratio_stats(metrics: Dict[str, Any]) -> Dict[str, Any]:
    share_values: List[float] = []
    crit_values: List[float] = []

    # Prefer normalized unlearning events when present.
    normalized_events = (
        metrics.get("normalized_results", {}).get("unlearning_events", [])
        if isinstance(metrics.get("normalized_results"), dict)
        else []
    )
    if isinstance(normalized_events, list):
        for event in normalized_events:
            if not isinstance(event, dict):
                continue
            overlap = event.get("overlap", {})
            if not isinstance(overlap, dict):
                continue
            s_share_ratio = parse_float(overlap.get("s_share_ratio"))
            s_share_crit_ratio = parse_float(overlap.get("s_share_crit_ratio"))
            if s_share_ratio is not None:
                share_values.append(s_share_ratio)
            if s_share_crit_ratio is not None:
                crit_values.append(s_share_crit_ratio)

    # Fallback to raw unlearning_events.
    if not share_values and not crit_values:
        raw_events = metrics.get("unlearning_events", [])
        if isinstance(raw_events, list):
            for event in raw_events:
                if not isinstance(event, dict):
                    continue
                overlap = event.get("overlap", {})
                if not isinstance(overlap, dict):
                    continue
                s_share_ratio = parse_float(overlap.get("s_share_ratio"))
                s_share_crit_ratio = parse_float(overlap.get("s_share_crit_ratio"))
                if s_share_ratio is not None:
                    share_values.append(s_share_ratio)
                if s_share_crit_ratio is not None:
                    crit_values.append(s_share_crit_ratio)

    return {
        "s_share_ratio_mean": mean_or_none(share_values),
        "s_share_ratio_last": share_values[-1] if share_values else None,
        "s_share_crit_ratio_mean": mean_or_none(crit_values),
        "s_share_crit_ratio_last": crit_values[-1] if crit_values else None,
    }


def find_overlap_files(root: Path) -> Iterable[Path]:
    for overlap_path in root.rglob("overlap.csv"):
        yield overlap_path


def build_row(overlap_path: Path) -> Optional[Dict[str, Any]]:
    parsed = parse_overlap_matrix(overlap_path)
    if parsed is None:
        print(f"[WARN] Skipping invalid overlap.csv: {overlap_path}", file=sys.stderr)
        return None

    run_dir = overlap_path.parent
    config = load_json(run_dir / "config.json") or {}
    metrics = load_json(run_dir / "metrics.json") or {}

    overlap_summary = summarize_overlap(parsed["matrix"])
    ratio_summary = extract_ratio_stats(metrics)

    run_meta = metrics.get("run", {}) if isinstance(metrics.get("run"), dict) else {}
    row = {
        "run_path": str(run_dir),
        "dataset": first_non_none(config.get("dataset"), run_meta.get("dataset")),
        "method": first_non_none(config.get("method"), run_meta.get("method")),
        "seed": first_non_none(config.get("seed"), run_meta.get("seed")),
        "arch": config.get("arch"),
    }
    row.update(overlap_summary)
    row.update(ratio_summary)
    return row


def write_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in OUTPUT_COLUMNS})


def fmt(value: Any, decimals: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def write_markdown(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_path",
        "method",
        "dataset",
        "seed",
        "avg_overlap_offdiag",
        "max_overlap_offdiag",
        "min_overlap_offdiag",
        "num_task_pairs",
        "s_share_ratio_last",
        "s_share_crit_ratio_last",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze overlap.csv files under a run root.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Root directory to scan recursively.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("overlap_summary.csv"),
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional Markdown summary path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Root directory does not exist: {args.root}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"[ERROR] Root path is not a directory: {args.root}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for overlap_path in sorted(find_overlap_files(args.root)):
        row = build_row(overlap_path)
        if row is not None:
            rows.append(row)

    write_csv(args.out_csv, rows)
    if args.out_md is not None:
        write_markdown(args.out_md, rows)

    print(f"[INFO] Parsed overlap.csv files: {len(rows)}")
    print(f"[INFO] Wrote CSV summary: {args.out_csv}")
    if args.out_md is not None:
        print(f"[INFO] Wrote Markdown summary: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
