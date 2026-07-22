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
    "regime",
    "experiment_tag",
    "config_id",
    "final_avg_acc",
    "avg_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "mia_auc_before",
    "mia_auc_after",
    "probe_acc_before",
    "probe_acc_after",
    "updated_param_ratio_mean",
    "adapter_param_ratio_mean",
    "critical_ratio",
    "protected_ratio",
    "updated_overlap_ratio",
    "shared_critical_count",
    "protected_params",
    "updated_params",
    "t_forget_total_mean",
]

CONFIG_COLUMNS = [
    "protect_importance",
    "protect_ratio",
    "lambda_protect",
    "protect_anchor",
    "adaptive_protect",
    "modified_component_mode",
    "adapter_bottleneck",
    "adapter_shared_bottleneck",
    "adapter_shared_forget_ratio",
    "adapter_shared_protect_ratio",
    "adapter_forget_steps",
    "adapter_shared_forget_lr",
    "adapter_shared_protect_strength",
    "retrain_steps",
    "adapter_train_classifier",
    "adapter_component_mode",
    "pretrained_input_norm",
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


def derive_regime(row: Dict[str, str]) -> str:
    explicit = str(row.get("regime", "")).strip()
    if explicit:
        return explicit
    tag = str(row.get("experiment_tag", "")).strip().lower()
    if "standard" in tag:
        return "standard_split"
    if "pretrained" in tag or tag in {"tiny_pretrained", "adapter_tune_pretrained_v1"}:
        return "pretrained_frozen"
    if tag:
        return "from_scratch"
    return ""


def normalize_config_value(value: Any) -> str:
    text = str(value or "").strip()
    if text == "" or text.lower() in {"none", "nan", "na"}:
        return ""
    number = parse_float(text)
    if number is not None:
        if abs(number - round(number)) < 1e-10:
            return str(int(round(number)))
        return f"{number:.12g}"
    return text


def config_id(row: Dict[str, str]) -> str:
    parts = []
    method = str(row.get("method", "")).strip()
    for column in CONFIG_COLUMNS:
        value = normalize_config_value(row.get(column))
        if value == "":
            continue
        if column == "protect_importance" and value == "gradient":
            continue
        if column == "lambda_protect" and value == "0":
            continue
        if column.startswith("adapter_") and method != "pall_adapter":
            continue
        if method == "pall_adapter":
            if column == "adapter_bottleneck" and value == "16":
                continue
            if column == "adapter_shared_bottleneck" and value == "0":
                continue
            if column in {"adapter_shared_forget_ratio", "adapter_shared_protect_ratio"} and value in {"0", "0.0"}:
                continue
            if column == "adapter_forget_steps" and value == "10":
                continue
            if column == "adapter_train_classifier" and value.lower() == "false":
                continue
        parts.append(f"{column}={value}")
    return "; ".join(parts) if parts else "default"


def dedupe_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return (
        str(row.get("dataset", "")).strip(),
        str(row.get("method", "")).strip(),
        derive_regime(row),
        str(row.get("experiment_tag", "")).strip(),
        config_id(row),
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
        key=lambda row: (
            str(row.get("dataset", "")).strip(),
            derive_regime(row),
            str(row.get("method", "")).strip(),
            str(row.get("experiment_tag", "")).strip(),
            config_id(row),
        ),
    )
    removed_count = max(0, len(rows) - len(deduped_rows))
    return deduped_rows, removed_count


def format_number(value: Any, decimals: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{decimals}f}"


def first_present_value(row: Dict[str, str], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if str(value).strip() == "":
            continue
        return value
    return None


def format_mean_std(row: Dict[str, str], mean_key: str, std_key: str, decimals: int = 4) -> str:
    mean_value = parse_float(row.get(mean_key))
    std_value = parse_float(row.get(std_key))
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.{decimals}f}"
    return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"


def format_mean_std_or_value(row: Dict[str, str], base_key: str, decimals: int = 4) -> str:
    formatted = format_mean_std(row, f"{base_key}_mean", f"{base_key}_std", decimals=decimals)
    if formatted:
        return formatted
    return format_number(row.get(base_key), decimals=decimals)


def compact_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "dataset": str(row.get("dataset", "")).strip(),
        "method": str(row.get("method", "")).strip(),
        "regime": derive_regime(row),
        "experiment_tag": str(row.get("experiment_tag", "")).strip(),
        "config_id": config_id(row),
        "final_avg_acc": format_mean_std(row, "final_avg_acc_mean", "final_avg_acc_std"),
        "avg_forgetting": format_mean_std(row, "avg_forgetting_mean", "avg_forgetting_std"),
        "Fu": format_mean_std(row, "Fu_mean", "Fu_std"),
        "WorstDrop": format_mean_std(row, "WorstDrop_mean", "WorstDrop_std"),
        "Au": format_mean_std(row, "Au_mean", "Au_std"),
        "mia_auc_before": format_mean_std_or_value(row, "mia_auc_before"),
        "mia_auc_after": format_mean_std_or_value(row, "mia_auc_after"),
        "probe_acc_before": format_mean_std_or_value(row, "probe_acc_before"),
        "probe_acc_after": format_mean_std_or_value(row, "probe_acc_after"),
        "updated_param_ratio_mean": format_number(row.get("updated_param_ratio_mean")),
        "adapter_param_ratio_mean": format_number(row.get("adapter_param_ratio_mean")),
        "critical_ratio": format_number(
            first_present_value(
                row,
                "overlap_shared_critical_ratio",
                "critical_ratio",
                "S_share_crit_ratio",
                "s_share_crit_ratio",
                "shared_critical_ratio",
            )
        ),
        "protected_ratio": format_number(
            first_present_value(
                row,
                "overlap_protected_ratio",
                "protected_ratio",
            )
        ),
        "updated_overlap_ratio": format_number(
            first_present_value(
                row,
                "overlap_updated_ratio",
                "updated_overlap_ratio",
            )
        ),
        "shared_critical_count": format_number(
            first_present_value(
                row,
                "overlap_shared_critical_count",
                "shared_critical_count",
                "S_share_crit",
                "s_share_crit",
            )
        ),
        "protected_params": format_number(
            first_present_value(
                row,
                "overlap_protected_params",
                "protected_params",
                "protected_adapter_params",
                "hard_protected_adapter_params",
            )
        ),
        "updated_params": format_number(
            first_present_value(
                row,
                "overlap_updated_params",
                "updated_params",
                "updated_adapter_params",
            )
        ),
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
