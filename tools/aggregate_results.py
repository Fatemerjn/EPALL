#!/usr/bin/env python3
"""
Aggregate experiment run metrics into one CSV table.

Usage
-----
python tools/aggregate_results.py --root runs --out results/aggregates/results_summary.csv

Expected input structure
------------------------
The script recursively scans `--root` for run directories that contain:
- config.json
- metrics.json (optional but recommended)

For each run, it prefers normalized metrics when available:
- normalized_results.final.final_avg_accuracy
- normalized_results.final.average_forgetting
- normalized_results.final.final_unlearning.*

If normalized fields are missing, it falls back to older fields when possible
(for example summary.final_avg_accuracy, forgetting.final, unlearning_events[-1]).
Missing values are written as empty CSV cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COLUMNS: List[str] = [
    "run_path",
    "experiment_tag",
    "dataset",
    "method",
    "method_variant",
    "variant",
    "seed",
    "arch",
    "class_per_task",
    "n_tasks",
    "n_forget",
    "n_epochs",
    "adapter_bottleneck",
    "adapter_train_classifier",
    "adapter_location",
    "adapter_shared_bottleneck",
    "adapter_shared_forget_ratio",
    "adapter_shared_protect_ratio",
    "adapter_shared_protect_strength",
    "protect_ratio",
    "protect_importance",
    "lambda_protect",
    "retrain_steps",
    "request_schedule_file",
    "final_avg_accuracy",
    "average_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "t_reset",
    "t_retrain",
    "t_forget_total",
    "num_updated_params",
    "overlap_s_t",
    "overlap_s_share",
    "overlap_s_share_crit",
    "overlap_s_share_ratio",
    "overlap_s_share_crit_ratio",
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
        if not isinstance(data, dict):
            print(f"[WARN] JSON root is not an object: {path}", file=sys.stderr)
            return None
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse JSON: {path} ({exc})", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[WARN] Failed to read file: {path} ({exc})", file=sys.stderr)
        return None


def find_run_dirs(root: Path) -> Iterable[Path]:
    # A run directory is any directory containing config.json.
    for config_path in root.rglob("config.json"):
        yield config_path.parent


def get_last_raw_unlearning_event(metrics: Dict[str, Any]) -> Dict[str, Any]:
    events = metrics.get("unlearning_events")
    if isinstance(events, list) and events:
        last = events[-1]
        if isinstance(last, dict):
            return last
    return {}


def get_final_unlearning_block(metrics: Dict[str, Any]) -> Dict[str, Any]:
    normalized_final_unlearning = nested_get(
        metrics,
        "normalized_results",
        "final",
        "final_unlearning",
    )
    if isinstance(normalized_final_unlearning, dict):
        return normalized_final_unlearning
    return {}


def extract_row(run_dir: Path, config: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    raw_last_event = get_last_raw_unlearning_event(metrics)
    raw_overlap = raw_last_event.get("overlap", {}) if isinstance(raw_last_event.get("overlap"), dict) else {}
    final_unlearning = get_final_unlearning_block(metrics)
    final_overlap = final_unlearning.get("overlap", {}) if isinstance(final_unlearning.get("overlap"), dict) else {}

    # Preferred normalized fields with graceful fallback to older schema.
    final_avg_accuracy = first_non_none(
        nested_get(metrics, "normalized_results", "final", "final_avg_accuracy"),
        nested_get(metrics, "summary", "final_avg_accuracy"),
    )
    average_forgetting = first_non_none(
        nested_get(metrics, "normalized_results", "final", "average_forgetting"),
        nested_get(metrics, "forgetting", "final"),
        nested_get(metrics, "summary", "final_avg_forgetting"),
    )
    t_reset = first_non_none(final_unlearning.get("t_reset"), raw_last_event.get("t_reset"))
    t_retrain = first_non_none(final_unlearning.get("t_retrain"), raw_last_event.get("t_retrain"))
    t_forget_total = first_non_none(final_unlearning.get("t_forget_total"), raw_last_event.get("t_forget_total"))
    if t_forget_total is None and (t_reset is not None or t_retrain is not None):
        t_forget_total = (t_reset or 0.0) + (t_retrain or 0.0)

    return {
        "run_path": str(run_dir),
        "experiment_tag": first_non_none(config.get("experiment_tag"), nested_get(metrics, "run", "experiment_tag")),
        "dataset": first_non_none(config.get("dataset"), nested_get(metrics, "run", "dataset")),
        "method": first_non_none(config.get("method"), nested_get(metrics, "run", "method")),
        "method_variant": config.get("method_variant"),
        "variant": config.get("variant"),
        "seed": first_non_none(config.get("seed"), nested_get(metrics, "run", "seed")),
        "arch": config.get("arch"),
        "class_per_task": config.get("class_per_task"),
        "n_tasks": first_non_none(config.get("n_tasks"), nested_get(metrics, "run", "n_tasks")),
        "n_forget": first_non_none(config.get("n_forget"), nested_get(metrics, "run", "n_forget")),
        "n_epochs": config.get("n_epochs"),
        "adapter_bottleneck": config.get("adapter_bottleneck"),
        "adapter_train_classifier": config.get("adapter_train_classifier"),
        "adapter_location": config.get("adapter_location"),
        "adapter_shared_bottleneck": config.get("adapter_shared_bottleneck"),
        "adapter_shared_forget_ratio": config.get("adapter_shared_forget_ratio"),
        "adapter_shared_protect_ratio": config.get("adapter_shared_protect_ratio"),
        "adapter_shared_protect_strength": config.get("adapter_shared_protect_strength"),
        "protect_ratio": config.get("protect_ratio"),
        "protect_importance": config.get("protect_importance"),
        "lambda_protect": config.get("lambda_protect"),
        "retrain_steps": config.get("retrain_steps"),
        "request_schedule_file": first_non_none(
            config.get("request_schedule_file"),
            metrics.get("request_schedule_file"),
            nested_get(metrics, "run", "request_schedule_file"),
        ),
        "final_avg_accuracy": final_avg_accuracy,
        "average_forgetting": average_forgetting,
        "Fu": first_non_none(final_unlearning.get("Fu"), raw_last_event.get("Fu")),
        "WorstDrop": first_non_none(final_unlearning.get("WorstDrop"), raw_last_event.get("WorstDrop")),
        "Au": first_non_none(final_unlearning.get("Au"), raw_last_event.get("Au")),
        "t_reset": t_reset,
        "t_retrain": t_retrain,
        "t_forget_total": t_forget_total,
        "num_updated_params": first_non_none(
            final_unlearning.get("num_updated_params"),
            raw_last_event.get("num_updated_params"),
        ),
        "overlap_s_t": first_non_none(final_overlap.get("s_t"), raw_overlap.get("s_t")),
        "overlap_s_share": first_non_none(final_overlap.get("s_share"), raw_overlap.get("s_share")),
        "overlap_s_share_crit": first_non_none(final_overlap.get("s_share_crit"), raw_overlap.get("s_share_crit")),
        "overlap_s_share_ratio": first_non_none(
            final_overlap.get("s_share_ratio"),
            raw_overlap.get("s_share_ratio"),
        ),
        "overlap_s_share_crit_ratio": first_non_none(
            final_overlap.get("s_share_crit_ratio"),
            raw_overlap.get("s_share_crit_ratio"),
        ),
    }


def value_matches_filter(value: Any, accepted_values: Optional[List[str]]) -> bool:
    if not accepted_values:
        return True
    if value is None:
        return False
    return str(value) in {str(item) for item in accepted_values}


_SMOKE_TAG_PREFIXES = ("smoke", "test")


def is_smoke_run(row: Dict[str, Any]) -> bool:
    tag = row.get("experiment_tag")
    if tag is None:
        return False
    return str(tag).lower().startswith(_SMOKE_TAG_PREFIXES)


def row_matches_filters(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    # By default, smoke/test runs (experiment_tag starting with 'smoke'/'test')
    # are excluded so they never leak into thesis tables. Pass --include-smoke
    # to keep them.
    if not getattr(args, "include_smoke", False) and is_smoke_run(row):
        return False
    return (
        value_matches_filter(row.get("dataset"), args.dataset)
        and value_matches_filter(row.get("method"), args.method)
        and value_matches_filter(row.get("n_tasks"), args.n_tasks)
        and value_matches_filter(row.get("n_forget"), args.n_forget)
        and value_matches_filter(row.get("seed"), args.seed)
        and value_matches_filter(row.get("experiment_tag"), args.experiment_tag)
    )


def aggregate_runs(root: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(find_run_dirs(root)):
        config = load_json(run_dir / "config.json")
        if config is None:
            print(f"[WARN] Skipping run without valid config.json: {run_dir}", file=sys.stderr)
            continue
        metrics = load_json(run_dir / "metrics.json") or {}
        row = extract_row(run_dir, config, metrics)
        if row_matches_filters(row, args):
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment runs into one CSV summary.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Root directory to scan recursively.")
    parser.add_argument("--out", type=Path, default=Path("results_summary.csv"), help="Output CSV file path.")
    parser.add_argument("--dataset", nargs="+", default=None, help="Filter by dataset value(s).")
    parser.add_argument("--method", nargs="+", default=None, help="Filter by method value(s).")
    parser.add_argument("--n-tasks", dest="n_tasks", nargs="+", default=None, help="Filter by n_tasks value(s).")
    parser.add_argument("--n-forget", dest="n_forget", nargs="+", default=None, help="Filter by n_forget value(s).")
    parser.add_argument("--seed", nargs="+", default=None, help="Filter by seed value(s).")
    parser.add_argument("--experiment-tag", nargs="+", default=None, help="Filter by experiment_tag value(s).")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/test runs (experiment_tag starting with 'smoke'/'test'); excluded by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"[ERROR] Root directory does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"[ERROR] Root path is not a directory: {root}", file=sys.stderr)
        return 1

    rows = aggregate_runs(root, args)
    write_csv(rows, args.out)
    print(f"[INFO] Aggregated {len(rows)} runs into: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
