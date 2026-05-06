#!/usr/bin/env python3
"""
Build a compact paper-oriented overlap experiment summary from run metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


CANONICAL_METHOD_VARIANTS = {
    "adapter_hard_critical_mask": "pall_adapter_hard_mask",
    "adapter_explicit_critical_mask": "pall_adapter_soft_mask",
}


OUTPUT_COLUMNS = [
    "dataset",
    "method",
    "method_variant",
    "experiment_tag",
    "n",
    "final_avg_acc",
    "avg_forgetting",
    "WorstDrop",
    "Au",
    "updated_param_ratio",
    "adapter_param_ratio",
    "critical_ratio",
    "protected_ratio",
    "updated_overlap_ratio",
    "shared_critical_count",
    "protected_params",
    "updated_params",
]

NUMERIC_COLUMNS = [
    "final_avg_acc",
    "avg_forgetting",
    "WorstDrop",
    "Au",
    "updated_param_ratio",
    "adapter_param_ratio",
    "critical_ratio",
    "protected_ratio",
    "updated_overlap_ratio",
    "shared_critical_count",
    "protected_params",
    "updated_params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize overlap-oriented experiment metrics from runs/**/metrics.json.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run root scanned recursively.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/thesis/overlap_experiment_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/thesis/overlap_experiment_summary.md"),
        help="Output Markdown path.",
    )
    parser.add_argument("--include-tags", nargs="+", default=None, help="Optional experiment_tag allow-list.")
    parser.add_argument("--method", default=None, help="Optional method filter.")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read JSON: {path} ({exc})", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


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


def nested_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def last_dict(items: Any) -> Dict[str, Any]:
    if not isinstance(items, list):
        return {}
    for item in reversed(items):
        if isinstance(item, dict):
            return item
    return {}


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


def to_int(value: Any) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return text


def canonicalize_method_variant(name: Any) -> str:
    text = normalize_text(name)
    if not text:
        return ""
    return CANONICAL_METHOD_VARIANTS.get(text, text)


def iter_run_dirs(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        yield run_dir


def metric_candidates(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
    normalized_results = nested_dict(metrics.get("normalized_results"))
    final_block = nested_dict(normalized_results.get("final"))
    normalized_last_event = last_dict(normalized_results.get("unlearning_events"))
    raw_last_event = last_dict(metrics.get("unlearning_events"))
    return (
        final_block,
        nested_dict(final_block.get("protection")),
        normalized_last_event,
        nested_dict(normalized_last_event.get("protection")),
        raw_last_event,
        nested_dict(raw_last_event.get("protection")),
        metrics,
        nested_dict(metrics.get("protection")),
    )


def extract_value(candidates: Sequence[Dict[str, Any]], *keys: str) -> Any:
    for candidate in candidates:
        for key in keys:
            if key in candidate and candidate.get(key) is not None:
                return candidate.get(key)
    return None


def extract_overlap_analysis(metrics: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for candidate in candidates:
        direct = nested_dict(candidate.get("overlap_analysis"))
        if direct:
            return direct
        nested = nested_dict(nested_dict(candidate.get("protection")).get("overlap_analysis"))
        if nested:
            return nested
    return {}


def extract_legacy_overlap_counts(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    shared_forget_count = to_int(extract_value(candidates, "shared_forget_count", "shared_forget", "S_share"))
    shared_critical_count = to_int(
        extract_value(candidates, "shared_critical_count", "shared_critical", "S_share_crit")
    )
    protected_params = to_int(
        extract_value(candidates, "protected_adapter_params", "hard_protected_adapter_params", "protected_params")
    )
    updated_params = to_int(extract_value(candidates, "updated_adapter_params", "updated_params"))
    return {
        "shared_forget_count": shared_forget_count,
        "shared_critical_count": shared_critical_count,
        "protected_params": protected_params,
        "updated_params": updated_params,
    }


def derive_updated_param_ratio(metrics: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Optional[float]:
    ratio = to_float(
        first_non_none(
            metrics.get("updated_param_ratio"),
            extract_value(candidates, "updated_param_ratio"),
            nested_get(metrics, "summary", "updated_param_ratio"),
        )
    )
    if ratio is not None:
        return ratio
    adapter_stats = nested_dict(metrics.get("adapter_stats"))
    model_stats = nested_dict(metrics.get("model"))
    total_model_params = to_float(
        first_non_none(
            adapter_stats.get("total_model_params"),
            model_stats.get("total_params"),
        )
    )
    num_updated_params = to_float(
        first_non_none(
            metrics.get("num_updated_params"),
            extract_value(candidates, "num_updated_params"),
        )
    )
    if total_model_params and num_updated_params is not None:
        return float(num_updated_params / total_model_params)
    return None


def derive_adapter_param_ratio(metrics: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Optional[float]:
    ratio = to_float(
        first_non_none(
            metrics.get("adapter_param_ratio"),
            extract_value(candidates, "adapter_param_ratio"),
            nested_get(metrics, "summary", "adapter_param_ratio"),
        )
    )
    if ratio is not None:
        return ratio
    adapter_stats = nested_dict(metrics.get("adapter_stats"))
    model_stats = nested_dict(metrics.get("model"))
    total_model_params = to_float(
        first_non_none(
            adapter_stats.get("total_model_params"),
            model_stats.get("total_params"),
        )
    )
    adapter_params = to_float(
        first_non_none(
            adapter_stats.get("adapter_params"),
            model_stats.get("adapter_params"),
        )
    )
    if total_model_params and adapter_params is not None:
        return float(adapter_params / total_model_params)
    return None


def extract_method_variant(metrics: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> str:
    generic_variants = {"", "adapter"}
    fallback_values: list[str] = []
    for candidate in candidates:
        for value in (candidate.get("method_variant"), nested_get(candidate, "protection", "method_variant")):
            text = canonicalize_method_variant(value)
            if not text:
                continue
            if text.lower() not in generic_variants:
                return text
            fallback_values.append(text)
    for value in (
        metrics.get("method_variant"),
        nested_get(metrics, "summary", "method_variant"),
        metrics.get("method"),
    ):
        text = canonicalize_method_variant(value)
        if text and text.lower() not in generic_variants:
            return text
        if text:
            fallback_values.append(text)
    return fallback_values[0] if fallback_values else "NA"


def build_row(run_dir: Path) -> Optional[Dict[str, Any]]:
    metrics = load_json(run_dir / "metrics.json")
    if metrics is None:
        return None
    config = load_json(run_dir / "config.json") or {}
    candidates = metric_candidates(metrics)
    overlap_analysis = extract_overlap_analysis(metrics, candidates)
    legacy_overlap = extract_legacy_overlap_counts(candidates)
    run_block = nested_dict(metrics.get("run"))

    dataset = normalize_text(first_non_none(config.get("dataset"), metrics.get("dataset"), run_block.get("dataset")))
    method = normalize_text(first_non_none(config.get("method"), metrics.get("method"), run_block.get("method")))
    experiment_tag = normalize_text(
        first_non_none(config.get("experiment_tag"), metrics.get("experiment_tag"), run_block.get("experiment_tag"))
    )
    seed = first_non_none(config.get("seed"), metrics.get("seed"), run_block.get("seed"))

    if not dataset or not method:
        return None

    shared_forget_count = to_int(first_non_none(overlap_analysis.get("shared_forget"), legacy_overlap["shared_forget_count"]))
    shared_critical_count = to_int(
        first_non_none(
            overlap_analysis.get("shared_critical"),
            legacy_overlap["shared_critical_count"],
            extract_value(candidates, "S_share_crit", "s_share_crit"),
        )
    )
    protected_params = to_int(first_non_none(overlap_analysis.get("protected_params"), legacy_overlap["protected_params"]))
    updated_params = to_int(first_non_none(overlap_analysis.get("updated_params"), legacy_overlap["updated_params"]))
    critical_ratio = to_float(overlap_analysis.get("critical_ratio"))
    protected_ratio = to_float(overlap_analysis.get("protected_ratio"))
    updated_overlap_ratio = to_float(overlap_analysis.get("updated_ratio"))

    if critical_ratio is None and shared_critical_count is not None and shared_forget_count is not None:
        critical_ratio = float(shared_critical_count / max(shared_forget_count, 1))
    if protected_ratio is None and protected_params is not None and shared_forget_count is not None:
        protected_ratio = float(protected_params / max(shared_forget_count, 1))
    if updated_overlap_ratio is None and updated_params is not None and shared_forget_count is not None:
        updated_overlap_ratio = float(updated_params / max(shared_forget_count, 1))

    return {
        "dataset": dataset,
        "method": method,
        "method_variant": extract_method_variant(metrics, candidates),
        "experiment_tag": experiment_tag,
        "seed": seed,
        "final_avg_acc": to_float(extract_value(candidates, "final_avg_acc", "final_avg_accuracy")),
        "avg_forgetting": to_float(extract_value(candidates, "avg_forgetting", "average_forgetting")),
        "Fu": to_float(extract_value(candidates, "Fu")),
        "WorstDrop": to_float(extract_value(candidates, "WorstDrop")),
        "Au": to_float(extract_value(candidates, "Au")),
        "updated_param_ratio": derive_updated_param_ratio(metrics, candidates),
        "adapter_param_ratio": derive_adapter_param_ratio(metrics, candidates),
        "critical_ratio": first_non_none(
            critical_ratio,
            to_float(extract_value(candidates, "S_share_crit_ratio", "s_share_crit_ratio")),
        ),
        "protected_ratio": protected_ratio,
        "updated_overlap_ratio": updated_overlap_ratio,
        "shared_critical_count": shared_critical_count,
        "protected_params": protected_params,
        "updated_params": updated_params,
    }


def row_matches_filters(row: Dict[str, Any], args: argparse.Namespace, include_tags: Optional[set[str]]) -> bool:
    if args.method is not None and row.get("method") != args.method:
        return False
    if include_tags is not None and normalize_text(row.get("experiment_tag")) not in include_tags:
        return False
    return True


def mean_std(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), None
    return float(statistics.mean(values)), float(statistics.stdev(values))


def format_stat(values: Sequence[float]) -> str:
    mean_value, std_value = mean_std(values)
    if mean_value is None:
        return "NA"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def aggregate_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {"n": len(rows)}
    for column in NUMERIC_COLUMNS:
        values = [float(value) for value in (row.get(column) for row in rows) if value is not None]
        aggregated[column] = format_stat(values)
    return aggregated


def build_summary(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str, str], list[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            normalize_text(row.get("dataset")),
            normalize_text(row.get("method")),
            normalize_text(row.get("method_variant")) or "NA",
            normalize_text(row.get("experiment_tag")),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[Dict[str, Any]] = []
    for key in sorted(grouped):
        dataset, method, method_variant, experiment_tag = key
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "method_variant": method_variant or "NA",
                "experiment_tag": experiment_tag,
                **aggregate_group(grouped[key]),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(OUTPUT_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(OUTPUT_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "NA") or "NA") for column in OUTPUT_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Root directory does not exist: {args.root}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"[ERROR] Root path is not a directory: {args.root}", file=sys.stderr)
        return 1

    include_tags = {normalize_text(tag) for tag in args.include_tags} if args.include_tags else None

    scanned_runs = 0
    included_rows: list[Dict[str, Any]] = []
    for run_dir in sorted(iter_run_dirs(args.root)):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        scanned_runs += 1
        row = build_row(run_dir)
        if row is None:
            continue
        if not row_matches_filters(row, args, include_tags):
            continue
        included_rows.append(row)

    summary_rows = build_summary(included_rows)
    write_csv(args.out_csv, summary_rows)
    write_markdown(args.out_md, summary_rows)

    print(f"[INFO] Runs scanned: {scanned_runs}")
    print(f"[INFO] Runs included: {len(included_rows)}")
    print(f"[INFO] Groups summarized: {len(summary_rows)}")
    print(f"[INFO] Wrote CSV: {args.out_csv}")
    print(f"[INFO] Wrote Markdown: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
