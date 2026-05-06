#!/usr/bin/env python3
"""
Analyze the relationship between overlap statistics and forgetting damage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    "experiment_tag",
    "seed",
    "final_avg_acc",
    "avg_forgetting",
    "Fu",
    "WorstDrop",
    "Au",
    "updated_param_ratio",
    "adapter_param_ratio",
    "S_share",
    "S_share_crit",
    "S_share_ratio",
    "S_share_crit_ratio",
    "overlap_protected_ratio",
    "overlap_updated_ratio",
    "overlap_protected_params",
    "overlap_updated_params",
]

MARKDOWN_COLUMNS = [
    "dataset",
    "method",
    "experiment_tag",
    "seed",
    "S_share_ratio",
    "S_share_crit_ratio",
    "final_avg_acc",
    "avg_forgetting",
    "WorstDrop",
    "Au",
    "updated_param_ratio",
]

PLOT_SPECS = (
    ("overlap_crit_vs_worstdrop.png", "WorstDrop", "WorstDrop vs Critical Shared Overlap"),
    ("overlap_crit_vs_avg_forgetting.png", "avg_forgetting", "Avg Forgetting vs Critical Shared Overlap"),
    ("overlap_crit_vs_final_acc.png", "final_avg_acc", "Final Avg Acc vs Critical Shared Overlap"),
)

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
CORRELATION_METRICS = ("WorstDrop", "avg_forgetting", "final_avg_acc", "Au")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze overlap-vs-damage trends from run artifacts.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run root scanned recursively.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/thesis"),
        help="Output directory for CSV/Markdown summaries and report_plots/.",
    )
    parser.add_argument("--method", default=None, help="Optional method filter.")
    parser.add_argument(
        "--include-tags",
        nargs="+",
        default=None,
        help="Optional experiment_tag allow-list.",
    )
    parser.add_argument(
        "--exclude-empty-tags",
        action="store_true",
        help="Skip runs with missing or 'None' experiment_tag values.",
    )
    parser.add_argument(
        "--min-overlap-crit-ratio",
        type=float,
        default=None,
        help="Optional minimum S_share_crit_ratio filter.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_int(value: Any) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


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


def nested_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def last_dict(items: Any) -> Dict[str, Any]:
    if not isinstance(items, list):
        return {}
    for item in reversed(items):
        if isinstance(item, dict):
            return item
    return {}


def extract_overlap_from_csv(path: Path) -> Dict[str, Optional[float]]:
    empty = {
        "S_share": None,
        "S_share_crit": None,
        "S_share_ratio": None,
        "S_share_crit_ratio": None,
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            normalized = {str(name).strip().lower(): name for name in fieldnames}
            required = ("s_share", "s_share_crit", "s_share_ratio", "s_share_crit_ratio")
            if not any(name in normalized for name in required):
                return empty

            last_values: Dict[str, Optional[float]] = dict(empty)
            seen = False
            for row in reader:
                row_values = {
                    "S_share": parse_float(row.get(normalized["s_share"])) if "s_share" in normalized else None,
                    "S_share_crit": (
                        parse_float(row.get(normalized["s_share_crit"])) if "s_share_crit" in normalized else None
                    ),
                    "S_share_ratio": (
                        parse_float(row.get(normalized["s_share_ratio"])) if "s_share_ratio" in normalized else None
                    ),
                    "S_share_crit_ratio": (
                        parse_float(row.get(normalized["s_share_crit_ratio"]))
                        if "s_share_crit_ratio" in normalized
                        else None
                    ),
                }
                if any(value is not None for value in row_values.values()):
                    last_values = row_values
                    seen = True
            return last_values if seen else empty
    except OSError:
        return empty


def metric_candidates(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
    normalized_results = nested_dict(metrics.get("normalized_results"))
    final_block = nested_dict(normalized_results.get("final"))
    normalized_last_event = last_dict(normalized_results.get("unlearning_events"))
    raw_last_event = last_dict(metrics.get("unlearning_events"))
    final_unlearning = nested_dict(final_block.get("final_unlearning"))
    top_level_final_unlearning = nested_dict(metrics.get("final_unlearning"))
    overlap_blocks = (
        nested_dict(final_unlearning.get("overlap")),
        nested_dict(top_level_final_unlearning.get("overlap")),
        nested_dict(normalized_last_event.get("overlap")),
        nested_dict(raw_last_event.get("overlap")),
    )
    return (
        final_block,
        final_unlearning,
        normalized_last_event,
        raw_last_event,
        metrics,
        top_level_final_unlearning,
        *overlap_blocks,
    )


def extract_overlap_analysis(metrics: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    analysis_candidates = [
        nested_dict(metrics.get("overlap_analysis")),
        nested_dict(nested_dict(metrics.get("protection")).get("overlap_analysis")),
    ]
    for candidate in candidates:
        analysis_candidates.append(nested_dict(candidate.get("overlap_analysis")))
        analysis_candidates.append(nested_dict(nested_dict(candidate.get("protection")).get("overlap_analysis")))
    for candidate in analysis_candidates:
        if candidate:
            return candidate
    return {}


def extract_value(candidates: Sequence[Dict[str, Any]], *keys: str) -> Any:
    for candidate in candidates:
        for key in keys:
            if key in candidate and candidate.get(key) is not None:
                return candidate.get(key)
    return None


def format_number(value: Any, decimals: int = 6) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if abs(number) < 0.5 * (10 ** (-decimals)):
        number = 0.0
    return f"{number:.{decimals}f}"


def format_markdown_value(column: str, value: Any) -> str:
    if column == "seed":
        return str(value) if value not in (None, "") else "NA"
    if column in {"dataset", "method", "experiment_tag"}:
        text = str(value).strip()
        return text if text else "NA"
    formatted = format_number(value, decimals=4)
    return formatted if formatted else "NA"


def normalize_tag(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return text


def iter_run_dirs(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        yield run_dir


def build_row(run_dir: Path) -> Optional[Dict[str, Any]]:
    config = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "metrics.json")
    if config is None or metrics is None:
        return None

    candidates = metric_candidates(metrics)
    overlap_analysis = extract_overlap_analysis(metrics, candidates)
    run_block = nested_dict(metrics.get("run"))
    overlap_csv_values = extract_overlap_from_csv(run_dir / "overlap.csv")
    preferred_critical_ratio = parse_float(
        first_non_none(
            overlap_analysis.get("critical_ratio"),
            extract_value(candidates, "S_share_crit_ratio", "s_share_crit_ratio"),
        )
    )

    row = {
        "dataset": first_non_none(config.get("dataset"), run_block.get("dataset")),
        "method": first_non_none(config.get("method"), run_block.get("method")),
        "method_variant": canonicalize_method_variant(
            first_non_none(
                nested_dict(metrics.get("normalized_results")).get("final", {}).get("method_variant")
                if isinstance(nested_dict(metrics.get("normalized_results")).get("final"), dict)
                else None,
                nested_dict(metrics.get("normalized_results")).get("final", {}).get("protection", {}).get("method_variant")
                if isinstance(nested_dict(metrics.get("normalized_results")).get("final"), dict)
                and isinstance(nested_dict(metrics.get("normalized_results")).get("final", {}).get("protection"), dict)
                else None,
                metrics.get("method_variant"),
                nested_dict(metrics.get("protection")).get("method_variant"),
            )
        ),
        "experiment_tag": first_non_none(config.get("experiment_tag"), run_block.get("experiment_tag")),
        "seed": first_non_none(config.get("seed"), run_block.get("seed")),
        "final_avg_acc": parse_float(extract_value(candidates, "final_avg_acc", "final_avg_accuracy")),
        "avg_forgetting": parse_float(extract_value(candidates, "avg_forgetting", "average_forgetting")),
        "Fu": parse_float(extract_value(candidates, "Fu")),
        "WorstDrop": parse_float(extract_value(candidates, "WorstDrop")),
        "Au": parse_float(extract_value(candidates, "Au")),
        "updated_param_ratio": parse_float(extract_value(candidates, "updated_param_ratio")),
        "adapter_param_ratio": parse_float(extract_value(candidates, "adapter_param_ratio")),
        "S_share": parse_int(first_non_none(overlap_analysis.get("shared_total"), extract_value(candidates, "S_share", "s_share"))),
        "S_share_crit": parse_int(
            first_non_none(overlap_analysis.get("shared_critical"), extract_value(candidates, "S_share_crit", "s_share_crit"))
        ),
        "S_share_ratio": parse_float(extract_value(candidates, "S_share_ratio", "s_share_ratio")),
        "S_share_crit_ratio": preferred_critical_ratio,
        "overlap_protected_ratio": parse_float(overlap_analysis.get("protected_ratio")),
        "overlap_updated_ratio": parse_float(overlap_analysis.get("updated_ratio")),
        "overlap_protected_params": parse_int(overlap_analysis.get("protected_params")),
        "overlap_updated_params": parse_int(overlap_analysis.get("updated_params")),
    }

    for key, value in overlap_csv_values.items():
        if row.get(key) is None and value is not None:
            row[key] = int(value) if key in {"S_share", "S_share_crit"} else value

    if row["S_share_crit_ratio"] is None:
        return None
    return row


def row_matches_filters(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    method = str(row.get("method") or "").strip()
    experiment_tag = normalize_tag(row.get("experiment_tag"))
    overlap_crit_ratio = parse_float(row.get("S_share_crit_ratio"))

    if args.method is not None and method != args.method:
        return False
    if args.exclude_empty_tags and experiment_tag == "":
        return False
    if args.include_tags is not None and experiment_tag not in set(args.include_tags):
        return False
    if args.min_overlap_crit_ratio is not None:
        if overlap_crit_ratio is None or overlap_crit_ratio < args.min_overlap_crit_ratio:
            return False
    return True


def sort_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset") or ""),
            str(row.get("method") or ""),
            str(row.get("experiment_tag") or ""),
            int(row.get("seed")) if row.get("seed") is not None else -1,
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            out_row = {
                "dataset": row.get("dataset") or "",
                "method": row.get("method") or "",
                "experiment_tag": row.get("experiment_tag") or "",
                "seed": row.get("seed") if row.get("seed") is not None else "",
                "final_avg_acc": format_number(row.get("final_avg_acc")),
                "avg_forgetting": format_number(row.get("avg_forgetting")),
                "Fu": format_number(row.get("Fu")),
                "WorstDrop": format_number(row.get("WorstDrop")),
                "Au": format_number(row.get("Au")),
                "updated_param_ratio": format_number(row.get("updated_param_ratio")),
                "adapter_param_ratio": format_number(row.get("adapter_param_ratio")),
                "S_share": row.get("S_share") if row.get("S_share") is not None else "",
                "S_share_crit": row.get("S_share_crit") if row.get("S_share_crit") is not None else "",
                "S_share_ratio": format_number(row.get("S_share_ratio")),
                "S_share_crit_ratio": format_number(row.get("S_share_crit_ratio")),
            }
            writer.writerow(out_row)


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_markdown_with_summary(path, rows, correlation_rows=[])


def write_markdown_with_summary(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    correlation_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Correlation summary:"]
    if correlation_rows:
        correlation_columns = ["metric", "pearson_r", "n"]
        lines.extend(
            [
                "| " + " | ".join(correlation_columns) + " |",
                "| " + " | ".join(["---"] * len(correlation_columns)) + " |",
            ]
        )
        for row in correlation_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("metric", "")).strip() or "NA",
                        format_markdown_value("pearson_r", row.get("pearson_r")),
                        str(row.get("n", "")) if row.get("n") not in (None, "") else "NA",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No correlation rows available.")

    lines.append("")
    lines.extend(
        [
            "| " + " | ".join(MARKDOWN_COLUMNS) + " |",
            "| " + " | ".join(["---"] * len(MARKDOWN_COLUMNS)) + " |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_markdown_value(column, row.get(column)) for column in MARKDOWN_COLUMNS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_by_method(rows: Sequence[Dict[str, Any]]) -> list[tuple[str, list[Dict[str, Any]]]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method") or "unknown")
        grouped.setdefault(method, []).append(row)
    return sorted(grouped.items(), key=lambda item: item[0])


def import_pandas():
    try:
        import pandas as pd

        return pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required to compute overlap_vs_damage correlations. "
            "Install pandas in the active Python environment and rerun the script."
        ) from exc


def build_correlation_summary(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    pd = import_pandas()
    dataframe = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary_rows: list[Dict[str, Any]] = []
    for metric in CORRELATION_METRICS:
        pair = dataframe[["S_share_crit_ratio", metric]].copy()
        pair["S_share_crit_ratio"] = pd.to_numeric(pair["S_share_crit_ratio"], errors="coerce")
        pair[metric] = pd.to_numeric(pair[metric], errors="coerce")
        pair = pair.dropna()
        n = int(len(pair))
        pearson_r: Optional[float] = None
        if n >= 2:
            pearson_value = pair["S_share_crit_ratio"].corr(pair[metric], method="pearson")
            pearson_r = parse_float(pearson_value)
        summary_rows.append({"metric": metric, "pearson_r": pearson_r, "n": n})
    return summary_rows


def write_correlation_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "pearson_r", "n"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row.get("metric") or "",
                    "pearson_r": format_number(row.get("pearson_r")),
                    "n": row.get("n") if row.get("n") is not None else "",
                }
            )


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate overlap_vs_damage plots. "
            "Install matplotlib in the active Python environment and rerun the script."
        ) from exc


def plot_metric(rows: Sequence[Dict[str, Any]], y_key: str, title: str, out_path: Path) -> None:
    plt = import_pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    plotted = False
    for index, (method, method_rows) in enumerate(grouped_by_method(rows)):
        xs: list[float] = []
        ys: list[float] = []
        for row in method_rows:
            x_value = parse_float(row.get("S_share_crit_ratio"))
            y_value = parse_float(row.get(y_key))
            if x_value is None or y_value is None:
                continue
            xs.append(x_value)
            ys.append(y_value)
        if not xs:
            continue
        marker = MARKERS[index % len(MARKERS)]
        ax.scatter(xs, ys, label=method, marker=marker, s=36, alpha=0.85)
        plotted = True

    ax.set_xlabel("S_share_crit_ratio")
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Root directory does not exist: {args.root}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"[ERROR] Root path is not a directory: {args.root}", file=sys.stderr)
        return 1

    scanned_runs = 0
    skipped_missing_overlap = 0
    extracted_rows: list[Dict[str, Any]] = []

    for run_dir in sorted(iter_run_dirs(args.root)):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        scanned_runs += 1
        row = build_row(run_dir)
        if row is None:
            skipped_missing_overlap += 1
            continue
        extracted_rows.append(row)

    rows = sort_rows([row for row in extracted_rows if row_matches_filters(row, args)])

    out_csv = args.outdir / "overlap_vs_damage.csv"
    out_md = args.outdir / "overlap_vs_damage.md"
    out_corr_csv = args.outdir / "overlap_correlation_summary.csv"
    plots_dir = args.outdir / "report_plots"

    write_csv(out_csv, rows)
    try:
        correlation_rows = build_correlation_summary(rows)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    write_correlation_csv(out_corr_csv, correlation_rows)
    write_markdown_with_summary(out_md, rows, correlation_rows)

    print(f"[INFO] Scanned runs: {scanned_runs}")
    print(f"[INFO] Rows included: {len(rows)}")
    print(f"[INFO] Rows skipped due to missing overlap: {skipped_missing_overlap}")
    print(f"[INFO] Wrote CSV summary: {out_csv}")
    print(f"[INFO] Wrote Markdown summary: {out_md}")
    print(f"[INFO] Wrote correlation summary: {out_corr_csv}")

    try:
        for filename, y_key, title in PLOT_SPECS:
            plot_metric(rows, y_key, title, plots_dir / filename)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Wrote plots under: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
