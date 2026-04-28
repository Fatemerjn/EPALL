#!/usr/bin/env python3
"""
Generate thesis/report plots from the compact report table CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BAR_PLOTS: Sequence[Tuple[str, str, str]] = (
    ("final_avg_acc", "Final Average Accuracy", "final_accuracy"),
    ("avg_forgetting", "Average Forgetting", "avg_forgetting"),
    ("WorstDrop", "WorstDrop", "worstdrop"),
    ("Fu", "Fu", "fu"),
    ("Au", "Au", "au"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report plots from report_table.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/thesis/report_table.csv"),
        help="Input report table CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/thesis/report_plots"),
        help="Directory for generated plots.",
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
    if text == "" or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def parse_mean_std(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    text = str(value).strip()
    if text == "" or text.upper() == "NA":
        return None, None
    if "+/-" in text:
        mean_text, std_text = [part.strip() for part in text.split("+/-", 1)]
        return parse_float(mean_text), parse_float(std_text)
    return parse_float(text), None


def group_by_dataset(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        dataset = (row.get("dataset") or "").strip()
        if not dataset:
            continue
        grouped.setdefault(dataset, []).append(row)
    return grouped


def sanitize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("/", "_")


def method_label(method: str) -> str:
    return method.replace("_", " ")


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)


def dedup_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    deduped: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in deduped:
            deduped[label] = handle
    if deduped:
        ax.legend(deduped.values(), deduped.keys(), loc="best")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_bar_plot(dataset: str, rows: Sequence[Dict[str, str]], metric_key: str, ylabel: str, stem: str, outdir: Path) -> Path:
    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for row in rows:
        mean_value, std_value = parse_mean_std(row.get(metric_key))
        if mean_value is None:
            continue
        labels.append(method_label((row.get("method") or "").strip()))
        means.append(mean_value)
        stds.append(0.0 if std_value is None else std_value)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not labels:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        positions = list(range(len(labels)))
        ax.bar(positions, means, yerr=stds, capsize=4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        style_axis(ax, f"{dataset}: {ylabel}", "Method", ylabel)

    output_path = outdir / f"{sanitize_name(dataset)}_{stem}.png"
    save_figure(fig, output_path)
    return output_path


def make_updated_ratio_bar_plot(dataset: str, rows: Sequence[Dict[str, str]], outdir: Path) -> Path:
    labels: List[str] = []
    values: List[float] = []
    for row in rows:
        value = parse_float(row.get("updated_param_ratio_mean"))
        if value is None:
            continue
        labels.append(method_label((row.get("method") or "").strip()))
        values.append(value)

    fig, ax = plt.subplots(figsize=(8, 5))
    if not labels:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        positions = list(range(len(labels)))
        ax.bar(positions, values)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        style_axis(ax, f"{dataset}: Updated Parameter Ratio", "Method", "Updated Parameter Ratio")

    output_path = outdir / f"{sanitize_name(dataset)}_updated_param_ratio.png"
    save_figure(fig, output_path)
    return output_path


def build_tradeoff_points(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    for row in rows:
        x_value = parse_float(row.get("updated_param_ratio_mean"))
        y_mean, _ = parse_mean_std(row.get("WorstDrop"))
        method = (row.get("method") or "").strip()
        if x_value is None or y_mean is None or not method:
            continue
        points.append((method, x_value, y_mean))
    return points


def make_tradeoff_scatter(dataset: str, rows: Sequence[Dict[str, str]], outdir: Path) -> Path:
    points = build_tradeoff_points(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not points:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for method, x_value, y_value in points:
            ax.scatter(x_value, y_value, s=60, label=method_label(method))
            ax.annotate(method_label(method), (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_title(f"{dataset}: Updated Ratio vs WorstDrop")
        ax.set_xlabel("Updated Parameter Ratio")
        ax.set_ylabel("WorstDrop")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        dedup_legend(ax)

    output_path = outdir / f"{sanitize_name(dataset)}_tradeoff_scatter.png"
    save_figure(fig, output_path)
    return output_path


def build_forgetting_quality_points(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    for row in rows:
        x_mean, _ = parse_mean_std(row.get("Fu"))
        y_mean, _ = parse_mean_std(row.get("Au"))
        method = (row.get("method") or "").strip()
        if x_mean is None or y_mean is None or not method:
            continue
        points.append((method, x_mean, y_mean))
    return points


def make_forgetting_quality_scatter(dataset: str, rows: Sequence[Dict[str, str]], outdir: Path) -> Path:
    points = build_forgetting_quality_points(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not points:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for method, x_value, y_value in points:
            ax.scatter(x_value, y_value, s=60, label=method_label(method))
            ax.annotate(method_label(method), (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_title(f"{dataset}: Fu vs Au")
        ax.set_xlabel("Fu")
        ax.set_ylabel("Au")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        dedup_legend(ax)

    output_path = outdir / f"{sanitize_name(dataset)}_forgetting_quality_scatter.png"
    save_figure(fig, output_path)
    return output_path


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if rows is None:
        return 1

    grouped = group_by_dataset(rows)
    written = 0
    for dataset in sorted(grouped.keys()):
        dataset_rows = grouped[dataset]
        for metric_key, ylabel, stem in BAR_PLOTS:
            output_path = make_bar_plot(dataset, dataset_rows, metric_key, ylabel, stem, args.outdir)
            print(f"[INFO] Wrote plot: {output_path}")
            written += 1
        output_path = make_updated_ratio_bar_plot(dataset, dataset_rows, args.outdir)
        print(f"[INFO] Wrote plot: {output_path}")
        written += 1
        output_path = make_tradeoff_scatter(dataset, dataset_rows, args.outdir)
        print(f"[INFO] Wrote plot: {output_path}")
        written += 1
        output_path = make_forgetting_quality_scatter(dataset, dataset_rows, args.outdir)
        print(f"[INFO] Wrote plot: {output_path}")
        written += 1

    print(f"[INFO] Figures generated: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
