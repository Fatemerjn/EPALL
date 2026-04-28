#!/usr/bin/env python3
"""
Plot thesis summary results from the aggregated thesis table CSV.
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


PlotSpec = Tuple[str, str, str, str, str]


PLOTS: Sequence[PlotSpec] = (
    (
        "updated_param_ratio_mean",
        "WorstDrop_mean",
        "tradeoff_plot.png",
        "Updated Parameter Ratio",
        "WorstDrop",
    ),
    (
        "Fu_mean",
        "Au_mean",
        "forgetting_quality_plot.png",
        "Fu",
        "Au",
    ),
    (
        "adapter_param_ratio_mean",
        "final_avg_acc_mean",
        "overlap_efficiency_proxy_plot.png",
        "Adapter Parameter Ratio",
        "Final Average Accuracy",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot thesis result summaries from thesis_table.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/thesis/thesis_table.csv"),
        help="Input thesis table CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/thesis/plots"),
        help="Directory to save plot images.",
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


def filtered_points(rows: Sequence[Dict[str, str]], x_key: str, y_key: str) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    for row in rows:
        method = (row.get("method") or "").strip()
        x_value = parse_float(row.get(x_key))
        y_value = parse_float(row.get(y_key))
        if not method or x_value is None or y_value is None:
            continue
        points.append((method, x_value, y_value))
    return points


def build_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    deduped: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in deduped:
            deduped[label] = handle
    if deduped:
        ax.legend(deduped.values(), deduped.keys(), loc="best")


def make_plot(
    rows: Sequence[Dict[str, str]],
    x_key: str,
    y_key: str,
    output_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    points = filtered_points(rows, x_key, y_key)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not points:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for method, x_value, y_value in points:
            ax.scatter(x_value, y_value, s=60, label=method)
            ax.annotate(method, (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        build_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if rows is None:
        return 1

    written = 0
    for x_key, y_key, filename, x_label, y_label in PLOTS:
        output_path = args.outdir / filename
        make_plot(rows, x_key, y_key, output_path, x_label, y_label)
        print(f"[INFO] Wrote plot: {output_path}")
        written += 1

    print(f"[INFO] Figures generated: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
