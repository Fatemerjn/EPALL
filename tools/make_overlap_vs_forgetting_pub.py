#!/usr/bin/env python3
"""
Create a publication-quality overlap-vs-forgetting plot.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a publication-quality overlap-vs-forgetting PDF plot.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/thesis/overlap_vs_damage.csv"),
        help="Input overlap-vs-damage CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/thesis/report_plots/overlap_vs_forgetting_pub.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument("--method", default=None, help="Optional method filter.")
    parser.add_argument("--min-overlap-crit-ratio", type=float, default=None, help="Optional x-axis filter.")
    return parser.parse_args()


def validate_input(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"S_share_crit_ratio", "avg_forgetting", "method"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(sorted(missing))}")

    filtered = dataframe.copy()
    filtered["S_share_crit_ratio"] = pd.to_numeric(filtered["S_share_crit_ratio"], errors="coerce")
    filtered["avg_forgetting"] = pd.to_numeric(filtered["avg_forgetting"], errors="coerce")
    filtered = filtered.dropna(subset=["S_share_crit_ratio", "avg_forgetting"])
    if filtered.empty:
        raise ValueError("No valid rows remain after dropping missing S_share_crit_ratio/avg_forgetting values.")
    return filtered


def compute_regression_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.size < 2:
        raise ValueError("At least two points are required for linear regression.")

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x_grid + intercept

    n = float(x.size)
    x_mean = float(np.mean(x))
    ssx = float(np.sum((x - x_mean) ** 2))
    if math.isclose(ssx, 0.0):
        raise ValueError("Cannot fit regression: S_share_crit_ratio has zero variance.")

    residuals = y - (slope * x + intercept)
    dof = x.size - 2
    if dof <= 0:
        t_crit = 12.706204736432095  # 95% CI for df=1
        residual_std = float(np.sqrt(np.sum(residuals**2) / max(1, x.size)))
    else:
        residual_std = float(np.sqrt(np.sum(residuals**2) / dof))
        t_crit_lookup = {
            1: 12.706204736432095,
            2: 4.302652729696142,
            3: 3.182446305284263,
            4: 2.7764451051977987,
            5: 2.570581835636305,
            6: 2.4469118511449692,
            7: 2.3646242510102993,
            8: 2.306004135204166,
            9: 2.2621571628540993,
            10: 2.2281388519649385,
            11: 2.200985160082949,
            12: 2.1788128296634177,
            13: 2.160368656461013,
            14: 2.1447866879169273,
            15: 2.131449545559323,
            16: 2.1199052992210112,
            17: 2.1098155778331806,
            18: 2.10092204024096,
            19: 2.093024054408263,
            20: 2.085963447265837,
            21: 2.079613844727662,
            22: 2.073873067904015,
            23: 2.068657610419041,
            24: 2.063898561628021,
            25: 2.0595385527532946,
            26: 2.0555294386428713,
            27: 2.0518305164802833,
            28: 2.048407141795244,
            29: 2.045229642132703,
            30: 2.042272456301238,
        }
        t_crit = t_crit_lookup.get(dof, 1.959963984540054)

    se_mean = residual_std * np.sqrt((1.0 / n) + ((x_grid - x_mean) ** 2 / ssx))
    ci_delta = t_crit * se_mean
    return y_fit, y_fit - ci_delta, y_fit + ci_delta


def plot_publication_figure(dataframe: pd.DataFrame, output_path: Path) -> None:
    methods = sorted(str(method) for method in dataframe["method"].fillna("unknown").unique())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    x = dataframe["S_share_crit_ratio"].to_numpy(dtype=float)
    y = dataframe["avg_forgetting"].to_numpy(dtype=float)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_fit, y_low, y_high = compute_regression_band(x, y, x_grid)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    for index, method in enumerate(methods):
        subset = dataframe[dataframe["method"].fillna("unknown") == method]
        ax.scatter(
            subset["S_share_crit_ratio"],
            subset["avg_forgetting"],
            s=58,
            alpha=0.9,
            marker=markers[index % len(markers)],
            edgecolors="white",
            linewidths=0.5,
            label=method,
        )

    ax.plot(x_grid, y_fit, color="black", linewidth=1.8, label="Linear fit", zorder=3)
    ax.fill_between(x_grid, y_low, y_high, color="black", alpha=0.12, linewidth=0, zorder=2)

    ax.set_xlabel("Critical Shared Overlap Ratio ($S_{share,crit}$ ratio)")
    ax.set_ylabel("Average Forgetting")
    ax.set_title("Shared Critical Overlap vs Forgetting Damage")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.grid(False)
    ax.legend(frameon=False, ncol=1, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"[ERROR] Input CSV not found: {args.input}")

    dataframe = pd.read_csv(args.input)
    dataframe = validate_input(dataframe)

    if args.method is not None:
        dataframe = dataframe[dataframe["method"].fillna("") == args.method]
    if args.min_overlap_crit_ratio is not None:
        dataframe = dataframe[dataframe["S_share_crit_ratio"] >= args.min_overlap_crit_ratio]
    if dataframe.empty:
        raise SystemExit("[ERROR] No rows remain after applying filters.")

    plot_publication_figure(dataframe, args.output)
    print(f"[INFO] Wrote publication plot: {args.output}")
    print(f"[INFO] Rows plotted: {len(dataframe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
