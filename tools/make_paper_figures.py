#!/usr/bin/env python3
"""Generate the four publication figures for the PALL-Adapter manuscript.

Outputs (vector PDF, IEEE-compliant style):
    - drop_decomposition.pdf   : conceptual decomposition of the WorstDrop budget
                                 (visualizes Theorem 1; soft-scaling vs. unconstrained).
    - worstdrop_chart.pdf      : grouped bar chart of WorstDrop per dataset/method.
    - overlap_results.pdf      : Critical-overlap ratio vs. final accuracy (with fit).
    - tradeoff_results.pdf     : updated-parameter ratio vs. WorstDrop scatter.

Data sourcing
-------------
Each data-driven figure first tries to read the corresponding result CSV(s).
If a CSV is absent or does not expose the required columns, the script falls
back to the exact numbers reported in the manuscript (Table I) and prints a
clear ``[fallback]`` notice. This keeps the figures reproducible from live
experiment outputs while remaining runnable on a clean checkout.

Usage
-----
    python tools/make_paper_figures.py --outdir /path/to/manuscript
    python tools/make_paper_figures.py \
        --worstdrop-csv results/stable/results_all_methods_v1.csv \
        --tradeoff-csv  results/stable/results_all_methods_v1.csv \
        --overlap-csv   results/thesis/overlap_vs_accuracy.csv \
        --outdir figures/

CSV column expectations (case-insensitive; first match wins)
    dataset col : {dataset}
    method col  : {method, algorithm, algo}
    worstdrop   : {worstdrop_mean, worstdrop}
    updated     : {updated_param_ratio_mean, updated_param_ratio, r_updated}
    overlap     : {overlap_s_share_crit_ratio, critical_overlap_ratio, s_share_crit_ratio}
    accuracy    : {final_avg_acc_mean, final_avg_accuracy, a_final}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# Headless-safe backend: never opens a window, so it is safe inside long,
# unattended training runs and on remote/CI machines without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional for fallback mode
    pd = None


# --------------------------------------------------------------------------- #
# IEEE-compliant global style                                                  #
# --------------------------------------------------------------------------- #
IEEE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "pdf.fonttype": 42,   # embed TrueType (editable text) rather than Type-3
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Single-column IEEE width is ~3.5 in; use it as the canonical figure width.
COL_WIDTH = 3.5

# Consistent, color-blind-friendly palette and per-method styling.
METHOD_STYLE = {
    "PALL-Original": {"color": "#2ca02c", "marker": "^", "hatch": "//"},
    "PALL-Modified": {"color": "#ff7f0e", "marker": "s", "hatch": ".."},
    "PALL-Adapter":  {"color": "#1f77b4", "marker": "o", "hatch": None},
}
DATASETS = ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
METHODS = ["PALL-Original", "PALL-Modified", "PALL-Adapter"]

# Maps raw experiment method identifiers to display names.
METHOD_ALIASES = {
    "pall_original": "PALL-Original",
    "pall": "PALL-Original",
    "pall_modified": "PALL-Modified",
    "pall_mod": "PALL-Modified",
    "pall_adapter": "PALL-Adapter",
    "adapter": "PALL-Adapter",
}
DATASET_ALIASES = {
    "cifar10": "CIFAR-10",
    "cifar-10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cifar-100": "CIFAR-100",
    "tinyimagenet": "TinyImageNet",
    "tiny-imagenet": "TinyImageNet",
    "tiny_imagenet": "TinyImageNet",
}

# --------------------------------------------------------------------------- #
# Manuscript fallback values (Table I)                                         #
# --------------------------------------------------------------------------- #
PAPER_WORSTDROP = {
    "CIFAR-10":     {"PALL-Original": 0.0160, "PALL-Modified": 0.0008, "PALL-Adapter": 0.0050},
    "CIFAR-100":    {"PALL-Original": 0.0710, "PALL-Modified": 0.0140, "PALL-Adapter": 0.0050},
    "TinyImageNet": {"PALL-Original": 0.0940, "PALL-Modified": 0.0600, "PALL-Adapter": 0.0080},
}
PAPER_UPDATED = {
    "CIFAR-10":     {"PALL-Original": 0.0475, "PALL-Modified": 0.0475, "PALL-Adapter": 0.0061},
    "CIFAR-100":    {"PALL-Original": 0.0763, "PALL-Modified": 0.0763, "PALL-Adapter": 0.0123},
    "TinyImageNet": {"PALL-Original": 0.0960, "PALL-Modified": 0.0960, "PALL-Adapter": 0.0292},
}
# Critical-overlap-ratio vs. final-accuracy operating points (illustrative trend).
PAPER_OVERLAP_POINTS = np.array(
    [(0.05, 0.50), (0.10, 0.53), (0.15, 0.55), (0.20, 0.53), (0.30, 0.45)]
)


# --------------------------------------------------------------------------- #
# CSV helpers                                                                  #
# --------------------------------------------------------------------------- #
def _find_col(df, candidates):
    """Return the first column in ``df`` matching any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_metric_by_method(csv_path, value_cols):
    """Load a {dataset: {method: value}} table from a tidy results CSV.

    Returns ``None`` when the file is missing, pandas is unavailable, or the
    required columns cannot be located -- signalling the caller to fall back.
    """
    if csv_path is None or pd is None:
        return None
    path = Path(csv_path)
    if not path.is_file():
        print(f"[fallback] CSV not found: {path}", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    ds_col = _find_col(df, ["dataset"])
    me_col = _find_col(df, ["method", "algorithm", "algo"])
    va_col = _find_col(df, value_cols)
    if not all([ds_col, me_col, va_col]):
        print(f"[fallback] required columns missing in {path.name} "
              f"(dataset/method/{value_cols}).", file=sys.stderr)
        return None
    table = {}
    for _, row in df.iterrows():
        ds = DATASET_ALIASES.get(str(row[ds_col]).strip().lower())
        me = METHOD_ALIASES.get(str(row[me_col]).strip().lower())
        if ds is None or me is None:
            continue
        try:
            val = float(row[va_col])
        except (TypeError, ValueError):
            continue
        # Keep the worst (largest) value if duplicates exist for a cell.
        table.setdefault(ds, {})
        if me not in table[ds] or val > table[ds][me]:
            table[ds][me] = val
    return table or None


def _resolve_table(csv_path, value_cols, fallback, label):
    table = _load_metric_by_method(csv_path, value_cols)
    if table is None:
        print(f"[fallback] {label}: using manuscript (Table I) values.")
        return fallback, True
    # Backfill any missing cells from the paper values so plots never break.
    merged = {ds: dict(fallback[ds]) for ds in DATASETS}
    for ds, by_method in table.items():
        merged.setdefault(ds, {})
        merged[ds].update(by_method)
    print(f"[ok] {label}: loaded from {Path(csv_path).name}.")
    return merged, False


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)


# --------------------------------------------------------------------------- #
# Figure 1: WorstDrop grouped bar chart                                        #
# --------------------------------------------------------------------------- #
def plot_worstdrop_chart(outdir, csv_path=None):
    table, _ = _resolve_table(
        csv_path, ["worstdrop_mean", "worstdrop"], PAPER_WORSTDROP, "worstdrop_chart"
    )
    x = np.arange(len(DATASETS))
    n = len(METHODS)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    for i, method in enumerate(METHODS):
        vals = [table[ds].get(method, np.nan) for ds in DATASETS]
        offset = (i - (n - 1) / 2) * width
        st = METHOD_STYLE[method]
        bars = ax.bar(
            x + offset, vals, width, label=method, color=st["color"],
            edgecolor="black", linewidth=0.5, hatch=st["hatch"], alpha=0.95,
        )
        ax.bar_label(bars, fmt="%.4f", padding=1.5, fontsize=5, rotation=90)

    ax.set_ylabel(r"WorstDrop $(\downarrow)$")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, max(0.1, ax.get_ylim()[1] * 1.18))
    ax.yaxis.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1, handlelength=1.4)
    _style_axes(ax)
    _save(fig, outdir / "worstdrop_chart.pdf")


# --------------------------------------------------------------------------- #
# Figure 2: Overlap ratio vs. final accuracy                                   #
# --------------------------------------------------------------------------- #
def plot_overlap_results(outdir, csv_path=None):
    points = None
    if csv_path is not None and pd is not None and Path(csv_path).is_file():
        df = pd.read_csv(csv_path)
        ox = _find_col(df, ["overlap_s_share_crit_ratio", "critical_overlap_ratio", "s_share_crit_ratio"])
        ay = _find_col(df, ["final_avg_acc_mean", "final_avg_accuracy", "a_final"])
        if ox and ay:
            sub = df[[ox, ay]].dropna().astype(float)
            points = sub.values
            print(f"[ok] overlap_results: loaded {len(points)} points from {Path(csv_path).name}.")
    if points is None or len(points) < 3:
        print("[fallback] overlap_results: using manuscript illustrative points.")
        points = PAPER_OVERLAP_POINTS

    xs, ys = points[:, 0], points[:, 1]
    # Least-squares quadratic fit to expose the non-linear retention peak.
    coeffs = np.polyfit(xs, ys, deg=2)
    grid = np.linspace(max(0.0, xs.min() - 0.02), xs.max() + 0.02, 200)
    fit = np.polyval(coeffs, grid)
    peak_x = grid[int(np.argmax(fit))]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    ax.plot(grid, fit, color="#d62728", linewidth=1.3, label="Quadratic fit", zorder=2)
    ax.scatter(xs, ys, color=METHOD_STYLE["PALL-Adapter"]["color"], edgecolor="black",
               linewidth=0.5, s=28, zorder=3, label="Operating points")
    ax.axvline(peak_x, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(peak_x, ax.get_ylim()[0], "  peak", fontsize=6, color="0.4",
            va="bottom", ha="left")

    ax.set_xlabel("Critical Shared Overlap Ratio")
    ax.set_ylabel("Final Average Accuracy")
    ax.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", handlelength=1.6)
    _style_axes(ax)
    _save(fig, outdir / "overlap_results.pdf")


# --------------------------------------------------------------------------- #
# Figure 3: Updated-parameter ratio vs. WorstDrop scatter                      #
# --------------------------------------------------------------------------- #
def plot_tradeoff_results(outdir, worstdrop_csv=None, updated_csv=None):
    wd, _ = _resolve_table(
        worstdrop_csv, ["worstdrop_mean", "worstdrop"], PAPER_WORSTDROP, "tradeoff (WorstDrop)"
    )
    rp, _ = _resolve_table(
        updated_csv, ["updated_param_ratio_mean", "updated_param_ratio", "r_updated"],
        PAPER_UPDATED, "tradeoff (R_updated)",
    )

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    for method in METHODS:
        st = METHOD_STYLE[method]
        xs = [rp[ds].get(method, np.nan) for ds in DATASETS]
        ys = [wd[ds].get(method, np.nan) for ds in DATASETS]
        ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=34,
                   edgecolor="black", linewidth=0.5, label=method, zorder=3)

    # Annotate each point with its dataset for readability.
    for method in METHODS:
        for ds in DATASETS:
            x, y = rp[ds].get(method), wd[ds].get(method)
            if x is None or y is None:
                continue
            ax.annotate(ds.replace("CIFAR-", "C").replace("TinyImageNet", "Tiny"),
                        (x, y), textcoords="offset points", xytext=(3, 3),
                        fontsize=5, color="0.35")

    ax.set_xlabel(r"Updated Parameter Ratio $R_{\mathrm{updated}}$ $(\downarrow)$")
    ax.set_ylabel(r"WorstDrop $(\downarrow)$")
    ax.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.2)
    # A point near the origin is best on both axes; mark that corner.
    ax.annotate("better", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6,
                color="0.4", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="<-", color="0.5", lw=0.7),
                xytext=(0.16, 0.16), textcoords="axes fraction")
    _style_axes(ax)
    _save(fig, outdir / "tradeoff_results.pdf")


# --------------------------------------------------------------------------- #
# Figure 4: Conceptual WorstDrop decomposition (visualizes Theorem 1)          #
# --------------------------------------------------------------------------- #
def plot_drop_decomposition(outdir, protect_strength=None):
    # Illustrative per-region contributions to the first-order drop. Frozen
    # weights (H u N) contribute exactly zero; the forget-exclusive region is
    # bounded by epsilon; the critical overlap is the only attenuated term.
    p = 2.0 / 3.0 if protect_strength is None else float(protect_strength)
    f_excl, frozen, crit_unc = 0.15, 0.0, 0.75
    regions = [r"Moves, irrelevant" + "\n" + r"$\mathcal{F}^{\circ}\ (\leq \epsilon)$",
               r"Relevant, frozen" + "\n" + r"$H \cup \mathcal{N}\ (=0)$",
               r"Critical overlap" + "\n" + r"$S_{\mathrm{share\_crit}}$"]
    unc = [f_excl, frozen, crit_unc]
    soft = [f_excl, frozen, (1.0 - p) * crit_unc]

    x = np.arange(len(regions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))
    b1 = ax.bar(x - width / 2, unc, width, label="Unconstrained ($m_i=1$)",
                color="0.55", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, soft, width, label="Soft-masked (ours)",
                color=METHOD_STYLE["PALL-Adapter"]["color"], edgecolor="black", linewidth=0.5)

    # WorstDrop reference lines = sum of contributions for each method.
    wd_unc, wd_soft = sum(unc), sum(soft)
    ax.axhline(wd_unc, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(wd_soft, color=METHOD_STYLE["PALL-Adapter"]["color"],
               linestyle="--", linewidth=0.9, zorder=0)
    ax.text(-0.45, wd_unc + 0.012, "WorstDrop (unconstr.)", fontsize=5.5, color="0.4")
    ax.text(-0.45, wd_soft + 0.012, "WorstDrop (soft)", fontsize=5.5,
            color=METHOD_STYLE["PALL-Adapter"]["color"])

    # Attenuation annotation on the critical-overlap bars.
    ax.annotate("", xy=(2 + width / 2, soft[2] + 0.01), xytext=(2 - width / 2, unc[2] - 0.01),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3))
    ax.text(2.08, (unc[2] + soft[2]) / 2, r"$\times(1-p)$", color="#d62728",
            fontsize=8, fontweight="bold", ha="left", va="center")

    ax.set_ylabel(r"Contribution to WorstDrop")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylim(0, 0.9)
    ax.yaxis.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", handlelength=1.4)
    _style_axes(ax)
    _save(fig, outdir / "drop_decomposition.pdf")


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)  # free the figure to avoid leaking memory across calls
    print(f"[saved] {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Directory to write the PDF figures into.")
    parser.add_argument("--worstdrop-csv", type=Path, default=None)
    parser.add_argument("--updated-csv", type=Path, default=None,
                        help="CSV for R_updated (defaults to --worstdrop-csv).")
    parser.add_argument("--overlap-csv", type=Path, default=None)
    parser.add_argument("--protect-strength", type=float, default=None,
                        help="Override p in the decomposition figure (default 2/3).")
    args = parser.parse_args(argv)

    plt.rcParams.update(IEEE_STYLE)
    args.outdir.mkdir(parents=True, exist_ok=True)
    updated_csv = args.updated_csv or args.worstdrop_csv

    plot_drop_decomposition(args.outdir, protect_strength=args.protect_strength)
    plot_worstdrop_chart(args.outdir, csv_path=args.worstdrop_csv)
    plot_overlap_results(args.outdir, csv_path=args.overlap_csv)
    plot_tradeoff_results(args.outdir, worstdrop_csv=args.worstdrop_csv, updated_csv=updated_csv)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
