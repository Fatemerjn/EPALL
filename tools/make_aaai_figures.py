#!/usr/bin/env python3
"""Generate compact, single-column AAAI paper figures from real experiment data.

Outputs (vector PDF, 3.4in wide, >=9pt fonts, colorblind-safe Okabe-Ito palette,
hatch patterns as a color-independent secondary encoding so the figures survive
grayscale conversion, as required by the AAAI author kit):

    aaai_worstdrop.pdf : grouped WorstDrop bars for the current scratch-regime
                         CIFAR runs (2 datasets x 3 PALL methods) with std bars.
    aaai_tradeoff.pdf  : updated-parameter ratio vs. WorstDrop scatter for the
                         same nine runs (log-x).
    aaai_mia.pdf       : MIA-AUC before/after forgetting, read from the
                         canonical server_thesis_table.csv aggregates.

The WorstDrop / updated-ratio numbers are the audited values of
thesis/chapters/results.tex tab:main-pall-results (source:
results/aggregates/server_thesis_table.csv; verified by
tools/check_thesis_numbers.py). Keep them in sync with that table.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Okabe-Ito colorblind-safe hues, fixed order: Original, Modified, Adapter.
COLORS = {"PALL-Original": "#0072B2", "PALL-Modified": "#E69F00", "PALL-Adapter": "#009E73"}
HATCH = {"PALL-Original": "", "PALL-Modified": "//", "PALL-Adapter": "xx"}
MARKER = {"PALL-Original": "o", "PALL-Modified": "s", "PALL-Adapter": "^"}
METHODS = ["PALL-Original", "PALL-Modified", "PALL-Adapter"]
DATASETS = ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
CURRENT_WORSTDROP_DATASETS = ["CIFAR-10", "CIFAR-100"]

# tab:main-pall-results (scratch regime; CIFAR: 3 seeds, Tiny: 2 seeds).
WORSTDROP_MEAN = {
    "CIFAR-10": {"PALL-Original": 0.0040, "PALL-Modified": 0.0008, "PALL-Adapter": 0.0107},
    "CIFAR-100": {"PALL-Original": 0.0187, "PALL-Modified": 0.0047, "PALL-Adapter": 0.0427},
    "TinyImageNet": {"PALL-Original": 0.0940, "PALL-Modified": 0.0600, "PALL-Adapter": 0.0080},
}
WORSTDROP_STD = {
    "CIFAR-10": {"PALL-Original": 0.0069, "PALL-Modified": 0.0033, "PALL-Adapter": 0.0101},
    "CIFAR-100": {"PALL-Original": 0.0095, "PALL-Modified": 0.0050, "PALL-Adapter": 0.0333},
    "TinyImageNet": {"PALL-Original": 0.0028, "PALL-Modified": 0.0368, "PALL-Adapter": 0.0000},
}
UPDATED_RATIO = {
    "CIFAR-10": {"PALL-Original": 0.0240, "PALL-Modified": 0.0240, "PALL-Adapter": 0.0030},
    "CIFAR-100": {"PALL-Original": 0.0210, "PALL-Modified": 0.0210, "PALL-Adapter": 0.0055},
    "TinyImageNet": {"PALL-Original": 0.0960, "PALL-Modified": 0.0960, "PALL-Adapter": 0.0292},
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
})


def fig_worstdrop(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(CURRENT_WORSTDROP_DATASETS))
    width = 0.26
    for i, m in enumerate(METHODS):
        means = [WORSTDROP_MEAN[d][m] for d in CURRENT_WORSTDROP_DATASETS]
        stds = [WORSTDROP_STD[d][m] for d in CURRENT_WORSTDROP_DATASETS]
        ax.bar(x + (i - 1) * width, means, width * 0.92, yerr=stds, capsize=2,
               color=COLORS[m], hatch=HATCH[m], edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 0.8, "ecolor": "#444444"},
               label=m.replace("PALL-", "") + (" (ours)" if m != "PALL-Original" else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(CURRENT_WORSTDROP_DATASETS)
    ax.set_ylabel("WorstDrop")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=1, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "aaai_worstdrop.pdf")
    plt.close(fig)


def fig_tradeoff(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in METHODS:
        xs = [UPDATED_RATIO[d][m] for d in DATASETS]
        ys = [WORSTDROP_MEAN[d][m] for d in DATASETS]
        ax.scatter(xs, ys, s=42, color=COLORS[m], marker=MARKER[m],
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=m.replace("PALL-", "") + (" (ours)" if m != "PALL-Original" else ""))
    # Direct dataset labels on the adapter points (the interesting frontier).
    offsets = {"CIFAR-10": (2, 3), "CIFAR-100": (2, 3), "TinyImageNet": (2, -8)}
    for d in DATASETS:
        ax.annotate(d, (UPDATED_RATIO[d]["PALL-Adapter"], WORSTDROP_MEAN[d]["PALL-Adapter"]),
                    textcoords="offset points", xytext=offsets[d],
                    fontsize=9, color="#444444")
    ax.set_xscale("log")
    ax.set_xlabel("Updated parameter ratio (log scale)")
    ax.set_ylabel("WorstDrop")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "aaai_tradeoff.pdf")
    plt.close(fig)


def fig_mia(outdir: Path) -> None:
    df = pd.read_csv(ROOT / "results/aggregates/server_thesis_table.csv")
    dataset_keys = {"CIFAR-10": "cifar10", "CIFAR-100": "cifar100"}
    tags = {
        "CIFAR-10": {"clpu": "cifar10_mia", "pall_modified": "anchor_ablation_v1",
                     "lora": "cifar10_pretrained_mia", "pall_adapter": "cifar10_pretrained_mia"},
        "CIFAR-100": {"clpu": "cifar100_mia", "pall_modified": "anchor_ablation_v1",
                      "lora": "cifar100_pretrained_mia", "pall_adapter": "cifar100_pretrained_mia"},
    }
    labels = {"clpu": "CLPU", "pall_modified": "Modified", "lora": "LoRA", "pall_adapter": "Adapter"}
    order = ["clpu", "pall_modified", "lora", "pall_adapter"]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), sharey=True)
    for ax, ds in zip(axes, ["CIFAR-10", "CIFAR-100"]):
        before, after = [], []
        for meth in order:
            sub = df[
                (df["dataset"] == dataset_keys[ds])
                & (df["experiment_tag"] == tags[ds][meth])
                & (df["method"] == meth)
            ]
            if meth == "pall_modified":
                sub = sub[sub["protect_anchor"] == "old"]
            if len(sub) != 1:
                raise ValueError(
                    f"expected one canonical MIA aggregate for {ds}/{meth}; found {len(sub)}"
                )
            before.append(float(sub.iloc[0]["mia_auc_before_mean"]))
            after.append(float(sub.iloc[0]["mia_auc_after_mean"]))
        x = np.arange(len(order))
        w = 0.38
        ax.bar(x - w / 2, before, w, color="#bbbbbb", edgecolor="white", linewidth=0.5, label="Before")
        ax.bar(x + w / 2, after, w, color="#0072B2", hatch="//", edgecolor="white",
               linewidth=0.5, label="After")
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in order], rotation=45, ha="right")
        ax.set_title(ds)
        ax.set_ylim(0.0, 0.78)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MIA-AUC")
    axes[0].legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.8,
                   handlelength=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "aaai_mia.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "paper/AuthorKit27/Figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_worstdrop(outdir)
    fig_tradeoff(outdir)
    fig_mia(outdir)
    print(f"Wrote aaai_worstdrop.pdf, aaai_tradeoff.pdf, aaai_mia.pdf to {outdir}")


if __name__ == "__main__":
    main()
