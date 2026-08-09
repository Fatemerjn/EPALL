#!/usr/bin/env python3
"""Generate compact, single-column AAAI paper figures from canonical aggregates.

Outputs (vector PDF, 3.4in wide, >=9pt fonts, colorblind-safe Okabe-Ito palette,
hatch patterns as a color-independent secondary encoding so the figures survive
grayscale conversion, as required by the AAAI author kit):

    aaai_worstdrop.pdf : grouped WorstDrop bars for the current scratch-regime
                         CIFAR runs (2 datasets x 3 PALL methods) with std bars.
    aaai_tradeoff.pdf  : updated-parameter ratio vs. WorstDrop scatter for the
                         same nine runs (log-x).
    aaai_mia.pdf       : MIA-AUC before/after forgetting, read from the
                         canonical server_thesis_table.csv aggregates.

WorstDrop and updated-ratio values are selected at runtime from
``results/aggregates/server_thesis_table.csv`` using strict dataset/tag/method
keys; no paper metric is hard-coded here.
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
COLORS = {"PALL-Original": "#0072B2", "EPALL": "#E69F00", "PALL-Adapter": "#009E73"}
HATCH = {"PALL-Original": "", "EPALL": "//", "PALL-Adapter": "xx"}
MARKER = {"PALL-Original": "o", "EPALL": "s", "PALL-Adapter": "^"}
METHODS = ["PALL-Original", "EPALL", "PALL-Adapter"]
DATASETS = ["CIFAR-10", "CIFAR-100"]
CURRENT_WORSTDROP_DATASETS = ["CIFAR-10", "CIFAR-100"]

_DATASET_KEYS = {"CIFAR-10": ("cifar10", "cifar10_main"), "CIFAR-100": ("cifar100", "cifar100_main")}
_METHOD_KEYS = {"PALL-Original": "pall_original", "EPALL": "pall_modified", "PALL-Adapter": "pall_adapter"}


def load_main_pall_metrics():
    df = pd.read_csv(ROOT / "results/aggregates/server_thesis_table.csv")
    means, stds, ratios = {}, {}, {}
    for dataset_label, (dataset, tag) in _DATASET_KEYS.items():
        means[dataset_label], stds[dataset_label], ratios[dataset_label] = {}, {}, {}
        for method_label, method in _METHOD_KEYS.items():
            rows = df[
                (df["dataset"] == dataset)
                & (df["experiment_tag"] == tag)
                & (df["method"] == method)
                & (df["protect_importance"].fillna("gradient") == "gradient")
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"expected one canonical row for {(dataset, tag, method)}, found {len(rows)}"
                )
            row = rows.iloc[0]
            means[dataset_label][method_label] = float(row["WorstDrop_mean"])
            stds[dataset_label][method_label] = float(row["WorstDrop_std"])
            ratios[dataset_label][method_label] = float(row["updated_param_ratio_mean"])
    return means, stds, ratios


WORSTDROP_MEAN, WORSTDROP_STD, UPDATED_RATIO = load_main_pall_metrics()

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
    "svg.fonttype": "none",
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
    fig.savefig(outdir / "aaai_worstdrop.svg")
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
    offsets = {"CIFAR-10": (2, 3), "CIFAR-100": (2, 3)}
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
    fig.savefig(outdir / "aaai_tradeoff.svg")
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
    fig.savefig(outdir / "aaai_mia.svg")
    plt.close(fig)



def fig_storage(outdir: Path) -> None:
    """Resident-state accounting: EPALL vs CLPU (total + per-active-task growth)."""
    df = pd.read_csv(ROOT / "results/aggregates/storage_accounting_summary.csv")
    ds_labels = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    methods = [("pall_modified", "EPALL", "#E69F00", "//"), ("clpu", "CLPU", "#0072B2", "")]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.3))
    panels = [
        ("accounted_total_mb_at_max", "Total (MiB)"),
        ("training_growth_mb_per_active_task", "MiB / active task"),
    ]
    x = np.arange(len(ds_labels))
    width = 0.34
    for ax, (col, ylabel) in zip(axes, panels):
        for i, (key, label, color, hatch) in enumerate(methods):
            vals = []
            for ds in ds_labels:
                rows = df[(df["dataset"] == ds) & (df["method"] == key)]
                if len(rows) != 1:
                    raise ValueError(f"expected one storage row for {(ds, key)}, found {len(rows)}")
                vals.append(float(rows.iloc[0][col]))
            ax.bar(x + (i - 0.5) * width, vals, width * 0.92, color=color, hatch=hatch,
                   edgecolor="white", linewidth=0.5, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([ds_labels[d] for d in ds_labels], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8, handlelength=1.2,
                   handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(outdir / "aaai_storage.pdf")
    fig.savefig(outdir / "aaai_storage.svg")
    plt.close(fig)


def fig_overlap_response(outdir: Path) -> None:
    """Plot the strict, de-duplicated controlled-overlap WorstDrop response."""
    import statistics

    try:
        from make_thesis_table import (
            dedupe_latest_rows,
            extract_run_row,
            group_key,
            load_json,
            seed_key,
        )
    except ImportError:  # pragma: no cover - supports package-style imports
        from tools.make_thesis_table import (
            dedupe_latest_rows,
            extract_run_row,
            group_key,
            load_json,
            seed_key,
        )

    levels = ("very_low", "low", "medium", "high", "very_high")
    level_ticks = ("VL", "L", "M", "H", "VH")
    datasets = ("cifar10", "cifar100")
    dataset_titles = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
    methods = (
        ("pall_original", "PALL-Original", "#0072B2", "o", "-"),
        ("pall_modified", "EPALL", "#E69F00", "s", "-"),
        ("salun", "SalUn", "#CC79A7", "^", "--"),
        ("clpu", "CLPU", "#009E73", "D", ":"),
    )
    method_keys = {method[0] for method in methods}
    trace_method_keys = method_keys | {"lora", "pall_adapter"}
    exact_tags = {level: f"overlap_curve_v1_{level}" for level in levels}
    tag_to_level = {tag: level for level, tag in exact_tags.items()}
    expected_seeds = {"0", "1", "2", "3", "4"}

    # A readable metrics.json is not sufficient evidence of completion: an
    # interrupted retry can write the file before the first forgetting event.
    # Keep only rows with a finite signed WorstDrop, then select the latest
    # completed full-config group+seed rather than treating retries as samples.
    candidates = []
    for metrics_path in sorted((ROOT / "runs").rglob("metrics.json")):
        row = extract_run_row(
            metrics_path,
            group_by_config=True,
            include_tags=set(tag_to_level),
        )
        if row is None:
            continue
        if row["dataset"] not in datasets or row["method"] not in trace_method_keys:
            continue

        config = load_json(metrics_path.with_name("config.json"))
        if config is None:
            raise ValueError(f"controlled-overlap run has no readable config: {metrics_path.parent}")
        if str(config.get("dataset")) != row["dataset"] or str(config.get("method")) != row["method"]:
            raise ValueError(
                "controlled-overlap config/metrics identity mismatch: "
                f"{metrics_path.parent}"
            )
        worst_drop = row.get("WorstDrop")
        if worst_drop is None or not np.isfinite(float(worst_drop)):
            continue
        row["WorstDrop"] = float(worst_drop)
        candidates.append(row)

    selected, retries_removed = dedupe_latest_rows(candidates, group_by_config=True)
    cells = {}
    for row in selected:
        level = tag_to_level[row["experiment_tag"]]
        cells.setdefault((row["dataset"], row["method"], level), []).append(row)

    expected_cells = {
        (dataset, method, level)
        for dataset in datasets
        for method in trace_method_keys
        for level in levels
    }
    unexpected_cells = set(cells) - expected_cells
    if unexpected_cells:
        raise ValueError(f"unexpected controlled-overlap cells: {sorted(unexpected_cells)}")

    trace_rows = []
    for dataset in datasets:
        for method in sorted(trace_method_keys):
            for level in levels:
                cell = (dataset, method, level)
                rows = cells.get(cell, [])
                seeds = [seed_key(row) for row in rows]
                if len(rows) != len(expected_seeds) or set(seeds) != expected_seeds:
                    raise ValueError(
                        f"incomplete controlled-overlap cell {cell}: "
                        f"expected seeds {sorted(expected_seeds)}, found {sorted(seeds)}"
                    )
                config_keys = {group_key(row, group_by_config=True) for row in rows}
                if len(config_keys) != 1:
                    raise ValueError(
                        f"incompatible full configs in controlled-overlap cell {cell}: "
                        f"found {len(config_keys)}"
                    )
                for row in sorted(rows, key=seed_key):
                    run_path = Path(row["run_path"])
                    try:
                        run_path = run_path.relative_to(ROOT)
                    except ValueError:
                        pass
                    trace_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "level": level,
                            "seed": seed_key(row),
                            "selected_run_path": run_path.as_posix(),
                            "WorstDrop": row["WorstDrop"],
                        }
                    )

    if len(trace_rows) != len(expected_cells) * len(expected_seeds):
        raise ValueError(
            f"expected {len(expected_cells) * len(expected_seeds)} selected runs, "
            f"found {len(trace_rows)}"
        )

    trace_path = ROOT / "paper/AuthorKit27/generated/overlap_response_trace.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False, float_format="%.17g")

    point_means = {}
    for dataset, method, level in expected_cells:
        values = [
            float(row["WorstDrop"])
            for row in trace_rows
            if (row["dataset"], row["method"], row["level"])
            == (dataset, method, level)
        ]
        point_means[(dataset, method, level)] = statistics.fmean(values)

    figure_rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "Times", "DejaVu Serif"],
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.8,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 9.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(figure_rc):
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), sharey=True)
        x = np.arange(len(levels))
        for ax, dataset in zip(axes, datasets):
            for method, label, color, marker, linestyle in methods:
                y = [point_means[(dataset, method, level)] for level in levels]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.35,
                    marker=marker,
                    markersize=4.8,
                    markeredgewidth=0.55,
                    label=label,
                    zorder=3,
                )
            ax.set_xticks(x, level_ticks)
            ax.set_title(dataset_titles[dataset], pad=3.0)
            ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.45)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", pad=2.0)
            ax.margins(x=0.06, y=0.10)

        fig.supylabel("WorstDrop ↓", x=0.015, fontsize=9.8)
        fig.supxlabel("Request-position grade", x=0.54, y=0.205, fontsize=9.8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.54, 0.012),
            ncol=2,
            frameon=False,
            columnspacing=1.25,
            handlelength=2.2,
            handletextpad=0.55,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.18, right=0.985, top=0.87, bottom=0.36, wspace=0.16)
        fig.savefig(outdir / "aaai_overlap_response.pdf")
        fig.savefig(outdir / "aaai_overlap_response.png", dpi=300)
        fig.savefig(outdir / "aaai_overlap_response.svg")
        plt.close(fig)

    print(
        f"[INFO] overlap response: selected {len(trace_rows)} completed runs; "
        f"removed {retries_removed} duplicate retries; trace={trace_path}"
    )

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(ROOT / "paper/AuthorKit27/Figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_worstdrop(outdir)
    fig_tradeoff(outdir)
    fig_mia(outdir)
    fig_storage(outdir)
    fig_overlap_response(outdir)
    print(f"Wrote aaai_worstdrop.pdf, aaai_tradeoff.pdf, aaai_mia.pdf, aaai_storage.pdf to {outdir}")


if __name__ == "__main__":
    main()
