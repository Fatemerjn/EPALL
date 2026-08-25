#!/usr/bin/env python3
"""Build compact horizontal thesis figures from the latest aggregate table."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "font-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "results/aggregates/server_thesis_table.csv"
OUTDIR = REPO_ROOT / "thesis/images"

METHOD_LABELS = {
    "pall_original": "PALL-Original",
    "pall_modified": "PALL-Modified",
    "pall_adapter": "PALL-Adapter",
}
METHODS = ["pall_original", "pall_modified", "pall_adapter"]
METHOD_COLORS = {
    "pall_original": "#4c78a8",
    "pall_modified": "#f58518",
    "pall_adapter": "#54a24b",
}
MAIN_ROWS: List[Tuple[str, str, str, str]] = [
    ("CIFAR-10\nfrom scratch", "cifar10", "pall_original", "cifar10_main"),
    ("CIFAR-10\nfrom scratch", "cifar10", "pall_modified", "cifar10_main"),
    ("CIFAR-10\nfrom scratch", "cifar10", "pall_adapter", "cifar10_main"),
    ("CIFAR-100\nfrom scratch", "cifar100", "pall_original", "cifar100_main"),
    ("CIFAR-100\nfrom scratch", "cifar100", "pall_modified", "cifar100_main"),
    ("CIFAR-100\nfrom scratch", "cifar100", "pall_adapter", "cifar100_main"),
    ("TinyImageNet\nfrom scratch", "tinyimagenet", "pall_original", "tiny_e3_original_v1"),
    ("TinyImageNet\nfrom scratch", "tinyimagenet", "pall_modified", "tiny_e3_modified_v1"),
    ("TinyImageNet\nfrom scratch", "tinyimagenet", "pall_adapter", "tiny_e3_adapter_v1"),
]


def load_rows() -> Dict[Tuple[str, str, str], Dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["dataset"], row["method"], row["experiment_tag"]): row
            for row in csv.DictReader(handle)
        }


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    if value == "":
        return float("nan")
    return float(value)


def selected_rows(rows: Dict[Tuple[str, str, str], Dict[str, str]]) -> Iterable[Tuple[str, str, Dict[str, str]]]:
    for group, dataset, method, tag in MAIN_ROWS:
        row = rows[(dataset, method, tag)]
        yield group, method, row


def grouped_values(rows: Dict[Tuple[str, str, str], Dict[str, str]], metric: str) -> Tuple[List[str], np.ndarray]:
    groups = []
    for group, _, _, _ in MAIN_ROWS:
        if group not in groups:
            groups.append(group)
    values = np.full((len(groups), len(METHODS)), np.nan, dtype=float)
    for group, method, row in selected_rows(rows):
        values[groups.index(group), METHODS.index(method)] = as_float(row, metric)
    return groups, values


def save_grouped_bar(rows: Dict[Tuple[str, str, str], Dict[str, str]], metric: str, ylabel: str, name: str) -> None:
    groups, values = grouped_values(rows, metric)
    x = np.arange(len(groups))
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    for idx, method in enumerate(METHODS):
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            values[:, idx],
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.45,
        )
        ax.bar_label(bars, fmt="%.4f", padding=1.2, fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUTDIR / name, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(rows: Dict[Tuple[str, str, str], Dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    markers = {"pall_original": "o", "pall_modified": "s", "pall_adapter": "^"}
    for group, method, row in selected_rows(rows):
        x = as_float(row, "updated_param_ratio_mean")
        y = as_float(row, "WorstDrop_mean")
        ax.scatter(
            x,
            y,
            s=46,
            marker=markers[method],
            color=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.5,
            label=METHOD_LABELS[method],
        )
        label = group.replace("\n", " ").replace("from scratch", "FS").replace("pretrained", "PT")
        label = label.replace("CIFAR-", "C").replace("TinyImageNet", "Tiny")
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6)

    ax.set_xlabel("Updated parameter ratio")
    ax.set_ylabel("WorstDrop")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=False)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUTDIR / "latest_pall_tradeoff_horizontal.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_grouped_bar(rows, "final_avg_acc_mean", "Final average accuracy", "latest_pall_final_accuracy_horizontal.pdf")
    save_grouped_bar(rows, "WorstDrop_mean", "WorstDrop", "latest_pall_worstdrop_horizontal.pdf")
    save_tradeoff(rows)


if __name__ == "__main__":
    main()
