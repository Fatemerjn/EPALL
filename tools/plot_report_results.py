#!/usr/bin/env python3
"""
Generate thesis/report plots from the compact report table CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "font-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from make_paper_figures import IEEE_STYLE
except Exception:  # pragma: no cover
    IEEE_STYLE = {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

try:
    from make_thesis_table import CONFIG_GROUP_COLUMNS, dedupe_latest_rows, extract_run_row, values_per_seed
except Exception:  # pragma: no cover
    CONFIG_GROUP_COLUMNS = [
        "experiment_tag",
        "protect_importance",
        "protect_ratio",
        "lambda_protect",
        "adapter_shared_bottleneck",
        "adapter_shared_forget_ratio",
        "adapter_shared_protect_ratio",
        "adapter_shared_forget_lr",
        "adapter_shared_protect_strength",
        "retrain_steps",
        "adapter_train_classifier",
    ]
    dedupe_latest_rows = None
    extract_run_row = None
    values_per_seed = None


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
    parser.add_argument(
        "--paper-figures",
        action="store_true",
        help="Generate the PDF thesis/paper figure set from a group-by-config thesis table.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root run directory for ablation and representative heatmap figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when saving figures.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples per method/config for paper-figure confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed for deterministic bootstrap confidence intervals.",
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


DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tinyimagenet": "TinyImageNet",
}
DATASET_ORDER = ["cifar10", "cifar100", "tinyimagenet"]
REGIME_ORDER = ["from_scratch", "pretrained_frozen", "standard_split"]
REGIME_LABELS = {
    "from_scratch": "From scratch",
    "pretrained_frozen": "Pretrained frozen",
    "standard_split": "Standard split",
}
CHANCE = {"cifar10": 0.5, "cifar100": 0.2, "tinyimagenet": 0.1}
METHOD_LABELS = {
    "clpu": "CLPU",
    "derpp": "DER++",
    "er": "ER",
    "ewc": "EWC",
    "lora": "LoRA",
    "lwf": "LwF",
    "pall_adapter": "PALL-Adapter",
    "pall_modified": "PALL-Modified",
    "pall_original": "PALL-Original",
    "salun": "SalUn",
    "ssd": "SSD",
}
METHOD_ORDER = {
    "pall_original": 0,
    "pall_modified": 1,
    "pall_adapter": 2,
    "lora": 3,
    "ssd": 4,
    "salun": 5,
    "er": 6,
    "derpp": 7,
    "ewc": 8,
    "lwf": 9,
    "clpu": 10,
}
CB_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#999999",
    "#332288",
    "#882255",
    "#44AA99",
]
METHOD_COLORS = {method: CB_PALETTE[idx % len(CB_PALETTE)] for method, idx in METHOD_ORDER.items()}
FORGET_LINE_COLOR = "#D55E00"
BOOTSTRAP_CONFIDENCE = 0.95
AGG_TO_RUN_METRIC = {
    "final_avg_acc_mean": "final_avg_accuracy",
    "WorstDrop_mean": "WorstDrop",
    "Au_mean": "Au",
    "mia_auc_before_mean": "mia_auc_before",
    "mia_auc_after_mean": "mia_auc_after",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


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


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def regime_label(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def classify_regime(tag: Any) -> Optional[str]:
    text = clean_text(tag).lower()
    if not text or text.startswith(("smoke", "test")):
        return None
    if "standard" in text:
        return "standard_split"
    if "pretrained" in text or text == "tiny_pretrained" or text == "adapter_tune_pretrained_v1":
        return "pretrained_frozen"
    if text in {
        "cifar10_main",
        "cifar100_main",
        "tiny_main",
        "cifar10_baselines_v2",
        "cifar100_baselines_v2",
        "cifar10_extra_baselines",
        "cifar100_extra_baselines",
        "cifar10_mia",
        "cifar100_mia",
        "thesis_c10_main_compare_v1",
        "thesis_c100_main_compare_v1",
        "c100_main_compare_v1",
    }:
        return "from_scratch"
    return None


def is_performance_row(row: Any) -> bool:
    tag = clean_text(row.get("experiment_tag")).lower()
    if not classify_regime(tag):
        return False
    excluded_bits = (
        "mia",
        "smoke",
        "candidate",
        "controlled",
        "sanity",
        "debug",
        "ablation",
        "bottleneck",
        "cache",
    )
    return not any(bit in tag for bit in excluded_bits)


def paper_method_label(row: Any, include_config: bool = True) -> str:
    method = clean_text(row.get("method"))
    label = METHOD_LABELS.get(method, method.replace("_", " ").title())
    tag = clean_text(row.get("experiment_tag"))
    if include_config and tag == "adapter_tune_pretrained_v1":
        f_ratio = to_float(row.get("adapter_shared_forget_ratio"))
        p_ratio = to_float(row.get("adapter_shared_protect_ratio"))
        forget_steps = to_float(row.get("adapter_forget_steps"))
        if f_ratio is not None and p_ratio is not None:
            label = f"{label} f={f_ratio:g}, p={p_ratio:g}"
            if forget_steps is not None:
                label = f"{label}, fs={forget_steps:g}"
    return label


def load_thesis_table(path: Path) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for --paper-figures mode")
    df = pd.read_csv(path)
    non_numeric_columns = {"dataset", "method", "experiment_tag", "protect_importance", "adapter_train_classifier"}
    for column in df.columns:
        if column not in non_numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    df["regime"] = df["experiment_tag"].apply(classify_regime)
    df["plot_label"] = df.apply(paper_method_label, axis=1)
    return df


def canonical_group_value(value: Any) -> str:
    text = clean_text(value)
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"nan", "none", "na"}:
        return ""
    if lower in {"true", "false"}:
        return lower.capitalize()
    number = to_float(value)
    if number is not None:
        if abs(number - round(number)) < 1e-10:
            return str(int(round(number)))
        return f"{number:.12g}"
    return text


def aggregate_group_key(row: Any) -> Tuple[str, ...]:
    return (
        canonical_group_value(row.get("dataset")),
        canonical_group_value(row.get("method")),
        *(canonical_group_value(row.get(column)) for column in CONFIG_GROUP_COLUMNS),
    )


def load_run_samples(runs_root: Path) -> Dict[Tuple[str, ...], Dict[str, List[float]]]:
    samples: Dict[Tuple[str, ...], Dict[str, List[float]]] = {}
    if extract_run_row is None or values_per_seed is None:
        print("[WARN] make_thesis_table helpers unavailable; CI bootstrap will fall back to aggregate means.", file=sys.stderr)
        return samples
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for metrics_path in sorted(runs_root.rglob("metrics.json")):
        row = extract_run_row(metrics_path, group_by_config=True)
        if row is None:
            continue
        key = aggregate_group_key(row)
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        if dedupe_latest_rows is not None:
            rows, _ = dedupe_latest_rows(rows, group_by_config=True)
        metric_samples: Dict[str, List[float]] = {}
        for aggregate_metric, run_metric in AGG_TO_RUN_METRIC.items():
            values = values_per_seed(rows, run_metric)
            if values:
                metric_samples[aggregate_metric] = [float(value) for value in values]
        if metric_samples:
            samples[key] = metric_samples
    return samples


def samples_for_row(
    row: Any,
    samples_by_key: Dict[Tuple[str, ...], Dict[str, List[float]]],
    metric: str,
) -> List[float]:
    samples = samples_by_key.get(aggregate_group_key(row), {}).get(metric, [])
    clean_samples = [float(value) for value in samples if to_float(value) is not None]
    if clean_samples:
        return clean_samples
    fallback = to_float(row.get(metric))
    return [float(fallback)] if fallback is not None else []


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None, None, None
    mean_value = float(np.mean(finite))
    if finite.size == 1:
        return mean_value, mean_value, mean_value
    n_bootstrap = max(1000, int(n_bootstrap))
    indices = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    boot_means = finite[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return mean_value, float(low), float(high)


def ci_error(center: float, low: Optional[float], high: Optional[float]) -> np.ndarray:
    if low is None or high is None:
        return np.array([[0.0], [0.0]])
    return np.array([[max(0.0, center - low)], [max(0.0, high - center)]])


def method_color(method: Any) -> str:
    method_text = clean_text(method)
    return METHOD_COLORS.get(method_text, CB_PALETTE[METHOD_ORDER.get(method_text, 99) % len(CB_PALETTE)])


def method_sort_key(method: Any) -> Tuple[int, str]:
    method_text = clean_text(method)
    return METHOD_ORDER.get(method_text, 99), METHOD_LABELS.get(method_text, method_text)


def present_methods(df: Any) -> List[str]:
    methods = {clean_text(method) for method in df["method"].dropna().tolist() if clean_text(method)}
    return sorted(methods, key=method_sort_key)


def add_global_legend(
    fig: plt.Figure,
    methods: Sequence[str],
    *,
    marker: bool = False,
    extra_handles: Optional[Sequence[Any]] = None,
) -> None:
    handles: List[Any] = []
    for method in methods:
        label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        if marker:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=method_color(method),
                    markeredgecolor="black",
                    markersize=5,
                    label=label,
                )
            )
        else:
            handles.append(Patch(facecolor=method_color(method), edgecolor="black", label=label))
    if extra_handles:
        handles.extend(extra_handles)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(6, max(1, len(handles))),
            frameon=False,
        )


def sorted_panel_keys(df: Any) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    for dataset in DATASET_ORDER:
        for regime in REGIME_ORDER:
            if not df[(df["dataset"] == dataset) & (df["regime"] == regime)].empty:
                keys.append((dataset, regime))
    return keys


def sorted_rows(df: Any) -> Any:
    order = df["method"].map(lambda method: METHOD_ORDER.get(str(method), 99))
    return df.assign(_order=order).sort_values(["_order", "plot_label"]).drop(columns=["_order"])


def setup_paper_style() -> None:
    plt.rcParams.update(IEEE_STYLE)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def make_panel_grid(n_panels: int, row_height: float = 3.0, col_width: float = 4.9) -> Tuple[plt.Figure, np.ndarray]:
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, row_height * n_rows), squeeze=False)
    return fig, axes.reshape(-1)


def save_pdf(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0, rect=(0.0, 0.055, 1.0, 0.98))
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def panel_title(dataset: str, regime: str) -> str:
    return f"{dataset_label(dataset)} | {regime_label(regime)}"


def plot_metric_bars(
    df: Any,
    metric: str,
    xlabel: str,
    out_path: Path,
    dpi: int,
    samples_by_key: Dict[Tuple[str, ...], Dict[str, List[float]]],
    n_bootstrap: int,
    rng: np.random.Generator,
    chance_line: bool = False,
) -> Optional[Path]:
    plot_df = df[df.apply(is_performance_row, axis=1)].copy()
    plot_df = plot_df[plot_df[metric].notna()]
    keys = sorted_panel_keys(plot_df)
    if not keys:
        return None
    fig, axes = make_panel_grid(len(keys), row_height=3.1)
    for ax, (dataset, regime) in zip(axes, keys):
        rows = sorted_rows(plot_df[(plot_df["dataset"] == dataset) & (plot_df["regime"] == regime)])
        labels = [textwrap.fill(label, 18) for label in rows["plot_label"]]
        values = rows[metric].astype(float).to_numpy()
        errors: List[List[float]] = [[], []]
        for center, (_, row) in zip(values, rows.iterrows()):
            _, ci_low, ci_high = bootstrap_mean_ci(
                samples_for_row(row, samples_by_key, metric),
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            error = ci_error(float(center), ci_low, ci_high)
            errors[0].append(float(error[0, 0]))
            errors[1].append(float(error[1, 0]))
        colors = [method_color(method) for method in rows["method"]]
        y_pos = np.arange(len(rows))
        ax.barh(
            y_pos,
            values,
            xerr=np.asarray(errors),
            color=colors,
            edgecolor="black",
            linewidth=0.35,
            capsize=2.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{xlabel} (mean with 95% bootstrap CI)")
        ax.set_title(panel_title(dataset, regime))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        if metric in {"final_avg_acc_mean", "Au_mean"}:
            ax.set_xlim(0.0, 1.0)
        if chance_line:
            chance = CHANCE.get(dataset)
            if chance is not None:
                ax.axvline(chance, color="black", linestyle="--", linewidth=1.0)
    for ax in axes[len(keys):]:
        ax.set_axis_off()
    extra_handles = None
    if chance_line:
        extra_handles = [Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="dataset chance line")]
    add_global_legend(fig, present_methods(plot_df), extra_handles=extra_handles)
    return save_pdf(fig, out_path, dpi)


def top_pareto_indices(rows: Any, limit: int = 3) -> set[int]:
    valid = rows[rows["final_avg_acc_mean"].notna() & rows["WorstDrop_mean"].notna()].copy()
    if valid.empty:
        return set()
    pareto: List[Tuple[int, float, float]] = []
    for idx, row in valid.iterrows():
        accuracy = float(row["final_avg_acc_mean"])
        worst_drop = float(row["WorstDrop_mean"])
        dominated = False
        for other_idx, other in valid.iterrows():
            if other_idx == idx:
                continue
            other_accuracy = float(other["final_avg_acc_mean"])
            other_worst_drop = float(other["WorstDrop_mean"])
            if (
                other_accuracy >= accuracy
                and other_worst_drop <= worst_drop
                and (other_accuracy > accuracy or other_worst_drop < worst_drop)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append((idx, accuracy, worst_drop))
    pareto.sort(key=lambda item: (item[1] - item[2], item[1], -item[2]), reverse=True)
    return {idx for idx, _accuracy, _worst_drop in pareto[:limit]}


def plot_tradeoff(df: Any, y_metric: str, ylabel: str, out_path: Path, dpi: int) -> Optional[Path]:
    plot_df = df[df.apply(is_performance_row, axis=1)].copy()
    plot_df = plot_df[plot_df["updated_param_ratio_mean"].notna() & plot_df[y_metric].notna()]
    keys = sorted_panel_keys(plot_df)
    if not keys:
        return None
    fig, axes = make_panel_grid(len(keys), row_height=3.35)
    offsets = [(5, 5), (5, -10), (-52, 5), (-52, -10), (8, 14), (-55, 14)]
    for ax, (dataset, regime) in zip(axes, keys):
        rows = sorted_rows(plot_df[(plot_df["dataset"] == dataset) & (plot_df["regime"] == regime)])
        annotate_indices = top_pareto_indices(rows, limit=3)
        for idx, (_, row) in enumerate(rows.iterrows()):
            method = clean_text(row.get("method"))
            color = method_color(method)
            x_val = float(row["updated_param_ratio_mean"])
            y_val = float(row[y_metric])
            label = paper_method_label(row, include_config=True)
            ax.scatter(x_val, y_val, s=34, color=color, edgecolor="black", linewidth=0.35, zorder=3)
            if row.name in annotate_indices:
                dx, dy = offsets[idx % len(offsets)]
                ax.annotate(
                    textwrap.fill(label, 14),
                    (x_val, y_val),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6.3,
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "0.75", "alpha": 0.9},
                )
        ax.set_xlabel("Updated parameter ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title(dataset, regime))
        ax.grid(True)
        if not rows.empty:
            xmax = max(0.01, float(rows["updated_param_ratio_mean"].max()) * 1.22)
            ax.set_xlim(left=0.0, right=xmax)
    for ax in axes[len(keys):]:
        ax.set_axis_off()
    add_global_legend(fig, present_methods(plot_df), marker=True)
    return save_pdf(fig, out_path, dpi)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def nested_get(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_final_unlearning(metrics: Dict[str, Any]) -> Dict[str, Any]:
    final = nested_get(metrics, "normalized_results", "final", "final_unlearning")
    if isinstance(final, dict) and final:
        return final
    events = metrics.get("unlearning_events")
    if isinstance(events, list) and events and isinstance(events[-1], dict):
        return events[-1]
    return {}


def plot_bottleneck_ablation(
    runs_root: Path,
    out_path: Path,
    dpi: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Optional[Path]:
    rows: List[Dict[str, Optional[float]]] = []
    for config_path in runs_root.rglob("config.json"):
        config = load_json(config_path) or {}
        if config.get("experiment_tag") != "adapter_bottleneck_ablation_v1":
            continue
        metrics = load_json(config_path.with_name("metrics.json")) or {}
        final = extract_final_unlearning(metrics)
        rows.append(
            {
                "adapter_bottleneck": to_float(config.get("adapter_bottleneck")),
                "final_accuracy": to_float(nested_get(metrics, "normalized_results", "final", "final_avg_accuracy")),
                "WorstDrop": to_float(final.get("WorstDrop")),
                "updated_param_ratio": to_float(
                    nested_get(metrics, "normalized_results", "final", "updated_param_ratio")
                    or metrics.get("updated_param_ratio")
                    or nested_get(metrics, "summary", "updated_param_ratio")
                ),
            }
        )
    if pd is None or not rows:
        return None
    ablation_df = pd.DataFrame(rows).dropna(subset=["adapter_bottleneck"]).sort_values("adapter_bottleneck")
    if ablation_df.empty:
        return None
    metrics = [
        ("final_accuracy", "Final average accuracy", "#0072B2"),
        ("WorstDrop", "WorstDrop", "#D55E00"),
        ("updated_param_ratio", "Updated parameter ratio", "#009E73"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.9), sharex=True)
    x_reference: Optional[np.ndarray] = None
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        plot_rows: List[Tuple[float, float, float, float]] = []
        for bottleneck, group in ablation_df.groupby("adapter_bottleneck"):
            values = [float(value) for value in group[metric].dropna().tolist()]
            mean_value, ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, rng=rng)
            if mean_value is None:
                continue
            plot_rows.append(
                (
                    float(bottleneck),
                    mean_value,
                    mean_value if ci_low is None else ci_low,
                    mean_value if ci_high is None else ci_high,
                )
            )
        if not plot_rows:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            continue
        plot_rows.sort(key=lambda item: item[0])
        x_vals = np.asarray([row[0] for row in plot_rows], dtype=float)
        means = np.asarray([row[1] for row in plot_rows], dtype=float)
        lows = np.asarray([row[2] for row in plot_rows], dtype=float)
        highs = np.asarray([row[3] for row in plot_rows], dtype=float)
        x_reference = x_vals
        if len(x_vals) > 2:
            x_dense = np.linspace(float(x_vals.min()), float(x_vals.max()), 240)
            mean_dense = np.interp(x_dense, x_vals, means)
            low_dense = np.interp(x_dense, x_vals, lows)
            high_dense = np.interp(x_dense, x_vals, highs)
            ax.fill_between(x_dense, low_dense, high_dense, color=color, alpha=0.16, linewidth=0)
            ax.plot(x_dense, mean_dense, color=color, linewidth=1.3)
        else:
            ax.fill_between(x_vals, lows, highs, color=color, alpha=0.16, linewidth=0)
            ax.plot(x_vals, means, color=color, linewidth=1.3)
        ax.scatter(x_vals, means, color=color, edgecolor="black", linewidth=0.4, s=26, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("Adapter bottleneck")
    axes[0].set_title("PALL-Adapter Bottleneck Ablation (CIFAR-10) | 95% bootstrap CI bands")
    if x_reference is not None:
        axes[-1].set_xticks(x_reference)
        axes[-1].set_xticklabels([f"{int(x)}" for x in x_reference])
    return save_pdf(fig, out_path, dpi)


def _overlap_critical_ratio(metrics: Dict[str, Any], final: Dict[str, Any]) -> Optional[float]:
    """overlap_shared_critical_ratio == overlap_analysis['critical_ratio'], read
    from the same nested metrics.json locations as
    tools/make_thesis_table.get_overlap_analysis (the actual dump lives at
    unlearning_events[-1].protection.overlap_analysis)."""
    events = metrics.get("unlearning_events")
    raw_last = events[-1] if isinstance(events, list) and events and isinstance(events[-1], dict) else {}
    candidates = (
        nested_get(metrics, "normalized_results", "final", "overlap_analysis"),
        nested_get(metrics, "normalized_results", "final", "protection", "overlap_analysis"),
        final.get("overlap_analysis") if isinstance(final, dict) else None,
        nested_get(final, "protection", "overlap_analysis"),
        raw_last.get("overlap_analysis") if isinstance(raw_last, dict) else None,
        nested_get(raw_last, "protection", "overlap_analysis"),
        nested_get(metrics, "overlap_analysis"),
        nested_get(metrics, "protection", "overlap_analysis"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("critical_ratio") is not None:
            return to_float(candidate.get("critical_ratio"))
    return None


def plot_shared_bottleneck_ablation(
    runs_root: Path,
    out_path: Path,
    dpi: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Optional[Path]:
    rows: List[Dict[str, Optional[float]]] = []
    for config_path in runs_root.rglob("config.json"):
        config = load_json(config_path) or {}
        if config.get("experiment_tag") != "shared_bottleneck_ablation_v1":
            continue
        metrics = load_json(config_path.with_name("metrics.json")) or {}
        final = extract_final_unlearning(metrics)
        rows.append(
            {
                "adapter_shared_bottleneck": to_float(config.get("adapter_shared_bottleneck")),
                "final_accuracy": to_float(nested_get(metrics, "normalized_results", "final", "final_avg_accuracy")),
                "WorstDrop": to_float(final.get("WorstDrop")),
                "updated_param_ratio": to_float(
                    nested_get(metrics, "normalized_results", "final", "updated_param_ratio")
                    or metrics.get("updated_param_ratio")
                    or nested_get(metrics, "summary", "updated_param_ratio")
                ),
                "overlap_shared_critical_ratio": _overlap_critical_ratio(metrics, final),
            }
        )
    if pd is None or not rows:
        return None
    ablation_df = (
        pd.DataFrame(rows)
        .dropna(subset=["adapter_shared_bottleneck"])
        .sort_values("adapter_shared_bottleneck")
    )
    if ablation_df.empty:
        return None
    metrics_spec = [
        ("final_accuracy", "Final average accuracy", "#0072B2"),
        ("WorstDrop", "WorstDrop", "#D55E00"),
        ("updated_param_ratio", "Updated parameter ratio", "#009E73"),
        ("overlap_shared_critical_ratio", "Shared critical-overlap ratio", "#CC79A7"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 9.0), sharex=True)
    x_reference: Optional[np.ndarray] = None
    for ax, (metric, ylabel, color) in zip(axes, metrics_spec):
        plot_rows: List[Tuple[float, float, float, float]] = []
        for bottleneck, group in ablation_df.groupby("adapter_shared_bottleneck"):
            values = [float(value) for value in group[metric].dropna().tolist()]
            mean_value, ci_low, ci_high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, rng=rng)
            if mean_value is None:
                continue
            plot_rows.append(
                (
                    float(bottleneck),
                    mean_value,
                    mean_value if ci_low is None else ci_low,
                    mean_value if ci_high is None else ci_high,
                )
            )
        if not plot_rows:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            continue
        plot_rows.sort(key=lambda item: item[0])
        x_vals = np.asarray([row[0] for row in plot_rows], dtype=float)
        means = np.asarray([row[1] for row in plot_rows], dtype=float)
        lows = np.asarray([row[2] for row in plot_rows], dtype=float)
        highs = np.asarray([row[3] for row in plot_rows], dtype=float)
        x_reference = x_vals
        if len(x_vals) > 2:
            x_dense = np.linspace(float(x_vals.min()), float(x_vals.max()), 240)
            mean_dense = np.interp(x_dense, x_vals, means)
            low_dense = np.interp(x_dense, x_vals, lows)
            high_dense = np.interp(x_dense, x_vals, highs)
            ax.fill_between(x_dense, low_dense, high_dense, color=color, alpha=0.16, linewidth=0)
            ax.plot(x_dense, mean_dense, color=color, linewidth=1.3)
        else:
            ax.fill_between(x_vals, lows, highs, color=color, alpha=0.16, linewidth=0)
            ax.plot(x_vals, means, color=color, linewidth=1.3)
        ax.scatter(x_vals, means, color=color, edgecolor="black", linewidth=0.4, s=26, zorder=3)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("Shared adapter bottleneck")
    axes[0].set_title("PALL-Adapter Shared-Bottleneck Ablation (CIFAR-10) | 95% bootstrap CI bands")
    if x_reference is not None:
        axes[-1].set_xticks(x_reference)
        axes[-1].set_xticklabels([f"{int(x)}" for x in x_reference])
    return save_pdf(fig, out_path, dpi)


def pick_heatmap_run(runs_root: Path) -> Optional[Path]:
    priority = {
        "cifar10_main": 0,
        "cifar10_pretrained": 1,
        "cifar100_main": 2,
        "cifar100_pretrained": 3,
        "tiny_pretrained": 4,
    }
    candidates: List[Tuple[int, str, Path]] = []
    for config_path in runs_root.rglob("config.json"):
        config = load_json(config_path) or {}
        if config.get("method") != "pall_adapter":
            continue
        results_path = config_path.with_name("results.pth")
        if not results_path.exists():
            continue
        tag = clean_text(config.get("experiment_tag"))
        rank = priority.get(tag, 99)
        if rank < 99:
            candidates.append((rank, clean_text(config.get("seed")), results_path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])))[0][2]


def plot_representative_heatmap(runs_root: Path, out_path: Path, dpi: int) -> Optional[Path]:
    results_path = pick_heatmap_run(runs_root)
    if results_path is None:
        return None
    try:
        import torch

        result = torch.load(results_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[WARN] Could not load heatmap run {results_path}: {exc}", file=sys.stderr)
        return None
    stats = result.get("stats", {}) if isinstance(result, dict) else {}
    accuracy = stats.get("accuracy")
    requests = result.get("user_requests_with_active_tasks", [])
    if accuracy is None or not requests:
        return None
    matrix = accuracy.detach().cpu().numpy().T.astype(float)
    normalized = np.zeros_like(matrix, dtype=float)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            normalized[row_idx] = 0.0
            continue
        row_min = float(np.min(finite))
        row_max = float(np.max(finite))
        if row_max - row_min < 1e-12:
            normalized[row_idx] = 0.5
        else:
            normalized[row_idx] = (row - row_min) / (row_max - row_min)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    labels = [f"{typ}{task}" for task, typ, _active in requests]
    forget_cols = [(idx, task) for idx, (task, typ, _active) in enumerate(requests) if typ == "F"]
    config = load_json(results_path.with_name("config.json")) or {}

    fig, ax = plt.subplots(figsize=(max(6.6, 0.55 * len(labels)), max(3.2, 0.32 * normalized.shape[0])))
    image = ax.imshow(normalized, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(
        "Row-normalized Per-task Accuracy Timeline | "
        f"{dataset_label(clean_text(config.get('dataset')))} "
        f"{METHOD_LABELS.get(clean_text(config.get('method')), config.get('method'))}"
    )
    ax.set_xlabel("Request sequence")
    ax.set_ylabel("Task")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(normalized.shape[0]))
    ax.set_yticklabels([f"Task {idx}" for idx in range(normalized.shape[0])])
    for col, task in forget_cols:
        ax.axvline(col, color=FORGET_LINE_COLOR, linestyle="--", linewidth=1.1)
        ax.scatter([col], [-0.55], marker="v", color=FORGET_LINE_COLOR, clip_on=False, s=28)
        ax.text(col, -0.95, f"F{task}", ha="center", va="top", color=FORGET_LINE_COLOR, fontsize=7, clip_on=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized accuracy")
    return save_pdf(fig, out_path, dpi)


def plot_mia(
    df: Any,
    out_path: Path,
    dpi: int,
    samples_by_key: Dict[Tuple[str, ...], Dict[str, List[float]]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Optional[Path]:
    plot_df = df[df["mia_auc_before_mean"].notna() & df["mia_auc_after_mean"].notna()].copy()
    plot_df = plot_df[plot_df["regime"].notna()]
    keys = sorted_panel_keys(plot_df)
    if not keys:
        return None
    fig, axes = make_panel_grid(len(keys), row_height=3.05)
    for ax, (dataset, regime) in zip(axes, keys):
        rows = sorted_rows(plot_df[(plot_df["dataset"] == dataset) & (plot_df["regime"] == regime)])
        labels = [textwrap.fill(paper_method_label(row, include_config=False), 16) for _, row in rows.iterrows()]
        x_pos = np.arange(len(rows))
        width = 0.36
        before = rows["mia_auc_before_mean"].astype(float).to_numpy()
        after = rows["mia_auc_after_mean"].astype(float).to_numpy()
        before_errors: List[List[float]] = [[], []]
        after_errors: List[List[float]] = [[], []]
        for before_center, after_center, (_, row) in zip(before, after, rows.iterrows()):
            _, before_low, before_high = bootstrap_mean_ci(
                samples_for_row(row, samples_by_key, "mia_auc_before_mean"),
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            _, after_low, after_high = bootstrap_mean_ci(
                samples_for_row(row, samples_by_key, "mia_auc_after_mean"),
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            before_error = ci_error(float(before_center), before_low, before_high)
            after_error = ci_error(float(after_center), after_low, after_high)
            before_errors[0].append(float(before_error[0, 0]))
            before_errors[1].append(float(before_error[1, 0]))
            after_errors[0].append(float(after_error[0, 0]))
            after_errors[1].append(float(after_error[1, 0]))
        colors = [method_color(method) for method in rows["method"]]
        ax.bar(
            x_pos - width / 2,
            before,
            width,
            yerr=np.asarray(before_errors),
            color=colors,
            alpha=0.48,
            edgecolor="black",
            linewidth=0.35,
            capsize=2.5,
        )
        ax.bar(
            x_pos + width / 2,
            after,
            width,
            yerr=np.asarray(after_errors),
            color=colors,
            hatch="///",
            edgecolor="black",
            linewidth=0.35,
            capsize=2.5,
        )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("MIA AUC (mean with 95% bootstrap CI)")
        ax.set_title(panel_title(dataset, regime))
        ax.grid(True, axis="y")
    for ax in axes[len(keys):]:
        ax.set_axis_off()
    extra_handles = [
        Patch(facecolor="white", edgecolor="black", alpha=0.48, label="Before"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="After"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="chance=0.5"),
    ]
    add_global_legend(fig, present_methods(plot_df), extra_handles=extra_handles)
    return save_pdf(fig, out_path, dpi)


def describe_best_row(row: Any, metric: str) -> str:
    value = to_float(row.get(metric))
    value_text = "NA" if value is None else f"{value:.4f}"
    return f"{paper_method_label(row)} [{row.get('regime')}] = {value_text}"


def print_best_summary(df: Any) -> None:
    plot_df = df[df.apply(is_performance_row, axis=1)].copy()
    print("[SUMMARY] Best methods by dataset from aggregated CSV")
    for dataset in DATASET_ORDER:
        rows = plot_df[plot_df["dataset"] == dataset].copy()
        if rows.empty:
            continue
        print(f"[SUMMARY] {dataset_label(dataset)}")
        accuracy_rows = rows[rows["final_avg_acc_mean"].notna()]
        if not accuracy_rows.empty:
            row = accuracy_rows.loc[accuracy_rows["final_avg_acc_mean"].astype(float).idxmax()]
            print(f"[SUMMARY]   accuracy: {describe_best_row(row, 'final_avg_acc_mean')}")
        drop_rows = rows[rows["WorstDrop_mean"].notna()]
        if not drop_rows.empty:
            row = drop_rows.loc[drop_rows["WorstDrop_mean"].astype(float).idxmin()]
            print(f"[SUMMARY]   worstdrop: {describe_best_row(row, 'WorstDrop_mean')}")
        au_rows = rows[rows["Au_mean"].notna()].copy()
        if not au_rows.empty:
            chance = CHANCE.get(dataset)
            if chance is not None:
                au_rows["_au_distance"] = (au_rows["Au_mean"].astype(float) - chance).abs()
                row = au_rows.loc[au_rows["_au_distance"].idxmin()]
                print(f"[SUMMARY]   au closest to chance ({chance:g}): {describe_best_row(row, 'Au_mean')}")
            else:
                row = au_rows.loc[au_rows["Au_mean"].astype(float).idxmin()]
                print(f"[SUMMARY]   au: {describe_best_row(row, 'Au_mean')}")


def _iter_unlearning_events(metrics: Dict[str, Any]):
    events = metrics.get("unlearning_events")
    if not isinstance(events, list):
        events = nested_get(metrics, "normalized_results", "unlearning_events")
    return events if isinstance(events, list) else []


def collect_bound_points(runs_root: Path) -> List[Dict[str, Any]]:
    """Gather (predicted_bound, measured_drop, method, dataset) points from every
    per-task bound_check entry (all forget events) written under --eval_bound."""
    points: List[Dict[str, Any]] = []
    for config_path in runs_root.rglob("config.json"):
        config = load_json(config_path) or {}
        metrics = load_json(config_path.with_name("metrics.json")) or {}
        method = clean_text(config.get("method"))
        dataset = clean_text(config.get("dataset"))
        for event in _iter_unlearning_events(metrics):
            bound = event.get("bound_check") if isinstance(event, dict) else None
            if not isinstance(bound, dict):
                continue
            for task_id, entry in (bound.get("per_task") or {}).items():
                predicted = to_float(entry.get("predicted_bound"))
                measured = to_float(entry.get("measured_drop"))
                if predicted is None or measured is None:
                    continue
                points.append({
                    "method": method,
                    "dataset": dataset,
                    "task_id": task_id,
                    "predicted": float(predicted),
                    "measured": float(measured),
                })
    return points


def plot_bound_verification(runs_root: Path, out_path: Path, dpi: int) -> Optional[Path]:
    """Scatter predicted first-order WorstDrop bound (x) vs measured per-task drop
    (y), on log-log axes with the y=x reference line. Points on/under y=x satisfy
    the bound. Colored by method, marker by dataset."""
    points = collect_bound_points(runs_root)
    if not points:
        return None
    floor = 1e-4  # log axes cannot show <=0 (e.g. a task that improved); clamp for display
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    datasets = sorted({p["dataset"] for p in points})
    dmarker = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}
    methods = sorted({p["method"] for p in points})

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    xs = [max(floor, p["predicted"]) for p in points]
    ys = [max(floor, p["measured"]) for p in points]
    lo = min(min(xs), min(ys)) * 0.5
    hi = max(max(xs), max(ys)) * 2.0
    ax.plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0, zorder=1,
            label="y = x (bound tight)")
    for p in points:
        ax.scatter(max(floor, p["predicted"]), max(floor, p["measured"]),
                   color=method_color(p["method"]), marker=dmarker[p["dataset"]],
                   edgecolor="black", linewidth=0.4, s=42, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted first-order bound  " r"$(1-p)E^{\mathrm{crit}}_t + \epsilon C_t$")
    ax.set_ylabel("Measured per-task drop  " r"$A_t^{\mathrm{before}}-A_t^{\mathrm{after}}$")
    n_ok = sum(1 for p in points if p["measured"] <= p["predicted"])
    ax.set_title(f"WorstDrop bound verification | {n_ok}/{len(points)} points satisfy measured $\\leq$ predicted")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    handles = [plt.Line2D([0], [0], color="0.4", linestyle="--", label="y = x")]
    handles += [plt.Line2D([0], [0], marker="o", linestyle="", color=method_color(m),
                           markeredgecolor="black", label=str(m)) for m in methods]
    handles += [plt.Line2D([0], [0], marker=dmarker[d], linestyle="", color="0.5",
                           markeredgecolor="black", label=str(d)) for d in datasets]
    ax.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.9)
    return save_pdf(fig, out_path, dpi)


def generate_paper_figures(
    input_csv: Path,
    runs_root: Path,
    outdir: Path,
    dpi: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> List[Path]:
    setup_paper_style()
    df = load_thesis_table(input_csv)
    samples_by_key = load_run_samples(runs_root)
    rng = np.random.default_rng(bootstrap_seed)
    n_bootstrap = max(1000, int(n_bootstrap))
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    specs = [
        ("final_avg_acc_mean", "Final average accuracy", "final_accuracy_bars_by_dataset_regime.pdf", False),
        ("WorstDrop_mean", "WorstDrop", "worstdrop_bars_by_dataset_regime.pdf", False),
        ("Au_mean", "Au (forgotten-task accuracy)", "au_bars_by_dataset_regime.pdf", True),
    ]
    for metric, xlabel, filename, chance_line in specs:
        path = plot_metric_bars(
            df,
            metric,
            xlabel,
            outdir / filename,
            dpi,
            samples_by_key,
            n_bootstrap,
            rng,
            chance_line=chance_line,
        )
        if path is not None:
            outputs.append(path)
    for y_metric, ylabel, filename in [
        ("WorstDrop_mean", "WorstDrop", "tradeoff_updated_vs_worstdrop_by_dataset_regime.pdf"),
        ("final_avg_acc_mean", "Final average accuracy", "tradeoff_updated_vs_final_accuracy_by_dataset_regime.pdf"),
    ]:
        path = plot_tradeoff(df, y_metric, ylabel, outdir / filename, dpi)
        if path is not None:
            outputs.append(path)
    for builder in [
        lambda: plot_bottleneck_ablation(
            runs_root,
            outdir / "adapter_bottleneck_ablation.pdf",
            dpi,
            n_bootstrap,
            rng,
        ),
        lambda: plot_shared_bottleneck_ablation(
            runs_root,
            outdir / "shared_bottleneck_ablation.pdf",
            dpi,
            n_bootstrap,
            rng,
        ),
        lambda: plot_representative_heatmap(runs_root, outdir / "representative_pall_adapter_accuracy_heatmap.pdf", dpi),
        lambda: plot_mia(df, outdir / "mia_before_after_by_dataset_regime.pdf", dpi, samples_by_key, n_bootstrap, rng),
        lambda: plot_bound_verification(runs_root, outdir / "bound_verification.pdf", dpi),
    ]:
        path = builder()
        if path is not None:
            outputs.append(path)
    print_best_summary(df)
    return outputs


def main() -> int:
    args = parse_args()
    if args.paper_figures:
        outdir = args.outdir
        if outdir == Path("results/thesis/report_plots"):
            outdir = Path("results/thesis/plots_v2")
        outputs = generate_paper_figures(
            args.input,
            args.runs_root,
            outdir,
            args.dpi,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        for output_path in outputs:
            print(f"[INFO] Wrote plot: {output_path}")
        print(f"[INFO] Figures generated: {len(outputs)}")
        return 0

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
