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
REGIME_ORDER = ["from_scratch", "pretrained", "standard"]
REGIME_LABELS = {
    "from_scratch": "From scratch",
    "pretrained": "Frozen ImageNet backbone",
    "standard": "Standard split",
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
        return "standard"
    if "pretrained" in text or text == "tiny_pretrained" or text == "adapter_tune_pretrained_v1":
        return "pretrained"
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
        if f_ratio is not None and p_ratio is not None:
            label = f"{label} f={f_ratio:g}, p={p_ratio:g}"
    return label


def load_thesis_table(path: Path) -> Any:
    if pd is None:
        raise RuntimeError("pandas is required for --paper-figures mode")
    df = pd.read_csv(path)
    for column in df.columns:
        if column not in {"dataset", "method", "experiment_tag", "adapter_train_classifier"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)
    df["regime"] = df["experiment_tag"].apply(classify_regime)
    df["plot_label"] = df.apply(paper_method_label, axis=1)
    return df


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
    fig.tight_layout(pad=1.0)
    fig.savefig(path, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def panel_title(dataset: str, regime: str) -> str:
    return f"{dataset_label(dataset)} | {regime_label(regime)}"


def plot_metric_bars(
    df: Any,
    metric: str,
    std_metric: str,
    xlabel: str,
    out_path: Path,
    dpi: int,
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
        errors = rows[std_metric].fillna(0.0).astype(float).to_numpy() if std_metric in rows else None
        colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(rows))]
        y_pos = np.arange(len(rows))
        ax.barh(y_pos, values, xerr=errors, color=colors, edgecolor="black", linewidth=0.35, capsize=2.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title(dataset, regime))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        if metric in {"final_avg_acc_mean", "Au_mean"}:
            ax.set_xlim(0.0, 1.0)
        if chance_line:
            chance = CHANCE.get(dataset)
            if chance is not None:
                ax.axvline(chance, color="black", linestyle="--", linewidth=1.0, label=f"chance={chance:g}")
                ax.legend(loc="lower right")
    for ax in axes[len(keys):]:
        ax.set_axis_off()
    return save_pdf(fig, out_path, dpi)


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
        for idx, (_, row) in enumerate(rows.iterrows()):
            method = clean_text(row.get("method"))
            color = CB_PALETTE[METHOD_ORDER.get(method, idx) % len(CB_PALETTE)]
            x_val = float(row["updated_param_ratio_mean"])
            y_val = float(row[y_metric])
            label = paper_method_label(row, include_config=True)
            ax.scatter(x_val, y_val, s=34, color=color, edgecolor="black", linewidth=0.35, zorder=3)
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


def plot_bottleneck_ablation(runs_root: Path, out_path: Path, dpi: int) -> Optional[Path]:
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
    grouped = ablation_df.groupby("adapter_bottleneck").agg(["mean", "std"])
    x_vals = grouped.index.to_numpy(dtype=float)
    metrics = [
        ("final_accuracy", "Final average accuracy"),
        ("WorstDrop", "WorstDrop"),
        ("updated_param_ratio", "Updated parameter ratio"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.9), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        means = grouped[(metric, "mean")].astype(float).to_numpy()
        stds = grouped[(metric, "std")].fillna(0.0).astype(float).to_numpy()
        ax.errorbar(x_vals, means, yerr=stds, marker="o", color="#0072B2", capsize=3)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[-1].set_xlabel("Adapter bottleneck")
    axes[0].set_title("PALL-Adapter Bottleneck Ablation (CIFAR-10)")
    axes[-1].set_xticks(x_vals)
    axes[-1].set_xticklabels([f"{int(x)}" for x in x_vals])
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
    matrix = accuracy.detach().cpu().numpy().T
    labels = [f"{typ}{task}" for task, typ, _active in requests]
    forget_cols = [(idx, task) for idx, (task, typ, _active) in enumerate(requests) if typ == "F"]
    config = load_json(results_path.with_name("config.json")) or {}

    fig, ax = plt.subplots(figsize=(max(6.6, 0.55 * len(labels)), max(3.2, 0.32 * matrix.shape[0])))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(
        "Per-task Accuracy Timeline | "
        f"{dataset_label(clean_text(config.get('dataset')))} "
        f"{METHOD_LABELS.get(clean_text(config.get('method')), config.get('method'))}"
    )
    ax.set_xlabel("Request sequence")
    ax.set_ylabel("Task")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([f"Task {idx}" for idx in range(matrix.shape[0])])
    for col, task in forget_cols:
        ax.axvline(col, color="#D55E00", linestyle="--", linewidth=1.1)
        ax.scatter([col], [-0.55], marker="v", color="#D55E00", clip_on=False, s=28)
        ax.text(col, -0.95, f"F{task}", ha="center", va="top", color="#D55E00", fontsize=7, clip_on=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy")
    return save_pdf(fig, out_path, dpi)


def plot_mia(df: Any, out_path: Path, dpi: int) -> Optional[Path]:
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
        ax.bar(x_pos - width / 2, before, width, label="Before", color="#0072B2", edgecolor="black", linewidth=0.35)
        ax.bar(x_pos + width / 2, after, width, label="After", color="#D55E00", edgecolor="black", linewidth=0.35)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="chance=0.5")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("MIA AUC")
        ax.set_title(panel_title(dataset, regime))
        ax.legend(loc="best")
        ax.grid(True, axis="y")
    for ax in axes[len(keys):]:
        ax.set_axis_off()
    return save_pdf(fig, out_path, dpi)


def generate_paper_figures(input_csv: Path, runs_root: Path, outdir: Path, dpi: int) -> List[Path]:
    setup_paper_style()
    df = load_thesis_table(input_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    specs = [
        ("final_avg_acc_mean", "final_avg_acc_std", "Final average accuracy", "final_accuracy_bars_by_dataset_regime.pdf", False),
        ("WorstDrop_mean", "WorstDrop_std", "WorstDrop", "worstdrop_bars_by_dataset_regime.pdf", False),
        ("Au_mean", "Au_std", "Au (forgotten-task accuracy)", "au_bars_by_dataset_regime.pdf", True),
    ]
    for metric, std_metric, xlabel, filename, chance_line in specs:
        path = plot_metric_bars(df, metric, std_metric, xlabel, outdir / filename, dpi, chance_line=chance_line)
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
        lambda: plot_bottleneck_ablation(runs_root, outdir / "adapter_bottleneck_ablation.pdf", dpi),
        lambda: plot_representative_heatmap(runs_root, outdir / "representative_pall_adapter_accuracy_heatmap.pdf", dpi),
        lambda: plot_mia(df, outdir / "mia_before_after_by_dataset_regime.pdf", dpi),
    ]:
        path = builder()
        if path is not None:
            outputs.append(path)
    return outputs


def main() -> int:
    args = parse_args()
    if args.paper_figures:
        outputs = generate_paper_figures(args.input, args.runs_root, args.outdir, args.dpi)
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
