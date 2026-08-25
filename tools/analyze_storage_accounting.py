#!/usr/bin/env python3
"""Summarize matched resident tensor-state accounting for PALL-Modified/CLPU."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


TAG = "storage_accounting_v1"
DATASETS = ("cifar10", "cifar100")
METHODS = ("pall_modified", "clpu")
BYTE_FIELDS = (
    "model_parameter_bytes",
    "side_network_parameter_bytes",
    "subnet_mask_bytes",
    "backup_index_bytes",
    "backup_value_bytes",
    "replay_image_bytes",
    "replay_label_bytes",
    "replay_logit_bytes",
    "accounted_total_bytes",
)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def select_runs(root, tag):
    selected = {}
    for config_path in root.glob("**/config.json"):
        config = load_json(config_path)
        if not isinstance(config, dict):
            continue
        dataset = config.get("dataset")
        method = config.get("method")
        if (
            config.get("experiment_tag") != tag
            or dataset not in DATASETS
            or method not in METHODS
            or int(config.get("seed", -1)) != 0
        ):
            continue
        metrics = load_json(config_path.with_name("metrics.json"))
        history = (metrics or {}).get("storage_history")
        if not isinstance(history, list) or not history:
            continue
        key = (dataset, method)
        candidate = (config_path.parent.name, config_path, config, metrics)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS}
    if set(selected) != expected:
        missing = sorted(expected.difference(selected))
        raise ValueError(f"missing completed storage runs for tag={tag!r}: {missing}")
    return selected


def mb(value):
    return float(value or 0) / (1024.0 ** 2)


def trace_rows(selected):
    rows = []
    for (dataset, method), (_, config_path, _config, metrics) in sorted(selected.items()):
        history = metrics["storage_history"]
        for event in history:
            storage = event.get("storage") or {}
            row = {
                "dataset": dataset,
                "method": method,
                "seed": 0,
                "request_id": int(event["request_id"]),
                "request_type": event["request_type"],
                "task_id": int(event["task_id"]),
                "active_tasks": len(event.get("active_tasks") or []),
                "active_side_networks": int(storage.get("active_side_networks", 0)),
                "source_run": str(config_path.parent),
            }
            for field in BYTE_FIELDS:
                row[field] = int(storage.get(field, 0) or 0)
            row["base_model_mb"] = mb(row["model_parameter_bytes"])
            row["task_network_mb"] = mb(row["side_network_parameter_bytes"])
            row["mask_mb"] = mb(row["subnet_mask_bytes"])
            row["backup_mb"] = mb(row["backup_index_bytes"] + row["backup_value_bytes"])
            row["replay_mb"] = mb(
                row["replay_image_bytes"] + row["replay_label_bytes"] + row["replay_logit_bytes"]
            )
            row["accounted_total_mb"] = mb(row["accounted_total_bytes"])
            rows.append(row)
    return rows


def linear_slope(points):
    if len(points) < 2:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    x_bar, y_bar = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom


def summarize(rows):
    output = []
    for dataset in DATASETS:
        for method in METHODS:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            max_active = max(row["active_tasks"] for row in group)
            at_max = [row for row in group if row["active_tasks"] == max_active][-1]
            peak = max(row["accounted_total_mb"] for row in group)
            training_points = [
                (row["active_tasks"], row["accounted_total_mb"])
                for row in group if row["request_type"] == "T"
            ]
            output.append({
                "dataset": dataset,
                "method": method,
                "seed": 0,
                "max_active_tasks": max_active,
                "base_model_mb_at_max": at_max["base_model_mb"],
                "task_network_mb_at_max": at_max["task_network_mb"],
                "mask_mb_at_max": at_max["mask_mb"],
                "backup_mb_at_max": at_max["backup_mb"],
                "replay_mb_at_max": at_max["replay_mb"],
                "accounted_total_mb_at_max": at_max["accounted_total_mb"],
                "peak_accounted_total_mb": peak,
                "training_growth_mb_per_active_task": linear_slope(training_points),
                "source_run": at_max["source_run"],
            })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "method"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "NA" if value is None else f"{value:.2f}"


def write_markdown(path, summary, tag):
    lines = [
        "# Matched Resident Tensor-State Accounting", "",
        f"Strict tag: `{tag}`; schedule seed 0. Values are MiB (2^20 bytes).", "",
        "The accounting includes resident model parameters, CLPU side networks, subnet masks, sparse backup indices/values, replay images/labels, and stored logits. It excludes Python/container overhead, serialized-file compression, optimizer state, activations, and allocator workspace.", "",
        "| Dataset | Method | Max active | Base model | Task networks | Masks | Sparse backups | Replay+logits | Total at max active | Peak total | Growth / active task |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['max_active_tasks']} | "
            f"{fmt(row['base_model_mb_at_max'])} | {fmt(row['task_network_mb_at_max'])} | "
            f"{fmt(row['mask_mb_at_max'])} | {fmt(row['backup_mb_at_max'])} | "
            f"{fmt(row['replay_mb_at_max'])} | {fmt(row['accounted_total_mb_at_max'])} | "
            f"{fmt(row['peak_accounted_total_mb'])} | {fmt(row['training_growth_mb_per_active_task'])} |"
        )
    lines.extend(("", "This table is an implementation-state comparison, not a claim about disk erasure, peak GPU memory, or end-to-end runtime."))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/aggregates/storage_accounting"))
    args = parser.parse_args()
    try:
        selected = select_runs(args.root, args.tag)
        traces = trace_rows(selected)
        summary = summarize(traces)
    except ValueError as exc:
        raise SystemExit(str(exc))
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_trace.csv"), traces)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), summary, args.tag)
    print(f"Selected {len(selected)} runs; wrote {len(traces)} request-level storage rows")


if __name__ == "__main__":
    main()
