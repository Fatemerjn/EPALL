#!/usr/bin/env python3
"""Summarize the matched PALL-Adapter component-control experiment.

The script reads completed run artifacts directly, selects the latest run for
each dataset/seed/component mode, and writes both per-run trace rows and an
across-seed summary.  It never mixes experiment tags or dataset protocols.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


TAG = "adapter_components_pretrained_v1"
MODES = (
    "reset_only",
    "reset_repair",
    "uniform_unprotected",
    "mask_no_ascent",
    "full",
)
STAGES = (
    "after_target_reset",
    "after_shared_update",
    "after_classifier_ascent",
    "after_retained_repair",
)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def accuracy_at(stage_eval, task_id):
    if not isinstance(stage_eval, dict):
        return None
    accuracy = stage_eval.get("accuracy")
    if isinstance(accuracy, list) and 0 <= task_id < len(accuracy):
        return as_float(accuracy[task_id])
    if isinstance(accuracy, dict):
        return as_float(accuracy.get(str(task_id), accuracy.get(task_id)))
    return None


def stage_metrics(event, stage):
    stage_eval = (event.get("stage_evals") or {}).get(stage)
    if not isinstance(stage_eval, dict):
        return None
    task_id = int(event["task_id"])
    remaining = [int(task) for task in event.get("remaining_tasks", [])]
    before = event.get("per_task_acc_before") or {}
    forgotten_acc = accuracy_at(stage_eval, task_id)
    retained_acc = [accuracy_at(stage_eval, task) for task in remaining]
    retained_acc = [value for value in retained_acc if value is not None]
    drops = []
    for task in remaining:
        before_value = as_float(before.get(str(task), before.get(task)))
        after_value = accuracy_at(stage_eval, task)
        if before_value is not None and after_value is not None:
            drops.append(before_value - after_value)
    return {
        "au": forgotten_acc,
        "retained_avg": statistics.fmean(retained_acc) if retained_acc else None,
        "worst_drop": max(drops) if drops else None,
    }


def mean_present(values):
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def extract_row(config_path, config, metrics):
    summary = metrics.get("summary") or {}
    normalized = (metrics.get("normalized_results") or {}).get("final") or {}
    mode = config.get("adapter_component_mode")
    chance = 1.0 / float(config["class_per_task"])
    au = as_float(normalized.get("Au", summary.get("Au")))
    row = {
        "dataset": config.get("dataset"),
        "seed": int(config.get("seed")),
        "mode": mode,
        "A_final": as_float(normalized.get("final_avg_accuracy", summary.get("final_avg_accuracy"))),
        "F_avg": as_float(normalized.get("average_forgetting", summary.get("average_forgetting"))),
        "WorstDrop": as_float(normalized.get("WorstDrop", summary.get("WorstDrop"))),
        "Au": au,
        "chance": chance,
        "Au_distance_to_chance": abs(au - chance) if au is not None else None,
        "T_f": as_float(normalized.get("t_forget_total", summary.get("t_forget_total"))),
        "source_run": str(config_path.parent),
    }
    events = metrics.get("unlearning_events") or []
    for stage in STAGES:
        per_event = [stage_metrics(event, stage) for event in events]
        per_event = [entry for entry in per_event if entry is not None]
        for metric in ("au", "retained_avg", "worst_drop"):
            row[f"{stage}_{metric}"] = mean_present([entry[metric] for entry in per_event])
    return row


def select_rows(root, tag):
    selected = {}
    for config_path in root.glob("**/config.json"):
        config = load_json(config_path)
        if not isinstance(config, dict):
            continue
        if config.get("method") != "pall_adapter" or config.get("experiment_tag") != tag:
            continue
        mode = config.get("adapter_component_mode")
        if mode not in MODES:
            continue
        metrics_path = config_path.with_name("metrics.json")
        metrics = load_json(metrics_path)
        if (
            not isinstance(metrics, dict)
            or (metrics.get("summary") or {}).get("final_avg_accuracy") is None
        ):
            continue
        key = (config.get("dataset"), int(config.get("seed")), mode)
        candidate = (config_path.parent.name, config_path, config, metrics)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    return [
        extract_row(config_path, config, metrics)
        for _, config_path, config, metrics in sorted(selected.values(), key=lambda item: (
            item[2].get("dataset"), int(item[2].get("seed")), MODES.index(item[2].get("adapter_component_mode"))
        ))
    ]


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def summarize(rows):
    columns = ("A_final", "F_avg", "WorstDrop", "Au", "Au_distance_to_chance", "T_f")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["mode"]), []).append(row)
    summary = []
    for (dataset, mode), group in sorted(grouped.items(), key=lambda item: (
        item[0][0], MODES.index(item[0][1])
    )):
        entry = {"dataset": dataset, "mode": mode, "n_seeds": len(group)}
        for column in columns:
            values = [row[column] for row in group if row[column] is not None]
            entry[f"{column}_mean"] = statistics.fmean(values) if values else None
            entry[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(entry)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["dataset", "seed", "mode", "source_run"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary, tag):
    lines = [
        "# PALL-Adapter Component Controls",
        "",
        f"Strict experiment tag: `{tag}`. Latest completed run per dataset/seed/mode.",
        "",
        "| Dataset | Mode | Seeds | A_final | F_avg | WorstDrop | Au | |Au-chance| | T_f (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cells = []
        for metric in ("A_final", "F_avg", "WorstDrop", "Au", "Au_distance_to_chance", "T_f"):
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            cells.append("NA" if mean is None else f"{fmt(mean)} ± {fmt(std)}")
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['n_seeds']} | " + " | ".join(cells) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/aggregates/adapter_components"))
    args = parser.parse_args()

    rows = select_rows(args.root, args.tag)
    if not rows:
        raise SystemExit(f"No completed component runs found for tag={args.tag!r} under {args.root}")
    summary = summarize(rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_runs.csv"), rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), summary, args.tag)
    print(f"Selected {len(rows)} latest run(s); wrote component summaries under {args.out_prefix.parent}")


if __name__ == "__main__":
    main()
