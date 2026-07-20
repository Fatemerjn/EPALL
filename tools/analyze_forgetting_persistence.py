#!/usr/bin/env python3
"""Read-only forgotten-task persistence analysis for matched MAIN runs.

For each deletion request, the analyzer records forgotten-task accuracy
immediately after the request and after every later deletion request in the
same run.  It reports maximum post-delete rebound without treating chance as a
monotone objective.  Current schedules contain no training requests after the
first deletion, so later observations correspond to subsequent deletions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/font-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_sequential_damage import DATASETS, METHODS, METHOD_COLORS, METHOD_LABELS, select_runs


RAW_COLUMNS = (
    "dataset",
    "method",
    "seed",
    "deleted_event",
    "forgotten_task",
    "observation_event",
    "requests_after_deletion",
    "forgotten_accuracy",
    "chance",
    "rebound_from_immediate",
    "experiment_tag",
    "run_path",
)
TASK_COLUMNS = (
    "dataset",
    "method",
    "seed",
    "deleted_event",
    "forgotten_task",
    "n_later_requests",
    "chance",
    "immediate_accuracy",
    "final_accuracy",
    "final_distance_to_chance",
    "max_post_delete_accuracy",
    "max_rebound",
    "final_rebound",
    "max_excess_above_chance",
    "experiment_tag",
    "run_path",
)
SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "n_seeds",
    "n_deleted_tasks",
    "immediate_accuracy_mean",
    "final_accuracy_mean",
    "final_distance_to_chance_mean",
    "max_rebound_mean",
    "max_rebound_worst",
    "fraction_rebound_gt_0_01",
    "max_excess_above_chance_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/aggregates/forgetting_persistence"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def accuracy_map(event, key, context):
    payload = event.get(key)
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: missing {key}")
    return {int(task): float(value) for task, value in payload.items()}


def analyze_run(row):
    events = row["events"]
    chance = 1.0 / float(row["config"]["class_per_task"])
    raw_rows = []
    task_rows = []
    for deleted_index, event in enumerate(events):
        task_id = int(event["task_id"])
        trajectory = []
        for observation_index in range(deleted_index, len(events)):
            observation = events[observation_index]
            post = accuracy_map(
                observation,
                "per_task_acc_after_retrain",
                f"{row['run_path']} event {observation_index + 1}",
            )
            if task_id not in post:
                raise ValueError(
                    f"{row['run_path']}: forgotten task {task_id} is absent from later evaluation"
                )
            accuracy = post[task_id]
            trajectory.append(accuracy)
            raw_rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "deleted_event": deleted_index + 1,
                    "forgotten_task": task_id,
                    "observation_event": observation_index + 1,
                    "requests_after_deletion": observation_index - deleted_index,
                    "forgotten_accuracy": accuracy,
                    "chance": chance,
                    "rebound_from_immediate": accuracy - trajectory[0],
                    "experiment_tag": row["experiment_tag"],
                    "run_path": row["run_path"],
                }
            )
        immediate = trajectory[0]
        final = trajectory[-1]
        later = trajectory[1:]
        max_rebound = max([0.0, *[accuracy - immediate for accuracy in later]])
        task_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "deleted_event": deleted_index + 1,
                "forgotten_task": task_id,
                "n_later_requests": len(later),
                "chance": chance,
                "immediate_accuracy": immediate,
                "final_accuracy": final,
                "final_distance_to_chance": abs(final - chance),
                "max_post_delete_accuracy": max(trajectory),
                "max_rebound": max_rebound,
                "final_rebound": final - immediate,
                "max_excess_above_chance": max(0.0, max(trajectory) - chance),
                "experiment_tag": row["experiment_tag"],
                "run_path": row["run_path"],
            }
        )
    return raw_rows, task_rows


def summarize(task_rows):
    output = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [
                row for row in task_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            if not rows:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_seeds": len({int(row["seed"]) for row in rows}),
                    "n_deleted_tasks": len(rows),
                    "immediate_accuracy_mean": statistics.fmean(row["immediate_accuracy"] for row in rows),
                    "final_accuracy_mean": statistics.fmean(row["final_accuracy"] for row in rows),
                    "final_distance_to_chance_mean": statistics.fmean(
                        row["final_distance_to_chance"] for row in rows
                    ),
                    "max_rebound_mean": statistics.fmean(row["max_rebound"] for row in rows),
                    "max_rebound_worst": max(row["max_rebound"] for row in rows),
                    "fraction_rebound_gt_0_01": statistics.fmean(
                        float(row["max_rebound"] > 0.01) for row in rows
                    ),
                    "max_excess_above_chance_mean": statistics.fmean(
                        row["max_excess_above_chance"] for row in rows
                    ),
                }
            )
    return output


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)


def write_markdown(path, summary, selected):
    lines = [
        "# Forgotten-Task Persistence Audit",
        "",
        "Read-only analysis of the latest matched MAIN-equivalent runs.",
        "Accuracy is followed from immediately after each deletion through every later deletion request.",
        "Current schedules contain no training request after deletion begins. Chance is a target, not a monotone lower-is-better objective.",
        "The curve follows only the first-deleted task in each run, which has a complete two-request follow-up horizon; the table summarizes all deleted tasks.",
        "",
        f"Selected runs: {len(selected)}; seeds per dataset/method: {sorted({int(row['seed']) for row in selected})}.",
        "",
        "| Dataset | Method | Seeds | Deleted tasks | Immediate Au | Final Au | Final abs(Au-chance) | Mean max rebound | Worst rebound | Rebound >1pt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {method} | {n_seeds} | {n_deleted_tasks} | {immediate:.4f} | "
            "{final:.4f} | {distance:.4f} | {mean_rebound:.4f} | {worst_rebound:.4f} | {fraction:.1%} |".format(
                dataset=row["dataset"],
                method=METHOD_LABELS[row["method"]],
                n_seeds=row["n_seeds"],
                n_deleted_tasks=row["n_deleted_tasks"],
                immediate=row["immediate_accuracy_mean"],
                final=row["final_accuracy_mean"],
                distance=row["final_distance_to_chance_mean"],
                mean_rebound=row["max_rebound_mean"],
                worst_rebound=row["max_rebound_worst"],
                fraction=row["fraction_rebound_gt_0_01"],
            )
        )
    lines.extend(["", "## Selected run trace", ""])
    for row in selected:
        lines.append(
            f"- `{row['dataset']}` / `{row['method']}` / seed {row['seed']}: `{row['run_path']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(path, raw_rows, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=False)
    for axis, dataset in zip(axes, ("cifar10", "cifar100")):
        # Use one fixed cohort: the first-deleted task is observed immediately
        # and after both later requests in every selected run. Mixing deletion
        # ordinals would change the contributing task set at each x value.
        dataset_rows = [
            row for row in raw_rows
            if row["dataset"] == dataset and int(row["deleted_event"]) == 1
        ]
        for method in METHODS:
            method_rows = [row for row in dataset_rows if row["method"] == method]
            xs = sorted({int(row["requests_after_deletion"]) for row in method_rows})
            ys = [
                statistics.fmean(
                    row["forgotten_accuracy"] for row in method_rows
                    if int(row["requests_after_deletion"]) == offset
                )
                for offset in xs
            ]
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.35,
                markersize=3.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        chance = 1.0 / (2 if dataset == "cifar10" else 5)
        axis.axhline(chance, color="#666666", linestyle="--", linewidth=1.0, label="Chance")
        axis.set_title(dataset.upper().replace("CIFAR", "CIFAR-"))
        axis.set_xlabel("Subsequent deletion requests")
        axis.set_xticks((0, 1, 2))
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Forgotten-task accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    selected = select_runs(args.runs_root)
    raw_rows = []
    task_rows = []
    for row in selected:
        run_raw, run_tasks = analyze_run(row)
        raw_rows.extend(run_raw)
        task_rows.extend(run_tasks)
    summary = summarize(task_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "persistence_trajectories.csv", raw_rows, RAW_COLUMNS)
    write_csv(args.outdir / "persistence_per_task.csv", task_rows, TASK_COLUMNS)
    write_csv(args.outdir / "persistence_summary.csv", summary, SUMMARY_COLUMNS)
    write_markdown(args.outdir / "PERSISTENCE_AUDIT.md", summary, selected)
    plot_curves(args.outdir / "forgetting_persistence.pdf", raw_rows, args.dpi)
    print(
        f"Selected {len(selected)} matched runs; analyzed {len(task_rows)} deletion trajectories; "
        f"outputs: {args.outdir}"
    )


if __name__ == "__main__":
    main()
