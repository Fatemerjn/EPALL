#!/usr/bin/env python3
"""Read-only forgotten-task persistence analysis for matched MAIN runs.

For every deletion request the analyzer follows the forgotten task through the
remainder of the schedule and reports how far it drifts from chance.  The
governing quantity is the distance to chance ``|A_u - c|``: for a forgotten task
both a rise above chance and a fall below chance are failures of forgetting, so
raw accuracy is not a monotone objective and is reported only as context.

The MAIN schedules interleave training requests with deletion requests, and the
interleaving is *seed dependent* -- seed 0 of ``cifar10_t5_f3`` is
T T T T F T F F while seed 1 is T T F T F T T F -- so each observation gap is
attributed to one of two causes:

* ``train`` -- the requests between the previous deletion and the next one, read
  from ``per_task_acc_before`` of the next deletion event;
* ``delete`` -- the next deletion request itself, read from
  ``per_task_acc_after_retrain`` of that event.

The two contributions sum exactly to the total drift, which lets the audit say
whether a forgotten task is disturbed by later learning or by later deletions.
Gaps are sized from the ``request_id`` of consecutive events, so a gap with zero
intervening training requests contributes exactly zero by construction.
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
    "observation_index",
    "observation_phase",
    "observation_request_id",
    "train_requests_since_previous",
    "forgotten_accuracy",
    "chance",
    "distance_to_chance",
    "distance_rebound_from_immediate",
    "accuracy_change_from_immediate",
    "experiment_tag",
    "run_path",
)
TASK_COLUMNS = (
    "dataset",
    "method",
    "seed",
    "deleted_event",
    "forgotten_task",
    "n_later_observations",
    "n_later_train_requests",
    "n_later_delete_requests",
    "chance",
    "immediate_accuracy",
    "immediate_distance_to_chance",
    "final_accuracy",
    "final_distance_to_chance",
    "max_distance_to_chance",
    "max_distance_rebound",
    "final_distance_rebound",
    "train_attributed_drift",
    "delete_attributed_drift",
    "experiment_tag",
    "run_path",
)
SUMMARY_COLUMNS = (
    "dataset",
    "method",
    "n_seeds",
    "n_deleted_tasks",
    "n_followed_tasks",
    "immediate_distance_mean",
    "final_distance_mean",
    "max_distance_rebound_mean",
    "max_distance_rebound_worst",
    "fraction_distance_rebound_gt_0_01",
    "train_attributed_drift_mean",
    "delete_attributed_drift_mean",
    "net_distance_drift_mean",
    "immediate_accuracy_mean",
    "final_accuracy_mean",
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


def request_id(event, context):
    value = event.get("request_id")
    if not isinstance(value, int):
        raise ValueError(f"{context}: missing integer request_id")
    return value


def observation_sequence(events, task_id, deleted_index, run_path):
    """Ordered observations of ``task_id`` from its deletion to the end of the run.

    The first entry is the immediate post-deletion measurement.  Every later
    deletion event contributes up to two entries: the state reached by the
    requests in between (``train``) and the state after that deletion itself
    (``delete``).  A gap with no intervening request yields no ``train`` entry.
    """

    context = f"{run_path} event {deleted_index + 1}"
    observations = [
        {
            "phase": "immediate",
            "request_id": request_id(events[deleted_index], context),
            "train_requests": 0,
            "accuracy": accuracy_map(events[deleted_index], "per_task_acc_after_retrain", context)[task_id],
        }
    ]
    for index in range(deleted_index + 1, len(events)):
        event = events[index]
        context = f"{run_path} event {index + 1}"
        current_id = request_id(event, context)
        previous_id = request_id(events[index - 1], context)
        intervening = current_id - previous_id - 1
        if intervening < 0:
            raise ValueError(f"{context}: non-monotone request_id sequence")
        if intervening > 0:
            observations.append(
                {
                    "phase": "train",
                    "request_id": current_id - 1,
                    "train_requests": intervening,
                    "accuracy": accuracy_map(event, "per_task_acc_before", context)[task_id],
                }
            )
        observations.append(
            {
                "phase": "delete",
                "request_id": current_id,
                "train_requests": 0,
                "accuracy": accuracy_map(event, "per_task_acc_after_retrain", context)[task_id],
            }
        )
    return observations


def analyze_run(row):
    events = row["events"]
    chance = 1.0 / float(row["config"]["class_per_task"])
    raw_rows = []
    task_rows = []
    for deleted_index, event in enumerate(events):
        task_id = int(event["task_id"])
        for index, observation in enumerate(events):
            post = accuracy_map(
                observation,
                "per_task_acc_after_retrain",
                f"{row['run_path']} event {index + 1}",
            )
            if index >= deleted_index and task_id not in post:
                raise ValueError(
                    f"{row['run_path']}: forgotten task {task_id} is absent from later evaluation"
                )
        observations = observation_sequence(events, task_id, deleted_index, row["run_path"])
        immediate = observations[0]["accuracy"]
        immediate_distance = abs(immediate - chance)
        distances = []
        for index, observation in enumerate(observations):
            distance = abs(observation["accuracy"] - chance)
            distances.append(distance)
            raw_rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed": int(row["seed"]),
                    "deleted_event": deleted_index + 1,
                    "forgotten_task": task_id,
                    "observation_index": index,
                    "observation_phase": observation["phase"],
                    "observation_request_id": observation["request_id"],
                    "train_requests_since_previous": observation["train_requests"],
                    "forgotten_accuracy": observation["accuracy"],
                    "chance": chance,
                    "distance_to_chance": distance,
                    "distance_rebound_from_immediate": distance - immediate_distance,
                    "accuracy_change_from_immediate": observation["accuracy"] - immediate,
                    "experiment_tag": row["experiment_tag"],
                    "run_path": row["run_path"],
                }
            )
        later = observations[1:]
        train_drift = 0.0
        delete_drift = 0.0
        for index in range(1, len(observations)):
            step = distances[index] - distances[index - 1]
            if observations[index]["phase"] == "train":
                train_drift += step
            else:
                delete_drift += step
        net_drift = distances[-1] - immediate_distance
        if abs(train_drift + delete_drift - net_drift) > 1e-9:
            raise ValueError(
                f"{row['run_path']} task {task_id}: attribution does not sum to the net drift"
            )
        task_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "deleted_event": deleted_index + 1,
                "forgotten_task": task_id,
                "n_later_observations": len(later),
                "n_later_train_requests": sum(item["train_requests"] for item in later),
                "n_later_delete_requests": sum(1 for item in later if item["phase"] == "delete"),
                "chance": chance,
                "immediate_accuracy": immediate,
                "immediate_distance_to_chance": immediate_distance,
                "final_accuracy": observations[-1]["accuracy"],
                "final_distance_to_chance": distances[-1],
                "max_distance_to_chance": max(distances),
                "max_distance_rebound": max([0.0, *[value - immediate_distance for value in distances[1:]]]),
                "final_distance_rebound": distances[-1] - immediate_distance,
                "train_attributed_drift": train_drift,
                "delete_attributed_drift": delete_drift,
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
            # A task deleted by the last request has no follow-up horizon; including
            # it would dilute the rebound statistics with structural zeros.
            followed = [row for row in rows if row["n_later_observations"] > 0]
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_seeds": len({int(row["seed"]) for row in rows}),
                    "n_deleted_tasks": len(rows),
                    "n_followed_tasks": len(followed),
                    "immediate_distance_mean": statistics.fmean(
                        row["immediate_distance_to_chance"] for row in rows
                    ),
                    "final_distance_mean": statistics.fmean(
                        row["final_distance_to_chance"] for row in rows
                    ),
                    "max_distance_rebound_mean": statistics.fmean(
                        row["max_distance_rebound"] for row in followed
                    ),
                    "max_distance_rebound_worst": max(row["max_distance_rebound"] for row in followed),
                    "fraction_distance_rebound_gt_0_01": statistics.fmean(
                        float(row["max_distance_rebound"] > 0.01) for row in followed
                    ),
                    "train_attributed_drift_mean": statistics.fmean(
                        row["train_attributed_drift"] for row in followed
                    ),
                    "delete_attributed_drift_mean": statistics.fmean(
                        row["delete_attributed_drift"] for row in followed
                    ),
                    # Equals the sum of the two attributed means by construction;
                    # reported so the decomposition can be checked from the table.
                    "net_distance_drift_mean": statistics.fmean(
                        row["final_distance_rebound"] for row in followed
                    ),
                    "immediate_accuracy_mean": statistics.fmean(row["immediate_accuracy"] for row in rows),
                    "final_accuracy_mean": statistics.fmean(row["final_accuracy"] for row in rows),
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
        "The forgotten task is followed from immediately after its deletion to the end of the schedule.",
        "Rebound is measured on the distance to chance `|A_u - c|`, not on raw accuracy: for a forgotten",
        "task, drifting below chance is as much a failure as drifting above it.",
        "The MAIN schedules do contain training requests after the first deletion, so each gap is",
        "attributed either to the intervening training requests (`train`) or to the next deletion (`delete`).",
        "The two attributed drifts sum to the total change in `|A_u - c|`.",
        "Rebound columns cover only tasks with a non-empty follow-up horizon; a task deleted by the last",
        "request of the schedule has none.",
        "",
        f"Selected runs: {len(selected)}; seeds per dataset/method: {sorted({int(row['seed']) for row in selected})}.",
        "",
        "| Dataset | Method | Seeds | Deleted | Followed | Immediate abs(Au-c) | Final abs(Au-c) | "
        "Mean max rebound | Worst rebound | Rebound >1pt | Train-attributed | Delete-attributed | Net |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {method} | {n_seeds} | {n_deleted_tasks} | {n_followed_tasks} | "
            "{immediate:.4f} | {final:.4f} | {mean_rebound:.4f} | {worst_rebound:.4f} | {fraction:.1%} | "
            "{train:+.4f} | {delete:+.4f} | {net:+.4f} |".format(
                net=row["net_distance_drift_mean"],
                dataset=row["dataset"],
                method=METHOD_LABELS[row["method"]],
                n_seeds=row["n_seeds"],
                n_deleted_tasks=row["n_deleted_tasks"],
                n_followed_tasks=row["n_followed_tasks"],
                immediate=row["immediate_distance_mean"],
                final=row["final_distance_mean"],
                mean_rebound=row["max_distance_rebound_mean"],
                worst_rebound=row["max_distance_rebound_worst"],
                fraction=row["fraction_distance_rebound_gt_0_01"],
                train=row["train_attributed_drift_mean"],
                delete=row["delete_attributed_drift_mean"],
            )
        )
    lines.extend(["", "## Selected run trace", ""])
    for row in selected:
        lines.append(
            f"- `{row['dataset']}` / `{row['method']}` / seed {row['seed']}: `{row['run_path']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def usable_offsets(dataset_rows):
    """Observation indices where every run contributes the same phase.

    Returns the longest prefix of indices for which (a) no index mixes a
    training observation with a deletion observation, and (b) no run has
    dropped out. Beyond that point an index-wise mean would compare unlike
    quantities, so the curve simply stops.

    The completeness test counts every contributing run of the dataset (methods
    times seeds), which is what makes a dropout detectable. It is not the sample
    size behind a plotted point: each method's curve averages that method's
    seeds only.
    """

    by_index = {}
    for row in dataset_rows:
        by_index.setdefault(int(row["observation_index"]), []).append(row)
    if not by_index:
        return []
    full = max(len(rows) for rows in by_index.values())
    usable = []
    for index in sorted(by_index):
        rows = by_index[index]
        phases = {str(row["observation_phase"]) for row in rows}
        if len(phases) != 1 or len(rows) != full:
            break
        usable.append(index)
    return usable


def plot_curves(path, raw_rows, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=False)
    for axis, dataset in zip(axes, ("cifar10", "cifar100")):
        # One fixed cohort: the first-deleted task, which is observed at every
        # later point in every selected run. Mixing deletion ordinals would
        # change the contributing task set at each x value.
        dataset_rows = [
            row for row in raw_rows
            if row["dataset"] == dataset and int(row["deleted_event"]) == 1
        ]
        # The MAIN schedules are not identical across seeds: the deletion
        # requests sit at different positions, so seed 0 and seed 1 can differ
        # both in how many observations the first-deleted task has and in what
        # each observation *is*. Averaging index by index is only meaningful
        # while every contributing run agrees on the phase and all runs are
        # still present, so the curve is truncated at the last such index.
        offsets = usable_offsets(dataset_rows)
        for method in METHODS:
            method_rows = [row for row in dataset_rows if row["method"] == method]
            ys = [
                statistics.fmean(
                    row["distance_to_chance"] for row in method_rows
                    if int(row["observation_index"]) == offset
                )
                for offset in offsets
            ]
            axis.plot(
                offsets,
                ys,
                marker="o",
                linewidth=1.35,
                markersize=3.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        phases = {
            int(row["observation_index"]): str(row["observation_phase"]) for row in dataset_rows
        }
        axis.set_xticks(offsets)
        axis.set_xticklabels(
            ["delete" if offset == 0 else phases[offset] for offset in offsets],
            fontsize=7,
        )
        axis.set_title(dataset.upper().replace("CIFAR", "CIFAR-"))
        axis.set_xlabel("Observation after the deletion request")
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(r"$|A_u - c|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", dpi=dpi)
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
