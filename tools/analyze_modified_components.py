#!/usr/bin/env python3
"""Strict, paired summary for the direct PALL-Modified mechanism ablation."""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


TAG = "pall_modified_components_overlapmatched_v2"
MODES = ("no_anchor", "overlap_only", "random_budget", "ranking_no_overlap", "full")
METRICS = ("A_final", "F_avg", "WorstDrop", "Au", "Au_distance_to_chance", "T_f")
PAIR_ORIENTATION = {
    "A_final": "full_minus_control",
    "WorstDrop": "control_minus_full",
    "Au_distance_to_chance": "control_minus_full",
}


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_rows(root, tag):
    selected = {}
    for config_path in root.glob("**/config.json"):
        config = load_json(config_path)
        if not isinstance(config, dict):
            continue
        if config.get("method") != "pall_modified" or config.get("experiment_tag") != tag:
            continue
        mode = config.get("modified_component_mode")
        if mode not in MODES:
            continue
        metrics = load_json(config_path.with_name("metrics.json"))
        summary = (metrics or {}).get("summary") or {}
        if summary.get("final_avg_accuracy") is None:
            continue
        key = (config.get("dataset"), int(config.get("seed")), mode)
        candidate = (config_path.parent.name, config_path, config, metrics)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate

    rows = []
    for _, config_path, config, metrics in sorted(selected.values(), key=lambda item: (
        item[2].get("dataset"), int(item[2].get("seed")), MODES.index(item[2].get("modified_component_mode"))
    )):
        summary = metrics.get("summary") or {}
        normalized = (metrics.get("normalized_results") or {}).get("final") or {}
        au = number(normalized.get("Au", summary.get("Au")))
        chance = 1.0 / float(config["class_per_task"])
        rows.append({
            "dataset": config.get("dataset"),
            "seed": int(config.get("seed")),
            "mode": config.get("modified_component_mode"),
            "A_final": number(normalized.get("final_avg_accuracy", summary.get("final_avg_accuracy"))),
            "F_avg": number(normalized.get("average_forgetting", summary.get("average_forgetting"))),
            "WorstDrop": number(normalized.get("WorstDrop", summary.get("WorstDrop"))),
            "Au": au,
            "chance": chance,
            "Au_distance_to_chance": abs(au - chance) if au is not None else None,
            "T_f": number(normalized.get("t_forget_total", summary.get("t_forget_total"))),
            "source_run": str(config_path.parent),
        })
    return rows


def summarize(rows):
    output = []
    keys = sorted({(row["dataset"], row["mode"]) for row in rows}, key=lambda key: (key[0], MODES.index(key[1])))
    for dataset, mode in keys:
        group = [row for row in rows if row["dataset"] == dataset and row["mode"] == mode]
        entry = {"dataset": dataset, "mode": mode, "n_seeds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            entry[f"{metric}_mean"] = statistics.fmean(values) if values else None
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(entry)
    return output


def bootstrap_ci(values, samples=10000, seed=2027):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def paired_summary(rows):
    by_key = {(row["dataset"], row["seed"], row["mode"]): row for row in rows}
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        seeds = sorted({row["seed"] for row in rows if row["dataset"] == dataset})
        for control in MODES:
            if control == "full":
                continue
            pairs = []
            for seed in seeds:
                full = by_key.get((dataset, seed, "full"))
                baseline = by_key.get((dataset, seed, control))
                if full is not None and baseline is not None:
                    pairs.append((full, baseline))
            entry = {"dataset": dataset, "control": control, "n_pairs": len(pairs)}
            for metric, orientation in PAIR_ORIENTATION.items():
                values = []
                for full, baseline in pairs:
                    if full[metric] is None or baseline[metric] is None:
                        continue
                    delta = full[metric] - baseline[metric]
                    values.append(delta if orientation == "full_minus_control" else -delta)
                mean = statistics.fmean(values) if values else None
                low, high = bootstrap_ci(values)
                entry[f"delta_{metric}_favor_full_mean"] = mean
                entry[f"delta_{metric}_favor_full_ci_low"] = low
                entry[f"delta_{metric}_favor_full_ci_high"] = high
            output.append(entry)
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "seed", "mode", "source_run"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def write_markdown(path, summary, pairs, tag):
    lines = [
        "# Direct PALL-Modified Mechanism Ablation", "",
        f"Strict tag: `{tag}`. Rows share dataset protocol, schedule seed, and training configuration.", "",
        "| Dataset | Mode | Seeds | A_final | F_avg | WorstDrop | Au | |Au-chance| | T_f (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        cells = []
        for metric in METRICS:
            cells.append(f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}")
        lines.append(f"| {row['dataset']} | {row['mode']} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    lines.extend([
        "", "## Paired differences relative to Full", "",
        "Positive deltas favor Full. Bootstrap intervals are descriptive across the matched seeds.", "",
        "| Dataset | Control | Pairs | ΔA_final | ΔWorstDrop | Δ|Au-chance| |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in pairs:
        cells = []
        for metric in PAIR_ORIENTATION:
            base = f"delta_{metric}_favor_full"
            mean, low, high = row.get(f"{base}_mean"), row.get(f"{base}_ci_low"), row.get(f"{base}_ci_high")
            cells.append("NA" if mean is None else f"{mean:.4f} [{low:.4f}, {high:.4f}]")
        lines.append(f"| {row['dataset']} | {row['control']} | {row['n_pairs']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/aggregates/modified_components"))
    args = parser.parse_args()
    rows = select_rows(args.root, args.tag)
    seeds_by_dataset = {
        dataset: {int(row["seed"]) for row in rows if row["dataset"] == dataset}
        for dataset in ("cifar10", "cifar100")
    }
    if not seeds_by_dataset["cifar10"] or seeds_by_dataset["cifar10"] != seeds_by_dataset["cifar100"]:
        raise SystemExit(
            f"Expected the same non-empty seed set for both datasets; got {seeds_by_dataset}"
        )
    selected_seeds = seeds_by_dataset["cifar10"]
    expected_keys = {
        (dataset, seed, mode)
        for dataset in ("cifar10", "cifar100")
        for seed in selected_seeds
        for mode in MODES
    }
    actual_keys = {(row["dataset"], int(row["seed"]), row["mode"]) for row in rows}
    if actual_keys != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        extra = sorted(actual_keys.difference(expected_keys))
        raise SystemExit(
            f"Expected a complete {len(expected_keys)}-run PALL-Modified matrix for tag={args.tag!r}; "
            f"found {len(actual_keys)} keys, missing={missing}, extra={extra}"
        )
    summary = summarize(rows)
    pairs = paired_summary(rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_runs.csv"), rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_paired_summary.csv"), pairs)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), summary, pairs, args.tag)
    print(f"Selected {len(rows)} latest run(s); wrote PALL-Modified component summaries")


if __name__ == "__main__":
    main()
