#!/usr/bin/env python3
"""Paired seed analysis of PALL-Modified versus PALL-Original in main tables."""

import argparse
import csv
import random
import statistics
from pathlib import Path


SPECS = {
    "cifar10": "cifar10_standard",
    "cifar100": "cifar100_standard",
}
METRICS = {
    "A_final": ("final_avg_accuracy", "modified_minus_original"),
    "F_avg": ("average_forgetting", "original_minus_modified"),
    "WorstDrop": ("WorstDrop", "original_minus_modified"),
    "Au_distance_to_chance": ("Au", "distance_original_minus_modified"),
    "T_f": ("t_forget_total", "original_minus_modified"),
}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bootstrap_ci(values, samples=10000, seed=2027):
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def paired_rows(rows):
    selected = {}
    for row in rows:
        dataset = row.get("dataset")
        if dataset not in SPECS or row.get("experiment_tag") != SPECS[dataset]:
            continue
        method = row.get("method")
        if method not in {"pall_original", "pall_modified"}:
            continue
        if method == "pall_modified":
            if number(row.get("protect_ratio")) != 0.2 or number(row.get("lambda_protect")) != 1.0:
                continue
            if row.get("protect_importance") not in {"", "gradient"}:
                continue
        selected[(dataset, int(row["seed"]), method)] = row

    output = []
    for dataset in SPECS:
        chance = 0.5 if dataset == "cifar10" else 0.1
        seeds = sorted({seed for ds, seed, _method in selected if ds == dataset})
        for seed in seeds:
            modified = selected.get((dataset, seed, "pall_modified"))
            original = selected.get((dataset, seed, "pall_original"))
            if modified is None or original is None:
                continue
            entry = {
                "dataset": dataset,
                "seed": seed,
                "modified_run": modified.get("run_path"),
                "original_run": original.get("run_path"),
            }
            for label, (column, orientation) in METRICS.items():
                mod_value, orig_value = number(modified.get(column)), number(original.get(column))
                if mod_value is None or orig_value is None:
                    entry[f"delta_{label}_favor_modified"] = None
                elif orientation == "modified_minus_original":
                    entry[f"delta_{label}_favor_modified"] = mod_value - orig_value
                elif orientation == "distance_original_minus_modified":
                    entry[f"delta_{label}_favor_modified"] = abs(orig_value - chance) - abs(mod_value - chance)
                else:
                    entry[f"delta_{label}_favor_modified"] = orig_value - mod_value
            output.append(entry)
    return output


def summaries(pairs):
    output = []
    for dataset in SPECS:
        group = [row for row in pairs if row["dataset"] == dataset]
        entry = {"dataset": dataset, "n_pairs": len(group)}
        for label in METRICS:
            column = f"delta_{label}_favor_modified"
            values = [row[column] for row in group if row[column] is not None]
            entry[f"{column}_mean"] = statistics.fmean(values) if values else None
            entry[f"{column}_all_positive"] = bool(values) and all(value > 0 for value in values)
            if values:
                low, high = bootstrap_ci(values)
            else:
                low, high = None, None
            entry[f"{column}_ci_low"] = low
            entry[f"{column}_ci_high"] = high
        output.append(entry)
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "seed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summaries):
    lines = [
        "# Paired Main-Table Analysis", "",
        "PALL-Modified minus PALL-Original on identical schedule seeds. Positive deltas favor PALL-Modified.",
        "Bootstrap intervals are descriptive because each dataset has only three paired seeds.", "",
        "| Dataset | Pairs | ΔA_final | ΔF_avg | ΔWorstDrop | Δ|Au-chance| | ΔT_f |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cells = []
        for label in METRICS:
            base = f"delta_{label}_favor_modified"
            mean, low, high = row.get(f"{base}_mean"), row.get(f"{base}_ci_low"), row.get(f"{base}_ci_high")
            cells.append("NA" if mean is None else f"{mean:.4f} [{low:.4f}, {high:.4f}]")
        lines.append(f"| {row['dataset']} | {row['n_pairs']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/aggregates/server_results.csv"))
    parser.add_argument("--out-prefix", type=Path, default=Path("results/aggregates/paired_main"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs = paired_rows(rows)
    summary = summaries(pairs)
    if any(row["n_pairs"] != 3 for row in summary):
        raise SystemExit(f"Expected exactly three matched seeds per dataset; got {summary}")
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_runs.csv"), pairs)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), summary)
    print(f"Wrote {len(pairs)} paired rows and {len(summary)} dataset summaries")


if __name__ == "__main__":
    main()
