#!/usr/bin/env python3
"""Strict summary of corrected, augmentation-matched PALL-Modified MIA runs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


TAG = "pall_modified_mia_corrected_v2"


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def value(item):
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def latest_rows(root, tag):
    selected = {}
    for config_path in root.glob("**/config.json"):
        config = load(config_path)
        if not isinstance(config, dict):
            continue
        if config.get("experiment_tag") != tag or config.get("method") != "pall_modified":
            continue
        metrics = load(config_path.with_name("metrics.json"))
        final = ((metrics or {}).get("normalized_results") or {}).get("final") or {}
        before, after = value(final.get("mia_auc_before")), value(final.get("mia_auc_after"))
        if before is None or after is None:
            continue
        key = (config.get("dataset"), int(config.get("seed")))
        candidate = (config_path.parent.name, {
            "dataset": config.get("dataset"),
            "seed": int(config.get("seed")),
            "auc_before": before,
            "auc_after": after,
            "distance_before": abs(before - 0.5),
            "distance_after": abs(after - 0.5),
            "delta_distance_toward_chance": abs(before - 0.5) - abs(after - 0.5),
            "source_run": str(config_path.parent),
        })
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    return [selected[key][1] for key in sorted(selected)]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, tag):
    lines = [
        "# Corrected PALL-Modified MIA Diagnostic", "",
        f"Strict tag: `{tag}`. Member and non-member examples use the same augmentation-free preprocessing; tied scores use average ranks.",
        "AUC near 0.5 is a low-power diagnostic null, not a privacy or exact-removal guarantee.", "",
        "| Dataset | Seeds | AUC before | AUC after | Δdistance toward 0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        def stat(key):
            values = [row[key] for row in group]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            return f"{mean:.4f} ± {std:.4f}"
        lines.append(f"| {dataset} | {len(group)} | {stat('auc_before')} | {stat('auc_after')} | {stat('delta_distance_toward_chance')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/aggregates/corrected_mia"))
    args = parser.parse_args()
    rows = latest_rows(args.root, args.tag)
    expected_keys = {(dataset, seed) for dataset in ("cifar10", "cifar100") for seed in (0, 1, 2)}
    actual_keys = {(row["dataset"], int(row["seed"])) for row in rows}
    if actual_keys != expected_keys:
        raise SystemExit(
            f"Expected exact corrected MIA matrix (2 datasets x 3 seeds); "
            f"found keys={sorted(actual_keys)}"
        )
    write_csv(args.out_prefix.with_suffix(".csv"), rows)
    write_markdown(args.out_prefix.with_suffix(".md"), rows, args.tag)
    print(f"Selected {len(rows)} corrected MIA runs")


if __name__ == "__main__":
    main()
