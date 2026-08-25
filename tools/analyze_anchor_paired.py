#!/usr/bin/env python3
"""Paired seed analysis of the EPALL anchor target: ``old`` versus ``reinit``.

``protect_anchor`` selects what the L2 anchor penalty on ``S_crit`` pulls toward
during retained-task repair: the pre-request weights (``old``) or a fresh
re-initialisation (``reinit``).  Because the flag only affects ``pall_modified``,
the two arms differ in exactly one configuration value and share the request
schedule per seed, so every seed forms a matched pair.

The two arms live under separate experiment tags (``*_standard`` for ``old``,
``*_standard_reinit_v1`` for ``reinit``), which keeps the reinit runs out of the
canonical result tables until an anchor is chosen.

Sign convention: a positive delta always favours ``old``.  ``A_u`` is scored as
distance to chance, so a positive delta means ``old`` lands closer to chance.
Exact one-sided sign tests are reported; with eight seeds the smallest
attainable p is 1/256 for a clean sweep, so effects are read alongside the
per-seed win counts rather than from p alone.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

OLD_TAG = "{dataset}_standard"
REINIT_TAG = "{dataset}_standard_reinit_v1"
DATASETS = ("cifar10", "cifar100")
CHANCE = {"cifar10": 0.5, "cifar100": 0.1}

# metric -> (source column, orientation)
#   higher_is_better: delta = old - reinit
#   lower_is_better:  delta = reinit - old
#   distance_to_chance: delta = |reinit - c| - |old - c|
METRICS = {
    "A_final": ("final_avg_accuracy", "higher_is_better"),
    "F_avg": ("average_forgetting", "lower_is_better"),
    "WorstDrop": ("WorstDrop", "lower_is_better"),
    "Au_distance_to_chance": ("Au", "distance_to_chance"),
    "T_f": ("t_forget_total", "lower_is_better"),
}


def number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_arm(rows, dataset: str, tag: str, anchor: str) -> Dict[int, dict]:
    """Matched EPALL runs of one arm, keyed by seed.

    The protection settings are pinned as well as the anchor: a run with a
    different protect ratio or penalty weight is a different method, not a
    different anchor.
    """

    selected: Dict[int, dict] = {}
    for row in rows:
        if row.get("dataset") != dataset or row.get("method") != "pall_modified":
            continue
        if row.get("experiment_tag") != tag or row.get("protect_anchor") != anchor:
            continue
        if number(row.get("protect_ratio")) != 0.2 or number(row.get("lambda_protect")) != 1.0:
            continue
        if row.get("protect_importance") not in {"", "gradient"}:
            continue
        seed = int(row["seed"])
        if seed in selected:
            raise SystemExit(f"{dataset}/{anchor}: duplicate seed {seed} after seed policy")
        selected[seed] = row
    return selected


def delta(metric: str, old_row: dict, reinit_row: dict, chance: float) -> Optional[float]:
    column, orientation = METRICS[metric]
    old_value, reinit_value = number(old_row.get(column)), number(reinit_row.get(column))
    if old_value is None or reinit_value is None:
        return None
    if orientation == "higher_is_better":
        return old_value - reinit_value
    if orientation == "lower_is_better":
        return reinit_value - old_value
    return abs(reinit_value - chance) - abs(old_value - chance)


def sign_test_p(favour_old: int, favour_reinit: int) -> float:
    """Exact one-sided binomial p for 'old is better', ties dropped."""

    n = favour_old + favour_reinit
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(favour_old, n + 1))
    return tail / (2 ** n)


def cohen_dz(deltas: List[float]) -> Optional[float]:
    if len(deltas) < 2:
        return None
    spread = statistics.stdev(deltas)
    if spread == 0:
        return None
    return statistics.fmean(deltas) / spread


def analyze(rows) -> tuple[List[dict], List[dict]]:
    pair_rows: List[dict] = []
    summary_rows: List[dict] = []
    for dataset in DATASETS:
        chance = CHANCE[dataset]
        old = select_arm(rows, dataset, OLD_TAG.format(dataset=dataset), "old")
        reinit = select_arm(rows, dataset, REINIT_TAG.format(dataset=dataset), "reinit")
        seeds = sorted(set(old) & set(reinit))
        if not seeds:
            raise SystemExit(f"{dataset}: no matched seeds between the two anchor arms")
        unmatched = sorted(set(old) ^ set(reinit))
        if unmatched:
            print(f"[WARN] {dataset}: seeds present in only one arm, dropped: {unmatched}")
        for seed in seeds:
            row = {"dataset": dataset, "seed": seed}
            for metric in METRICS:
                row[f"delta_{metric}"] = delta(metric, old[seed], reinit[seed], chance)
            pair_rows.append(row)
        for metric in METRICS:
            deltas = [
                value
                for value in (delta(metric, old[seed], reinit[seed], chance) for seed in seeds)
                if value is not None
            ]
            if not deltas:
                continue
            favour_old = sum(1 for value in deltas if value > 0)
            favour_reinit = sum(1 for value in deltas if value < 0)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "favour_old": favour_old,
                    "favour_reinit": favour_reinit,
                    "ties": len(deltas) - favour_old - favour_reinit,
                    "mean_delta": statistics.fmean(deltas),
                    "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "cohen_dz": cohen_dz(deltas),
                    "sign_p_old_better": sign_test_p(favour_old, favour_reinit),
                    "min_attainable_p": sign_test_p(len(deltas), 0),
                }
            )
    holm_correct(summary_rows)
    return pair_rows, summary_rows


ARM_COLUMNS = ("final_avg_accuracy", "average_forgetting", "WorstDrop", "Au")


def arm_summaries(rows) -> List[dict]:
    """Per-arm mean and sd, so the thesis table has one generated source."""

    output: List[dict] = []
    for dataset in DATASETS:
        for tag, anchor in (
            (OLD_TAG.format(dataset=dataset), "old"),
            (REINIT_TAG.format(dataset=dataset), "reinit"),
        ):
            arm = select_arm(rows, dataset, tag, anchor)
            if not arm:
                continue
            record = {
                "dataset": dataset,
                "anchor": anchor,
                "experiment_tag": tag,
                "n_seeds": len(arm),
            }
            for column in ARM_COLUMNS:
                values = [number(row.get(column)) for row in arm.values()]
                values = [value for value in values if value is not None]
                record[f"{column}_mean"] = statistics.fmean(values)
                record[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(record)
    return output


def holm_correct(summary_rows: List[dict]) -> None:
    """Holm step-down correction across every test in this family.

    Ten tests are run here (five metrics on two datasets); reading any raw p at
    the 0.05 level without correcting would overstate the evidence.
    """

    order = sorted(range(len(summary_rows)), key=lambda i: summary_rows[i]["sign_p_old_better"])
    total = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * summary_rows[index]["sign_p_old_better"])
        running = max(running, adjusted)  # step-down: adjusted p is non-decreasing
        summary_rows[index]["holm_p"] = running
        summary_rows[index]["survives_holm_05"] = running < 0.05


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: List[dict], pairs: List[dict], arms: List[dict]) -> None:
    lines = [
        "# EPALL Anchor Target: `old` versus `reinit`",
        "",
        "## Arm means (standard protocol, eight seeds)",
        "",
        "| Dataset | Anchor | Seeds | A_final | F_avg | WorstDrop | A_u |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        cells = " | ".join(
            f"{row[f'{column}_mean']:.4f} ± {row[f'{column}_std']:.4f}" for column in ARM_COLUMNS
        )
        lines.append(f"| {row['dataset']} | {row['anchor']} | {row['n_seeds']} | {cells} |")
    lines += [
        "",
        "## Paired tests",
        "",
        "Matched-seed comparison of the L2 anchor target on `S_crit`. The two arms differ",
        "in `protect_anchor` alone and share the per-seed request schedule.",
        "A positive delta favours `old`. `A_u` is scored as distance to chance, so a",
        "positive delta means `old` sits closer to chance.",
        "",
        "| Dataset | Metric | Pairs | Favour old | Favour reinit | Ties | Mean delta | d_z | Sign p (old better) | Holm p | Survives Holm | Min attainable p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary:
        dz = "NA" if row["cohen_dz"] is None else f"{row['cohen_dz']:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['n_pairs']} | {row['favour_old']} | "
            f"{row['favour_reinit']} | {row['ties']} | {row['mean_delta']:+.6f} | {dz} | "
            f"{row['sign_p_old_better']:.6f} | {row['holm_p']:.4f} | "
            f"{'yes' if row['survives_holm_05'] else 'no'} | {row['min_attainable_p']:.6f} |"
        )
    lines.extend(["", "## Per-seed deltas", "", "| Dataset | Seed | " + " | ".join(METRICS) + " |",
                  "|---|---:|" + "---:|" * len(METRICS)])
    for row in pairs:
        cells = []
        for metric in METRICS:
            value = row[f"delta_{metric}"]
            cells.append("NA" if value is None else f"{value:+.4f}")
        lines.append(f"| {row['dataset']} | {row['seed']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/aggregates/server_results_g29_local.csv"),
        help="Flat per-run aggregate containing both anchor arms.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("results/aggregates/anchor_paired"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs, summary = analyze(rows)
    arms = arm_summaries(rows)

    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_runs.csv"), pairs)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_arms.csv"), arms)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), summary, pairs, arms)
    print(f"Wrote {len(pairs)} paired rows and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
