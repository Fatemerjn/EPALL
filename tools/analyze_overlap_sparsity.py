#!/usr/bin/env python3
"""Direct manipulation of structural overlap via subnetwork sparsity.

The request-position benchmark varies *when* a task is deleted, which moves
structural overlap only as a side effect and simultaneously moves task identity,
task age, and the amount of post-deletion training.  This analysis instead holds
the schedule, the target task, and its position fixed and varies the subnetwork
sparsity, which changes how many coordinates each task mask claims and therefore
moves |S_share| directly.

Both arms of every pair see the same sparsity and the same seed, so the measured
overlap is matched between them.  The quantity of interest is the *interaction*:
does the benefit of protection grow as overlap grows?  That is the claim the
thesis makes and the one the request-position benchmark can only gesture at.

Sparsity also changes capacity, and that confound cannot be removed by design.
It is reported explicitly: A_final before protection is the capacity check, and
the interaction is read against it rather than in isolation.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

DATASETS = ("cifar10", "cifar100")
ARMS = ("pall_original", "pall_modified")
TAG = "{dataset}_overlap_sparsity_{level}_v1"


def number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def levels_present(rows) -> List[str]:
    found = set()
    for row in rows:
        tag = str(row.get("experiment_tag"))
        if "_overlap_sparsity_" in tag and tag.endswith("_v1"):
            found.add(tag.rsplit("_", 2)[1])
    return sorted(found, key=lambda s: float(s.replace("p", ".")))


def arm(rows, dataset: str, level: str, method: str) -> Dict[int, dict]:
    tag = TAG.format(dataset=dataset, level=level)
    return {
        int(row["seed"]): row
        for row in rows
        if row.get("experiment_tag") == tag and row.get("method") == method
    }


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Rank correlation, used because five levels cannot support a linear fit."""

    if len(xs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def spearman_exact_p(xs, ys):
    """Exact one-sided permutation p for a positive rank correlation.

    Five levels give 120 orderings, so the null is enumerated rather than
    approximated; with n=5 the smallest attainable p is 1/120.
    """

    from itertools import permutations

    observed = spearman(xs, ys)
    if observed is None:
        return None
    total = hits = 0
    for perm in permutations(ys):
        value = spearman(xs, list(perm))
        total += 1
        if value is not None and value >= observed - 1e-12:
            hits += 1
    return hits / total


def mean_of(by_seed, seeds, column):
    """Mean over seeds, skipping runs where the column was not recorded."""

    values = [number(by_seed[s].get(column)) for s in seeds]
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def analyze(rows):
    cells: List[dict] = []
    for dataset in DATASETS:
        for level in levels_present(rows):
            original = arm(rows, dataset, level, "pall_original")
            modified = arm(rows, dataset, level, "pall_modified")
            seeds = sorted(set(original) & set(modified))
            if not seeds:
                continue
            gaps = []
            for seed in seeds:
                a, b = original[seed], modified[seed]
                wd_a, wd_b = number(a["WorstDrop"]), number(b["WorstDrop"])
                ac_a, ac_b = number(a["final_avg_accuracy"]), number(b["final_avg_accuracy"])
                gaps.append(
                    {
                        "seed": seed,
                        # positive = protection helps
                        "worstdrop_gap": None if None in (wd_a, wd_b) else wd_a - wd_b,
                        "accuracy_gap": None if None in (ac_a, ac_b) else ac_b - ac_a,
                        "s_share": number(a.get("overlap_s_share")),
                    }
                )
            wd_gaps = [g["worstdrop_gap"] for g in gaps if g["worstdrop_gap"] is not None]
            ac_gaps = [g["accuracy_gap"] for g in gaps if g["accuracy_gap"] is not None]
            shares = [g["s_share"] for g in gaps if g["s_share"] is not None]
            cells.append(
                {
                    "dataset": dataset,
                    "sparsity": float(level.replace("p", ".")),
                    "n_seeds": len(seeds),
                    "s_share_mean": statistics.fmean(shares) if shares else None,
                    "worstdrop_original": mean_of(original, seeds, "WorstDrop"),
                    "worstdrop_modified": mean_of(modified, seeds, "WorstDrop"),
                    "worstdrop_gap_mean": statistics.fmean(wd_gaps),
                    "worstdrop_gap_wins": sum(1 for g in wd_gaps if g > 0),
                    "worstdrop_gap_p": sign_test_p(
                        sum(1 for g in wd_gaps if g > 0), sum(1 for g in wd_gaps if g < 0)
                    ),
                    "accuracy_original": mean_of(original, seeds, "final_avg_accuracy"),
                    "accuracy_modified": mean_of(modified, seeds, "final_avg_accuracy"),
                    "accuracy_gap_mean": statistics.fmean(ac_gaps),
                }
            )
    return cells


def trends(cells):
    """Does the protection benefit track measured overlap, per dataset?"""

    out = []
    for dataset in DATASETS:
        sub = [c for c in cells if c["dataset"] == dataset and c["s_share_mean"]]
        if len(sub) < 3:
            continue
        shares = [c["s_share_mean"] for c in sub]
        out.append(
            {
                "dataset": dataset,
                "n_levels": len(sub),
                "s_share_min": min(shares),
                "s_share_max": max(shares),
                "s_share_fold_range": max(shares) / min(shares),
                "rho_share_vs_worstdrop_original": spearman(
                    shares, [c["worstdrop_original"] for c in sub]
                ),
                "rho_share_vs_worstdrop_modified": spearman(
                    shares, [c["worstdrop_modified"] for c in sub]
                ),
                "rho_share_vs_worstdrop_gap": spearman(
                    shares, [c["worstdrop_gap_mean"] for c in sub]
                ),
                "exact_p_worstdrop_gap": spearman_exact_p(
                    shares, [c["worstdrop_gap_mean"] for c in sub]
                ),
                "rho_share_vs_accuracy_gap": spearman(
                    shares, [c["accuracy_gap_mean"] for c in sub]
                ),
                "exact_p_accuracy_gap": spearman_exact_p(
                    shares, [c["accuracy_gap_mean"] for c in sub]
                ),
                # capacity check: how far the sparsity sweep moves raw accuracy
                "accuracy_original_min": min(c["accuracy_original"] for c in sub),
                "accuracy_original_max": max(c["accuracy_original"] for c in sub),
            }
        )
    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, cells, trend_rows):
    lines = [
        "# Overlap Manipulated Directly Through Subnetwork Sparsity",
        "",
        "Schedule, target task, and request position are held fixed; sparsity varies, which",
        "moves the measured structural overlap |S_share| directly. Both arms share sparsity and",
        "seed, so overlap is matched within every pair. A positive gap favours EPALL.",
        "",
        "## Per level",
        "",
        "| Dataset | Sparsity | Seeds | mean \\|S_share\\| | WorstDrop PALL | WorstDrop EPALL | Gap | Wins | Sign p | A_final PALL | A_final EPALL | Acc gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(
            f"| {c['dataset']} | {c['sparsity']:.1f} | {c['n_seeds']} | {c['s_share_mean']:,.0f} | "
            f"{c['worstdrop_original']:.4f} | {c['worstdrop_modified']:.4f} | {c['worstdrop_gap_mean']:+.4f} | "
            f"{c['worstdrop_gap_wins']}/{c['n_seeds']} | {c['worstdrop_gap_p']:.4f} | "
            f"{c['accuracy_original']:.4f} | {c['accuracy_modified']:.4f} | {c['accuracy_gap_mean']:+.4f} |"
        )
    lines += ["", "## Trend across levels (Spearman over the five level means)", ""]
    for t in trend_rows:
        def fmt(key):
            value = t[key]
            return "NA" if value is None else f"{value:+.3f}"

        lines += [
            f"### {t['dataset']}",
            "",
            f"- |S_share| range: {t['s_share_min']:,.0f} to {t['s_share_max']:,.0f} "
            f"({t['s_share_fold_range']:.1f}x fold change over {t['n_levels']} levels)",
            f"- rho(overlap, WorstDrop of PALL-Original): {fmt('rho_share_vs_worstdrop_original')}",
            f"- rho(overlap, WorstDrop of EPALL): {fmt('rho_share_vs_worstdrop_modified')}",
            f"- rho(overlap, protection benefit on WorstDrop): {fmt('rho_share_vs_worstdrop_gap')} "
            f"(exact permutation p = {t['exact_p_worstdrop_gap']:.4f})",
            f"- rho(overlap, protection benefit on A_final): {fmt('rho_share_vs_accuracy_gap')} "
            f"(exact permutation p = {t['exact_p_accuracy_gap']:.4f})",
            f"- capacity check: A_final of PALL-Original moves "
            f"{t['accuracy_original_min']:.4f} to {t['accuracy_original_max']:.4f} across the sweep, "
            "so sparsity changes capacity as well as overlap; the interaction is read against this.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/aggregates/server_results_final.csv")
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=Path("results/aggregates/overlap_sparsity")
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = analyze(rows)
    if not cells:
        raise SystemExit("no overlap-sparsity runs found in the input")
    trend_rows = trends(cells)

    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_levels.csv"), cells)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_trends.csv"), trend_rows)
    write_markdown(
        args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), cells, trend_rows
    )
    print(f"Wrote {len(cells)} level cells and {len(trend_rows)} dataset trends")


if __name__ == "__main__":
    main()
