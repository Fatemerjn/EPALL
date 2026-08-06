#!/usr/bin/env python3
"""Paired tests for the EPALL vs PALL-Original main comparison.

The analyzer reports exact one-sided tests, their attainable discrete floor,
direction consistency, and Cohen's d_z.  The manuscript treats these as a
small-sample paired audit rather than using significance as a stopping rule.

Reads results/aggregates/paired_main_runs.csv (per-seed paired deltas produced by
tools/analyze_paired_main.py) and writes
results/aggregates/significance_tests.{csv,md}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Metric column -> (display name, direction in which a positive delta is good).
METRICS = {
    "delta_A_final_favor_modified": "A_final",
    "delta_WorstDrop_favor_modified": "WorstDrop",
    "delta_F_avg_favor_modified": "F_avg",
}


def exact_sign_test_p(deltas: list[float]) -> float:
    """One-sided exact sign test: P(#positive >= observed | p=0.5), ties dropped."""
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = sum(1 for d in nonzero if d > 0)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return tail / (2 ** n)


def exact_wilcoxon_p(deltas: list[float]) -> float:
    """One-sided exact Wilcoxon signed-rank p-value (small n, ties dropped).

    Enumerates all 2^n sign assignments of the observed |d| ranks under H0.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    # Midranks: tied |d| values must share the average of the ranks they span.
    # Assigning distinct integer ranks to ties (the previous behaviour) biases
    # the statistic whenever two absolute deltas coincide.
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        midrank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    count = 0
    for signs in itertools.product([0, 1], repeat=n):
        stat = sum(ranks[i] for i in range(n) if signs[i])
        if stat >= w_plus:
            count += 1
    return count / (2 ** n)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values for a family of tests.

    Sorts ascending, multiplies the k-th smallest by (m - k), then enforces
    monotonicity and caps at 1.  Returned in the original input order.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted


def cohens_dz(deltas: list[float]) -> float | None:
    n = len(deltas)
    if n < 2:
        return None
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return math.inf if mean > 0 else (0.0 if mean == 0 else -math.inf)
    return mean / sd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path,
                        default=REPO / "results/aggregates/paired_main_runs.csv")
    parser.add_argument("--out-prefix", type=Path,
                        default=REPO / "results/aggregates/significance_tests")
    args = parser.parse_args()

    with args.runs.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    datasets = sorted({row["dataset"] for row in rows})
    out_rows = []
    for dataset in datasets:
        subset = [r for r in rows if r["dataset"] == dataset]
        for column, label in METRICS.items():
            deltas = [float(r[column]) for r in subset]
            n = len(deltas)
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            n_tie = sum(1 for d in deltas if d == 0.0)
            dz = cohens_dz(deltas)
            out_rows.append({
                "dataset": dataset,
                "metric": label,
                "n_pairs": n,
                "n_favor_epall": n_pos,
                "n_favor_original": n_neg,
                "n_ties": n_tie,
                "mean_delta": round(sum(deltas) / n, 6) if n else "",
                "cohens_dz": ("inf" if dz == math.inf else
                              ("" if dz is None else round(dz, 4))),
                "wilcoxon_exact_p_onesided": round(exact_wilcoxon_p(deltas), 6),
                "sign_exact_p_onesided": round(exact_sign_test_p(deltas), 6),
                "min_attainable_p": round(1 / (2 ** max(1, n - n_tie)), 6),
            })

    # Holm-Bonferroni over the whole family of Wilcoxon tests (all datasets x
    # metrics).  Reported alongside the raw p so the manuscript can state which
    # improvements survive multiplicity correction.
    raw = [r["wilcoxon_exact_p_onesided"] for r in out_rows]
    for row, adj in zip(out_rows, holm_adjust(raw)):
        row["wilcoxon_holm_p"] = round(adj, 6)
        row["survives_holm_05"] = "yes" if adj <= 0.05 else "no"

    csv_path = args.out_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "# Paired Significance Tests (EPALL vs PALL-Original)",
        "",
        "Exact one-sided tests on matched schedule seeds. Positive deltas favour EPALL.",
        "",
        "**Small-sample interpretation.** The attainable exact p-value is discrete",
        "and depends on non-tied pairs. These tests document direction consistency",
        "and effect size; the seed count was fixed independently of the outcome.",
        "",
        "| Dataset | Metric | Pairs | Favour EPALL | Ties | Mean delta | Cohen's d_z | Wilcoxon p | Holm p | Survives Holm | Sign p | Min attainable p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in out_rows:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['n_pairs']} | "
            f"{row['n_favor_epall']} | {row['n_ties']} | {row['mean_delta']} | "
            f"{row['cohens_dz']} | {row['wilcoxon_exact_p_onesided']} | "
            f"{row['wilcoxon_holm_p']} | {row['survives_holm_05']} | "
            f"{row['sign_exact_p_onesided']} | {row['min_attainable_p']} |"
        )
    md_path = args.out_prefix.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path.name} and {md_path.name} ({len(out_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
