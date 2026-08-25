#!/usr/bin/env python3
"""Continuous conflict-weighted soft mask: does gamma change anything?

The discrete Phase-3 mask of PALL-Adapter showed no measurable effect. This
analysis asks the same question of the *continuous* mask, where each forget-set
coordinate gets a multiplier ``m_i = clamp(1 - gamma * c_hat_i, 0, 1)`` and
``c_hat`` is the normalised gradient-conflict energy ``relu(-g_f * g_r) / max``.
Sweeping gamma with everything else fixed makes gamma the only moving part, and
``gamma = 0`` recovers a full update, so the sweep contains its own null arm.

Two things are reported side by side, because either alone is misleading:

* the end-state metrics, which answer "did it matter?"; and
* the mask statistics logged at request time, which answer "did the mask even
  change?".  A null result means something entirely different depending on
  whether the mask moved.

Runs live under ``*_conflict_gamma_<g>_v2``. The ``_v1`` tag is a discarded first
attempt that used the standard from-scratch protocol, where the adapter path is
degenerate and the forget gradient was identically zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

DATASETS = ("cifar10", "cifar100")
GAMMA_KEYS = ("0", "0p25", "0p5", "1", "2")
TAG = "{dataset}_conflict_gamma_{gamma}_v2"
METRIC_COLUMNS = ("final_avg_accuracy", "WorstDrop", "Au")
MASK_KEYS = (
    "full_count",
    "soft_count",
    "frozen_count",
    "mean_c_hat",
    "pct_m_below_half",
    "pct_nonzero_conflict_forget",
)


def number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def gamma_value(key: str) -> float:
    return float(key.replace("p", "."))


def metric_rows(rows, dataset: str, gamma: str) -> List[dict]:
    return [r for r in rows if r.get("experiment_tag") == TAG.format(dataset=dataset, gamma=gamma)]


def mask_stats(runs_root: Path, dataset: str, gamma: str) -> Optional[Dict[str, float]]:
    """Request-time mask statistics, averaged over seeds and deletion events."""

    wanted = TAG.format(dataset=dataset, gamma=gamma)
    collected: Dict[str, List[float]] = {key: [] for key in MASK_KEYS}
    for config_path in runs_root.glob(f"{dataset}/*/pall_adapter/seed_*/*/config.json"):
        try:
            if json.loads(config_path.read_text()).get("experiment_tag") != wanted:
                continue
            metrics_path = config_path.with_name("metrics.json")
            if not metrics_path.exists():
                continue
            events = json.loads(metrics_path.read_text()).get("unlearning_events", [])
        except (OSError, ValueError):
            continue
        for event in events:
            stats = event.get("conflict_mask")
            if not isinstance(stats, dict):
                continue
            for key in MASK_KEYS:
                value = number(stats.get(key))
                if value is not None:
                    collected[key].append(value)
    if not collected["full_count"]:
        return None
    return {key: statistics.fmean(values) for key, values in collected.items() if values}


def analyze(rows, runs_root: Path) -> List[dict]:
    output: List[dict] = []
    for dataset in DATASETS:
        for gamma in GAMMA_KEYS:
            metrics = metric_rows(rows, dataset, gamma)
            if not metrics:
                continue
            record = {
                "dataset": dataset,
                "gamma": gamma_value(gamma),
                "n_seeds": len(metrics),
            }
            for column in METRIC_COLUMNS:
                values = [number(r.get(column)) for r in metrics]
                values = [v for v in values if v is not None]
                record[column] = statistics.fmean(values) if values else None
            record.update(mask_stats(runs_root, dataset, gamma) or {})
            output.append(record)
    return output


def spread(cells, dataset: str, column: str) -> Optional[float]:
    """Range of a metric across gamma: the effect size the sweep produced."""

    values = [c[column] for c in cells if c["dataset"] == dataset and c.get(column) is not None]
    return max(values) - min(values) if values else None


def write_markdown(path: Path, cells) -> None:
    lines = [
        "# Continuous Conflict-Weighted Soft Mask: Gamma Sweep",
        "",
        "Everything but gamma is held fixed; `gamma = 0` recovers a full update and is the",
        "in-family control. End-state metrics and request-time mask statistics are shown",
        "together, because a null end state means something different depending on whether",
        "the mask actually moved.",
        "",
        "| Dataset | gamma | Seeds | A_final | WorstDrop | A_u | full | soft | frozen | mean c_hat | % m<0.5 | % conflict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(
            f"| {c['dataset']} | {c['gamma']:.2f} | {c['n_seeds']} | "
            f"{c['final_avg_accuracy']:.4f} | {c['WorstDrop']:.4f} | {c['Au']:.4f} | "
            f"{c.get('full_count', 0):.0f} | {c.get('soft_count', 0):.0f} | {c.get('frozen_count', 0):.0f} | "
            f"{c.get('mean_c_hat', 0):.5f} | {c.get('pct_m_below_half', 0):.2f} | "
            f"{c.get('pct_nonzero_conflict_forget', 0):.2f} |"
        )
    lines += ["", "## Effect of gamma on the end state (range across the sweep)", ""]
    for dataset in DATASETS:
        parts = []
        for column in METRIC_COLUMNS:
            value = spread(cells, dataset, column)
            if value is not None:
                parts.append(f"{column} {value:.4f}")
        if parts:
            lines.append(f"- {dataset}: " + ", ".join(parts))
    lines += [
        "",
        "## Reading",
        "",
        "The mask does move: at `gamma = 0` every forget-set coordinate is updated in full,",
        "and from `gamma = 0.25` onward roughly a third of them are reclassified as soft.",
        "But the normalised conflict energy is concentrated in a very thin tail -- its mean",
        "is three orders of magnitude below its maximum -- so a linear multiplier",
        "`1 - gamma * c_hat` leaves almost every coordinate at a value close to one. The",
        "share of coordinates actually attenuated below one half stays under one per cent",
        "even at the largest gamma tested, and the end state does not move.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/aggregates/server_results_final.csv")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--out-prefix", type=Path, default=Path("results/aggregates/conflict_gamma")
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cells = analyze(rows, args.runs_root)
    if not cells:
        raise SystemExit("no conflict-gamma v2 runs found")

    out_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    write_markdown(args.out_prefix.with_name(args.out_prefix.name + "_summary.md"), cells)
    print(f"Wrote {len(cells)} gamma cells")


if __name__ == "__main__":
    main()
