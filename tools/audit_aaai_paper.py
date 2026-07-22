#!/usr/bin/env python3
"""Read-only, strict numeric audit for the AAAI manuscript tables.

The main standard comparison is regenerated in memory from the canonical
``server_report_table.csv`` and compared byte-for-byte with the included file.
The three hand-written endpoint tables are then checked row-by-row against
strictly selected aggregate rows.  Adapter component provenance is additionally
checked against every source run's experiment tag so stale v2 diagnostics cannot
silently pass the final-paper audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from generate_main_standard_table import read_rows, render_table, select_rows  # noqa: E402


EXPECTED_COMPONENT_TAG = "adapter_components_pretrained_imagenetnorm_rngneutral_v3"
METRICS = (
    ("final_avg_acc_mean", "final_avg_acc_std"),
    ("WorstDrop_mean", "WorstDrop_std"),
    ("Au_mean", "Au_std"),
)


def fail(message: str) -> None:
    raise ValueError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    matches = [
        row for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in conditions.items())
    ]
    if len(matches) != 1:
        fail(f"expected one row for {conditions}; found {len(matches)}")
    return matches[0]


def table_block(tex: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    label_at = tex.find(marker)
    if label_at < 0:
        fail(f"missing table label {label}")
    begin = max(tex.rfind(r"\begin{table}", 0, label_at), tex.rfind(r"\begin{table*}", 0, label_at))
    if begin < 0:
        fail(f"cannot find table start for {label}")
    end_plain = tex.find(r"\end{table}", label_at)
    end_star = tex.find(r"\end{table*}", label_at)
    ends = [value for value in (end_plain, end_star) if value >= 0]
    if not ends:
        fail(f"cannot find table end for {label}")
    end = min(ends)
    return tex[begin:end]


def numeric_tokens(line: str) -> list[float]:
    normalized = line.replace("$-$", "-").replace(r"\!", "")
    return [float(value) for value in re.findall(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\.\d+)", normalized)]


def row_line(block: str, dataset_marker: str, next_marker: str | None, label: str) -> str:
    start = block.find(dataset_marker)
    if start < 0:
        fail(f"missing dataset marker {dataset_marker!r}")
    end = block.find(next_marker, start + len(dataset_marker)) if next_marker else len(block)
    section = block[start:end if end >= 0 else len(block)]
    for line in section.splitlines():
        if label in line and "&" in line and r"\\" in line:
            return line
    fail(f"missing row {label!r} under {dataset_marker!r}")


def expected_triplets(row: dict[str, str]) -> list[float]:
    output: list[float] = []
    for mean_key, std_key in METRICS:
        output.extend((float(row[mean_key]), float(row[std_key])))
    return output


def assert_values(context: str, line: str, expected: list[float]) -> None:
    found = numeric_tokens(line)
    if len(found) != len(expected):
        fail(f"{context}: expected {len(expected)} numeric tokens, found {len(found)} in {line!r}")
    for index, (actual, target) in enumerate(zip(found, expected), start=1):
        if abs(actual - target) > 5e-5:
            fail(f"{context}: token {index} is {actual:.4f}, expected {target:.4f}")


def audit_generated_main(tex: str, report_path: Path, generated_path: Path) -> list[str]:
    if r"\input{generated/main_standard_comparison.tex}" not in tex:
        fail("main manuscript does not input generated/main_standard_comparison.tex")
    rows = read_rows(report_path)
    selected = select_rows(rows)
    expected = render_table(selected, report_path)
    actual = generated_path.read_text(encoding="utf-8")
    if actual != expected:
        fail("generated Table 1 is stale; rerun tools/generate_main_standard_table.py")
    return [
        f"Table 1: {dataset}/{method} <- {row['experiment_tag']} [{row['config_id']}]"
        for (dataset, method), row in sorted(selected.items())
    ]


def audit_pretrained(tex: str, thesis_rows: list[dict[str, str]]) -> list[str]:
    block = table_block(tex, "tab:pretrained")
    specs = (
        ("cifar10", "C10", "C100", "lora", "LoRA", "imagenet"),
        ("cifar10", "C10", "C100", "pall_adapter", "Adapter", "imagenet"),
        ("cifar100", "C100", "Tiny-IN", "lora", "LoRA", "imagenet"),
        ("cifar100", "C100", "Tiny-IN", "pall_adapter", "Adapter", "imagenet"),
        ("tinyimagenet", "Tiny-IN", None, "lora", "LoRA", "imagenet_equivalent"),
        ("tinyimagenet", "Tiny-IN", None, "pall_adapter", "Adapter", "imagenet_equivalent"),
    )
    provenance = []
    for dataset, marker, next_marker, method, label, norm in specs:
        row = one(
            thesis_rows,
            dataset=dataset,
            method=method,
            experiment_tag={"cifar10": "cifar10_pretrained", "cifar100": "cifar100_pretrained", "tinyimagenet": "tiny_pretrained"}[dataset],
            pretrained_input_norm=norm,
        )
        line = row_line(block, marker, next_marker, label)
        assert_values(f"Table 2 {dataset}/{method}", line, expected_triplets(row))
        provenance.append(f"Table 2: {dataset}/{method} <- {row['experiment_tag']} norm={norm}")
    return provenance


def audit_components(
    tex: str,
    summary_path: Path,
    runs_path: Path,
    expected_tag: str,
) -> list[str]:
    summary = read_csv(summary_path)
    runs = read_csv(runs_path)
    expected_keys = {(dataset, seed, mode) for dataset in ("cifar10", "cifar100") for seed in (0, 1, 2) for mode in (
        "reset_only", "reset_repair", "uniform_unprotected", "mask_no_ascent", "full"
    )}
    actual_keys = {(row["dataset"], int(row["seed"]), row["mode"]) for row in runs}
    if actual_keys != expected_keys:
        fail(f"component run matrix mismatch: expected 30 exact keys, found {len(actual_keys)}")
    for row in runs:
        config_path = REPO / row["source_run"] / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("experiment_tag") != expected_tag:
            fail(
                f"component provenance {config_path} has tag={config.get('experiment_tag')!r}; "
                f"expected {expected_tag!r}"
            )

    block = table_block(tex, "tab:adapter-components")
    labels = {
        "reset_only": "Reset only",
        "reset_repair": "Reset+repair",
        "uniform_unprotected": "Uniform",
        "mask_no_ascent": "Mask/no-ascent",
        "full": "Full",
    }
    provenance = []
    for dataset, marker, next_marker in (("cifar10", "C10", "C100"), ("cifar100", "C100", None)):
        for mode, label in labels.items():
            row = one(summary, dataset=dataset, mode=mode)
            expected = [
                float(row["A_final_mean"]), float(row["A_final_std"]),
                float(row["WorstDrop_mean"]), float(row["WorstDrop_std"]),
                float(row["Au_distance_to_chance_mean"]), float(row["Au_distance_to_chance_std"]),
            ]
            line = row_line(block, marker, next_marker, label)
            assert_values(f"Table 3 {dataset}/{mode}", line, expected)
            provenance.append(f"Table 3: {dataset}/{mode} <- {expected_tag}, seeds 0/1/2")
    return provenance


def audit_overlap_heavy(tex: str, thesis_rows: list[dict[str, str]]) -> list[str]:
    block = table_block(tex, "tab:pall_overlap_heavy")
    labels = {
        "pall_original": "PALL-Original",
        "pall_modified": "PALL-Modified",
        "pall_adapter": "PALL-Adapter",
    }
    provenance = []
    for dataset, marker, next_marker in (("cifar10", "CIFAR-10", "CIFAR-100"), ("cifar100", "CIFAR-100", None)):
        for method, label in labels.items():
            row = one(thesis_rows, dataset=dataset, method=method, experiment_tag=f"{dataset}_main")
            line = row_line(block, marker, next_marker, label)
            assert_values(f"Table 4 {dataset}/{method}", line, expected_triplets(row))
            provenance.append(f"Table 4: {dataset}/{method} <- {row['experiment_tag']}")
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=REPO / "paper/AuthorKit27/OverlapAwareUnlearning.tex")
    parser.add_argument("--report", type=Path, default=REPO / "results/aggregates/server_report_table.csv")
    parser.add_argument("--thesis", type=Path, default=REPO / "results/aggregates/server_thesis_table.csv")
    parser.add_argument("--generated", type=Path, default=REPO / "paper/AuthorKit27/generated/main_standard_comparison.tex")
    parser.add_argument("--component-summary", type=Path, default=REPO / "results/aggregates/adapter_components_summary.csv")
    parser.add_argument("--component-runs", type=Path, default=REPO / "results/aggregates/adapter_components_runs.csv")
    parser.add_argument("--component-tag", default=EXPECTED_COMPONENT_TAG)
    args = parser.parse_args()

    try:
        tex = args.tex.read_text(encoding="utf-8")
        thesis_rows = read_csv(args.thesis)
        provenance = []
        provenance.extend(audit_generated_main(tex, args.report, args.generated))
        provenance.extend(audit_pretrained(tex, thesis_rows))
        provenance.extend(audit_components(tex, args.component_summary, args.component_runs, args.component_tag))
        provenance.extend(audit_overlap_heavy(tex, thesis_rows))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[FAIL] AAAI paper audit: {exc}", file=sys.stderr)
        return 1

    print(f"[PASS] AAAI paper audit: {len(provenance)} table rows/cells traced")
    print("\n".join(provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
