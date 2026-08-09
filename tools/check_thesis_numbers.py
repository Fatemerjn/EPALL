#!/usr/bin/env python3
"""Verify every numeric cell typeset in the thesis against the aggregate results.

The thesis externalises each table into ``thesis/tables/<name>.tex`` and pulls it
in with ``\\input``.  This checker parses those generated files directly, so it
audits what is actually typeset rather than a hand-maintained copy of the
numbers.  Each table is bound to the aggregate artifact it was derived from:
either a Markdown summary under ``results/aggregates/`` or a row subset of
``server_thesis_table.csv`` / ``server_report_table.csv``.

A value passes when it equals some source value rounded to the precision at
which the thesis displays it.  Structural constants that legitimately appear in
a table but not in the source (seed counts, bottleneck widths, chance levels)
are listed per table in ``SKIP`` and are the only values exempt from the check.

A second pass scans the prose of ``thesis/chapters/*.tex`` for inline decimals
and reports any that do not appear in a parsed table, which catches numbers that
drifted out of sync with a regenerated table.

Read-only: this tool reports; it never edits the thesis.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
AGGREGATES = ROOT / "results" / "aggregates"
TABLES = ROOT / "thesis" / "tables"
CHAPTERS = ROOT / "thesis" / "chapters"

# Columns of server_thesis_table.csv that a thesis table may quote.
CSV_VALUE_COLUMNS = (
    "final_avg_acc_mean",
    "final_avg_acc_std",
    "avg_forgetting_mean",
    "WorstDrop_mean",
    "WorstDrop_std",
    "Au_mean",
    "Au_std",
    "updated_param_ratio_mean",
)

MAIN_METHODS = (
    "clpu",
    "derpp",
    "er",
    "ewc",
    "lwf",
    "ssd",
    "salun",
    "pall_original",
    "pall_modified",
    "lora",
    "pall_adapter",
)
MAIN_COLUMNS = ("final_avg_acc", "avg_forgetting", "WorstDrop", "Au")


class Report:
    def __init__(self) -> None:
        self.passes = 0
        self.failures: List[str] = []

    def ok(self) -> None:
        self.passes += 1

    def fail(self, message: str) -> None:
        self.failures.append(message)


# --------------------------------------------------------------------------- #
# LaTeX parsing                                                                #
# --------------------------------------------------------------------------- #
def table_cells(name: str) -> List[str]:
    """Every ``\\lr{...}`` payload in a table file, from the first rule onwards.

    Numeric cells in the Persian tables are always wrapped in ``\\lr`` so they
    typeset left-to-right, which makes the wrapper a reliable cell marker.
    """

    text = (TABLES / f"{name}.tex").read_text(encoding="utf-8")
    text = text[text.index("\\midrule"):]
    return [cell.replace("$-$", "-").strip() for cell in re.findall(r"\\lr\{([^{}]*)\}", text)]


def table_numbers(name: str) -> List[float]:
    return [
        float(match.group())
        for cell in table_cells(name)
        for match in re.finditer(r"-?\d+\.\d+", cell)
    ]


# --------------------------------------------------------------------------- #
# Sources                                                                      #
# --------------------------------------------------------------------------- #
def numbers_in_file(path: Path) -> List[float]:
    return [float(m.group()) for m in re.finditer(r"-?\d+\.\d+", path.read_text(encoding="utf-8"))]


def matches(value: float, source: Sequence[float]) -> bool:
    """True when some source value rounds to the displayed value."""

    decimals = len(str(value).split(".")[1])
    return any(abs(round(candidate, decimals) - value) < 1e-9 for candidate in source)


# Tables whose every number lives in one aggregate artifact, with the structural
# constants that are legitimately absent from that artifact.
MARKDOWN_SOURCES: Dict[str, tuple[Path, tuple[float, ...]]] = {
    "res_paired": (AGGREGATES / "paired_main_summary.md", (8.0,)),
    "res_significance": (AGGREGATES / "significance_tests.md", (8.0,)),
    "res_components": (AGGREGATES / "modified_components_summary.md", ()),
    "res_components_paired": (AGGREGATES / "modified_components_summary.md", ()),
    "res_adapter_components": (AGGREGATES / "adapter_components_summary.md", ()),
    "res_mia": (AGGREGATES / "corrected_mia.md", (3.0, 0.5)),
    "res_retrain": (AGGREGATES / "retraining_reference.md", (0.0, 1.0)),
    "res_storage": (
        AGGREGATES / "storage_accounting_summary.md",
        # The thesis merges masks + sparse backups into one column; the merged
        # totals are checked separately in check_storage_merge.
        (4.0, 8.0, 45.99, 169.29),
    ),
    "res_overlap_sparsity": (
        AGGREGATES / "overlap_sparsity_summary.md",
        # sparsity levels and the seed count are structural, not results
        (0.5, 0.6, 0.7, 0.8, 0.9),
    ),
    "res_persistence": (
        AGGREGATES / "forgetting_persistence" / "PERSISTENCE_AUDIT.md",
        (),
    ),
    # The reinit arm lives under its own experiment tag and is aggregated
    # separately, so this table is bound to the anchor analysis rather than to
    # the canonical result CSVs.
    "res_anchor": (AGGREGATES / "anchor_paired_summary.md", (3.0, 6.0, 7.0, 2.0, 8.0)),
    "res_slopes": (
        ROOT / "results" / "thesis" / "plots_v2" / "overlap_response_slopes.md",
        (11.0, 12.0, 14.0, 86.0, 92.0, 95.0),
    ),
}


def check_markdown_tables(report: Report) -> None:
    for name, (source_path, skip) in MARKDOWN_SOURCES.items():
        if not source_path.exists():
            report.fail(f"{name}: source artifact missing: {source_path}")
            continue
        source = numbers_in_file(source_path)
        for value in table_numbers(name):
            if value in skip:
                continue
            if matches(value, source):
                report.ok()
            else:
                report.fail(f"{name}: {value} not found in {source_path.name}")


def check_storage_merge(report: Report) -> None:
    """The merged storage column must equal masks + sparse backups."""

    path = AGGREGATES / "storage_accounting_summary.md"
    if not path.exists():
        report.fail(f"res_storage: source artifact missing: {path}")
        return
    text = path.read_text(encoding="utf-8")
    for dataset, expected in (("cifar10", 45.99), ("cifar100", 169.29)):
        rows = [
            line
            for line in text.splitlines()
            if line.startswith(f"| {dataset}") and "pall_modified" in line
        ]
        if not rows:
            report.fail(f"res_storage {dataset}: no pall_modified row in the source")
            continue
        fields = [field.strip() for field in rows[0].split("|")]
        masks, backups = float(fields[6]), float(fields[7])
        if abs(masks + backups - expected) < 0.01:
            report.ok()
        else:
            report.fail(f"res_storage {dataset}: {masks}+{backups} != {expected}")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


CSV_SUBSETS: Dict[str, tuple[Callable[[Dict[str, str]], bool], tuple[float, ...]]] = {
    "res_pretrained": (
        lambda row: row["experiment_tag"]
        in ("cifar10_pretrained", "cifar100_pretrained", "tiny_pretrained"),
        (),
    ),
    "res_overlap_heavy": (
        lambda row: row["experiment_tag"]
        in (
            "cifar10_main",
            "cifar100_main",
            "tiny_e3_original_v1",
            "tiny_e3_modified_v1",
            "tiny_e3_adapter_v1",
        ),
        (3.0,),
    ),
    "res_overlap_levels": (
        lambda row: row["experiment_tag"].startswith("overlap_curve_v1_"),
        (),
    ),
    "res_conflict": (
        lambda row: row["experiment_tag"]
        in ("conflict_ablation_v1", "adapter_bottleneck_ablation_v1"),
        (),
    ),
    "res_vit": (
        lambda row: row["experiment_tag"] in ("cifar10_vit_v1", "cifar100_vit_v1"),
        (),
    ),
    "res_bottleneck": (
        lambda row: row["experiment_tag"] == "adapter_bottleneck_ablation_v1",
        (4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    ),
}


def check_csv_tables(report: Report, rows: Sequence[Dict[str, str]]) -> None:
    for name, (predicate, skip) in CSV_SUBSETS.items():
        source: List[float] = []
        for row in rows:
            if not predicate(row):
                continue
            for column in CSV_VALUE_COLUMNS:
                try:
                    source.append(float(row[column]))
                except (KeyError, ValueError):
                    continue
        if not source:
            report.fail(f"{name}: no source rows matched the CSV subset")
            continue
        for value in table_numbers(name):
            if value in skip:
                continue
            if matches(value, source):
                report.ok()
            else:
                report.fail(f"{name}: {value} not in its CSV subset")


def check_main_tables(report: Report, report_rows: Sequence[Dict[str, str]], seeds: int) -> None:
    """The two headline tables are checked cell by cell, not as a value pool.

    Every ``mean ± sd`` cell must come from the matching dataset/method/tag row,
    and the source row must carry the full seed count the thesis claims.
    """

    def source_cell(dataset: str, method: str, column: str):
        tag = (
            "standard_unlearning_ssd_salun_v1"
            if method in ("ssd", "salun")
            else f"{dataset}_standard"
        )
        for row in report_rows:
            if (
                row["dataset"] == dataset
                and row["method"] == method
                and row["experiment_tag"] == tag
            ):
                return row[column], row["n_seeds"]
        return None, None

    for dataset, name in (("cifar10", "res_main_cifar10"), ("cifar100", "res_main_cifar100")):
        numeric = [cell for cell in table_cells(name) if re.search(r"\d", cell)]
        values = [cell for cell in numeric if "±" in cell]
        timings = [cell for cell in numeric if "±" not in cell]
        expected = len(MAIN_METHODS) * len(MAIN_COLUMNS)
        if len(values) != expected:
            report.fail(f"{name}: expected {expected} mean±sd cells, got {len(values)}")
            continue
        for method_index, method in enumerate(MAIN_METHODS):
            for column_index, column in enumerate(MAIN_COLUMNS):
                cell = values[method_index * len(MAIN_COLUMNS) + column_index]
                source, n_seeds = source_cell(dataset, method, column)
                if source is None:
                    report.fail(f"{name}/{method}/{column}: no source row")
                    continue
                if n_seeds != str(seeds):
                    report.fail(
                        f"{name}/{method}: source n_seeds={n_seeds}, thesis claims {seeds}"
                    )
                    continue
                got = [float(x) for x in re.findall(r"-?\d+\.\d+", cell)]
                want = [float(x) for x in re.findall(r"-?\d+\.\d+", source)]
                if len(got) == 2 and len(want) == 2 and all(
                    abs(a - b) < 5e-5 for a, b in zip(got, want)
                ):
                    report.ok()
                else:
                    report.fail(f"{name}/{method}/{column}: '{cell}' vs source '{source}'")
        if len(timings) != len(MAIN_METHODS):
            report.fail(f"{name}: expected {len(MAIN_METHODS)} timing cells, got {len(timings)}")
            continue
        for method_index, method in enumerate(MAIN_METHODS):
            source, _ = source_cell(dataset, method, "t_forget_total_mean")
            if source is None:
                report.fail(f"{name}/{method}/T_f: no source row")
                continue
            if abs(float(timings[method_index]) - float(source)) < 5e-4:
                report.ok()
            else:
                report.fail(f"{name}/{method}/T_f: {timings[method_index]} vs {source}")


# --------------------------------------------------------------------------- #
# Prose scan                                                                   #
# --------------------------------------------------------------------------- #
def all_table_numbers() -> set[float]:
    values: set[float] = set()
    for path in sorted(TABLES.glob("*.tex")):
        values.update(table_numbers(path.stem))
    return values


def check_prose(report: Report, quiet: bool) -> List[str]:
    """Report prose decimals of table precision that match no typeset cell.

    Only four-decimal numbers are considered: that is the precision the result
    tables use, so a four-decimal number in the text is almost always a quoted
    table cell. Coarser numbers (hyperparameters, ratios, page counts) are not
    table quotations and would only produce noise.
    """

    # Prose usually quotes a table cell at lower precision than the table prints
    # it (0.0469 for a typeset 0.046872), and sometimes quotes an aggregate value
    # the table does not display at all, so both pools are matched with rounding.
    known = sorted(all_table_numbers())
    known += numbers_in_file(AGGREGATES / "server_thesis_table.csv")
    signed = known + [-value for value in known]
    orphans: List[str] = []
    for path in sorted(CHAPTERS.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in re.finditer(r"-?\d+\.\d{4}\b", line):
                value = float(match.group())
                if matches(value, signed):
                    continue
                orphans.append(f"{path.name}:{line_number}: {match.group()} matches no table cell")
    if not quiet:
        for orphan in orphans:
            print(f"  ORPHAN {orphan}")
    return orphans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Seed count the headline tables claim (default: 8).",
    )
    parser.add_argument(
        "--strict-prose",
        action="store_true",
        help="Treat unmatched prose decimals as failures instead of warnings.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-orphan output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()

    check_markdown_tables(report)
    check_storage_merge(report)
    check_csv_tables(report, load_csv(AGGREGATES / "server_thesis_table.csv"))
    check_main_tables(report, load_csv(AGGREGATES / "server_report_table.csv"), args.seeds)
    orphans = check_prose(report, args.quiet)

    print(f"PASS: {report.passes}")
    print(f"FAIL: {len(report.failures)}")
    for failure in report.failures:
        print(f"  - {failure}")
    print(f"PROSE UNMATCHED: {len(orphans)}")

    if report.failures:
        return 1
    if args.strict_prose and orphans:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
