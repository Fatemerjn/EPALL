#!/usr/bin/env python3
"""
Cross-check every numeric cell in the LaTeX tables of thesis/chapters/results.tex
against the aggregate CSVs (results/aggregates/server_thesis_table.csv, with
server_report_table.csv available as a corroborating source).

For each audited table cell it prints PASS/FAIL with: table label, row, column,
the thesis value, the source CSV value, and the source-row identifier. Rows are
matched by dataset + method + experiment_tag and, where those still collide, by
the make_thesis_table CONFIG_GROUP_COLUMNS (e.g. adapter_shared_forget_ratio /
adapter_shared_protect_ratio for the adapter ablation). A cell whose CSV lookup
is empty or ambiguous is FAIL ("no unique match"), never silently skipped.

It also scans the prose paragraphs (everything outside table/figure/landscape
environments) for inline decimal numbers that repeat table values and flags any
that do not match a cell in the parsed tables.

Read-only: this tool reports; it never edits results.tex.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# make_thesis_table's CONFIG_GROUP_COLUMNS is what disambiguates rows that share
# dataset+method; we import it so this tool tracks the canonical list.
try:
    from make_thesis_table import CONFIG_GROUP_COLUMNS
except ImportError:  # pragma: no cover - support `python tools/check_thesis_numbers.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_thesis_table import CONFIG_GROUP_COLUMNS

TEX_PATH = Path("thesis/chapters/results.tex")
THESIS_CSV = Path("results/aggregates/server_thesis_table.csv")
REPORT_CSV = Path("results/aggregates/server_report_table.csv")
PRIVACY_CSV = Path("results/aggregates/privacy_audit.csv")

METHOD_MAP = {
    "PALL-Original": "pall_original",
    "PALL-Modified": "pall_modified",
    "PALL-Adapter": "pall_adapter",
    "CLPU": "clpu",
    "DER++": "derpp",
    "ER": "er",
    "EWC": "ewc",
    "LoRA": "lora",
    "LwF": "lwf",
}
DATASET_MAP = {"CIFAR-10": "cifar10", "CIFAR-100": "cifar100", "TinyImageNet": "tinyimagenet"}


# --------------------------------------------------------------------------- #
# LaTeX parsing helpers                                                        #
# --------------------------------------------------------------------------- #
def strip_latex(cell: str) -> str:
    s = cell.strip()
    s = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", s)
    prev = None
    while prev != s:  # unwrap nested \textbf{...}, \lr{...}, ...
        prev = s
        s = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)  # drop bare control words (e.g. \toprule leftovers)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def _skip_braces(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def table_body(text: str, label: str) -> str:
    i = text.index("\\label{" + label + "}")
    b = text.index("\\begin{tabular}", i)
    arg = text.index("{", b + len("\\begin{tabular}"))
    start = _skip_braces(text, arg)
    end = text.index("\\end{tabular}", start)
    return text[start:end]


def data_rows(body: str) -> List[List[str]]:
    """Return the data rows of a tabular body as lists of raw cell strings."""
    rows: List[List[str]] = []
    for chunk in body.split("\\\\"):
        # drop rule macros; a row with no '&' and no digits is structural
        stripped = chunk.strip()
        if not stripped:
            continue
        content = re.sub(r"\\(top|mid|bottom|c)rule(\[[^\]]*\])?(\{[^}]*\})?", "", stripped).strip()
        if not content:
            continue
        cells = re.split(r"(?<!\\)&", content)
        rows.append(cells)
    return rows


def numbers_in(cell: str) -> List[str]:
    """Ordered numeric literals in a cell (keeps decimal places as written)."""
    return re.findall(r"-?\d+\.\d+|-?\d+", strip_latex(cell))


# --------------------------------------------------------------------------- #
# Numeric comparison                                                           #
# --------------------------------------------------------------------------- #
def values_match(thesis_str: str, csv_str: Optional[str]) -> Tuple[bool, str]:
    if csv_str is None or str(csv_str).strip() == "":
        return False, "<empty>"
    try:
        cv = float(csv_str)
    except (TypeError, ValueError):
        return False, str(csv_str)
    ndec = len(thesis_str.split(".")[1]) if "." in thesis_str else 0
    tv = float(thesis_str)
    tol = 0.5 * (10 ** (-ndec)) + 1e-9
    return abs(cv - tv) <= tol, f"{cv:.{ndec}f}"


# --------------------------------------------------------------------------- #
# CSV lookup                                                                   #
# --------------------------------------------------------------------------- #
class CsvTable:
    def __init__(self, path: Path):
        self.path = path
        with path.open() as handle:
            self.rows = list(csv.DictReader(handle))

    def find(self, **filters) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]], str]:
        matches = []
        for row in self.rows:
            if all(str(row.get(k, "")).strip() == str(v).strip() for k, v in filters.items()):
                matches.append(row)
        ident = self.path.name + "[" + ", ".join(f"{k}={v}" for k, v in filters.items()) + "]"
        return (matches[0] if len(matches) == 1 else None), matches, ident


def no_match_reason(matches: List[Dict[str, str]]) -> str:
    """Actionable message when a CSV lookup is not exactly one row."""
    if not matches:
        return "no match: 0 CSV rows"
    cols = [c for c in CONFIG_GROUP_COLUMNS if len({str(m.get(c, "")) for m in matches}) > 1]
    detail = ("; differ in " + ", ".join(cols)) if cols else ""
    return f"no unique match: {len(matches)} rows{detail}"


# --------------------------------------------------------------------------- #
# Audit record                                                                 #
# --------------------------------------------------------------------------- #
class Record:
    __slots__ = ("table", "row", "column", "thesis", "csv", "source", "status")

    def __init__(self, table, row, column, thesis, csv_val, source, status):
        self.table = table
        self.row = row
        self.column = column
        self.thesis = thesis
        self.csv = csv_val
        self.source = source
        self.status = status


# table_values maps a formatted number -> list of "label | row | column" it appears in.
TABLE_VALUES: Dict[str, List[str]] = {}


def register_value(num_str: str, where: str) -> None:
    TABLE_VALUES.setdefault(num_str, []).append(where)


def audit_metric_cell(records, table, row_label, col_label, cell, csv_row, mean_field,
                      std_field, source, reason):
    """Compare one tex cell (mean or mean±std) against CSV field(s).

    ``reason`` is None when the CSV row is unique; otherwise it is the
    ``no_match_reason`` string and every sub-cell is a FAIL."""
    nums = numbers_in(cell)
    if not nums:
        records.append(Record(table, row_label, col_label, "<missing>",
                              reason or csv_row.get(mean_field, "<empty>"), source, "FAIL"))
        return
    subs = [("mean", mean_field, nums[0])]
    if std_field is not None and len(nums) > 1:
        subs.append(("std", std_field, nums[1]))
    for kind, field, num in subs:
        col = f"{col_label}|{kind}" if std_field is not None else col_label
        register_value(num, f"{table} | {row_label} | {col}")
        if reason is not None:
            records.append(Record(table, row_label, col, num, reason, source, "FAIL"))
            continue
        ok, csv_val = values_match(num, csv_row.get(field))
        records.append(Record(table, row_label, col, num, csv_val, source, "PASS" if ok else "FAIL"))
    if (
        std_field is not None
        and reason is None
        and str(csv_row.get(std_field, "")).strip() != ""
        and len(nums) < 2
    ):
        records.append(Record(table, row_label, f"{col_label}|std", "<missing>",
                              csv_row.get(std_field), source, "FAIL"))


def audit_scalar(records, table, row_label, col_label, thesis_num, csv_value, source):
    """Audit and register one already-extracted scalar value."""
    register_value(thesis_num, f"{table} | {row_label} | {col_label}")
    ok, csv_val = values_match(thesis_num, csv_value)
    records.append(Record(table, row_label, col_label, thesis_num, csv_val, source,
                          "PASS" if ok else "FAIL"))


# --------------------------------------------------------------------------- #
# Per-table audits                                                             #
# --------------------------------------------------------------------------- #
MAIN_METRIC_COLS = [
    ("Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
    ("Avg. Forgetting", "avg_forgetting_mean", "avg_forgetting_std"),
    ("Fu", "Fu_mean", "Fu_std"),
    ("WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
    ("Au", "Au_mean", "Au_std"),
    ("Updated Ratio", "updated_param_ratio_mean", None),
]


def audit_main_pall(text, thesis, records):
    label = "tab:main-pall-results"
    body = table_body(text, label)
    setting_to_tag = {
        ("cifar10", "Scratch"): "cifar10_main",
        ("cifar100", "Scratch"): "cifar100_main",
        ("tinyimagenet", "Scratch"): "tiny_main",
        ("tinyimagenet", "Pretrained"): "tiny_pretrained",
    }
    tiny_scratch_tags = {
        "pall_original": "tiny_e3_original_v1",
        "pall_modified": "tiny_e3_modified_v1",
        "pall_adapter": "tiny_e3_adapter_v1",
    }
    cur_ds = cur_set = None
    for cells in data_rows(body):
        vals = [strip_latex(c) for c in cells]
        if len(vals) < 9:
            continue
        if vals[0] in DATASET_MAP:
            cur_ds = DATASET_MAP[vals[0]]
        if vals[1] in ("Scratch", "Pretrained"):
            cur_set = vals[1]
        method = METHOD_MAP.get(vals[2])
        if method is None or cur_ds is None or cur_set is None:
            continue
        tag = (
            tiny_scratch_tags.get(method)
            if cur_ds == "tinyimagenet" and cur_set == "Scratch"
            else setting_to_tag.get((cur_ds, cur_set))
        )
        row_label = f"{cur_ds}/{cur_set}/{vals[2]}"
        csv_row, matches, source = thesis.find(dataset=cur_ds, method=method, experiment_tag=tag)
        reason = None if len(matches) == 1 else no_match_reason(matches)
        for idx, (col_label, mean_f, std_f) in enumerate(MAIN_METRIC_COLS):
            audit_metric_cell(records, label, row_label, col_label, cells[3 + idx],
                              csv_row or {}, mean_f, std_f, source, reason)


STANDARD_METRIC_COLS = [
    ("Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
    ("Avg. Forgetting", "avg_forgetting_mean", "avg_forgetting_std"),
    ("Fu", "Fu_mean", "Fu_std"),
    ("WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
    ("Au", "Au_mean", "Au_std"),
    ("Forget Time (s)", "t_forget_total_mean", None),
]


def audit_standard(text, thesis, records, label, dataset, tag):
    body = table_body(text, label)
    for cells in data_rows(body):
        vals = [strip_latex(c) for c in cells]
        if len(vals) < 7:
            continue
        method = METHOD_MAP.get(vals[0])
        if method is None:
            continue
        row_label = f"{dataset}/standard/{vals[0]}"
        csv_row, matches, source = thesis.find(dataset=dataset, method=method, experiment_tag=tag)

        # The corrected lr=1e-3 CIFAR-100 LoRA launch exists, but the current
        # aggregates still contain only the invalid, diverged lr=1e-2 pair.
        # Require an explicit non-numeric pending row while that remains true;
        # once a corrected aggregate arrives this check fails and forces the
        # thesis table to be regenerated with real values.
        if dataset == "cifar100" and method == "lora" and not any(
            numbers_in(cell) for cell in cells[1:7]
        ):
            collapsed = (
                len(matches) == 1
                and values_match("0.1000", csv_row.get("final_avg_acc_mean"))[0]
                and values_match("0.0000", csv_row.get("final_avg_acc_std"))[0]
                and values_match("0.1000", csv_row.get("Au_mean"))[0]
            )
            status = "PASS" if collapsed else "FAIL"
            csv_value = "only collapsed lr=1e-2 aggregate" if collapsed else no_match_reason(matches)
            records.append(Record(label, row_label, "corrected-run availability", "pending", csv_value,
                                  source, status))
            continue
        reason = None if len(matches) == 1 else no_match_reason(matches)
        for idx, (col_label, mean_f, std_f) in enumerate(STANDARD_METRIC_COLS):
            audit_metric_cell(records, label, row_label, col_label, cells[1 + idx],
                              csv_row or {}, mean_f, std_f, source, reason)


ABLATION_METRIC_COLS = [
    (3, "Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
    (4, "WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
    (5, "Au", "Au_mean", "Au_std"),
    (6, "Updated Ratio", "updated_param_ratio_mean", None),
    (7, "Critical Ratio", "overlap_shared_critical_ratio", None),
    (8, "Protected Ratio", "overlap_protected_ratio", None),
]


def audit_adapter_ablation(text, thesis, records):
    label = "tab:adapter-ablation"
    body = table_body(text, label)
    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(c) for c in cells]
        if len(vals) < 9:
            continue
        af, ap, steps = vals[0], vals[1], vals[2]
        if not re.match(r"^\d", af) or not re.match(r"^\d", ap) or not re.match(r"^\d", steps):
            continue
        key = (af, ap, steps)
        if key in seen:
            records.append(Record(label, "/".join(key), "row uniqueness", "duplicate",
                                  "one row expected", THESIS_CSV.name, "FAIL"))
        seen.add(key)
        row_label = f"cifar100/pretrained/af={af}/ap={ap}/steps={steps}"
        csv_row, matches, source = thesis.find(
            dataset="cifar100", method="pall_adapter", experiment_tag="adapter_tune_pretrained_v1",
            adapter_shared_forget_ratio=af, adapter_shared_protect_ratio=ap,
            adapter_forget_steps=steps,
        )
        reason = None if len(matches) == 1 else no_match_reason(matches)
        # Audit the sweep coordinates too (they must equal the matched config columns).
        for idx, col_label, field in ((0, "alpha_f", "adapter_shared_forget_ratio"),
                                      (1, "alpha_p", "adapter_shared_protect_ratio"),
                                      (2, "forget_steps", "adapter_forget_steps")):
            register_value(vals[idx], f"{label} | {row_label} | {col_label}")
            if reason is not None:
                records.append(Record(label, row_label, col_label, vals[idx], reason, source, "FAIL"))
            else:
                ok, cv = values_match(vals[idx], csv_row.get(field))
                records.append(Record(label, row_label, col_label, vals[idx], cv, source, "PASS" if ok else "FAIL"))
        for idx, col_label, mean_f, std_f in ABLATION_METRIC_COLS:
            audit_metric_cell(records, label, row_label, col_label, cells[idx],
                              csv_row or {}, mean_f, std_f, source, reason)

    expected = {
        (row["adapter_shared_forget_ratio"], row["adapter_shared_protect_ratio"],
         row["adapter_forget_steps"])
        for row in thesis.rows
        if row.get("experiment_tag") == "adapter_tune_pretrained_v1"
    }
    for af, ap, steps in sorted(expected - seen):
        records.append(Record(label, f"af={af}/ap={ap}/steps={steps}", "row presence",
                              "missing", "aggregate row exists", THESIS_CSV.name, "FAIL"))


ANCHOR_METRIC_COLS = [
    (2, "Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
    (3, "Fu", "Fu_mean", "Fu_std"),
    (4, "WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
    (5, "Au", "Au_mean", "Au_std"),
    (6, "MIA before", "mia_auc_before_mean", "mia_auc_before_std"),
    (7, "MIA after", "mia_auc_after_mean", "mia_auc_after_std"),
]


def audit_anchor_ablation(text, thesis, records):
    label = "tab:anchor-ablation"
    body = table_body(text, label)
    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(cell) for cell in cells]
        if len(vals) < 8:
            continue
        dataset = DATASET_MAP.get(vals[0])
        anchor = vals[1].lower()
        if dataset is None or anchor not in {"old", "reinit"}:
            continue
        seen.add((dataset, anchor))
        row_label = f"{dataset}/anchor={anchor}"
        csv_row, matches, source = thesis.find(
            dataset=dataset,
            method="pall_modified",
            experiment_tag="anchor_ablation_v1",
            protect_anchor=anchor,
        )
        reason = None if len(matches) == 1 else no_match_reason(matches)
        for idx, col_label, mean_f, std_f in ANCHOR_METRIC_COLS:
            audit_metric_cell(records, label, row_label, col_label, cells[idx], csv_row or {},
                              mean_f, std_f, source, reason)

    expected = {
        (row["dataset"], row["protect_anchor"])
        for row in thesis.rows
        if row.get("experiment_tag") == "anchor_ablation_v1"
    }
    for dataset, anchor in sorted(expected - seen):
        records.append(Record(label, f"{dataset}/anchor={anchor}", "row presence", "missing",
                              "aggregate row exists", THESIS_CSV.name, "FAIL"))


CONFLICT_METRIC_COLS = [
    (4, "Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
    (5, "WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
    (6, "Au", "Au_mean", "Au_std"),
    (7, "Grad norm ratio", "grad_norm_ratio_mean", "grad_norm_ratio_std"),
]


def audit_conflict_ablation(text, thesis, records):
    label = "tab:conflict-ablation"
    body = table_body(text, label)
    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(cell) for cell in cells]
        if len(vals) < 8:
            continue
        dataset = DATASET_MAP.get(vals[0])
        method = METHOD_MAP.get(vals[1])
        importance = vals[2].lower()
        if dataset is None or method not in {"pall_modified", "pall_adapter"}:
            continue
        if importance not in {"gradient", "conflict"}:
            continue
        if method == "pall_adapter":
            adaptive = "False"
        elif vals[3].lower() == "on":
            adaptive = "True"
        elif vals[3].lower() == "off":
            adaptive = "False"
        else:
            continue
        seen.add((dataset, method, importance, adaptive))
        row_label = f"{dataset}/{method}/{importance}/adaptive={adaptive}"
        csv_row, matches, source = thesis.find(
            dataset=dataset,
            method=method,
            experiment_tag="conflict_ablation_v1",
            protect_importance=importance,
            adaptive_protect=adaptive,
        )
        reason = None if len(matches) == 1 else no_match_reason(matches)
        for idx, col_label, mean_f, std_f in CONFLICT_METRIC_COLS:
            audit_metric_cell(records, label, row_label, col_label, cells[idx], csv_row or {},
                              mean_f, std_f, source, reason)

    expected = {
        (row["dataset"], row["method"], row["protect_importance"], row["adaptive_protect"])
        for row in thesis.rows
        if row.get("experiment_tag") == "conflict_ablation_v1"
    }
    for dataset, method, importance, adaptive in sorted(expected - seen):
        row_label = f"{dataset}/{method}/{importance}/adaptive={adaptive}"
        records.append(Record(label, row_label, "row presence", "missing", "aggregate row exists",
                              THESIS_CSV.name, "FAIL"))


def audit_privacy_table(text, privacy, records):
    label = "tab:privacy-eps-hat"
    body = table_body(text, label)
    buckets: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in privacy.rows:
        before = str(row.get("eps_hat_before", "")).strip()
        after = str(row.get("eps_hat_after", "")).strip()
        if before == "" or after == "":
            continue
        key = (row.get("dataset", ""), row.get("method", ""))
        bucket = buckets.setdefault(key, {"before": [], "after": [], "regimes": set()})
        bucket["before"].append(float(before))
        bucket["after"].append(float(after))
        bucket["regimes"].add(row.get("regime", ""))

    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(cell) for cell in cells]
        if len(vals) < 7:
            continue
        dataset = DATASET_MAP.get(vals[0])
        method = METHOD_MAP.get(vals[1])
        if dataset is None or method is None or (dataset, method) not in buckets:
            continue
        before_nums = numbers_in(cells[3])
        after_nums = numbers_in(cells[4])
        max_nums = numbers_in(cells[5])
        count_nums = numbers_in(cells[6])
        bucket = buckets[(dataset, method)]
        regimes = bucket["regimes"]
        regime_ok = len(regimes) == 1 and vals[2] == next(iter(regimes))
        if not before_nums or not after_nums or not max_nums or not count_nums or not regime_ok:
            records.append(Record(label, f"{dataset}/{method}", "summary values", "missing",
                                  "privacy rows exist", privacy.path.name, "FAIL"))
            continue
        seen.add((dataset, method))
        before_values = bucket["before"]
        after_values = bucket["after"]
        source = f"{privacy.path.name}[dataset={dataset}, method={method}, eps_hat_after]"
        audit_scalar(records, label, f"{dataset}/{method}", "mean eps_hat_before",
                     before_nums[0], str(sum(before_values) / len(before_values)), source)
        audit_scalar(records, label, f"{dataset}/{method}", "mean eps_hat_after",
                     after_nums[0], str(sum(after_values) / len(after_values)), source)
        audit_scalar(records, label, f"{dataset}/{method}", "max eps_hat_after",
                     max_nums[0], str(max(after_values)), source)
        audit_scalar(records, label, f"{dataset}/{method}", "n", count_nums[0],
                     str(len(after_values)), source)

    for dataset, method in sorted(set(buckets) - seen):
        records.append(Record(label, f"{dataset}/{method}", "row presence", "missing",
                              "privacy summary exists", privacy.path.name, "FAIL"))


def audit_probe_table(text, thesis, records):
    """Validate probe_v1 before/after mean±std cells and signed mean changes."""
    label = "tab:probe-results"
    body = table_body(text, label)
    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(cell) for cell in cells]
        if len(vals) < 5:
            continue
        dataset = DATASET_MAP.get(vals[0])
        method = METHOD_MAP.get(vals[1])
        if dataset is None or method is None:
            continue
        key = (dataset, method)
        seen.add(key)
        row_label = f"{dataset}/{method}"
        csv_row, matches, source = thesis.find(
            dataset=dataset,
            method=method,
            experiment_tag="probe_v1",
        )
        reason = None if len(matches) == 1 else no_match_reason(matches)
        audit_metric_cell(
            records, label, row_label, "Probe Before", cells[2], csv_row or {},
            "probe_acc_before_mean", "probe_acc_before_std", source, reason,
        )
        audit_metric_cell(
            records, label, row_label, "Probe After", cells[3], csv_row or {},
            "probe_acc_after_mean", "probe_acc_after_std", source, reason,
        )
        change_nums = numbers_in(cells[4])
        if reason is not None or not change_nums:
            records.append(Record(
                label, row_label, "Signed Change",
                change_nums[0] if change_nums else "<missing>",
                reason or "probe means unavailable", source, "FAIL",
            ))
            continue
        signed_change = float(csv_row["probe_acc_after_mean"]) - float(csv_row["probe_acc_before_mean"])
        audit_scalar(records, label, row_label, "Signed Change", change_nums[0], str(signed_change), source)

    expected = {
        (row["dataset"], row["method"])
        for row in thesis.rows
        if row.get("experiment_tag") == "probe_v1"
    }
    for dataset, method in sorted(expected - seen):
        records.append(Record(
            label, f"{dataset}/{method}", "row presence", "missing",
            "probe_v1 aggregate row exists", THESIS_CSV.name, "FAIL",
        ))


def audit_vit_availability(text, thesis, records):
    """Audit ViT metrics when present, or an explicit pending row when absent."""
    label = "tab:vit-results"
    body = table_body(text, label)
    expected = {
        ("cifar10", "pall_original", "cifar10_vit_v1"),
        ("cifar10", "pall_modified", "cifar10_vit_v1"),
        ("cifar100", "pall_original", "cifar100_vit_v1"),
        ("cifar100", "pall_modified", "cifar100_vit_v1"),
    }
    seen = set()
    for cells in data_rows(body):
        vals = [strip_latex(cell) for cell in cells]
        if len(vals) < 6:
            continue
        dataset = DATASET_MAP.get(vals[0])
        method = METHOD_MAP.get(vals[1])
        if dataset is None or method not in {"pall_original", "pall_modified"}:
            continue
        tag = f"{dataset}_vit_v1"
        key = (dataset, method, tag)
        seen.add(key)
        _, matches, source = thesis.find(dataset=dataset, method=method, experiment_tag=tag)
        row_label = f"{dataset}/{method}"
        if len(matches) == 1:
            metric_cols = (
                ("Final Acc.", "final_avg_acc_mean", "final_avg_acc_std"),
                ("WorstDrop", "WorstDrop_mean", "WorstDrop_std"),
                ("Au", "Au_mean", "Au_std"),
            )
            for idx, (col_label, mean_field, std_field) in enumerate(metric_cols):
                audit_metric_cell(records, label, row_label, col_label, cells[2 + idx],
                                  matches[0], mean_field, std_field, source, None)
            continue

        has_metrics = any(numbers_in(cell) for cell in cells[2:5])
        marked_unavailable = "No aggregate row" in vals[5]
        ok = not matches and not has_metrics and marked_unavailable
        csv_value = "0 aggregate rows" if not matches else f"{len(matches)} aggregate row(s)"
        records.append(Record(label, row_label, "availability", vals[5], csv_value,
                              source, "PASS" if ok else "FAIL"))

    for dataset, method, tag in sorted(expected - seen):
        records.append(Record(label, f"{dataset}/{method}", "row presence", "missing",
                              "expected explicit unavailable row", tag, "FAIL"))


OVERLAP_METRIC_COLS = [
    (2, "Critical Ratio", "overlap_shared_critical_ratio"),
    (3, "Protected Ratio", "overlap_protected_ratio"),
    (4, "Updated-in-Shared Ratio", "overlap_updated_ratio"),
    (5, "Critical Count", "overlap_shared_critical_count"),
]


def audit_overlap(text, thesis, records):
    label = "tab:overlap-correlation"
    body = table_body(text, label)
    setting_to_tag = {
        ("cifar10", "scratch"): "cifar10_main",
        ("cifar10", "pretrained"): "cifar10_pretrained",
        ("cifar100", "scratch"): "cifar100_main",
        ("cifar100", "pretrained"): "cifar100_pretrained",
        ("tinyimagenet", "pretrained"): "tiny_pretrained",
    }
    for cells in data_rows(body):
        vals = [strip_latex(c) for c in cells]
        if len(vals) < 6:
            continue
        dataset = DATASET_MAP.get(vals[0])
        if dataset is None:
            continue
        setting = "scratch" if "صفر" in vals[1] else ("pretrained" if "پیش" in vals[1] else None)
        if setting is None:
            continue
        tag = setting_to_tag.get((dataset, setting))
        row_label = f"{dataset}/{setting}/PALL-Adapter"
        csv_row, matches, source = thesis.find(dataset=dataset, method="pall_adapter", experiment_tag=tag)
        reason = None if len(matches) == 1 else no_match_reason(matches)
        for idx, col_label, field in OVERLAP_METRIC_COLS:
            audit_metric_cell(records, label, row_label, col_label, cells[idx],
                              csv_row or {}, field, None, source, reason)


# hparam Persian-key substring -> (csv field, reference-run filter)
HPARAM_FIELDS = [
    ("نسبت محافظت (روش", "protect_ratio", dict(dataset="cifar10", method="pall_modified", experiment_tag="cifar10_main")),
    ("ضریب جریمه", "lambda_protect", dict(dataset="cifar10", method="pall_modified", experiment_tag="cifar10_main")),
    ("بعد گلوگاه", "adapter_bottleneck", dict(dataset="cifar10", method="pall_adapter", experiment_tag="cifar10_main")),
    ("نسبت فراموشی مشترک", "adapter_shared_forget_ratio", dict(dataset="cifar10", method="pall_adapter", experiment_tag="cifar10_main")),
    ("نسبت محافظت مشترک", "adapter_shared_protect_ratio", dict(dataset="cifar10", method="pall_adapter", experiment_tag="cifar10_main")),
]


def audit_hparams(text, thesis, records):
    label = "tab:hparams"
    body = table_body(text, label)
    for cells in data_rows(body):
        if len(cells) < 2:
            continue
        key = strip_latex(cells[0])
        nums = numbers_in(cells[1])
        if not nums:
            continue
        value = nums[0]
        register_value(value, f"{label} | {key} | value")
        field = None
        filt = None
        for needle, csv_field, run_filter in HPARAM_FIELDS:
            if needle in cells[0] or needle in key:
                field, filt = csv_field, run_filter
                break
        if field is None:
            records.append(Record(label, key, "value", value, "(training constant; no CSV column)",
                                  "-", "SKIP-NOCSV"))
            continue
        csv_row, matches, source = thesis.find(**filt)
        if len(matches) != 1:
            records.append(Record(label, key, field, value, no_match_reason(matches), source, "FAIL"))
            continue
        ok, cv = values_match(value, csv_row.get(field))
        records.append(Record(label, key, field, value, cv, source, "PASS" if ok else "FAIL"))


# --------------------------------------------------------------------------- #
# Prose scan                                                                   #
# --------------------------------------------------------------------------- #
def strip_environments(text: str) -> str:
    for env in ("table", "figure", "landscape", "tabular"):
        text = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", " ", text, flags=re.DOTALL)
    return text


def audit_prose(text, records):
    prose = strip_environments(text)
    prose = prose.replace("$", " ")
    label = "prose"
    seen = set()
    for match in re.finditer(r"(?<![\d.])(\d+\.\d{3,})", prose):
        num = match.group(1)
        start = match.start()
        # a light context window to identify the paragraph occurrence
        snippet = re.sub(r"\s+", " ", prose[max(0, start - 28): start]).strip()[-28:]
        key = (num, snippet)
        if key in seen:
            continue
        seen.add(key)
        pdec = len(num.split(".")[1])
        pval = float(num)
        hits = []
        for tv_str, locs in TABLE_VALUES.items():
            try:
                tv = float(tv_str)
            except ValueError:
                continue
            tdec = len(tv_str.split(".")[1]) if "." in tv_str else 0
            tol = 0.5 * (10 ** (-min(pdec, tdec))) + 1e-9
            if abs(tv - pval) <= tol:
                hits.extend(locs)
        if hits:
            records.append(Record(label, f"...{snippet}", "inline number", num,
                                  hits[0] + (f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""),
                                  "results.tex tables", "PASS"))
        else:
            records.append(Record(label, f"...{snippet}", "inline number", num,
                                  "(no matching table cell)", "results.tex tables", "FAIL"))


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def print_records(records: List[Record]) -> None:
    order = ["tab:main-pall-results", "tab:standard-cifar10-results", "tab:standard-cifar100-results",
             "tab:vit-results", "tab:anchor-ablation", "tab:adapter-ablation",
             "tab:conflict-ablation", "tab:overlap-correlation", "tab:privacy-eps-hat", "tab:probe-results",
             "tab:hparams", "prose"]
    by_table = {name: [r for r in records if r.table == name] for name in order}
    for name in order:
        recs = by_table[name]
        if not recs:
            continue
        npass = sum(1 for r in recs if r.status == "PASS")
        nfail = sum(1 for r in recs if r.status == "FAIL")
        nskip = sum(1 for r in recs if r.status.startswith("SKIP"))
        print("\n" + "=" * 100)
        print(f"{name}   (PASS {npass} / FAIL {nfail}" + (f" / SKIP {nskip}" if nskip else "") + ")")
        print("=" * 100)
        print(f"{'STATUS':7} {'ROW':34} {'COLUMN':22} {'THESIS':16} {'CSV':16} SOURCE")
        print("-" * 100)
        for r in recs:
            row = (r.row[:33]) if len(r.row) > 33 else r.row
            col = (r.column[:21]) if len(r.column) > 21 else r.column
            print(f"{r.status:7} {row:34} {col:22} {r.thesis:16} {str(r.csv):16} {r.source}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit thesis result numbers against aggregate CSVs.")
    parser.add_argument("--tex", type=Path, default=TEX_PATH)
    parser.add_argument("--thesis-csv", type=Path, default=THESIS_CSV)
    parser.add_argument("--privacy-csv", type=Path, default=PRIVACY_CSV)
    args = parser.parse_args()

    for path in (args.tex, args.thesis_csv, args.privacy_csv):
        if not path.exists():
            print(f"[ERROR] missing: {path}", file=sys.stderr)
            return 1

    text = args.tex.read_text(encoding="utf-8")
    thesis = CsvTable(args.thesis_csv)
    privacy = CsvTable(args.privacy_csv)
    records: List[Record] = []

    audit_main_pall(text, thesis, records)
    audit_standard(text, thesis, records, "tab:standard-cifar10-results", "cifar10", "cifar10_standard")
    audit_standard(text, thesis, records, "tab:standard-cifar100-results", "cifar100", "cifar100_standard")
    audit_vit_availability(text, thesis, records)
    audit_anchor_ablation(text, thesis, records)
    audit_adapter_ablation(text, thesis, records)
    audit_conflict_ablation(text, thesis, records)
    audit_overlap(text, thesis, records)
    audit_privacy_table(text, privacy, records)
    audit_probe_table(text, thesis, records)
    audit_hparams(text, thesis, records)
    audit_prose(text, records)

    print_records(records)

    total_pass = sum(1 for r in records if r.status == "PASS")
    total_fail = sum(1 for r in records if r.status == "FAIL")
    total_skip = sum(1 for r in records if r.status.startswith("SKIP"))
    print("\n" + "#" * 100)
    print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL, {total_skip} SKIP "
          f"(config columns tracked: {', '.join(CONFIG_GROUP_COLUMNS)})")
    print("#" * 100)
    print("Note: read-only audit. results.tex was NOT modified.")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
