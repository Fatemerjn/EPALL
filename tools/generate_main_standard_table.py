#!/usr/bin/env python3
"""Generate the matched standard Split-CIFAR comparison used as Table 1."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METHOD_GROUPS: Sequence[Tuple[str, Sequence[str]]] = (
    ("Isolation", ("clpu",)),
    ("Shared CL", ("derpp", "er", "ewc", "lwf")),
    ("Unlearning", ("ssd", "salun")),
    ("Full PALL", ("pall_original", "pall_modified")),
    (r"PEFT$^\dagger$", ("lora", "pall_adapter")),
)

METHOD_LABELS = {
    "clpu": "CLPU",
    "derpp": "DER++",
    "er": "ER",
    "ewc": "EWC",
    "lwf": "LwF",
    "lora": r"LoRA$^\dagger$",
    "ssd": "SSD",
    "salun": "SalUn",
    "pall_original": "PALL-Original",
    "pall_modified": r"\textbf{EPALL}",
    "pall_adapter": r"\textbf{PALL-Adapter}$^\dagger$",
}

DATASETS = (
    ("cifar10", "Panel A: Standard Split-CIFAR-10", "0.5"),
    ("cifar100", "Panel B: Standard Split-CIFAR-100", "0.1"),
)

METRIC_COLUMNS = (
    ("final_avg_acc", r"$A_{\mathrm{final}}\!\uparrow$"),
    ("avg_forgetting", r"$F_{\mathrm{avg}}\!\downarrow$"),
    ("WorstDrop", r"WorstDrop $\downarrow$"),
    ("Au", r"$A_u\!\rightarrow\!$ chance"),
)

MEAN_STD_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\+/-\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "results/aggregates/server_report_table.csv",
        help="Canonical aggregate CSV (server_report_table.csv only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "paper/AuthorKit27/generated/main_standard_comparison.tex",
        help="Generated LaTeX table path.",
    )
    return parser.parse_args()


def fail(message: str) -> "None":
    raise ValueError(message)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if path.name != "server_report_table.csv":
        fail(f"refusing non-canonical input {path}; expected server_report_table.csv")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "dataset",
                "method",
                "regime",
                "experiment_tag",
                "config_id",
                "n_seeds",
                "final_avg_acc",
                "avg_forgetting",
                "WorstDrop",
                "Au",
                "t_forget_total_mean",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                fail(f"input is missing required columns: {', '.join(sorted(missing))}")
            return list(reader)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def expected_tag(dataset: str, method: str) -> str:
    if method in {"ssd", "salun"}:
        return "standard_unlearning_ssd_salun_v1"
    return f"{dataset}_standard"


def select_rows(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    selected: Dict[Tuple[str, str], Dict[str, str]] = {}
    methods = [method for _, group in METHOD_GROUPS for method in group]
    for dataset, _, _ in DATASETS:
        for method in methods:
            tag = expected_tag(dataset, method)
            matches = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["method"] == method
                and row["regime"] == "standard_split"
                and row["experiment_tag"] == tag
            ]
            if len(matches) != 1:
                fail(
                    "expected exactly one aggregate row for "
                    f"dataset={dataset}, method={method}, regime=standard_split, "
                    f"experiment_tag={tag}; found {len(matches)}"
                )
            selected[(dataset, method)] = matches[0]

    # Every row must rest on the same seed set; the count itself may grow as the
    # standard-table seed extension lands, so it is read from the data.
    seed_counts = {str(row["n_seeds"]) for row in selected.values()}
    if len(seed_counts) != 1:
        fail(f"all rows must share one seed count; found {sorted(seed_counts)}")

    if len(selected) != 22:
        fail(f"expected exactly 22 selected aggregate rows; found {len(selected)}")
    if any(row["experiment_tag"] == "ssd_tune_v1" for row in selected.values()):
        fail("ssd_tune_v1 must not appear in the matched standard comparison")
    return selected


def decimal_4(value: str, *, field: str, key: Tuple[str, str]) -> str:
    text = str(value).strip()
    if not text:
        fail(f"missing {field} for dataset={key[0]}, method={key[1]}")
    try:
        number = Decimal(text)
    except InvalidOperation:
        fail(f"invalid {field}={text!r} for dataset={key[0]}, method={key[1]}")
    return format(number.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), "f")


def mean_std_4(value: str, *, field: str, key: Tuple[str, str]) -> str:
    match = MEAN_STD_RE.fullmatch(str(value))
    if match is None:
        fail(
            f"expected aggregate mean +/- std in {field} for "
            f"dataset={key[0]}, method={key[1]}; got {value!r}"
        )
    mean = decimal_4(match.group(1), field=f"{field} mean", key=key)
    std = decimal_4(match.group(2), field=f"{field} std", key=key)
    return rf"${mean}\mathbin{{\pm}}{std}$"


def displayed_metrics(row: Dict[str, str], key: Tuple[str, str]) -> List[str]:
    values = [mean_std_4(row[column], field=column, key=key) for column, _ in METRIC_COLUMNS]
    values.append(decimal_4(row["t_forget_total_mean"], field="t_forget_total_mean", key=key))
    return values


def header() -> str:
    metric_headers = " & ".join(rf"\textbf{{{label}}}" for _, label in METRIC_COLUMNS)
    return r"\textbf{Family} & \textbf{Method} & " + metric_headers + r" & \textbf{$T_f\!\downarrow$ (s)} \\"


def panel_lines(
    dataset: str,
    panel_title: str,
    selected: Dict[Tuple[str, str], Dict[str, str]],
) -> Iterable[str]:
    yield rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{panel_title}}}}} \\"
    yield r"\cmidrule(lr){1-7}"
    yield header()
    yield r"\midrule"
    for group_index, (group_label, methods) in enumerate(METHOD_GROUPS):
        if group_index:
            yield r"\addlinespace[2pt]"
        for method_index, method in enumerate(methods):
            key = (dataset, method)
            metrics = displayed_metrics(selected[key], key)
            family = rf"\multirow{{{len(methods)}}}{{*}}{{\textit{{{group_label}}}}}" if method_index == 0 else ""
            yield f"{family} & {METHOD_LABELS[method]} & " + " & ".join(metrics) + r" \\"


SEED_WORDS = {
    3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
}


def caption(selected: Dict[Tuple[str, str], Dict[str, str]]) -> str:
    count = int(next(iter(selected.values()))["n_seeds"])
    seeds = f"{SEED_WORDS.get(count, count)} seeds"
    return (
        r"\caption{Matched standard Split-CIFAR comparison (20 epochs/task; "
        + seeds
        + r"). Rows group by operating assumption; bold names are our methods. Numeric"
        r" cells are not ranked: the metrics must be read jointly. CLPU is the"
        r" full-model isolation reference. $^\dagger$PEFT rows freeze a random backbone"
        r" here; their pretrained setting is Table~\ref{tab:pretrained}. Chance $A_u$ is"
        r" $0.5$/$0.1$. All metrics but the non-normalized $T_f$ are mean${\pm}$std.}"
    )


def render_table(selected: Dict[Tuple[str, str], Dict[str, str]], source: Path) -> str:
    lines = [
        "% AUTO-GENERATED by tools/generate_main_standard_table.py; do not edit.",
        f"% Source: {source.name}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
    ]
    for panel_index, (dataset, panel_title, _) in enumerate(DATASETS):
        if panel_index:
            lines.extend((r"\addlinespace[4pt]", r"\midrule", r"\addlinespace[2pt]"))
        lines.extend(panel_lines(dataset, panel_title, selected))
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            # AAAI-27 requires table captions below the tabular body.
            caption(selected),
            r"\label{tab:main_standard}",
            r"\end{table*}",
            "",
        )
    )
    return "\n".join(lines)


def provenance_lines(selected: Dict[Tuple[str, str], Dict[str, str]]) -> Iterable[str]:
    yield "Selected-row provenance (22 aggregate rows):"
    methods = [method for _, group in METHOD_GROUPS for method in group]
    for dataset, _, _ in DATASETS:
        for method in methods:
            key = (dataset, method)
            row = selected[key]
            metrics = displayed_metrics(row, key)
            plain_metrics = [value.replace("$", "").replace(r"\mathbin{\pm}", " +/- ") for value in metrics]
            yield (
                f"dataset={dataset} | method={method} | regime={row['regime']} | "
                f"experiment_tag={row['experiment_tag']} | config_id={row['config_id']} | "
                f"seeds={row['n_seeds']} | "
                f"A_final={plain_metrics[0]} | F_avg={plain_metrics[1]} | "
                f"WorstDrop={plain_metrics[2]} | Au={plain_metrics[3]} | T_f_s={plain_metrics[4]}"
            )


def main() -> int:
    args = parse_args()
    try:
        rows = read_rows(args.input)
        selected = select_rows(rows)
        output = render_table(selected, args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("\n".join(provenance_lines(selected)))
    print(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
