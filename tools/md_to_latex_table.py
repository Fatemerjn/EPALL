#!/usr/bin/env python3
"""
Convert a Markdown table to a LaTeX table environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert report_table.md to a LaTeX table.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/thesis/report_table.md"),
        help="Input Markdown table path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/thesis/report_table.tex"),
        help="Output LaTeX table path.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> Optional[List[str]]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"[ERROR] Input Markdown file not found: {path}", file=sys.stderr)
        return None
    except OSError as exc:
        print(f"[ERROR] Failed to read Markdown file: {path} ({exc})", file=sys.stderr)
        return None


def is_table_line(line: str) -> bool:
    text = line.strip()
    return text.startswith("|") and text.endswith("|")


def split_markdown_row(line: str) -> List[str]:
    cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
    return cells


def is_separator_row(cells: List[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        stripped = cell.replace("-", "").replace(":", "").strip()
        if stripped != "":
            return False
    return True


def extract_table(lines: List[str]) -> Optional[List[List[str]]]:
    table_rows: List[List[str]] = []
    for line in lines:
        if not is_table_line(line):
            continue
        cells = split_markdown_row(line)
        if is_separator_row(cells):
            continue
        table_rows.append(cells)
    if not table_rows:
        return None
    return table_rows


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in text)
    return escaped.replace("+/-", r"$\pm$")


def to_latex_table(rows: List[List[str]]) -> str:
    header = rows[0]
    body = rows[1:]
    n_cols = len(header)
    col_spec = "l" * n_cols

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        " & ".join(escape_latex(cell) for cell in header) + r" \\",
        r"\hline",
    ]
    for row in body:
        padded = row + [""] * max(0, n_cols - len(row))
        lines.append(" & ".join(escape_latex(cell) for cell in padded[:n_cols]) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Comparison of methods}",
            r"\label{tab:results}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    lines = read_lines(args.input)
    if lines is None:
        return 1

    rows = extract_table(lines)
    if rows is None:
        print(f"[ERROR] No Markdown table found in: {args.input}", file=sys.stderr)
        return 1

    latex = to_latex_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")

    print(f"[INFO] Parsed table rows: {len(rows)}")
    print(f"[INFO] Wrote LaTeX table: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
