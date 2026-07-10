#!/usr/bin/env python3
"""Shared helpers for selecting canonical completed run rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def normalize_group_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "na"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isnan(number):
        return ""
    if abs(number - round(number)) < 1e-10:
        return str(int(round(number)))
    return f"{number:.12g}"


def seed_key(row: Dict[str, Any], *, seed_column: str = "seed", path_column: str = "run_path") -> str:
    seed = row.get(seed_column)
    if seed is None:
        return f"run:{row.get(path_column, '')}"
    seed_text = str(seed).strip()
    if seed_text == "":
        return f"run:{row.get(path_column, '')}"
    return seed_text


def run_recency_key(row: Dict[str, Any], *, path_column: str = "run_path") -> Tuple[str, str]:
    path = Path(str(row.get(path_column, "")))
    return path.name, str(path)


def select_latest_seed_rows(
    rows: Sequence[Dict[str, Any]],
    group_columns: Sequence[str],
    *,
    seed_column: str = "seed",
    path_column: str = "run_path",
) -> Tuple[List[Dict[str, Any]], int]:
    """Keep only the newest run for each group+seed pair.

    Raw run directories use timestamp-like leaf names, so lexical ordering of
    the final path component gives a stable latest-run policy without touching
    the raw artifacts.
    """

    best_by_key: Dict[Tuple[Tuple[str, ...], str], Dict[str, Any]] = {}
    for row in rows:
        group_key = tuple(normalize_group_value(row.get(column)) for column in group_columns)
        key = (group_key, seed_key(row, seed_column=seed_column, path_column=path_column))
        current = best_by_key.get(key)
        if current is None or run_recency_key(row, path_column=path_column) > run_recency_key(
            current,
            path_column=path_column,
        ):
            best_by_key[key] = row
    selected = sorted(best_by_key.values(), key=lambda row: str(row.get(path_column, "")))
    return selected, max(0, len(rows) - len(selected))
