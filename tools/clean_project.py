#!/usr/bin/env python3
"""
Project cleanup helper.

Default mode is dry-run. Pass --apply to perform the planned actions.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT_MOVE_TARGET = Path("results/tmp_root_outputs")
ROOT_MOVE_PATTERNS = ("comparison_*.csv", "comparison_*.md", "results_*.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean caches, backups, and stray root outputs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply the cleanup actions.")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying the filesystem (default).",
    )
    return parser.parse_args()


def collect_pycache_dirs(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("__pycache__") if path.is_dir())


def collect_pyc_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.pyc") if path.is_file())


def collect_backup_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*~") if path.is_file())


def collect_root_output_files(root: Path) -> List[Tuple[Path, Path]]:
    planned_moves: List[Tuple[Path, Path]] = []
    for pattern in ROOT_MOVE_PATTERNS:
        for source in sorted(root.glob(pattern)):
            if not source.is_file():
                continue
            destination = root / ROOT_MOVE_TARGET / source.name
            planned_moves.append((source, destination))
    return planned_moves


def collect_incomplete_run_dirs(root: Path) -> List[Path]:
    runs_root = root / "runs"
    if not runs_root.exists():
        return []

    incomplete: List[Path] = []
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        if not (run_dir / "metrics.json").exists():
            incomplete.append(run_dir)
    return incomplete


def ensure_parent(path: Path, apply: bool) -> None:
    if apply:
        path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path, apply: bool) -> None:
    if not apply:
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def move_file(source: Path, destination: Path, apply: bool) -> None:
    if not apply:
        return
    ensure_parent(destination.parent, apply=True)
    shutil.move(str(source), str(destination))


def print_section(title: str, items: Sequence[str]) -> None:
    print(f"\n{title}")
    if not items:
        print("  none")
        return
    for item in items:
        print(f"  - {item}")


def main() -> int:
    args = parse_args()
    apply = bool(args.apply)
    root = Path.cwd()

    pycache_dirs = collect_pycache_dirs(root)
    pyc_files = collect_pyc_files(root)
    backup_files = collect_backup_files(root)
    root_moves = collect_root_output_files(root)
    incomplete_runs = collect_incomplete_run_dirs(root)

    print(f"[MODE] {'apply' if apply else 'dry-run'}")
    print(f"[ROOT] {root}")

    print_section(
        "Planned __pycache__ removals",
        [str(path.relative_to(root)) for path in pycache_dirs],
    )
    print_section(
        "Planned .pyc removals",
        [str(path.relative_to(root)) for path in pyc_files],
    )
    print_section(
        "Planned backup-file removals",
        [str(path.relative_to(root)) for path in backup_files],
    )
    print_section(
        "Planned root-output moves",
        [
            f"{source.relative_to(root)} -> {destination.relative_to(root)}"
            for source, destination in root_moves
        ],
    )
    print_section(
        "Run directories missing metrics.json",
        [str(path.relative_to(root)) for path in incomplete_runs],
    )

    if apply:
        for path in pycache_dirs:
            remove_path(path, apply=True)
        for path in pyc_files:
            remove_path(path, apply=True)
        for path in backup_files:
            remove_path(path, apply=True)
        for source, destination in root_moves:
            move_file(source, destination, apply=True)

    deleted_count = len(pycache_dirs) + len(pyc_files) + len(backup_files)
    moved_count = len(root_moves)

    print("\nSummary")
    print(f"  mode: {'apply' if apply else 'dry-run'}")
    print(f"  pycache_dirs: {len(pycache_dirs)}")
    print(f"  pyc_files: {len(pyc_files)}")
    print(f"  backup_files: {len(backup_files)}")
    print(f"  root_output_moves: {moved_count}")
    print(f"  incomplete_run_dirs: {len(incomplete_runs)}")
    print(f"  filesystem_changes: {deleted_count + moved_count if apply else 0}")
    print(f"  next_step: {'cleanup applied' if apply else 'rerun with --apply to execute changes'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
