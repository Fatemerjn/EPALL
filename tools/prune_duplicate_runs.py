#!/usr/bin/env python3
"""
Prune duplicate / partial run directories by ARCHIVING (moving) them.

This is the write-side companion to ``tools/audit_seed_completeness.py``. It
reuses that module's spec-matching so "duplicate completed run" and "partial
run" mean exactly what the audit reports:

  * A duplicate is detected the same way ``audit()`` does: for every EXPECTED
    spec (from ``expected_specs``) and each of its expected seeds, we look up the
    completed runs indexed by ``load_actual_runs`` under
    ``(spec.key(), seed)``. When that lookup returns more than one run
    directory we KEEP the one with the latest timestamp and archive the rest.
    Restricting to expected specs is deliberate: it makes this tool's duplicate
    set identical to the audit's "duplicate completed seeds" list, so the audit
    drops to zero after ``--apply`` and exploratory runs that are not part of
    the thesis result sets (e.g. the ``T*_F1`` overlap-search sweeps read by
    other tooling) are left untouched.
  * ``find_partial_dirs`` returns run directories that have ``config.json`` but
    no ``metrics.json`` (incomplete runs). Those are archived unconditionally.

Nothing is deleted: archived directories are MOVED under
``runs_archive/pruned_<YYYYMMDD>/<same path relative to the run root>`` so the
original directory structure is preserved and the move is reversible.

Default mode is ``--dry-run`` (only prints the plan). Pass ``--apply`` to move.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

# Run directories are laid out as
#   runs/<dataset>/<T*_F*>/<method>/seed_<n>/<timestamp>/
# so the timestamp is the leaf directory name in ``YYYYMMDD_HHMMSS`` form, which
# sorts chronologically as a plain string.
from audit_seed_completeness import expected_specs, find_partial_dirs, load_actual_runs


def _timestamp(path: Path) -> str:
    """Leaf directory name == run timestamp (``YYYYMMDD_HHMMSS``)."""
    return path.name


def _spec_label(path: Path, root: Path) -> str:
    """Human-readable ``dataset/T*_F*/method`` derived from the run path."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    # parts == (dataset, T*_F*, method, seed_<n>, timestamp)
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return str(path.parent)


def _seed_label(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if part.startswith("seed_"):
            return part.split("seed_", 1)[1]
    return ""


def _archive_target(path: Path, root: Path, archive_root: Path) -> Path:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(*path.parts[-5:]) if len(path.parts) >= 5 else Path(path.name)
    return archive_root / rel


def build_plan(root: Path, archive_root: Path, schedules_dir: Path) -> List[Dict[str, str]]:
    """Return the ordered list of archive actions (duplicates then partials)."""
    plan: List[Dict[str, str]] = []

    # (1) Duplicate completed runs, scoped exactly like ``audit()``: iterate the
    #     expected specs and, for each expected seed, keep the latest-timestamp
    #     run and archive the rest. A given run directory is only archived once
    #     even if two expected specs happened to resolve to the same key.
    actual = load_actual_runs(root)
    seen: set = set()
    for spec in expected_specs(schedules_dir):
        key = spec.key()
        for seed in spec.expected_seeds:
            paths = actual.get((key, seed), [])
            if len(paths) <= 1:
                continue
            ordered = sorted(paths, key=_timestamp)
            keep = ordered[-1]
            for path in ordered[:-1]:
                if path in seen:
                    continue
                seen.add(path)
                plan.append(
                    {
                        "action": "archive-duplicate",
                        "spec": _spec_label(path, root),
                        "seed": seed or _seed_label(path, root),
                        "timestamp": _timestamp(path),
                        "target": str(_archive_target(path, root, archive_root)),
                        "note": f"keep {_timestamp(keep)}",
                        "path": str(path),
                    }
                )

    # (2) Partial runs: config.json without metrics.json -> archive all.
    for path in find_partial_dirs(root):
        plan.append(
            {
                "action": "archive-partial",
                "spec": _spec_label(path, root),
                "seed": _seed_label(path, root),
                "timestamp": _timestamp(path),
                "target": str(_archive_target(path, root, archive_root)),
                "note": "no metrics.json",
                "path": str(path),
            }
        )

    plan.sort(key=lambda entry: (entry["action"], entry["path"]))
    return plan


def print_plan(plan: List[Dict[str, str]], apply: bool) -> None:
    header = "APPLY" if apply else "DRY-RUN"
    dup = sum(1 for entry in plan if entry["action"] == "archive-duplicate")
    part = sum(1 for entry in plan if entry["action"] == "archive-partial")
    print(f"[{header}] prune plan: {len(plan)} directories "
          f"({dup} duplicate, {part} partial)")
    if not plan:
        print("  nothing to prune.")
        return

    cols = ("action", "spec", "seed", "timestamp", "note", "target")
    widths = {col: len(col) for col in cols}
    for entry in plan:
        for col in cols:
            widths[col] = max(widths[col], len(str(entry.get(col, ""))))
    line = "  " + "  ".join(col.ljust(widths[col]) for col in cols)
    print(line)
    print("  " + "  ".join("-" * widths[col] for col in cols))
    for entry in plan:
        print("  " + "  ".join(str(entry.get(col, "")).ljust(widths[col]) for col in cols))


def apply_plan(plan: List[Dict[str, str]]) -> int:
    moved = 0
    for entry in plan:
        src = Path(entry["path"])
        dst = Path(entry["target"])
        if not src.exists():
            print(f"[WARN] source vanished, skipping: {src}", file=sys.stderr)
            continue
        if dst.exists():
            print(f"[WARN] target already exists, skipping: {dst}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1
        print(f"  moved {src} -> {dst}")
    print(f"[APPLY] archived {moved}/{len(plan)} directories.")
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive duplicate/partial run directories.")
    parser.add_argument("--root", type=Path, default=Path("runs"), help="Run artifact root.")
    parser.add_argument(
        "--schedules-dir",
        type=Path,
        default=Path("schedules"),
        help="Schedule directory (passed to expected_specs; infers optional TinyImageNet seed 1).",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive destination root (default: runs_archive/pruned_<YYYYMMDD>).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Only print the plan (default).")
    group.add_argument("--apply", action="store_true", help="Perform the moves.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.exists():
        print(f"[ERROR] Run root does not exist: {args.root}", file=sys.stderr)
        return 1
    archive_root = args.archive_root or Path("runs_archive") / f"pruned_{date.today():%Y%m%d}"
    plan = build_plan(args.root, archive_root, args.schedules_dir)
    print_plan(plan, apply=args.apply)
    if args.apply:
        apply_plan(plan)
    else:
        print("\n[DRY-RUN] no changes made. Re-run with --apply to archive the above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
