#!/usr/bin/env python3
"""
Generate or run candidate-overlap search experiments over candidate schedules.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DatasetPreset:
    class_per_task: int
    n_tasks: int
    n_forget: int
    candidate_csv: str


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "cifar10": DatasetPreset(
        class_per_task=2,
        n_tasks=5,
        n_forget=1,
        candidate_csv="results/thesis/cifar10_candidate_schedules.csv",
    ),
    "cifar100": DatasetPreset(
        class_per_task=5,
        n_tasks=10,
        n_forget=1,
        candidate_csv="results/thesis/cifar100_candidate_schedules.csv",
    ),
}
SUPPORTED_METHODS = ("pall_adapter", "pall_modified", "pall_original")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_main = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Generate or run candidate-overlap search commands.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--epochs", type=int, default=1, help="Value passed to --n_epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for all candidate runs.")
    parser.add_argument("--methods", nargs="+", default=["pall_adapter"], help="Methods to run.")
    parser.add_argument(
        "--protect-ratios",
        type=float,
        nargs="+",
        default=[0.1],
        help="Shared protection ratios for pall_adapter. Ignored for non-adapter methods.",
    )
    parser.add_argument("--forget-tasks", type=int, nargs="+", default=None, help="Optional forget_task_id filter.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--run", action="store_true", help="Execute commands sequentially.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py.")
    parser.add_argument("--main-file", default=str(default_main), help="Path to main.py.")
    return parser.parse_args()


def read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_args(args: argparse.Namespace, repo_root: Path) -> list[dict[str, str]]:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}.")
    if args.seed < 0:
        raise ValueError(f"--seed must be >= 0, got {args.seed}.")
    if args.run and args.dry_run:
        raise ValueError("Choose either --run or --dry-run, not both.")
    invalid_methods = [method for method in args.methods if method not in SUPPORTED_METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}. Valid methods: {list(SUPPORTED_METHODS)}")
    invalid_protect_ratios = [ratio for ratio in args.protect_ratios if not (0.0 <= ratio <= 1.0)]
    if invalid_protect_ratios:
        raise ValueError(
            f"Invalid --protect-ratios values: {invalid_protect_ratios}. Expected each ratio in [0.0, 1.0]."
        )

    main_file = Path(args.main_file).expanduser().resolve()
    if not main_file.exists():
        raise ValueError(f"main.py not found: {main_file}")
    if not main_file.is_file():
        raise ValueError(f"--main-file is not a file: {main_file}")

    preset = DATASET_PRESETS[args.dataset]
    if args.forget_tasks is not None:
        invalid_forget_tasks = [task_id for task_id in args.forget_tasks if not (0 <= task_id < preset.n_tasks)]
        if invalid_forget_tasks:
            raise ValueError(
                f"Invalid forget_task_id values for dataset={args.dataset}: {invalid_forget_tasks}. "
                f"Expected range [0, {preset.n_tasks - 1}]."
            )
    candidate_csv = repo_root / preset.candidate_csv
    if not candidate_csv.exists():
        raise ValueError(f"Candidate schedule CSV not found: {candidate_csv}")
    if not candidate_csv.is_file():
        raise ValueError(f"Candidate schedule CSV path is not a file: {candidate_csv}")

    rows = read_candidate_rows(candidate_csv)
    if not rows:
        raise ValueError(f"Candidate schedule CSV is empty: {candidate_csv}")

    filtered_rows: list[dict[str, str]] = []
    required_columns = {"dataset", "seed", "forget_task_id", "schedule_path", "experiment_tag"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Candidate schedule CSV is missing columns: {sorted(missing_columns)}")

    for row in rows:
        row_dataset = str(row.get("dataset", "")).strip()
        if row_dataset != args.dataset:
            continue
        row_seed_text = str(row.get("seed", "")).strip()
        try:
            row_seed = int(row_seed_text)
        except ValueError as exc:
            raise ValueError(f"Invalid seed value in candidate CSV: {row_seed_text!r}") from exc
        if row_seed != args.seed:
            continue

        schedule_rel = str(row.get("schedule_path", "")).strip()
        if schedule_rel == "":
            raise ValueError("Encountered candidate row with empty schedule_path.")
        schedule_path = repo_root / schedule_rel
        if not schedule_path.exists():
            raise ValueError(f"Candidate schedule file not found: {schedule_path}")
        if not schedule_path.is_file():
            raise ValueError(f"Candidate schedule path is not a file: {schedule_path}")

        if args.forget_tasks is not None:
            forget_task_text = str(row.get("forget_task_id", "")).strip()
            try:
                forget_task_id = int(forget_task_text)
            except ValueError as exc:
                raise ValueError(f"Invalid forget_task_id value in candidate CSV: {forget_task_text!r}") from exc
            if forget_task_id not in set(args.forget_tasks):
                continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError(
            f"No candidate rows found for dataset={args.dataset} seed={args.seed}. "
            "Generate them first with tools/search_overlap_schedules.py."
        )
    return filtered_rows


def method_args(method: str, protect_ratio: float | None = None) -> list[str]:
    if method == "pall_adapter":
        if protect_ratio is None:
            raise ValueError("protect_ratio is required for pall_adapter.")
        return [
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            str(protect_ratio),
            "--adapter_train_classifier",
            "--dump_overlap",
        ]
    if method == "pall_modified":
        return [
            "--protect_ratio",
            "0.2",
            "--lambda_protect",
            "0.1",
            "--retrain_steps",
            "50",
            "--dump_overlap",
        ]
    if method == "pall_original":
        return ["--dump_overlap"]
    raise ValueError(f"Unsupported method: {method}")


def protect_ratio_tag(protect_ratio: float) -> str:
    scaled = int(round(protect_ratio * 100))
    return f"p{scaled:03d}"


def build_commands(
    args: argparse.Namespace, repo_root: Path, rows: Sequence[dict[str, str]]
) -> list[tuple[int, str, str, float | None, list[str]]]:
    preset = DATASET_PRESETS[args.dataset]
    main_file = Path(args.main_file).expanduser().resolve()
    commands: list[tuple[int, str, str, float | None, list[str]]] = []

    for row in rows:
        forget_task_id = int(str(row.get("forget_task_id", "")).strip())
        schedule_rel = str(row.get("schedule_path", "")).strip()
        for method in args.methods:
            method_protect_ratios = args.protect_ratios if method == "pall_adapter" else [None]
            for protect_ratio in method_protect_ratios:
                if protect_ratio is None:
                    experiment_tag = f"candidate_{args.dataset}_forget{forget_task_id}_{method}_e{args.epochs}_v1"
                else:
                    experiment_tag = (
                        f"candidate_{args.dataset}_forget{forget_task_id}_{method}_"
                        f"{protect_ratio_tag(protect_ratio)}_e{args.epochs}_v1"
                    )
                command = [
                    args.python,
                    str(main_file),
                    "--dataset",
                    args.dataset,
                    "--class_per_task",
                    str(preset.class_per_task),
                    "--n_tasks",
                    str(preset.n_tasks),
                    "--n_forget",
                    str(preset.n_forget),
                    "--n_epochs",
                    str(args.epochs),
                    "--arch",
                    "resnet18",
                    "--method",
                    method,
                    "--seed",
                    str(args.seed),
                    "--request_schedule_file",
                    schedule_rel,
                    "--experiment_tag",
                    experiment_tag,
                    "--device",
                    args.device,
                    "--num_workers",
                    "0",
                    "--no-pin-memory",
                ]
                command.extend(method_args(method, protect_ratio=protect_ratio))
                commands.append((forget_task_id, method, schedule_rel, protect_ratio, command))
    return commands


def print_command(
    index: int,
    total: int,
    forget_task_id: int,
    method: str,
    schedule_path: str,
    protect_ratio: float | None,
    command: Sequence[str],
) -> None:
    details = f"[{index}/{total}] forget_task_id={forget_task_id} method={method} schedule={schedule_path}"
    if protect_ratio is not None:
        details += f" protect_ratio={protect_ratio}"
    print(details)
    print(f"  {shlex.join(list(command))}")


def run_commands(commands: Sequence[tuple[int, str, str, float | None, list[str]]], cwd: Path, execute: bool) -> int:
    total = len(commands)
    for index, (forget_task_id, method, schedule_path, protect_ratio, command) in enumerate(commands, start=1):
        print_command(
            index=index,
            total=total,
            forget_task_id=forget_task_id,
            method=method,
            schedule_path=schedule_path,
            protect_ratio=protect_ratio,
            command=command,
        )
        if not execute:
            continue
        result = subprocess.run(command, cwd=str(cwd), check=False)
        if result.returncode != 0:
            print(
                f"[ERROR] Command failed with exit code {result.returncode}: {shlex.join(command)}",
                file=sys.stderr,
            )
            return result.returncode
    return 0


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    try:
        rows = validate_args(args, repo_root)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    commands = build_commands(args, repo_root, rows)
    execute = args.run
    if not execute and not args.dry_run:
        print("[INFO] No execution flag provided; printing commands only. Use --run to execute.")
    return run_commands(commands=commands, cwd=repo_root, execute=execute)


if __name__ == "__main__":
    raise SystemExit(main())
