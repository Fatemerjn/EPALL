#!/usr/bin/env python3
"""
Generate or run controlled-overlap position-proxy experiments.
"""

from __future__ import annotations

import argparse
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
    schedule_template: str


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "cifar10": DatasetPreset(
        class_per_task=2,
        n_tasks=5,
        n_forget=1,
        schedule_template="schedules/cifar10_controlled_{setting}_seed{seed}.json",
    ),
    "cifar100": DatasetPreset(
        class_per_task=5,
        n_tasks=10,
        n_forget=1,
        schedule_template="schedules/cifar100_controlled_{setting}_seed{seed}.json",
    ),
}

SUPPORTED_METHODS = ("pall_original", "pall_modified", "pall_adapter")
SUPPORTED_SETTINGS = ("low", "medium", "high")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_main = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Generate or run controlled-overlap experiments.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seed list.")
    parser.add_argument("--epochs", type=int, required=True, help="Value passed to --n_epochs.")
    parser.add_argument("--methods", nargs="+", default=list(SUPPORTED_METHODS), help="Methods to run.")
    parser.add_argument("--settings", nargs="+", default=list(SUPPORTED_SETTINGS), help="Controlled settings to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--run", action="store_true", help="Execute commands sequentially.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py.")
    parser.add_argument("--main-file", default=str(default_main), help="Path to main.py.")
    return parser.parse_args()


def schedule_path_for(repo_root: Path, dataset: str, setting: str, seed: int) -> Path:
    preset = DATASET_PRESETS[dataset]
    return repo_root / preset.schedule_template.format(setting=setting, seed=seed)


def validate_args(args: argparse.Namespace, repo_root: Path) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if args.run and args.dry_run:
        raise ValueError("Choose either --run or --dry-run, not both.")

    invalid_methods = [method for method in args.methods if method not in SUPPORTED_METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}. Valid methods: {list(SUPPORTED_METHODS)}")
    invalid_settings = [setting for setting in args.settings if setting not in SUPPORTED_SETTINGS]
    if invalid_settings:
        raise ValueError(f"Unsupported settings: {invalid_settings}. Valid settings: {list(SUPPORTED_SETTINGS)}")

    main_file = Path(args.main_file).expanduser().resolve()
    if not main_file.exists():
        raise ValueError(f"main.py not found: {main_file}")
    if not main_file.is_file():
        raise ValueError(f"--main-file is not a file: {main_file}")

    for seed in args.seeds:
        if seed < 0:
            raise ValueError(f"Seed values must be >= 0, got {seed}.")
        for setting in args.settings:
            schedule_path = schedule_path_for(repo_root, args.dataset, setting, seed)
            if not schedule_path.exists():
                raise ValueError(f"Controlled schedule not found: {schedule_path}")
            if not schedule_path.is_file():
                raise ValueError(f"Controlled schedule path is not a file: {schedule_path}")


def method_args(method: str) -> list[str]:
    if method == "pall_adapter":
        return [
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.1",
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


def build_commands(args: argparse.Namespace, repo_root: Path) -> list[tuple[str, str, int, list[str]]]:
    preset = DATASET_PRESETS[args.dataset]
    main_file = Path(args.main_file).expanduser().resolve()
    commands: list[tuple[str, str, int, list[str]]] = []

    for seed in args.seeds:
        for setting in args.settings:
            schedule_path = schedule_path_for(repo_root, args.dataset, setting, seed)
            for method in args.methods:
                experiment_tag = f"controlled_{args.dataset}_{setting}_{method}_e{args.epochs}_v1"
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
                    str(seed),
                    "--device",
                    args.device,
                    "--num_workers",
                    "0",
                    "--no-pin-memory",
                    "--deterministic",
                    "--request_schedule_file",
                    str(schedule_path.relative_to(repo_root)),
                    "--experiment_tag",
                    experiment_tag,
                ]
                command.extend(method_args(method))
                commands.append((setting, method, seed, command))
    return commands


def print_command(index: int, total: int, setting: str, method: str, seed: int, command: Sequence[str]) -> None:
    print(f"[{index}/{total}] setting={setting} method={method} seed={seed}")
    print(f"  {shlex.join(list(command))}")


def run_commands(commands: Sequence[tuple[str, str, int, list[str]]], cwd: Path, execute: bool) -> int:
    total = len(commands)
    for index, (setting, method, seed, command) in enumerate(commands, start=1):
        print_command(index=index, total=total, setting=setting, method=method, seed=seed, command=command)
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
        validate_args(args, repo_root)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    commands = build_commands(args, repo_root)
    execute = args.run
    if not execute and not args.dry_run:
        print("[INFO] No execution flag provided; printing commands only. Use --run to execute.")
    return run_commands(commands=commands, cwd=repo_root, execute=execute)


if __name__ == "__main__":
    raise SystemExit(main())
