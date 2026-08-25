#!/usr/bin/env python3
"""
Generate or run fixed-schedule pall_adapter ablations.

Example:
python3 tools/run_adapter_ablation.py --dataset cifar10 --epochs 20 --dry-run
python3 tools/run_adapter_ablation.py --dataset cifar100 --seeds 0 1 --epochs 20 --run
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
    fixed_schedule: bool = False


@dataclass(frozen=True)
class AblationConfig:
    name: str
    extra_args: tuple[str, ...]


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "cifar10": DatasetPreset(
        class_per_task=2,
        n_tasks=5,
        n_forget=3,
        schedule_template="schedules/cifar10_t5_f3_fixed_seed{seed}.json",
    ),
    "cifar100": DatasetPreset(
        class_per_task=5,
        n_tasks=10,
        n_forget=3,
        schedule_template="schedules/cifar100_t10_f3_seed{seed}.json",
    ),
    "tinyimagenet": DatasetPreset(
        class_per_task=10,
        n_tasks=20,
        n_forget=3,
        schedule_template="schedules/tinyimagenet_t20_f3_seed0.json",
        fixed_schedule=True,
    ),
}

ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="adapter_no_shared",
        extra_args=(
            "--adapter_shared_bottleneck",
            "0",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_no_critical",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.0",
            "--adapter_shared_protect_ratio",
            "0.0",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_no_protection",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.0",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_critical",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.2",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_critical_p005",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.05",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_critical_p010",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.10",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_shared_critical_p020",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.3",
            "--adapter_shared_protect_ratio",
            "0.20",
            "--adapter_train_classifier",
        ),
    ),
    AblationConfig(
        name="adapter_high_forget_low_protect",
        extra_args=(
            "--adapter_shared_bottleneck",
            "16",
            "--adapter_shared_forget_ratio",
            "0.6",
            "--adapter_shared_protect_ratio",
            "0.1",
            "--adapter_train_classifier",
        ),
    ),
)
VALID_CONFIG_NAMES: tuple[str, ...] = tuple(config.name for config in ABLATION_CONFIGS)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_main = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Generate or run pall_adapter ablation commands.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), default="cifar10")
    parser.add_argument("--epochs", type=int, default=20, help="Value passed to --n_epochs.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seed list.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional ablation config names. Defaults to all configs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--run", action="store_true", help="Execute commands sequentially.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py.")
    parser.add_argument("--main-file", default=str(default_main), help="Path to main.py.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, repo_root: Path) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}.")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed.")
    if args.run and args.dry_run:
        raise ValueError("Choose either --run or --dry-run, not both.")
    if args.configs is not None:
        invalid_configs = [config_name for config_name in args.configs if config_name not in VALID_CONFIG_NAMES]
        if invalid_configs:
            valid_names = ", ".join(VALID_CONFIG_NAMES)
            raise ValueError(
                "Invalid config name(s): {invalid}. Valid config names are: {valid}".format(
                    invalid=", ".join(invalid_configs),
                    valid=valid_names,
                )
            )

    main_file = Path(args.main_file).expanduser().resolve()
    if not main_file.exists():
        raise ValueError(f"main.py not found: {main_file}")
    if not main_file.is_file():
        raise ValueError(f"--main-file is not a file: {main_file}")

    preset = DATASET_PRESETS[args.dataset]
    for seed in args.seeds:
        if seed < 0:
            raise ValueError(f"Seed values must be >= 0, got {seed}.")
        schedule_path = resolve_schedule_path(repo_root=repo_root, seed=seed, preset=preset)
        if not schedule_path.exists():
            if preset.fixed_schedule:
                raise ValueError(
                    "Schedule file not found for tinyimagenet fixed schedule: "
                    f"{schedule_path}. tinyimagenet currently uses only the seed0 schedule."
                )
            raise ValueError(
                f"Schedule file not found for dataset={args.dataset} seed={seed}: {schedule_path}"
            )
        if not schedule_path.is_file():
            raise ValueError(f"Schedule path is not a file: {schedule_path}")


def resolve_schedule_path(repo_root: Path, seed: int, preset: DatasetPreset) -> Path:
    if preset.fixed_schedule:
        return repo_root / preset.schedule_template
    return repo_root / preset.schedule_template.format(seed=seed)


def build_command(
    *,
    python_executable: str,
    main_file: Path,
    dataset: str,
    preset: DatasetPreset,
    repo_root: Path,
    schedule_path: Path,
    seed: int,
    epochs: int,
    device: str,
    config: AblationConfig,
) -> list[str]:
    experiment_tag = f"ablation_{dataset}_{config.name}_e{epochs}_v1"
    command = [
        python_executable,
        str(main_file),
        "--dataset",
        dataset,
        "--class_per_task",
        str(preset.class_per_task),
        "--n_tasks",
        str(preset.n_tasks),
        "--n_forget",
        str(preset.n_forget),
        "--n_epochs",
        str(epochs),
        "--arch",
        "resnet18",
        "--method",
        "pall_adapter",
        "--adapter_bottleneck",
        "16",
        "--seed",
        str(seed),
        "--device",
        device,
        "--num_workers",
        "0",
        "--no-pin-memory",
        "--deterministic",
        "--request_schedule_file",
        str(schedule_path.relative_to(repo_root)),
        "--experiment_tag",
        experiment_tag,
    ]
    command.extend(config.extra_args)
    return command


def build_commands(args: argparse.Namespace, repo_root: Path) -> list[tuple[str, int, list[str]]]:
    preset = DATASET_PRESETS[args.dataset]
    main_file = Path(args.main_file).expanduser().resolve()
    commands: list[tuple[str, int, list[str]]] = []
    selected_configs = ABLATION_CONFIGS
    if args.configs is not None:
        selected_names = set(args.configs)
        selected_configs = tuple(config for config in ABLATION_CONFIGS if config.name in selected_names)

    for seed in args.seeds:
        schedule_path = resolve_schedule_path(repo_root=repo_root, seed=seed, preset=preset)
        for config in selected_configs:
            command = build_command(
                python_executable=args.python,
                main_file=main_file,
                dataset=args.dataset,
                preset=preset,
                repo_root=repo_root,
                schedule_path=schedule_path,
                seed=seed,
                epochs=args.epochs,
                device=args.device,
                config=config,
            )
            commands.append((config.name, seed, command))
    return commands


def print_command(index: int, total: int, config_name: str, seed: int, command: Sequence[str]) -> None:
    print(f"[{index}/{total}] config={config_name} seed={seed}")
    print(f"  {shlex.join(list(command))}")


def run_commands(commands: Sequence[tuple[str, int, list[str]]], cwd: Path, execute: bool) -> int:
    total = len(commands)
    for index, (config_name, seed, command) in enumerate(commands, start=1):
        print_command(index=index, total=total, config_name=config_name, seed=seed, command=command)
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
