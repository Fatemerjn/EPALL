#!/usr/bin/env python3
"""
Run a small fixed-schedule ablation for PALL modified.

Default grid:
- protect_ratio: 0.1, 0.2
- lambda_protect: 0.0, 0.1
- retrain_steps: fixed at 50 unless overridden

Example:
python tools/run_small_ablation.py \
  --dataset cifar10 --arch resnet18 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --seeds 0 1 --request_schedule_file schedules/cifar10_t5_f3_seed0.json --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


DEFAULT_PROTECT_RATIOS = [0.1, 0.2]
DEFAULT_LAMBDA_PROTECT = [0.0, 0.1]


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset == "cifar100":
        # data.py currently enforces CIFAR-100 superclass tasks.
        if args.class_per_task != 5:
            raise ValueError("For cifar100, class_per_task must be 5 with current codebase.")
        if not (1 <= args.n_tasks <= 20):
            raise ValueError("For cifar100, n_tasks must be in [1, 20] with current codebase.")
    elif args.dataset == "cifar10":
        if args.class_per_task * args.n_tasks > 10:
            raise ValueError("For cifar10, class_per_task * n_tasks must be <= 10.")
    elif args.dataset == "tinyimagenet":
        if args.class_per_task * args.n_tasks > 200:
            raise ValueError("For tinyimagenet, class_per_task * n_tasks must be <= 200.")

    schedule_path = Path(args.request_schedule_file).expanduser()
    if not schedule_path.exists():
        raise ValueError(f"request_schedule_file not found: {schedule_path}")
    if not schedule_path.is_file():
        raise ValueError(f"request_schedule_file is not a file: {schedule_path}")
    args.request_schedule_file = str(schedule_path.resolve())

    if any(r < 0.0 or r > 1.0 for r in args.protect_ratios):
        raise ValueError("protect_ratio values must be in [0, 1].")
    if any(l < 0.0 for l in args.lambda_protect_grid):
        raise ValueError("lambda_protect values must be >= 0.")
    if args.retrain_steps < 0:
        raise ValueError("retrain_steps must be >= 0.")


def build_commands(args: argparse.Namespace, main_file: Path) -> List[List[str]]:
    commands: List[List[str]] = []
    grid = itertools.product(args.seeds, args.protect_ratios, args.lambda_protect_grid)

    for seed, protect_ratio, lambda_protect in grid:
        cmd = [
            args.python,
            str(main_file),
            "--dataset",
            args.dataset,
            "--arch",
            args.arch,
            "--class_per_task",
            str(args.class_per_task),
            "--n_tasks",
            str(args.n_tasks),
            "--n_forget",
            str(args.n_forget),
            "--seed",
            str(seed),
            "--method",
            "pall_modified",
            "--data_dir",
            args.data_dir,
            "--request_schedule_file",
            args.request_schedule_file,
            "--protect_ratio",
            str(protect_ratio),
            "--lambda_protect",
            str(lambda_protect),
            "--retrain_steps",
            str(args.retrain_steps),
        ]
        if not args.no_deterministic:
            cmd.append("--deterministic")
        if args.dump_overlap:
            cmd.append("--dump_overlap")
        if args.debug_unlearning:
            cmd.append("--debug_unlearning")
        commands.append(cmd)

    return commands


def run_commands(commands: List[List[str]], cwd: Path, dry_run: bool, continue_on_error: bool) -> int:
    exit_code = 0
    total = len(commands)
    for idx, cmd in enumerate(commands, start=1):
        shell_line = shlex.join(cmd)
        seed = cmd[cmd.index("--seed") + 1]
        protect_ratio = cmd[cmd.index("--protect_ratio") + 1]
        lambda_protect = cmd[cmd.index("--lambda_protect") + 1]
        print(f"[{idx}/{total}] seed={seed} protect_ratio={protect_ratio} lambda_protect={lambda_protect}")
        print(f"  {shell_line}")
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(cwd), check=False)
        if result.returncode != 0:
            exit_code = result.returncode
            print(f"[ERROR] Command failed with exit code {result.returncode}", file=sys.stderr)
            if not continue_on_error:
                return exit_code
    return exit_code


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_main = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Run a small fixed-schedule PALL modified ablation.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py.")
    parser.add_argument("--main-file", default=str(default_main), help="Path to main.py.")
    parser.add_argument(
        "--output-root",
        default=".",
        help="Working directory for subprocess runs (runs/ will be created under this root).",
    )

    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--class_per_task", type=int, default=2)
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--n_forget", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Seed list.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--request_schedule_file", required=True, help="Path to fixed request schedule JSON file.")

    parser.add_argument("--protect-ratios", type=float, nargs="+", default=DEFAULT_PROTECT_RATIOS)
    parser.add_argument("--lambda-protect-grid", type=float, nargs="+", default=DEFAULT_LAMBDA_PROTECT)
    parser.add_argument("--retrain_steps", type=int, default=50, help="Fixed retrain_steps for all runs.")

    parser.add_argument("--dump-overlap", action="store_true")
    parser.add_argument("--debug-unlearning", action="store_true")
    parser.add_argument("--no-deterministic", action="store_true", help="Disable --deterministic.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if a run command fails.")
    parser.add_argument(
        "--commands-file",
        default=None,
        help="Path for command log file. Defaults to <output-root>/small_ablation_commands_<timestamp>.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    main_file = Path(args.main_file).resolve()
    if not main_file.exists():
        print(f"[ERROR] main.py not found: {main_file}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    commands = build_commands(args, main_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.commands_file is None:
        commands_file = output_root / f"small_ablation_commands_{timestamp}.txt"
    else:
        commands_file = Path(args.commands_file).resolve()
        commands_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [shlex.join(cmd) for cmd in commands]
    commands_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Saved {len(lines)} commands to: {commands_file}")

    return run_commands(
        commands=commands,
        cwd=output_root,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
