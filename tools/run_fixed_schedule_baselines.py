#!/usr/bin/env python3
"""
Run baseline comparisons with a fixed learn/forget request schedule.

Default methods:
- pall_original
- pall_modified
- er
- derpp

Optional methods can be selected with --methods, including ewc and lwf.
All runs receive the same --request_schedule_file.

Example:
python tools/run_fixed_schedule_baselines.py \
  --dataset cifar10 --arch resnet18 --class_per_task 2 --n_tasks 5 --n_forget 3 \
  --seeds 0 1 --request_schedule_file schedules/cifar10_t5_f3_seed0.json --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


DEFAULT_METHODS = ["pall_original", "pall_modified", "er", "derpp"]
SUPPORTED_METHODS = {"pall_original", "pall_modified", "er", "derpp", "ewc", "lwf"}


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

    invalid = [method for method in args.methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods for this runner: {invalid}")
    if args.forget_iters < 0:
        raise ValueError("forget_iters must be >= 0.")


def build_commands(args: argparse.Namespace, main_file: Path) -> List[List[str]]:
    commands: List[List[str]] = []
    for seed in args.seeds:
        for method in args.methods:
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
                method,
                "--data_dir",
                args.data_dir,
                "--request_schedule_file",
                args.request_schedule_file,
            ]
            if not args.no_deterministic:
                cmd.append("--deterministic")
            # ER-style methods require forget_iters for forgetting runs.
            if method in {"er", "derpp"}:
                cmd.extend(["--forget_iters", str(args.forget_iters)])
            commands.append(cmd)
    return commands


def write_commands_log(commands: Iterable[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [shlex.join(cmd) for cmd in commands]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_commands(commands: List[List[str]], cwd: Path, dry_run: bool, continue_on_error: bool) -> int:
    exit_code = 0
    total = len(commands)
    for idx, cmd in enumerate(commands, start=1):
        shell_line = shlex.join(cmd)
        method = cmd[cmd.index("--method") + 1]
        seed = cmd[cmd.index("--seed") + 1]
        print(f"[{idx}/{total}] method={method} seed={seed}")
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

    parser = argparse.ArgumentParser(description="Run fixed-schedule baseline comparisons.")
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

    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods to run.")
    parser.add_argument(
        "--forget-iters",
        type=int,
        default=50,
        help="forget_iters passed only to ER/DERPP methods.",
    )
    parser.add_argument("--no-deterministic", action="store_true", help="Disable --deterministic.")

    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue if a run command fails.")
    parser.add_argument(
        "--commands-file",
        default=None,
        help="Path for command log file. Defaults to <output-root>/fixed_schedule_baseline_commands_<timestamp>.txt",
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
        commands_file = output_root / f"fixed_schedule_baseline_commands_{timestamp}.txt"
    else:
        commands_file = Path(args.commands_file).resolve()

    write_commands_log(commands, commands_file)
    print(f"[INFO] Saved {len(commands)} commands to: {commands_file}")

    return run_commands(
        commands=commands,
        cwd=output_root,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
