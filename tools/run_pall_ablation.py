#!/usr/bin/env python3
"""
Run ablations for overlap-aware selective forgetting (PALL modified).

This script generates and executes `main.py` commands over a practical grid of:
- protect_ratio (or protect_threshold)
- lambda_protect
- adaptive_retrain
- optional retrain_steps

It prints every exact command, and writes the full launched command list to a
text file for reproducibility.

Example:
python tools/run_pall_ablation.py --dataset cifar10 --seeds 0 1 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


# Practical default grid (not exhaustive brute force).
DEFAULT_PROTECT_RATIOS = [0.1, 0.2]
DEFAULT_LAMBDA_PROTECT = [0.0, 0.1]
DEFAULT_ADAPTIVE_RETRAIN = [False, True]
DEFAULT_RETRAIN_STEPS = ["50"]  # use ["none", "50"] to include k_shot fallback runs.


def parse_bool_token(token: str) -> bool:
    v = token.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool token: {token}")


def parse_retrain_steps(tokens: Iterable[str]) -> List[Optional[int]]:
    steps: List[Optional[int]] = []
    for token in tokens:
        t = token.strip().lower()
        if t in {"none", "null"}:
            steps.append(None)
            continue
        value = int(t)
        if value < 0:
            raise ValueError(f"retrain_steps must be >= 0, got: {value}")
        steps.append(value)
    return steps


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset == "cifar100":
        # Current data.py enforces CIFAR-100 superclass task split.
        if args.class_per_task != 5:
            raise ValueError("For cifar100, class_per_task must be 5 with current codebase.")
        if args.n_tasks > 20:
            raise ValueError("For cifar100, n_tasks must be <= 20 with current codebase.")
    elif args.dataset == "cifar10":
        if args.class_per_task * args.n_tasks > 10:
            raise ValueError("For cifar10, class_per_task * n_tasks must be <= 10.")
    elif args.dataset == "tinyimagenet":
        if args.class_per_task * args.n_tasks > 200:
            raise ValueError("For tinyimagenet, class_per_task * n_tasks must be <= 200.")

    if any(r < 0.0 or r > 1.0 for r in args.protect_ratios):
        raise ValueError("protect_ratio values must be in [0, 1].")
    if any(t < 0.0 for t in args.protect_thresholds):
        raise ValueError("protect_threshold values must be >= 0.")
    if any(l < 0.0 for l in args.lambda_protect_grid):
        raise ValueError("lambda_protect values must be >= 0.")


def build_commands(args: argparse.Namespace, main_file: Path) -> List[List[str]]:
    commands: List[List[str]] = []

    if args.protection_mode == "ratio":
        protection_grid = [("ratio", ratio) for ratio in args.protect_ratios]
    elif args.protection_mode == "threshold":
        protection_grid = [("threshold", thr) for thr in args.protect_thresholds]
    else:
        # Explicitly requested combined mode (includes both flags in each command).
        protection_grid = [
            ("combined", (ratio, thr))
            for ratio in args.protect_ratios
            for thr in args.protect_thresholds
        ]

    combo_iter = itertools.product(
        args.seeds,
        protection_grid,
        args.lambda_protect_grid,
        args.adaptive_retrain_grid,
        args.retrain_steps_grid,
    )

    for seed, protection_item, lambda_protect, adaptive_retrain, retrain_steps in combo_iter:
        cmd = [
            args.python,
            str(main_file),
            "--dataset",
            args.dataset,
            "--class_per_task",
            str(args.class_per_task),
            "--n_tasks",
            str(args.n_tasks),
            "--n_forget",
            str(args.n_forget),
            "--arch",
            args.arch,
            "--method",
            args.method,
            "--seed",
            str(seed),
            "--lambda_protect",
            str(lambda_protect),
            "--data_dir",
            args.data_dir,
            "--k_shot",
            str(args.k_shot),
        ]

        if not args.no_deterministic:
            cmd.append("--deterministic")
        if args.debug_unlearning:
            cmd.append("--debug_unlearning")
        if args.dump_overlap:
            cmd.append("--dump_overlap")
        if adaptive_retrain:
            cmd.append("--adaptive_retrain")
        if retrain_steps is not None:
            cmd.extend(["--retrain_steps", str(retrain_steps)])

        mode, value = protection_item
        if mode == "ratio":
            cmd.extend(["--protect_ratio", str(value)])
        elif mode == "threshold":
            cmd.extend(["--protect_threshold", str(value)])
        else:
            ratio, threshold = value
            cmd.extend(["--protect_ratio", str(ratio), "--protect_threshold", str(threshold)])

        commands.append(cmd)

    # Keep generated command list stable and deduplicate exact duplicates.
    seen = set()
    unique_commands: List[List[str]] = []
    for cmd in commands:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            unique_commands.append(cmd)
    return unique_commands


def to_shell_lines(commands: List[List[str]]) -> List[str]:
    return [shlex.join(cmd) for cmd in commands]


def run_commands(commands: List[List[str]], cwd: Path, dry_run: bool, continue_on_error: bool) -> int:
    exit_code = 0
    for idx, cmd in enumerate(commands, start=1):
        shell_line = shlex.join(cmd)
        print(f"[{idx}/{len(commands)}] {shell_line}")
        if dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(cwd), check=False)
        if result.returncode != 0:
            exit_code = result.returncode
            print(f"[ERROR] Command failed with exit code {result.returncode}: {shell_line}", file=sys.stderr)
            if not continue_on_error:
                return exit_code
    return exit_code


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_main = repo_root / "main.py"

    parser = argparse.ArgumentParser(description="Automate PALL modified ablation runs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for running main.py.")
    parser.add_argument("--main-file", default=str(default_main), help="Path to main.py.")
    parser.add_argument("--output-root", default=".", help="Working directory for runs/ outputs.")

    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--class_per_task", type=int, default=2)
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--n_forget", type=int, default=3)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Seed list.")

    parser.add_argument("--method", default="pall_modified", choices=["pall_modified", "pall_original", "pall"])
    parser.add_argument("--k_shot", type=int, default=50, help="Fallback retrain steps when no override is set.")
    parser.add_argument("--no-deterministic", action="store_true", help="Disable --deterministic flag.")

    parser.add_argument(
        "--protection-mode",
        choices=["ratio", "threshold", "combined"],
        default="ratio",
        help="ratio/threshold are alternative modes; combined is explicit mixed mode.",
    )
    parser.add_argument("--protect-ratios", type=float, nargs="+", default=DEFAULT_PROTECT_RATIOS)
    parser.add_argument("--protect-thresholds", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--lambda-protect-grid", type=float, nargs="+", default=DEFAULT_LAMBDA_PROTECT)
    parser.add_argument(
        "--adaptive-retrain-grid",
        nargs="+",
        default=["false", "true"],
        help="Bool tokens (e.g., false true).",
    )
    parser.add_argument(
        "--retrain-steps-grid",
        nargs="+",
        default=DEFAULT_RETRAIN_STEPS,
        help='Retrain steps grid, use "none" to omit --retrain_steps and use k_shot fallback.',
    )

    parser.add_argument("--debug-unlearning", action="store_true")
    parser.add_argument("--dump-overlap", action="store_true")

    parser.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining commands if one command fails.",
    )
    parser.add_argument(
        "--commands-file",
        default=None,
        help="Path to save generated command list. Defaults to <output-root>/pall_ablation_commands_<timestamp>.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        args.adaptive_retrain_grid = [parse_bool_token(v) for v in args.adaptive_retrain_grid]
        args.retrain_steps_grid = parse_retrain_steps(args.retrain_steps_grid)
        validate_args(args)
    except (ValueError, TypeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    main_file = Path(args.main_file).resolve()
    if not main_file.exists():
        print(f"[ERROR] main.py not found: {main_file}", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    commands = build_commands(args, main_file)
    shell_lines = to_shell_lines(commands)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.commands_file is None:
        commands_file = output_root / f"pall_ablation_commands_{timestamp}.txt"
    else:
        commands_file = Path(args.commands_file).resolve()
        commands_file.parent.mkdir(parents=True, exist_ok=True)

    commands_file.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Saved {len(shell_lines)} commands to: {commands_file}")

    return run_commands(
        commands=commands,
        cwd=output_root,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
