#!/usr/bin/env python3
"""
Generate controlled-overlap forgetting schedules compatible with main.py.

This script controls overlap at the request-position level only. The current
pipeline uses fixed task IDs and class-per-task splits, so true class-level
overlap would require a future dataset/task-split extension. The schedules
generated here remain fully compatible with the existing request_schedule_file
JSON format used by main.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List


DATASET_PRESETS = {
    "cifar10": {"n_tasks": 5, "n_forget": 1},
    "cifar100": {"n_tasks": 10, "n_forget": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled-overlap forgetting schedules.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("schedules"))
    return parser.parse_args()


def build_requests_from_plan(n_tasks: int, forget_task: int) -> List[Dict[str, Any]]:
    active_tasks: List[int] = []
    requests: List[Dict[str, Any]] = []

    for task_id in range(n_tasks):
        active_tasks.append(task_id)
        requests.append(
            {
                "task_id": task_id,
                "request_type": "T",
                "active_tasks": list(active_tasks),
            }
        )

    if forget_task not in active_tasks:
        raise ValueError(f"Invalid forget plan: task {forget_task} is not active after sequential training.")
    active_tasks.remove(forget_task)
    requests.append(
        {
            "task_id": forget_task,
            "request_type": "F",
            "active_tasks": list(active_tasks),
        }
    )
    return requests


def schedule_plan(dataset: str) -> Dict[str, int]:
    if dataset == "cifar10":
        return {
            # Request-level overlap proxy only: all tasks are trained
            # sequentially, then one task is forgotten. Class-level overlap
            # still requires a future dataset/task-split extension.
            "low": 4,
            "medium": 2,
            "high": 0,
        }
    if dataset == "cifar100":
        return {
            "low": 9,
            "medium": 5,
            "high": 0,
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_payload(dataset: str, seed: int, setting: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    preset = DATASET_PRESETS[dataset]
    return {
        "n_tasks": preset["n_tasks"],
        "n_forget": preset["n_forget"],
        "seed": seed,
        "sequence_length_requested": preset["n_tasks"] + preset["n_forget"],
        "sequence_length_actual": len(requests),
        "schedule_type": "controlled_overlap_position_proxy",
        "controlled_overlap_setting": setting,
        "notes": (
            "This schedule controls overlap at the request-position level only. "
            "Class-level overlap requires a future dataset/task-split extension."
        ),
        "requests": requests,
    }


def load_main_schedule_validator(repo_root: Path) -> Callable[[str, int], Dict[str, Any]]:
    main_path = repo_root / "main.py"
    source = main_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(main_path))
    target_names = {
        "parse_schedule_entries",
        "parse_schedule_request",
        "build_requests_with_active_tasks",
        "load_request_schedule",
    }
    selected_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in target_names]
    if len(selected_nodes) != len(target_names):
        missing = sorted(target_names.difference({node.name for node in selected_nodes}))
        raise ValueError(f"Failed to locate schedule loader functions in main.py: {missing}")

    extracted_module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: Dict[str, Any] = {"json": json, "Path": Path}
    exec(compile(extracted_module, filename=str(main_path), mode="exec"), namespace)
    return namespace["load_request_schedule"]


def write_schedule(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        print("[ERROR] --seed must be >= 0.", file=sys.stderr)
        return 2

    preset = DATASET_PRESETS[args.dataset]
    plans = schedule_plan(args.dataset)
    repo_root = Path(__file__).resolve().parents[1]
    validate_schedule = load_main_schedule_validator(repo_root)

    generated: List[Path] = []
    for setting in ("low", "medium", "high"):
        plan = plans[setting]
        requests = build_requests_from_plan(
            n_tasks=preset["n_tasks"],
            forget_task=plan,
        )
        payload = build_payload(args.dataset, args.seed, setting, requests)
        out_path = args.out_dir / f"{args.dataset}_controlled_{setting}_seed{args.seed}.json"
        write_schedule(out_path, payload)

        try:
            validate_schedule(str(out_path), preset["n_tasks"])
        except Exception as exc:  # pragma: no cover - surfaced as CLI validation error
            print(f"[ERROR] main.py schedule loader validation failed for {out_path}: {exc}", file=sys.stderr)
            return 2

        generated.append(out_path)
        print(f"[INFO] Generated: {out_path}")
        print(json.dumps(payload, indent=2))
        print()

    print(f"[INFO] Verified {len(generated)} schedules with main.py-compatible loader semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
