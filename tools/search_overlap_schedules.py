#!/usr/bin/env python3
"""
Generate candidate single-forget schedules for overlap search experiments.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List


DATASET_PRESETS = {
    "cifar10": {"n_tasks": 5, "n_forget": 1},
    "cifar100": {"n_tasks": 10, "n_forget": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate single-forget schedules.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("schedules"))
    return parser.parse_args()


def build_requests(n_tasks: int, forget_task: int) -> List[Dict[str, Any]]:
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
        raise ValueError(f"Invalid forget task {forget_task} for n_tasks={n_tasks}.")
    active_tasks.remove(forget_task)
    requests.append(
        {
            "task_id": forget_task,
            "request_type": "F",
            "active_tasks": list(active_tasks),
        }
    )
    return requests


def build_payload(dataset: str, seed: int, forget_task_id: int, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    preset = DATASET_PRESETS[dataset]
    return {
        "n_tasks": preset["n_tasks"],
        "n_forget": preset["n_forget"],
        "seed": seed,
        "sequence_length_requested": preset["n_tasks"] + preset["n_forget"],
        "sequence_length_actual": len(requests),
        "schedule_type": "candidate_single_forget_search",
        "forget_task_id": forget_task_id,
        "notes": (
            "Candidate schedule for measured overlap search. "
            "All tasks are trained sequentially, then one task is forgotten."
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


def write_index_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "seed", "forget_task_id", "schedule_path", "experiment_tag"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.seed < 0:
        print("[ERROR] --seed must be >= 0.", file=sys.stderr)
        return 2

    preset = DATASET_PRESETS[args.dataset]
    repo_root = Path(__file__).resolve().parents[1]
    validate_schedule = load_main_schedule_validator(repo_root)

    index_rows: List[Dict[str, str]] = []
    for forget_task_id in range(preset["n_tasks"]):
        requests = build_requests(preset["n_tasks"], forget_task_id)
        payload = build_payload(args.dataset, args.seed, forget_task_id, requests)
        out_path = args.out_dir / f"{args.dataset}_candidate_forget_task{forget_task_id}_seed{args.seed}.json"
        write_schedule(out_path, payload)

        try:
            validate_schedule(str(out_path), preset["n_tasks"])
        except Exception as exc:  # pragma: no cover - surfaced as CLI validation error
            print(f"[ERROR] main.py schedule loader validation failed for {out_path}: {exc}", file=sys.stderr)
            return 2

        experiment_tag = f"candidate_{args.dataset}_forget{forget_task_id}_adapter_e{{epochs_placeholder}}_v1"
        index_rows.append(
            {
                "dataset": args.dataset,
                "seed": str(args.seed),
                "forget_task_id": str(forget_task_id),
                "schedule_path": str(out_path),
                "experiment_tag": experiment_tag,
            }
        )
        print(f"[INFO] Generated: {out_path}")

    index_path = repo_root / "results" / "thesis" / f"{args.dataset}_candidate_schedules.csv"
    write_index_csv(index_path, index_rows)
    print(f"[INFO] Wrote candidate index CSV: {index_path}")
    print(f"[INFO] Verified {len(index_rows)} schedules with main.py-compatible loader semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
