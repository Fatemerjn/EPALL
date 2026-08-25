#!/usr/bin/env python3
"""
Export a reproducible learn/forget request schedule to JSON.

Usage:
  python tools/export_request_schedule.py --n_tasks 5 --n_forget 3 --seed 0 --out schedules/cifar10_t5_f3_seed0.json

The exported file can be consumed by:
  python main.py --request_schedule_file <path> ...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def generate_user_requests(num_tasks: int, sequence_length: int) -> List[Tuple[int, str]]:
    """Mirror main.py request generation behavior."""
    if sequence_length < num_tasks:
        raise ValueError("Sequence length must be at least the number of tasks.")

    user_requests = [(i, "T") for i in range(num_tasks)]
    trained_tasks = list(range(num_tasks))

    remaining_slots = sequence_length - num_tasks
    f_requests = []
    while remaining_slots > 0 and trained_tasks:
        task = random.choice(trained_tasks)
        f_requests.append((task, "F"))
        trained_tasks.pop(trained_tasks.index(task))
        remaining_slots -= 1

    for f_request in f_requests:
        t_index = user_requests.index((f_request[0], "T"))
        valid_positions = list(range(t_index + 1, len(user_requests) + 1))
        insert_position = random.choice(valid_positions)
        user_requests.insert(insert_position, f_request)

    return user_requests


def with_active_tasks(user_requests: List[Tuple[int, str]]):
    active_tasks: List[int] = []
    requests = []
    for task_id, request_type in user_requests:
        if request_type == "T":
            active_tasks.append(task_id)
        else:
            active_tasks.remove(task_id)
        requests.append(
            {
                "task_id": task_id,
                "request_type": request_type,
                "active_tasks": list(active_tasks),
            }
        )
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed learn/forget request schedule.")
    parser.add_argument("--n_tasks", type=int, required=True, help="Number of tasks.")
    parser.add_argument("--n_forget", type=int, required=True, help="Number of forget requests to simulate.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for schedule generation.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON schedule path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_tasks <= 0:
        raise ValueError("--n_tasks must be > 0.")
    if args.n_forget < 0:
        raise ValueError("--n_forget must be >= 0.")

    random.seed(args.seed)
    sequence_length = int(args.n_tasks + args.n_forget)
    requests = generate_user_requests(num_tasks=args.n_tasks, sequence_length=sequence_length)
    requests_with_active = with_active_tasks(requests)

    payload = {
        "n_tasks": args.n_tasks,
        "n_forget": args.n_forget,
        "seed": args.seed,
        "sequence_length_requested": sequence_length,
        "sequence_length_actual": len(requests_with_active),
        "requests": requests_with_active,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[INFO] Wrote schedule to: {args.out}")
    print(f"[INFO] Requests: {len(requests_with_active)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
