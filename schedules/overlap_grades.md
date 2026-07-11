# Controlled overlap grades

These schedules vary the same request-position proxy used by
`tools/generate_controlled_overlap_schedules.py`.  The target knob `K` is the
number of training requests after the task that is eventually forgotten:
`forget_task = n_tasks - 1 - K`.  A larger `K` means a higher intended
request-position overlap.  The normalized target shown below is
`K / (n_tasks - 1)`; it is a schedule-design target, **not** a claim about the
measured parameter-mask overlap produced by a method.

All schedules train tasks in ascending order and append one forget request.
The seed is recorded in the schedule metadata and selects the corresponding
experiment seed; request positions themselves are deterministic.

| Dataset | Grade | Target knob `K` (later tasks) | Normalized target | Forgotten task | Seed | Schedule file |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | `very_low` | 0 | 0.0000 | 4 | 0 | `cifar10_controlled_very_low_later0_seed0.json` |
| CIFAR-10 | `very_low` | 0 | 0.0000 | 4 | 1 | `cifar10_controlled_very_low_later0_seed1.json` |
| CIFAR-10 | `low` | 1 | 0.2500 | 3 | 0 | `cifar10_controlled_low_later1_seed0.json` |
| CIFAR-10 | `low` | 1 | 0.2500 | 3 | 1 | `cifar10_controlled_low_later1_seed1.json` |
| CIFAR-10 | `medium` | 2 | 0.5000 | 2 | 0 | `cifar10_controlled_medium_later2_seed0.json` |
| CIFAR-10 | `medium` | 2 | 0.5000 | 2 | 1 | `cifar10_controlled_medium_later2_seed1.json` |
| CIFAR-10 | `high` | 3 | 0.7500 | 1 | 0 | `cifar10_controlled_high_later3_seed0.json` |
| CIFAR-10 | `high` | 3 | 0.7500 | 1 | 1 | `cifar10_controlled_high_later3_seed1.json` |
| CIFAR-10 | `very_high` | 4 | 1.0000 | 0 | 0 | `cifar10_controlled_very_high_later4_seed0.json` |
| CIFAR-10 | `very_high` | 4 | 1.0000 | 0 | 1 | `cifar10_controlled_very_high_later4_seed1.json` |
| CIFAR-100 | `very_low` | 0 | 0.0000 | 9 | 0 | `cifar100_controlled_very_low_later0_seed0.json` |
| CIFAR-100 | `very_low` | 0 | 0.0000 | 9 | 1 | `cifar100_controlled_very_low_later0_seed1.json` |
| CIFAR-100 | `low` | 2 | 0.2222 | 7 | 0 | `cifar100_controlled_low_later2_seed0.json` |
| CIFAR-100 | `low` | 2 | 0.2222 | 7 | 1 | `cifar100_controlled_low_later2_seed1.json` |
| CIFAR-100 | `medium` | 4 | 0.4444 | 5 | 0 | `cifar100_controlled_medium_later4_seed0.json` |
| CIFAR-100 | `medium` | 4 | 0.4444 | 5 | 1 | `cifar100_controlled_medium_later4_seed1.json` |
| CIFAR-100 | `high` | 7 | 0.7778 | 2 | 0 | `cifar100_controlled_high_later7_seed0.json` |
| CIFAR-100 | `high` | 7 | 0.7778 | 2 | 1 | `cifar100_controlled_high_later7_seed1.json` |
| CIFAR-100 | `very_high` | 9 | 1.0000 | 0 | 0 | `cifar100_controlled_very_high_later9_seed0.json` |
| CIFAR-100 | `very_high` | 9 | 1.0000 | 0 | 1 | `cifar100_controlled_very_high_later9_seed1.json` |

## Generation and validation

From the repository root, the complete set is regenerated deterministically
with:

```bash
./.venv/bin/python tools/generate_controlled_overlap_schedules.py --dataset cifar10 --seed 0
./.venv/bin/python tools/generate_controlled_overlap_schedules.py --dataset cifar10 --seed 1
./.venv/bin/python tools/generate_controlled_overlap_schedules.py --dataset cifar100 --seed 0
./.venv/bin/python tools/generate_controlled_overlap_schedules.py --dataset cifar100 --seed 1
```

The generator constructs the ascending train requests plus the final forget
request, writes the target knob, normalized target, and forgotten task into the
JSON metadata, and immediately validates every file through the schedule-loader
functions extracted from `main.py`.  A nonzero generator exit therefore means
generation or loader validation failed.

## Legacy filenames

The six pre-existing seed-0 files named
`{dataset}_controlled_{low,medium,high}_seed0.json` are retained unchanged for
compatibility with `tools/run_controlled_overlap_experiments.py`.  In that old
three-point convention, `low` was the `K=0` endpoint and `high` was the maximum
`K` endpoint.  The canonical five-point benchmark uses only the filenames in
the table above, whose grade and knob are both explicit.
