# Forgotten-Task Persistence Audit

Read-only analysis of the latest matched MAIN-equivalent runs.
The forgotten task is followed from immediately after its deletion to the end of the schedule.
Rebound is measured on the distance to chance `|A_u - c|`, not on raw accuracy: for a forgotten
task, drifting below chance is as much a failure as drifting above it.
The MAIN schedules do contain training requests after the first deletion, so each gap is
attributed either to the intervening training requests (`train`) or to the next deletion (`delete`).
The two attributed drifts sum to the total change in `|A_u - c|`.
Rebound columns cover only tasks with a non-empty follow-up horizon; a task deleted by the last
request of the schedule has none.

Selected runs: 20; seeds per dataset/method: [0, 1].

| Dataset | Method | Seeds | Deleted | Followed | Immediate abs(Au-c) | Final abs(Au-c) | Mean max rebound | Worst rebound | Rebound >1pt | Train-attributed | Delete-attributed | Net |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar100 | PALL Original | 2 | 6 | 4 | 0.0087 | 0.0017 | 0.0030 | 0.0100 | 0.0% | -0.0090 | -0.0015 | -0.0105 |
| cifar100 | PALL Modified | 2 | 6 | 4 | 0.0137 | 0.0040 | 0.0040 | 0.0100 | 25.0% | -0.0120 | -0.0025 | -0.0145 |
| cifar100 | PALL Adapter | 2 | 6 | 4 | 0.0230 | 0.0260 | 0.0145 | 0.0500 | 25.0% | +0.0100 | -0.0055 | +0.0045 |
| cifar100 | LoRA | 2 | 6 | 4 | 0.0053 | 0.0053 | 0.0000 | 0.0000 | 0.0% | +0.0000 | +0.0000 | +0.0000 |
| cifar100 | CLPU | 2 | 6 | 4 | 0.0140 | 0.0140 | 0.0000 | 0.0000 | 0.0% | +0.0000 | +0.0000 | +0.0000 |
| cifar10 | PALL Original | 2 | 6 | 4 | 0.0011 | 0.0097 | 0.0179 | 0.0490 | 50.0% | +0.0019 | +0.0111 | +0.0130 |
| cifar10 | PALL Modified | 2 | 6 | 4 | 0.0018 | 0.0024 | 0.0136 | 0.0370 | 50.0% | +0.0063 | -0.0053 | +0.0010 |
| cifar10 | PALL Adapter | 2 | 6 | 4 | 0.0182 | 0.0098 | 0.0024 | 0.0060 | 0.0% | -0.0150 | +0.0024 | -0.0126 |
| cifar10 | LoRA | 2 | 6 | 4 | 0.0071 | 0.0071 | 0.0000 | 0.0000 | 0.0% | +0.0000 | +0.0000 | +0.0000 |
| cifar10 | CLPU | 2 | 6 | 4 | 0.0198 | 0.0198 | 0.0000 | 0.0000 | 0.0% | +0.0000 | +0.0000 | +0.0000 |

## Selected run trace

- `cifar100` / `pall_original` / seed 0: `runs/cifar100/T10_F3/pall_original/seed_0/20260712_194740`
- `cifar100` / `pall_original` / seed 1: `runs/cifar100/T10_F3/pall_original/seed_1/20260712_204605`
- `cifar100` / `pall_modified` / seed 0: `runs/cifar100/T10_F3/pall_modified/seed_0/20260712_195858`
- `cifar100` / `pall_modified` / seed 1: `runs/cifar100/T10_F3/pall_modified/seed_1/20260712_205711`
- `cifar100` / `pall_adapter` / seed 0: `runs/cifar100/T10_F3/pall_adapter/seed_0/20260712_201029`
- `cifar100` / `pall_adapter` / seed 1: `runs/cifar100/T10_F3/pall_adapter/seed_1/20260712_210833`
- `cifar100` / `lora` / seed 0: `runs/cifar100/T10_F3/lora/seed_0/20260712_201246`
- `cifar100` / `lora` / seed 1: `runs/cifar100/T10_F3/lora/seed_1/20260712_211046`
- `cifar100` / `clpu` / seed 0: `runs/cifar100/T10_F3/clpu/seed_0/20260712_201448`
- `cifar100` / `clpu` / seed 1: `runs/cifar100/T10_F3/clpu/seed_1/20260712_211248`
- `cifar10` / `pall_original` / seed 0: `runs/cifar10/T5_F3/pall_original/seed_0/20260712_191843`
- `cifar10` / `pall_original` / seed 1: `runs/cifar10/T5_F3/pall_original/seed_1/20260712_201728`
- `cifar10` / `pall_modified` / seed 0: `runs/cifar10/T5_F3/pall_modified/seed_0/20260712_192947`
- `cifar10` / `pall_modified` / seed 1: `runs/cifar10/T5_F3/pall_modified/seed_1/20260712_202826`
- `cifar10` / `pall_adapter` / seed 0: `runs/cifar10/T5_F3/pall_adapter/seed_0/20260712_194051`
- `cifar10` / `pall_adapter` / seed 1: `runs/cifar10/T5_F3/pall_adapter/seed_1/20260712_203930`
- `cifar10` / `lora` / seed 0: `runs/cifar10/T5_F3/lora/seed_0/20260712_194308`
- `cifar10` / `lora` / seed 1: `runs/cifar10/T5_F3/lora/seed_1/20260712_204137`
- `cifar10` / `clpu` / seed 0: `runs/cifar10/T5_F3/clpu/seed_0/20260712_194501`
- `cifar10` / `clpu` / seed 1: `runs/cifar10/T5_F3/clpu/seed_1/20260712_204329`
