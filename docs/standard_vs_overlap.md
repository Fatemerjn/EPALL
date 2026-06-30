# Our overlap-heavy setting vs the standard Split-CIFAR / PALL protocol

**Why this matters:** the default experiments in this repo deliberately use a task
construction that makes tasks share more parameters/representations than the
standard Split-CIFAR benchmark. That is the point of the project (selective
*un*learning is hard precisely when tasks overlap), but it means **our headline
numbers are NOT directly comparable to published Split-CIFAR / PALL results.** To
get literature-comparable numbers, run the standard group:

```bash
bash tools/run_server_experiments.sh g6_standard
```

This document records *exactly* what differs, with code references.

---

## At a glance

| Axis | Our overlap-heavy setting (default groups g1–g5) | Standard setting (`g6_standard`) |
|---|---|---|
| CIFAR-10 tasks | 5 tasks × 2 classes, random disjoint | 5 tasks × 2 classes, random disjoint *(same — already standard)* |
| CIFAR-100 tasks | **20 semantic superclasses × 5 fine classes**, `--class_per_task 5` forced; with `--n_tasks 10` only the **first 10 superclasses = 50/100 classes** | **standard Split-CIFAR-100: 10 tasks × 10 random disjoint classes** (all 100), `--cifar100_split standard` |
| CIFAR-100 task semantics | fine-grained, semantically coherent (e.g. task = *aquatic_mammals* = {beaver, dolphin, otter, seal, whale}) | heterogeneous, unrelated classes per task |
| Training length | `--n_epochs 3` | `--n_epochs 20` (reference length) |
| PALL subnet sparsity | `--sparsity 0.8` (C10) / `0.9` (C100) | same (`0.8` / `0.9`) — kept at the reference value |
| Schedules | `cifar10_t5_f3`, `cifar100_t10_f3` (3 interleaved forget requests) | same schedules |
| Methods | proposed + a subset of baselines | **all 9**: pall_original, pall_modified, pall_adapter, lora, er, derpp, ewc, lwf, clpu |

So the **two differentiators are CIFAR-100 task construction and the number of
training epochs.** CIFAR-10, the schedules, sparsity, optimizer, buffer, and
hyperparameters are identical, which keeps the comparison clean.

---

## What increases parameter / representation sharing (with code references)

### 1. CIFAR-100 is built from semantic *superclasses*, not random splits
`data.py` (`get_cifar100_superclass_tasks`, `_CIFAR100_SUPERCLASSES`) groups
CIFAR-100 into its 20 official superclasses and makes **each task one superclass
with its 5 fine-grained classes**:

```
task 0 = aquatic_mammals = {beaver, dolphin, otter, seal, whale}
task 1 = fish            = {aquarium_fish, flatfish, ray, shark, trout}
...
```

The standard Split-CIFAR-100 instead assigns **10 random, unrelated classes** to
each of 10 tasks. The superclass design raises sharing in two ways:

- **Within a task** the 5 classes are visually similar (fine-grained), so a task
  is separated mostly by *subtle* features sitting on top of *coarse* features
  (edges, textures, "animal-ness", "vehicle-ness") that are common to many other
  superclasses. The discriminative subspaces therefore ride on a largely shared
  representation, so the per-task subnetworks / shared-adapter directions
  overlap more.
- **Across tasks** related superclasses (e.g. *aquatic_mammals*, *fish*,
  *large_carnivores*, *small_mammals*) reuse the same mid-level features, whereas
  a random 10-class task mixes animals, vehicles, scenes, etc. and is forced to
  spread over a broader, less reused feature set.

### 2. Fewer classes per task, and only half the label space
`--class_per_task` is **forced to 5** for CIFAR-100 (`data.py` raises, and
`main.py:validate_experiment_args` errors otherwise). With `--n_tasks 10` only the
**first 10 of 20 superclasses** are used (`labels_per_task[:T]`), i.e. **50 of 100
classes**. Smaller, easier 5-way tasks converge onto generic, shared features more
readily than the standard 10-way tasks that span the full 100-class space.

### 3. Short training (n_epochs 3)
The overlap runs train only `--n_epochs 3`. Under-trained features are less
task-specialized and more generic, so they are reused (shared) across tasks more
than the reference's longer `--n_epochs 20` training. `g6_standard` uses 20.

### 4. The PALL subnetwork sparsity (context, not a differentiator)
For the subnet methods (`pall_original`, `pall_modified`), each task selects a
sparse subnetwork via `MaskByScores` (`models/subnet_layers.py`):
`k = 1 + round(sparsity*(numel-1))`, keeping the **top ~(1 − sparsity)** of weights.
So `--sparsity 0.8` ⇒ ~20% of weights active per task, `0.9` ⇒ ~10%. The measured
critical overlap `S_share_crit` is the intersection of the forget task's active
weights with the retained tasks' active weights. **`g6_standard` keeps the same
sparsity (0.8 / 0.9)**, so sparsity is *not* what changes between the two
settings — it is held fixed so the comparison isolates task construction + epochs.

### 5. Forgetting-centric schedules (same in both)
`schedules/cifar10_t5_f3_*` and `cifar100_t10_f3_*` interleave 3 forget requests
into the task stream. The unlearning then has to remove a task whose
subnet/representation overlaps the retained tasks; `S_share_crit` quantifies that
overlap. The schedules are identical in both settings — only the underlying task
construction (superclass vs random) changes how large that overlap is on
CIFAR-100.

---

## What `g6_standard` runs

On-demand only (intentionally **NOT** part of `all`):

- **CIFAR-10:** standard 5 tasks × 2 classes, `--arch resnet18`, `--sparsity 0.8`,
  `--n_epochs 20`, schedules `cifar10_t5_f3_fixed_seed{0,1}`, tag `cifar10_standard`.
- **CIFAR-100:** standard Split-CIFAR-100 = 10 tasks × 10 **random disjoint**
  classes (`--cifar100_split standard --class_per_task 10 --n_tasks 10`),
  `--arch resnet34`, `--sparsity 0.9`, `--n_epochs 20`, schedules
  `cifar100_t10_f3_seed{0,1}`, tag `cifar100_standard`.
- **Methods (all 9):** pall_original, pall_modified (gradient, protected),
  pall_adapter (paper config), lora (r=8, α=16), er, derpp (both with
  `--forget_iters 50`), ewc, lwf, clpu. Seeds 0 and 1 → 9 × 2 datasets × 2 seeds
  = **36 runs**.

The new `--cifar100_split {superclass,standard}` flag (default `superclass`) keeps
every existing run byte-for-byte unchanged; only `standard` routes CIFAR-100
through the random-disjoint-split path (`data.py`) and relaxes the
`class_per_task==5` check (`main.py`).

> Note on reference choice: we use **Split-CIFAR-100 = 10×10** (the common
> Mammoth/DER++/PALL convention, and it matches the existing 10-task forget
> schedule). If the reference you compare against uses 20×5, regenerate a
> 20-task schedule and set `--class_per_task 5 --n_tasks 20 --cifar100_split standard`.
