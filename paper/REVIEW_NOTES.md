# Code Review Notes & Open Decisions (for advisor meeting)

Generated during a deep technical review. These are the items that block a clean
"the code matches the paper" claim. Resolve the **OPEN DECISIONS** before
finalizing results.

## NEW CONTRIBUTION (2026-06-16): gradient-CONFLICT protection for high overlap

The core thesis goal is: improve PALL when the forget/retain parameter overlap is
HIGH. Added `--protect_importance conflict` to `pall_modified`. Instead of ranking
the critical shared set by retained-task gradient magnitude, it ranks by the
per-coordinate **conflict energy** `relu(-g_forget * g_retain)` (computed on the
rehearsal buffer; the signed forget gradient is captured before the forget buffer
is removed). This selects exactly the parameters where forgetting and retention
pull in OPPOSITE directions -- the true source of WorstDrop under high overlap. It
also aligns the method with the Theorem (whose `E_crit` uses retained-task
gradient sensitivity). Optional `--adaptive_protect` scales the L2 penalty by the
measured critical-overlap ratio (1x at zero overlap -> 2x at full overlap).

Smoke (CIFAR-10, 1 epoch -- directional only): WorstDrop conflict+adapt 0.0065 <
gradient 0.0085 < original 0.0095. Needs full multi-seed runs (see checklist).

Code split for clarity (1:1 paper mapping):
`methods/pall_base.py` (PALLBase, all machinery) + `pall_modified.py`
(PALLModified, main) + `pall_original.py` (PALLOriginal, baseline);
`pall.py` is a back-compat shim (`PALL = PALLBase`).

## Method taxonomy (finalized)

See README "Method taxonomy". Summary: main method = `pall_modified` with
`--protect_importance gradient` (variant `pall_modified_grad`, "PALL-Modified").
Weight version is an ablation (`pall_modified_weight`, "PALL-Modified-W").
Baseline = `pall_original`. Adapter modes: `adapter_reset` / `adapter_shared` /
`adapter_protected`. Every run records `variant` in config.json + the aggregated
CSV, so tables separate cleanly.

## Bug-audit findings (full-codebase pass)

- FIXED: `RehearsalMemory.remove()` returned a malformed tuple (operator
  precedence); was harmless (return value unused everywhere) but now correct.
- OK (verified, no bug): seed is set BEFORE task-split sampling, so CIFAR-10 /
  TinyImageNet random splits are seed-reproducible. CIFAR-100 uses fixed semantic
  superclasses (seed-independent by design). Metrics (WorstDrop/Fu/Au/forgetting)
  computed correctly before vs. after. Subnet masking + label remapping correct.
- PERF (not correctness): `data.SubDataset.__getitem__` calls
  `permutation.index()` (O(#classes)) per sample. Optional: precompute a dict.
- PROTOCOL NOTE: CIFAR-100 with `n_tasks=10` uses only the first 10 superclasses
  (50 of 100 classes), matching the paper. Use `n_tasks=20` for all 100 classes.

## 1. RESOLVED (implemented 2026-06-10): `pall_modified` now uses gradient-magnitude importance

- **Decision taken:** option (a). Gradient-magnitude importance is now the MAIN
  PALL-Modified criterion; the weight-magnitude version is kept as a selectable
  legacy ablation (`--protect_importance weight`). Default is `gradient`.
- **What changed:** `methods/pall.py` gained `_compute_retain_grad_importance`
  (one backward pass of retained-task CE over the rehearsal buffer ->
  `|grad L_retain|` per parameter, with grads isolated before/after). The
  critical-selection (`_select_critical_shared_masks`) now ranks the shared
  overlap by this gradient importance when in gradient mode, by `|w|` otherwise.
  Added `--protect_importance {gradient,weight}` (main.py); captured in
  config.json and `tools/aggregate_results.py`.
- **Consistency win:** this matches both the paper's Methodology text AND the
  Theorem, whose `E_crit` energy already uses retained-task gradient sensitivity
  `|d R_t / d w|`. Code, method section, and theory are now aligned.
- **Consequence:** every PALL-Modified row that used protection
  (`protect_ratio` set, `lambda_protect > 0`) must be RE-RUN; old numbers are
  superseded. See the "rerun" checklist below.

## 2. RESOLVED (2026-06-16): `pall_adapter` Phase-3 is now the iterative uniform-target loop

- **Code** (`methods/pall_adapter.py`): shared-adapter forgetting is now
  `_run_phase3_shared_forgetting` -- an ITERATIVE loop (`--adapter_forget_steps`,
  default 10) that each step minimises the uniform-target loss
  (`_uniform_forget_loss`) on replayed forget samples and applies the soft-masked
  gradient DESCENT `w -= lr * m_soft * grad` with masks held fixed. This matches
  Algorithm 1 / Phase 3 and the WorstDrop theorem. (`--adapter_forget_steps 1`
  recovers the old single-step behaviour.) The structural invariant
  `hard_protected + updated == shared_forget` still holds (smoke-verified).
- Additionally, `--protect_importance conflict` now also applies to the adapter:
  the hard-protected subset is ranked by `relu(-g_forget*g_retain)` (gradient
  conflict), consistent with `pall_modified`.
- **Consequence:** all adapter forgetting results must be RE-RUN (old single-step
  numbers superseded). The paper's Phase-3 description and Algorithm 1 now match
  the code.

## 3. RESOLVED (documented): the two methods are two operationalizations of ONE concept

- `pall_modified`: overlap = intersection of subnet masks (`S_share = m_u ∧ M_a`);
  criticality = retained-task GRADIENT magnitude `I_w` (per item #1; the old
  `|w|` weight version is now only the `--protect_importance weight` ablation).
- `pall_adapter`: overlap = intersection of gradient-importance top-K sets
  (`S_forget ∩ S_active`) on the shared adapter `phi_s`.
- **Resolution:** rather than force-reconciling them, `thesis/chapters/proposed.tex`
  now has a synthesizing section `\section{...}` (`\label{sec:unified-view}`) that
  states explicitly they are two _operationalizations of a single concept_
  (protecting the critical shared region `S_crit` = forget-footprint ∩
  retain-sensitivity), differing only in the SPACE where membership/importance is
  measured. It includes a "definition / space / criticality criterion" table
  (`tab:unified-view`) and shows the downstream protection (ℓ2 anchor vs soft mask
  `1-p`) and Theorem `thm:worstdrop` cover both at the concept level.

## 4. `pall_adapter` defaults make it a reset-only prototype

- Defaults: `adapter_shared_bottleneck=0`, `adapter_shared_forget_ratio=0.0`,
  `adapter_shared_protect_ratio=0.0` → no shared adapter, no overlap handling;
  forgetting degenerates to adapter + classifier RESET.
- The paper config (bottleneck 16, alpha_f 0.3, alpha_p 0.2) MUST be passed
  explicitly. Confirm every reported adapter run used these flags
  (`tools/run_adapter_ablation.py` does).

## 5. RESOLVED (documented as a limitation): `pall_adapter` overlap matrix is degenerate

- `compute_overlap_matrix` returns the SAME off-diagonal value for every task
  pair (shared adapter is shared equally by all tasks). It cannot support a
  per-pair "overlap vs damage" correlation for the adapter method. Use the
  scalar `s_share_crit_ratio` across schedules instead.
- **Resolution:** this is now stated explicitly as a **limitation** in the
  `sec:unified-view` section of `thesis/chapters/proposed.tex` (the "محدودیت:
  تباهیدگی ماتریس هم‌پوشانی آداپتر" paragraph): the pairwise matrix is degenerate,
  so the scalar critical-shared ratio `s_share_crit` across schedules is used as
  the overlap axis for the adapter, and the limitation is scoped to overlap
  _analysis_ only (it does not affect the forgetting mechanism or the theorem).

## 6. `shared_protect_strength` semantics

- When `--adapter_shared_protect_strength` is unset (default `None`), the soft
  scale is auto-derived as `critical_count / forget_count`, a data-dependent
  quantity — NOT the paper's fixed `protect_strength`. Set it explicitly for
  controlled ablations.

## 7. FIXED in this pass (engineering / reproducibility)

- `requirements.txt`: added `torch==2.9.1`, `torchvision==0.24.1`; corrected
  non-existent pins (`pandas==3.0.2`→`2.3.3`, `numpy==2.4.4`→`2.4.1`). A clean
  `pip install -r requirements.txt` previously failed outright.
- Added explanatory docstrings to `_select_critical_shared_masks` (pall.py) and
  the `PALLAdapter` class documenting the algorithm and the paper gaps above.
- Added `tools/smoke_test.sh` (CIFAR-10 baseline / pall_modified / pall_adapter /
  aggregation).

## 8. Stale documentation

- `example_run.sh` CIFAR-100 line uses `--class_per_task 10`, which now violates
  the enforced constraint (`main.py` requires `class_per_task == 5` for CIFAR-100).
  Running the documented example errors immediately. Update to `--class_per_task 5`.

## What is CORRECT (verified)

- Seeding (`set_seed`) covers python/numpy/torch/CUDA + deterministic algorithms;
  DataLoader workers + shuffling are now seeded too.
- Metrics: WorstDrop = max retained-task (pre - post) drop; Au = forgotten-task
  accuracy after; Fu = retained-avg drop; average-forgetting = standard CL
  definition. All computed before vs. after correctly.
- CIFAR-100 constraints (cpt==5, n_tasks in [1,20]) and TinyImageNet
  (cpt\*n_tasks<=200) are enforced.
- `pall_adapter` `_compute_shared_importance` DOES do a correct backward on the
  rehearsal buffer with proper grad zeroing before/after; masks match shapes/devices.
- The protection regularization in `pall_modified` (L2 anchor to pre-forget
  values on the critical set, applied during retrain) is implemented correctly.
  </content>
