# PALL Modified: Hyperparameters (Overlap-Aware Selective Forgetting)

This document summarizes method-specific knobs used by `pall_modified` and `pall_original`, where they are parsed, and where they affect behavior.

## Scope

- `--method pall` is normalized to `pall_modified`.
- `--method pall_original` and `--method pall_modified` share the same class implementation (`methods/pall.py`) and differ via `method_variant`.

## Hyperparameter Table

| Flag | Default | Meaning | Parse Location | Main Use Location(s) |
| --- | --- | --- | --- | --- |
| `--protect_ratio` | `None` | Fraction of shared parameters to mark as critical/protected (`S_share_crit`) by absolute magnitude. | `main.py` (`argparse`) | `methods/pall.py::_select_critical_shared_masks` |
| `--protect_threshold` | `None` | Absolute magnitude threshold to mark critical/protected shared parameters. | `main.py` (`argparse`) | `methods/pall.py::_select_critical_shared_masks` |
| `--lambda_protect` | `0.0` | Regularization weight to keep protected shared params close to pre-reset values during retraining. | `main.py` (`argparse`) | `methods/pall.py::_forget_impl` (protection refs + regularizer term) |
| `--retrain_steps` | `None` | Explicit retraining steps after reset (overlap recovery). | `main.py` (`argparse`) | `methods/pall.py::_compute_retrain_steps`, `_forget_impl` |
| `--retrain_epochs` | `None` | Alias for retraining steps; takes precedence over `retrain_steps`. | `main.py` (`argparse`) | `methods/pall.py::_compute_retrain_steps`, `_forget_impl` |
| `--allow_zero_retrain` | `False` | If `True`, allows skipping retraining when resolved steps are `0` (no fallback). | `main.py` (`argparse`) | `methods/pall.py::_forget_impl` |
| `--adaptive_retrain` | `False` | Scales resolved retraining steps by overlap ratio (`ceil(steps * s_share_ratio)`). | `main.py` (`argparse`) | `methods/pall.py::_compute_retrain_steps` |
| `--debug_unlearning` | `False` | Enables per-unlearning debug artifact dump (`masks.pt`, indices, norms). | `main.py` (`argparse`) | `main.py` (debug dir/context), `methods/pall.py::_dump_unlearning_debug` |
| `--dump_overlap` | `False` | Dumps full task-overlap matrix CSV at run end. | `main.py` (`argparse`) | `main.py` (`compute_overlap_matrix` + `write_overlap_csv`) |
| `--k_shot` | `1` | Baseline fallback retrain-step count when no retrain override is given. | `main.py` (`argparse`) | `methods/pall.py::_compute_retrain_steps` |

## Resolution Rules (Current Behavior)

1. Method variant:
   - `pall -> pall_modified`
   - `method_variant = modified/original` is set in `main.py` and consumed by `methods/pall.py`.
2. Protection mask selection:
   - If both `protect_ratio` and `protect_threshold` are set, `protect_ratio` is used.
3. Retrain steps:
   - `retrain_epochs` > `retrain_steps` > `k_shot`
   - If `adaptive_retrain=True`, resolved steps are scaled by overlap ratio.
   - If resolved steps become `0` and `allow_zero_retrain=False`, code applies fallback `50`.

## Config Serialization

All above flags are parsed in `main.py` and written to `config.json` via:
- `serialize_config(args, run_dir, timestamp)` (copies `vars(args)`),
- then `write_json(run_dir / "config.json", config)`.

## Consistency Check

Current status: **consistent**.

- All listed knobs are available via CLI.
- All are serialized in `config.json`.
- All are used in code paths where expected.
- No algorithmic refactor is required.

## Recommended Ablation Grid

Use this as a practical starting point; narrow based on budget.

| Parameter | Suggested Values |
| --- | --- |
| `protect_ratio` | `0.0, 0.05, 0.1, 0.2, 0.4` |
| `protect_threshold` | `None` when ratio is active; otherwise `0.0, 1e-4, 1e-3, 1e-2` |
| `lambda_protect` | `0.0, 1e-3, 1e-2, 1e-1, 1.0` |
| `retrain_steps` | `0, 10, 25, 50, 100` |
| `adaptive_retrain` | `False, True` |
| `allow_zero_retrain` | `False, True` (mainly for `retrain_steps=0`) |
| `dump_overlap` | `False, True` (diagnostic only) |
| `debug_unlearning` | `False, True` (diagnostic only; may increase storage/time) |

Suggested baseline controls:
- Compare `pall_original` vs `pall_modified`.
- Keep shared training settings fixed (`seed`, `arch`, `n_tasks`, `n_forget`, memory/optimizer).
