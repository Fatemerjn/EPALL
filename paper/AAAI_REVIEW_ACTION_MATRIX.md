# AAAI Review Action Matrix

This document maps the attached 4/10-style review to the current repository. It distinguishes corrections already made, evidence that exists but is incomplete, and experiments that genuinely remain necessary.

## Bottom line

The review's central objection is valid: current PALL-Adapter results do not separate the effect of resetting the target adapter/classifier slice from the value of the overlap-aware shared update. Prose alone cannot close this issue. The minimum credible empirical response is a component ablation headed by `reset only`, plus a retraining reference.

The review is partly stale on implementation details. The repository already logs three request events, pre/post retained accuracies, intermediate pre-repair evaluation, model agreement with a retrained reference, gradient ratios, probes, and MIA. These artifacts do not yet constitute the requested controls:

- `after_reset_eval` is measured after reset **and** shared forgetting **and** classifier ascent; it is not a reset-only baseline.
- `agreement_v1` trains a single-forget retrained reference and reports argmax agreement, but it has only CIFAR-10, two seeds, and no KL/logit/feature distance.
- `g22a_tiny_seed2` adds a seed to the legacy overlap-heavy arm; it does not convert legacy seed 0/1 runs to the current uniform-target algorithm.

## Closed in the manuscript without new training

| Review concern | Action now in the paper |
|---|---|
| PALL-Adapter overframed as representation removal | Abstract and conclusion now state decision-path suppression and disclose recoverable representation information. |
| Multi-step theorem compared with an actual unconstrained rerun | The theorem is explicitly a fixed-gradient-sequence bound, not a comparison of two optimization trajectories. |
| Hard-protected set omitted | `H` is defined as the top `ceil(alpha_p * |S_crit|)` retained-gradient-ranked coordinates inside the adapter intersection. |
| Bound hides hard protection | The theorem separates the soft subset and `H`; the unmasked fixed-gradient comparator restores both contributions. |
| Same notation used for two different overlaps | The text explicitly distinguishes binary-subnetwork overlap from gradient-selected adapter overlap. |
| Undefined `Phi` | The target deployed path is defined. |
| Scalar regularizer written with vector norm | Replaced by a scalar square. |
| `F_avg`, WorstDrop and aggregation ambiguous | The formula, task scope, signed/non-clipped behavior, final-event selection, and across-seed aggregation are now stated. |
| Au treated monotonically | It is interpreted by absolute distance to chance. |
| 87.5% presented as general efficiency | Renamed and restricted to nominal retained-repair scope; no compute/memory/runtime claim remains. |
| MIA near chance before deletion | The paper now calls it a low-power diagnostic null, not positive forgetting evidence. |
| Anchor comparison unverifiable | Matched old/reinit two-seed values are reported side by side. |
| Legacy Tiny result used in Figure 2 | Removed from the main WorstDrop figure and excluded from the current-method interpretation. |
| Schedule and memory unclear | Three distinct requests, no task re-entry, fixed total buffer partition, and no post-delete reallocation are stated. |

## Existing evidence that can be reused, with limits

### Retrained reference

`main.py --eval_agreement` trains a `Sequential` reference in which the forgotten task is never trained. Existing `agreement_v1` rows cover PALL-Modified and PALL-Adapter on CIFAR-10 with two seeds. This is useful pilot evidence, not a full oracle: it reports argmax agreement only, is restricted to one forget request, and is absent on CIFAR-100/TinyImageNet.

### Sequential requests

Raw `unlearning_events` already contain per-task accuracies before and after each of three deletion requests. `tools/analyze_sequential_damage.py` measures cumulative retained-task damage. A forgotten-task persistence analyzer can therefore be built read-only for completed runs, but it must follow each deleted task through later stream events rather than only reading final Au.

### Anchor ablation

The matched `anchor_ablation_v1` rows already provide old versus reinitialized anchor values for accuracy, WorstDrop, Au and MIA. No new two-seed training is required; seed 2 remains pending in `g22b_mia_anchor_seed2`.

## Experiments that remain genuinely necessary

### P0 — PALL-Adapter component attribution

Run on the intended pretrained-frozen regime, CIFAR-10 and CIFAR-100, with the same schedules and at least seeds 0/1/2:

| Arm | Reset target path | Shared uniform update | Soft mask | Cached ascent | Retained repair |
|---|---:|---:|---:|---:|---:|
| Reset only | yes | no | no | no | no |
| Reset + repair | yes | no | no | no | yes |
| Uniform, no mask | yes | yes | no | yes | yes |
| Mask, no ascent | yes | yes | yes | no | yes |
| Full PALL-Adapter | yes | yes | yes | yes | yes |

Primary outputs: retained accuracy, WorstDrop, `|Au-chance|`, MIA/probe where affordable, and per-stage values. This is the single most important addition.

### P0 — Retraining reference

Extend the existing reference evaluator to record, on retained and forgotten test sets:

- prediction agreement;
- mean KL in both directions or Jensen--Shannon divergence;
- logit L2 distance;
- feature cosine/CKA-style similarity;
- optional relearning speed.

Use identical schedule, seed, architecture and training budget. Start with one forget request per run to keep the oracle tractable, then cover both CIFAR datasets and the full method/reset-only arms.

### P1 — Current TinyImageNet evidence

Do **not** treat `g22a_tiny_seed2` as closure of the legacy criticism. Either remove the legacy adapter row from the main paper, or rerun PALL-Adapter with the current deterministic uniform-target algorithm on schedules `tinyimagenet_t20_f3_seed{0,1,2}.json`. Compare only against methods under that same schedule and regime.

### P1 — Persistence

From raw request histories, report forgotten-task accuracy immediately after deletion and after every later training/deletion event. A task must not silently recover above chance. Report a per-task curve and the maximum post-delete rebound.

### P1 — Paired uncertainty

Schedules are shared within seed, so use paired seed-level differences. With only three seeds, report paired bootstrap intervals and avoid significance language. Five seeds would be stronger but should not be mixed with three-seed rows until every method in the compared table is complete.

### P2 — Stronger privacy attacks

The current custom score is weak because pre-delete AUC is near chance. Add standard loss, confidence, entropy and modified-entropy scores first. Shadow/LiRA-style attacks are desirable but lower priority than reset attribution and the retraining oracle.

### P2 — Resource accounting

Report total/trainable parameters, task-growing stored parameters, rehearsal/logit storage, and wall-clock only on controlled hardware. Existing `T_f` is request latency and is not a full resource comparison.

## Recommended submission decision

Do not try to answer this review by adding more prose or more unrelated baselines. The paper becomes materially stronger if the next results table contains the five component arms and a retraining-reference distance. If those experiments show the full method is not better than reset + repair, the honest conclusion is that the current PALL-Adapter contribution is architectural task suppression; PALL-Modified should then be the primary method. If the full method improves retained stability or distance to the retrained reference at matched Au, the overlap-aware contribution becomes directly defensible.
