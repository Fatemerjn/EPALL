# AAAI Review Action Matrix

This document maps the latest hypothetical AAAI review (5/10, borderline/weak reject) to the current manuscript, code, and evidence. A change is marked closed only when its evidence is already available and audited; implemented experiment infrastructure is kept separate from completed server results.

## Current position

The manuscript is now centered on PALL-Modified as the supported positive contribution. PALL-Adapter is retained as an exploratory negative mechanism audit because its controlled endpoint is reset-dominated. The remaining acceptance-critical work is experimental rather than rhetorical: direct PALL-Modified component controls, a same-method retraining reference, RNG-neutral replacement of the Adapter component matrix, and corrected privacy diagnostics.

## Latest-review actions

| Review concern | Status | Current action/evidence |
|---|---|---|
| Two positive methods are still implied | **Closed in manuscript** | Title, abstract, contributions, method headings, results, limitations, and conclusion now center PALL-Modified. Adapter is explicitly an exploratory negative mechanism audit. |
| PALL-Modified lacks direct mechanism controls | **Implemented; server runs pending** | `g25_modified_components` contains full, no-anchor, overlap-only, same-budget random coordinates inside the eligible structural overlap, and same-budget ranking without an explicit overlap filter on CIFAR-10/100, seeds 0/1/2. The corrected strict tag is `pall_modified_components_overlapmatched_v2`. |
| Retraining oracle is absent | **Implemented; server runs pending** | `g24_retraining_reference` compares PALL-Modified with a same-seed model never trained on the deleted task on CIFAR-10/100. It records prediction agreement, Jensen--Shannon divergence, logit L2, and feature cosine similarity. |
| Privacy section has no valid evidence | **Correctly removed from Results; rerun ready** | The confounded legacy values remain withdrawn. `g26_corrected_mia` is augmentation-free, uses average ranks for ties, and covers PALL-Modified on both CIFAR datasets and seeds 0/1/2. A near-chance result will remain a diagnostic null, not a privacy guarantee. |
| Main results lack paired analysis | **Closed** | Strict standard-protocol rows are paired by schedule/seed. CIFAR-100 has positive `A_final` gains in all three seeds; paired bootstrap intervals and raw sign consistency are reported without significance language. |
| Adapter Full differs between endpoint/component tables | **Code fixed; replacement runs pending** | Diagnostic evaluations now preserve Python, NumPy, Torch CPU/CUDA, and loader-generator RNG states. The replacement component tag is `adapter_components_pretrained_imagenetnorm_rngneutral_v3`; old v2 rows must not be mixed with it. |
| Legacy TinyImageNet weakens the main story | **Closed in main manuscript** | Legacy TinyImageNet rows were removed from the main PALL comparison and current-method interpretation. |
| CLPU comparison lacks storage accounting | **Implemented; four matched runs pending** | PALL subnet backups now store only selected indices and values rather than dense layer clones. Every request logs model, CLPU side-network, mask, sparse-backup, replay-image, label, and stored-logit bytes. `g27_storage_accounting` runs PALL-Modified and CLPU on the same seed-0 schedules; its table explicitly excludes optimizer/activation/Python overhead. |
| AI-specific figure filename | **Closed** | The included file is now `Figures/adapter_mechanism.pdf`. |
| Paper-number audit is incomplete | **Implemented; final v3 gate pending** | `tools/audit_aaai_paper.py` regenerates Table 1 in memory, checks the three hand-written tables cell-by-cell, traces 44 rows, and refuses legacy Adapter component tags by default. It currently passes only with the explicit v2 override and will pass the default gate after the RNG-neutral v3 table is installed. |
| Checklist assumption answer too strong | **Closed** | The checklist no longer claims a theoretical contribution after removal of the theorem, and the assumptions/restrictions item is `partial`. |
| Anonymous code link is commented | **Open before submission** | Prepare an anonymous repository and activate the code link only after verifying that it contains no identity or hidden metadata. |

## Evidence already safe to use

### PALL-Modified standard comparison

The standard Split-CIFAR table uses matched schedules and three seeds. On CIFAR-100, the PALL-Modified minus PALL-Original paired differences are positive for `A_final` in all three seeds. Current descriptive paired summaries are:

- CIFAR-10: `Delta A_final = +0.0043` with paired bootstrap interval `[0.0000, 0.0065]`.
- CIFAR-100: `Delta A_final = +0.0191` with paired bootstrap interval `[0.0103, 0.0290]`.
- CIFAR-100 also changes `WorstDrop` by `+0.0200` under the signed retained-task improvement convention used by the analyzer; manuscript wording must remain consistent with the displayed metric direction.

These are descriptive three-seed intervals, not hypothesis tests.

### PALL-Adapter negative component finding

The v2 component matrix supports only a provisional qualitative finding: the endpoint is reset-dominated and the soft mask/classifier ascent do not show an independent benefit. The exact v2 numbers should be replaced by v3 because stage diagnostics previously consumed RNG. The conclusion may remain negative only if it is verified by the RNG-neutral reruns.

### Sequential persistence

Existing raw request histories follow each deleted task through later deletion events. They support a descriptive rebound audit, but the schedules do not contain renewed task training after deletion and therefore do not establish relapse resistance under continued learning.

## Runs required before final evidence freeze

Run one group at a time and synchronize/aggregate after each group so failures are attributable.

1. **P0: `g25_modified_components`** -- decisive attribution for the primary contribution.
2. **P0: `g24_retraining_reference`** -- same-method oracle on CIFAR-10/100.
3. **P1: `g26_corrected_mia`** -- corrected diagnostic only; omit if it does not finish cleanly.
4. **P1: `g23_adapter_components`** -- RNG-neutral v3 replacement for Table 3.
5. **P2: `g27_storage_accounting`** -- four matched seed-0 runs for quantitative resident tensor-state accounting.

Do not update the manuscript from partial groups, failed runs, or mixed tags. Each final table must be generated from a strict tag/config selection and retain a row-level provenance artifact.

## Manuscript changes after results return

1. Insert the five-arm PALL-Modified ablation as the primary mechanism table.
2. Add the retraining-reference distances next to that table or immediately after it.
3. Replace or remove the current Adapter component table using only v3 rows.
4. Restore a short privacy diagnostic only if all six corrected rows pass; otherwise keep the limitation text.
5. Add a storage-accounting table only after the logged categories are validated against model objects and serialized artifacts.
6. Re-run the paper-only number audit, compile, rasterize every page, and freeze the evidence hashes.

## Submission claim boundary

The defensible claim is task-level decision-path suppression with improved retained-task stability in a shared-model continual-learning setting. The paper does not establish exact unlearning, certified removal, record-level GDPR compliance, representation erasure, privacy protection, raw-data deletion, or checkpoint/external-copy deletion. CLPU remains a strong isolation reference; PALL-Modified is motivated when retaining one full network per active task is unacceptable, but that storage trade-off must be quantified before it is claimed as an empirical advantage.
