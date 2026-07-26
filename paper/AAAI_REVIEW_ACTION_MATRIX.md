# AAAI Review Action Matrix

This document maps the latest hypothetical AAAI review (5/10, borderline/weak reject) to the current manuscript, code, and evidence. A change is marked closed only when its evidence is already available and audited; implemented experiment infrastructure is kept separate from completed server results.


## Status snapshot (2026-07-25, final pre-submission)

Manuscript: content fits **7 pages**, references (43) on pp. 8--9. Audit gate PASS
(48 traced cells). Clean pdfLaTeX compile: 0 errors, 0 overfull, 0 undefined.
Bundle `paper/submission_OverlapAwareUnlearning.zip` rebuilt from the master `.tex`
(clean-room compile verified byte-identical to the master PDF; metadata scrubbed).

All experiment groups are complete and installed: g23 v3 (RNG-neutral adapter
components), g24 (retraining reference), g25 (EPALL mechanism controls), g26
(corrected MIA), g27 (storage accounting), plus the 120-run g17 overlap-response
benchmark. Claims corrected in this pass: the adapter is reset-dominated *except*
for a small classifier-ascent effect on \(A_u\); MIA before/after values and the
retraining-reference agreement ranges are now stated numerically; "pre-registered"
downgraded to "pre-specified"; figure caption reordered to match the implementation
(buffer deleted at the request boundary, anchors cached afterwards).

Open items are packaging-only: anonymous code repository + activating the links
block, confirming whether AAAI-27 wants the reproducibility checklist inside the PDF,
and any AI-use disclosure the CFP requires.

## Current position

The manuscript is centered on EPALL as the supported positive contribution. PALL-Adapter is retained as an exploratory negative mechanism audit because its controlled endpoint is reset-dominated. The direct EPALL controls, same-method retraining reference, RNG-neutral Adapter matrix, corrected privacy diagnostics, and storage accounting are completed, audited, and installed. Remaining work is submission packaging and final anonymity checks.

## Latest-review actions

| Review concern | Status | Current action/evidence |
|---|---|---|
| Two positive methods are still implied | **Closed in manuscript** | Title, abstract, contributions, method headings, results, limitations, and conclusion now center EPALL. Adapter is explicitly an exploratory negative mechanism audit. |
| EPALL lacks direct mechanism controls | **Closed (installed)** | `g25_modified_components` contains full, no-anchor, overlap-only, same-budget random coordinates inside the eligible structural overlap, and same-budget ranking without an explicit overlap filter on CIFAR-10/100, seeds 0/1/2. The corrected strict tag is `pall_modified_components_overlapmatched_v2`. |
| Retraining oracle is absent | **Closed (installed as Limitations note)** | `g24_retraining_reference` compares EPALL with a same-seed model never trained on the deleted task on CIFAR-10/100. It records prediction agreement, Jensen--Shannon divergence, logit L2, and feature cosine similarity. |
| Privacy section has no valid evidence | **Closed (corrected MIA installed in Limitations)** | The confounded legacy values remain withdrawn. `g26_corrected_mia` is augmentation-free, uses average ranks for ties, and covers EPALL on both CIFAR datasets and seeds 0/1/2. A near-chance result will remain a diagnostic null, not a privacy guarantee. |
| Main results lack paired analysis | **Closed** | Strict standard-protocol rows are paired by schedule/seed. CIFAR-100 has positive `A_final` gains in all three seeds; paired bootstrap intervals and raw sign consistency are reported without significance language. |
| Adapter Full differs between endpoint/component tables | **Closed (v3 table installed)** | Diagnostic evaluations now preserve Python, NumPy, Torch CPU/CUDA, and loader-generator RNG states. The replacement component tag is `adapter_components_pretrained_imagenetnorm_rngneutral_v3`; old v2 rows must not be mixed with it. |
| Legacy TinyImageNet weakens the main story | **Closed in main manuscript** | Legacy TinyImageNet rows were removed from the main PALL comparison and current-method interpretation. |
| CLPU comparison lacks storage accounting | **Closed (storage table installed)** | PALL subnet backups now store only selected indices and values rather than dense layer clones. Every request logs model, CLPU side-network, mask, sparse-backup, replay-image, label, and stored-logit bytes. `g27_storage_accounting` runs EPALL and CLPU on the same seed-0 schedules; its table explicitly excludes optimizer/activation/Python overhead. |
| AI-specific figure filename | **Closed** | The included file is `Figures/adapter_mechanism.pdf`. |
| Paper-number audit is incomplete | **Closed (gate PASS, 58 cells)** | `tools/audit_aaai_paper.py` regenerates Table 1 in memory, checks the hand-written tables cell-by-cell, traces 58 auditable cells, and rejects legacy Adapter component tags by default. The installed RNG-neutral v3 tag passes the default gate. |
| Checklist assumption answer too strong | **Closed** | The checklist no longer claims a theoretical contribution after removal of the theorem; all theory-only follow-up items are marked `NA`. |
| Anonymous code link is commented | **Ready for final activation** | The anonymous code archive has been scrubbed for direct author, affiliation, contact, path, and local-timezone disclosures. Activate its blinded link only after the final archive scan. |

## Evidence already safe to use

### EPALL standard comparison

The standard Split-CIFAR table uses matched schedules and three seeds. On CIFAR-100, the EPALL minus PALL-Original paired differences are positive for `A_final` in all three seeds. Current descriptive paired summaries are:

- CIFAR-10: `Delta A_final = +0.0043` with paired bootstrap interval `[0.0000, 0.0065]`.
- CIFAR-100: `Delta A_final = +0.0191` with paired bootstrap interval `[0.0103, 0.0290]`.
- CIFAR-100 also changes `WorstDrop` by `+0.0200` under the signed retained-task improvement convention used by the analyzer; manuscript wording must remain consistent with the displayed metric direction.

These are descriptive three-seed intervals, not hypothesis tests.

### PALL-Adapter negative component finding

The v2 component matrix supports only a provisional qualitative finding: the endpoint is reset-dominated and the soft mask/classifier ascent do not show an independent benefit. The exact v2 numbers should be replaced by v3 because stage diagnostics previously consumed RNG. The conclusion may remain negative only if it is verified by the RNG-neutral reruns.

### Sequential persistence

Existing raw request histories follow each deleted task through later deletion events. They support a descriptive rebound audit, but the schedules do not contain renewed task training after deletion and therefore do not establish relapse resistance under continued learning.

## Completed evidence groups

**Completed (2026-07-23):** `g23_adapter_components` v3, `g24_retraining_reference`, `g25_modified_components`, `g26_corrected_mia`, and `g27_storage_accounting` finished, synchronized, aggregated, and installed. The paper-number gate passes with 58 auditable cells. Each installed table comes from a strict tag/config selection with row-level provenance.

## Manuscript changes after results return

1. Insert the five-arm EPALL ablation as the primary mechanism table.
2. Add the retraining-reference distances next to that table or immediately after it.
3. Replace or remove the current Adapter component table using only v3 rows.
4. Restore a short privacy diagnostic only if all six corrected rows pass; otherwise keep the limitation text.
5. Add a storage-accounting table only after the logged categories are validated against model objects and serialized artifacts.
6. Re-run the paper-only number audit, compile, rasterize every page, and freeze the evidence hashes.

## Submission claim boundary

The defensible claim is task-level decision-path suppression with improved retained-task stability in a shared-model continual-learning setting. The paper does not establish exact unlearning, certified removal, record-level GDPR compliance, representation erasure, privacy protection, raw-data deletion, or checkpoint/external-copy deletion. CLPU remains a strong isolation reference; EPALL is motivated when retaining one full network per active task is unacceptable, but that storage trade-off must be quantified before it is claimed as an empirical advantage.
