# AAAI Review Action Matrix

This document maps the latest hypothetical AAAI review (5/10, borderline/weak reject) to the current manuscript, code, and evidence. A change is marked closed only when its evidence is already available and audited; implemented experiment infrastructure is kept separate from completed server results.


## Status snapshot (2026-07-26, five-seed checkpoint)

The synchronized checkpoint contains a complete, balanced five-seed primary table
and five-seed direct EPALL component matrix. Runs from the next server stage are
excluded by the canonical seed filter, so no table mixes five- and six-seed rows.
The five-seed manuscript passes its 48-row numeric gate, 23 tests, shell syntax,
pdfLaTeX/BibTeX compilation, font, metadata, and visual checks. Technical content
ends on page 7; references and the included checklist bring the PDF to 10 pages.

All experiment groups are complete and installed: g23 v3 (RNG-neutral adapter
components), g24 (retraining reference), g25 (EPALL mechanism controls), g26
(corrected MIA), g27 (storage accounting), plus the 120-run g17 overlap-response
benchmark. Claims corrected in this pass: the adapter is reset-dominated *except*
for a small classifier-ascent effect on \(A_u\); MIA before/after values and the
retraining-reference agreement ranges are now stated numerically; "pre-registered"
downgraded to "pre-specified"; figure caption reordered to match the implementation
(buffer deleted at the request boundary, anchors cached afterwards).

Open items are packaging-only: publishing/validating the anonymous code repository,
activating its links block, flattening the checklist input if the portal requires a
single TeX source, rebuilding the clean-room bundle, and copying the manuscript's
AI-use disclosure into any corresponding submission-form field.

## Current position

The manuscript is centered on EPALL as the supported positive contribution. PALL-Adapter is retained as an exploratory negative mechanism audit because its controlled endpoint is reset-dominated. The direct EPALL controls, same-method retraining reference, RNG-neutral Adapter matrix, corrected privacy diagnostics, and storage accounting are completed, audited, and installed. Remaining work is submission packaging and final anonymity checks.

## Latest-review actions

| Review concern | Status | Current action/evidence |
|---|---|---|
| Two positive methods are still implied | **Closed in manuscript** | Title, abstract, contributions, method headings, results, limitations, and conclusion now center EPALL. Adapter is explicitly an exploratory negative mechanism audit. |
| EPALL lacks direct mechanism controls | **Closed (installed)** | `g25_modified_components` contains full, no-anchor, overlap-only, same-budget random coordinates inside the eligible structural overlap, and same-budget ranking without an explicit overlap filter on CIFAR-10/100, seeds 0--4. The strict tag is `pall_modified_components_overlapmatched_v2`. |
| Retraining oracle is absent | **Closed (installed as Limitations note)** | `g24_retraining_reference` compares EPALL with a same-seed model never trained on the deleted task on CIFAR-10/100. It records prediction agreement, Jensen--Shannon divergence, logit L2, and feature cosine similarity. |
| Privacy section has no valid evidence | **Closed (corrected MIA installed in Limitations)** | The confounded legacy values remain withdrawn. `g26_corrected_mia` is augmentation-free, uses average ranks for ties, and covers EPALL on both CIFAR datasets and seeds 0/1/2. A near-chance result will remain a diagnostic null, not a privacy guarantee. |
| Main results lack paired analysis | **Closed** | Strict standard-protocol rows are paired on seeds 0--4. CIFAR-100 favours EPALL in all five pairs for `A_final`, `F_avg`, and `WorstDrop`; exact one-sided Wilcoxon and sign tests give `p=0.03125`. CIFAR-10 is reported as directional because its exact tests remain above 0.05. |
| Adapter Full differs between endpoint/component tables | **Closed (v3 table installed)** | Diagnostic evaluations now preserve Python, NumPy, Torch CPU/CUDA, and loader-generator RNG states. The replacement component tag is `adapter_components_pretrained_imagenetnorm_rngneutral_v3`; old v2 rows must not be mixed with it. |
| Legacy TinyImageNet weakens the main story | **Closed in main manuscript** | Legacy TinyImageNet rows were removed from the main PALL comparison and current-method interpretation. |
| CLPU comparison lacks storage accounting | **Closed (storage table installed)** | PALL subnet backups now store only selected indices and values rather than dense layer clones. Every request logs model, CLPU side-network, mask, sparse-backup, replay-image, label, and stored-logit bytes. `g27_storage_accounting` runs EPALL and CLPU on the same seed-0 schedules; its table explicitly excludes optimizer/activation/Python overhead. |
| AI-use and figure provenance | **Closed in manuscript** | A concise disclosure records language editing, code review, and drafting of the conceptual mechanism illustration; the included mechanism file is `Figures/EPALL_mechanism.png`. Mirror the same disclosure in the submission form if requested. |
| Paper-number audit is incomplete | **Closed (gate PASS, 48 rows/cells)** | `tools/audit_aaai_paper.py` regenerates Table 1 in memory, checks the hand-written tables cell-by-cell, traces 48 auditable rows/cells, and rejects legacy Adapter component tags by default. The installed RNG-neutral v3 tag passes the strict analyzer. |
| Checklist assumption answer too strong | **Closed** | The checklist no longer claims a theoretical contribution after removal of the theorem; all theory-only follow-up items are marked `NA`. |
| Anonymous code link is commented | **Ready for final activation** | The anonymous code archive has been scrubbed for direct author, affiliation, contact, path, and local-timezone disclosures. Activate its blinded link only after the final archive scan. |

## Evidence already safe to use

### EPALL standard comparison

The standard Split-CIFAR table uses matched schedules and five canonical seeds.
Current paired summaries are:

- CIFAR-10: `Delta A_final = +0.0114` with descriptive bootstrap interval `[0.0038, 0.0206]`; exact one-sided `p=0.0625` because one pair ties.
- CIFAR-100: `Delta A_final = +0.0176` with interval `[0.0084, 0.0268]`; all five pairs favour EPALL and exact one-sided Wilcoxon/sign `p=0.03125`.
- CIFAR-100 `F_avg` and `WorstDrop` also favour EPALL in all five pairs (`+0.0135` and `+0.0212`, respectively; exact one-sided `p=0.03125`).

The seed set was fixed independently of the outcome. Significance language is
limited to the matched CIFAR-100 result.

### PALL-Adapter negative component finding

The strict RNG-neutral v3 matrix confirms a reset-dominated endpoint and no measured
soft-mask benefit. Cached classifier ascent has a small effect on forgotten-task
accuracy (`0.0040` on CIFAR-100 and `0.0017` on CIFAR-10 in distance-to-chance
units) without changing retained accuracy; the manuscript states that exception.

### Sequential persistence

Existing raw request histories follow each deleted task through later deletion events. They support a descriptive rebound audit, but the schedules do not contain renewed task training after deletion and therefore do not establish relapse resistance under continued learning.

## Completed evidence groups

**Completed:** `g23_adapter_components` v3, `g24_retraining_reference`, `g25_modified_components`, `g26_corrected_mia`, and `g27_storage_accounting` finished, synchronized, aggregated, and installed. The current paper-number gate passes with 48 auditable rows/cells. Each installed table comes from a strict tag/config selection with row-level provenance.

## Remaining submission operations

1. Publish and anonymously open-test the scrubbed `epall-code` copy.
2. Insert the verified anonymous URL and recompile.
3. Flatten the checklist input if the portal enforces a single TeX source.
4. Rebuild and clean-room compile the source bundle, then repeat metadata/anonymity checks.

## Submission claim boundary

The defensible claim is task-level decision-path suppression with improved retained-task stability in a shared-model continual-learning setting. The paper does not establish exact unlearning, certified removal, record-level GDPR compliance, representation erasure, privacy protection, raw-data deletion, or checkpoint/external-copy deletion. CLPU remains a strong isolation reference; EPALL is motivated when retaining one full network per active task is unacceptable, but that storage trade-off must be quantified before it is claimed as an empirical advantage.
