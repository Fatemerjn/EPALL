# سامانهٔ بصری و پرامپت دقیق شکل‌های پایان‌نامه

این سند پس از بررسی نسخهٔ کامل پایان‌نامه، قالب LaTeX پروژه، ویرایش هشتم دستورالعمل
تهیهٔ پایان‌نامهٔ دانشگاه صنعتی شریف و راهنمای نگارش و ماشین‌نویسی تهیه شده است.

## تم منتخب: «مینیمال فنی شریف»

- زمینه: سفید خالص، بدون بافت، سایه، گرادیان و جلوهٔ سه‌بعدی.
- نسبت شکل‌های مفهومی: افقی و کم‌ارتفاع، ترجیحاً `16:6` یا `16:7`.
- قلم نوشته‌های لاتین و ریاضی: `Times New Roman` با جایگزین `DejaVu Serif`.
- حداقل اندازهٔ نوشته پس از درج در صفحهٔ A4: ۹ پوینت.
- ضخامت خط اصلی: ۱ تا ۱٫۴ پوینت در اندازهٔ نهایی؛ خطوط فرعی ۰٫۶ تا ۰٫۸ پوینت.
- گوشه‌ها: گرد و کم‌انحنا؛ بدون کادرهای ضخیم تزئینی.
- عنوان شکل داخل تصویر تکرار نمی‌شود؛ عنوان فارسی زیر شکل و توسط LaTeX می‌آید.
- متن داخل شکل کوتاه، فنی و ترجیحاً انگلیسی/نماد ریاضی است؛ توضیح بلند در کپشن می‌آید.
- برای چاپ خاکستری، رنگ با هاشور، نوع خط یا نشانگر پشتیبانی می‌شود.

### پالت ثابت

| معنا | رنگ |
|---|---|
| دقت، ساختار و مسیر اصلی | `#0072B2` |
| نگه‌داری، حفاظت و کارایی | `#009E73` |
| حذف، افت و هشدار | `#D55E00` |
| هم‌پوشانی بحرانی | `#CC79A7` |
| تأکید ثانویهٔ ناحیهٔ بحرانی | `#6F3C8F` |
| اجزای منجمد و خنثی | `#69757F` |
| متن اصلی | `#263238` |

## قواعد مشترک همهٔ پرامپت‌ها

عبارت زیر باید به انتهای هر پرامپت تصویری افزوده شود:

```text
Use a white background and the Sharif Technical Minimal theme. Use only flat vector
geometry, thin strokes, rounded rectangles, no shadows, no gradients, no 3D, no
decorative icons, and no long prose inside the figure. Use Times New Roman for Latin
and math labels. Use the fixed palette: blue #0072B2, green #009E73, orange #D55E00,
magenta #CC79A7, purple #6F3C8F, neutral gray #69757F, and ink #263238. Keep all text
editable. Deliver SVG with a 16:6 or 16:7 horizontal viewBox and also export a PDF.
Do not add a title inside the artwork; the Persian LaTeX caption is outside the figure.
```

برای نمودارهای تجربی استفاده از مدل تولید تصویر ممنوع است. پرامپت آن‌ها برای یک عامل
کدنویسی است و باید از داده‌های واقعی پروژه بخواند.

---

## شکل‌های مفهومی فعلی و جایگزین‌هایشان

### ۱. `selective_forgetting_pipeline.png`

**وضعیت:** جایگزین شود. نسخهٔ فعلی عمودی است و فضای A4 را هدر می‌دهد.

**SVG آماده:** `thesis/images/selective_forgetting_pipeline.svg`

```text
Redesign the selective-forgetting lifecycle as one compact horizontal timeline. Show
four events only: Learn T1, Learn T2, Learn T3, Forget T2. Under the events show the
exact active-task sets {T1}, {T1,T2}, {T1,T2,T3}, and after deletion {T1,T3}. From the
Forget T2 event branch to two outcomes: target-task accuracy Au approaches chance in
orange, while retained-task accuracy remains near its pre-request level in green. Do
not show any learning event after deletion because the reported persistence experiment
does not test fresh learning after a forget request. Add only the small footer
“Schematic lifecycle; no post-deletion learning is implied.”
```

### ۲. `parameter_overlap_concept.png`

**وضعیت:** جایگزین شود. ون‌دایگرام فعلی بیش از حد ساده است.

**SVG آماده:** `thesis/images/parameter_overlap_concept.svg`

```text
Create three aligned panels that reuse exactly the same fixed parameter grid. Panel
(a): color forgotten-mask-only cells orange, retained-mask-only cells blue, and their
intersection S_share = M_f ∩ M_r magenta. Panel (b): fade non-overlap cells and color
only S_share by the retained sensitivity q_i = |∇_i L_retain| from light to dark
purple. Panel (c): reuse the identical grid, keep all overlap cells visible, and hatch
exactly the top 20 percent as S_crit = Top-rho(S_share; q_i), with rho=0.2. Enforce
S_crit as a strict subset of S_share; never move or invent grid coordinates between
panels.
```

### ۳. شکل تازهٔ فصل مرور ادبیات

**محل:** بلافاصله پیش از بخش «مقایسهٔ مفهومی و جایگاه این پژوهش».

**SVG آماده:** `thesis/images/literature_taxonomy.svg`

```text
Create a horizontal conceptual taxonomy with five connected cards: Continual learning
(replay, regularization), Parameter isolation (task subnetworks), Machine unlearning
(data/model removal), Task-aware forgetting (PALL), and Overlap-aware repair
(EPALL and PALL-Adapter). Under the cards, draw one left-to-right conceptual axis from
“preserve previously learned behavior” to “selectively remove while protecting
overlap”. Mark the thesis position as overlap-aware, task-level, and empirical. This is
a taxonomy, not a historical claim and not experimental evidence.
```

### ۴. `EPALL_mechanism.png`

**وضعیت:** محتوای علمی حفظ، ولی شکل فشرده و هم‌تم شود.

**SVG آماده:** `thesis/images/EPALL_mechanism_compact.svg`

```text
Compress the EPALL mechanism into three equal horizontal panels. Panel (a): show one
forgotten path and one retained path sharing several coordinates; a naive reset of a
shared coordinate causes retained-task drift. Panel (b): show the exact sequence
S_share = M_f ∩ M_r, rank overlap coordinates with q_i = |∇_i L_retain| using retained
buffers only, then select S_crit = Top-rho(S_share; q_i). Panel (c): show five request
steps: delete the target buffer and retire M_f; form M_r and cache retained anchors;
apply branch-dependent target reset; run retained-only anchored repair; rebuild the
active mask and discard temporary state. End with “Empirical—not certified removal”.
Do not add claims of certified deletion.
```

### ۵. `pall_adapter_architecture.png`

**وضعیت:** جایگزین شود؛ نسخهٔ فعلی عمودی است.

**SVG آماده:** `thesis/images/pall_adapter_architecture.svg`

```text
Draw the exact implemented PALL-Adapter architecture left to right: input
x ∈ R^(B×3×H×W), frozen feature extractor theta_base producing g(x) ∈ R^512, shared
residual bottleneck adapter phi_s, one active task-specific residual bottleneck adapter
phi_t among phi_1…phi_T, shared classifier W_cls ∈ R^(C×512) with a highlighted
task-specific block of class rows C_t, and prediction y_hat_t. Add one top inset showing
the residual bottleneck z -> W_down(512->r) -> ReLU -> W_up(r->512) plus skip
connection, with W_down Kaiming-initialized and W_up initialized to zero. State that
r=16 is a compact fixed default, not an optimum. Add a dashed forget(u) callout with
the exact operations: soft-masked update on phi_s, reset phi_u, reset classifier row
block C_u, theta_base unchanged. Never depict the classifier as a T×K matrix or a
single task row.
```

### ۶. `softmask_drop_decomposition.png`

**وضعیت:** جایگزین شود؛ شکل فعلی بلند است و نسخهٔ تازهٔ ChatGPT نیز انقباض را درست
نمایش نمی‌دهد.

**SVG آماده:** `thesis/images/softmask_drop_decomposition.svg`

```text
Create two adjacent stacked bars with a common baseline. The unconstrained bar has
three visible segments: H subset of S_crit with m_i=1, S_crit\H with m_i=1, and
F°=S_forget\S_active with m_i=1. The soft-mask bar must preserve the F° segment,
contract the S_crit\H segment visibly to exactly (1-p) of its left-panel height, and
show H as a zero-update outlined slot with m_i=0. Draw dashed total levels Delta_unc
and Delta_soft and one bracket labelled ×(1-p). Add the footer “Schematic subphase
only; not a bound on end-to-end WorstDrop.” Do not imply an empirical measurement or
an end-to-end theorem.
```

### ۷. `bound_verification.pdf`

**وضعیت:** از پایان‌نامه حذف شود؛ بازطراحی تصویری توصیه نمی‌شود.

```text
Do not generate a replacement figure. Preserve the surrounding theoretical caveat in
prose: the plotted quantities have different units and uncalibrated constants, so the
diagnostic cannot verify or falsify the stated subphase result. Remove the figure and
its in-text figure reference instead of making it visually more persuasive.
```

---

## نمودارهای داده‌محور استفاده‌شده در پایان‌نامه

### ۸. سه شکل `main_metrics_*_1x3.pdf`

```text
Modify only the layout and visual styling of the real-data main-metrics dashboard in
tools/plot_report_results.py. Read completed run samples and the canonical thesis CSV;
never hard-code or fabricate values. Retain final average accuracy, signed WorstDrop,
and forgotten-task accuracy Au for CIFAR-10, CIFAR-100, and TinyImageNet. Preserve the
observed means and real 95% bootstrap confidence intervals. Produce three separate
horizontal 1×3 figures—one for final average accuracy, one for signed WorstDrop, and
one for forgotten-task accuracy Au—with columns CIFAR-10, CIFAR-100, and TinyImageNet
in that order. Use the fixed thesis palette, thin black edges, light dashed grids, and
the real chance lines. Keep negative WorstDrop values and its zero line. Export each
row as PDF, editable SVG with text preserved, and a 300-DPI preview. No text may be
smaller than 9 pt at final A4 size. Fail clearly when matching samples are unavailable.
```

### ۹. `overlap_response_cifar10.pdf`

```text
Restyle only the real pooled CIFAR-10 overlap-response plot. Preserve every individual
completed-run observation, OLS fit, HC3 95% confidence band, signed WorstDrop values,
and the zero line. Use circular observations with thin black edges; differentiate
methods by the fixed thesis colors and line styles. Keep the warning that full-network
mask IoU and adapter critical/shared ratio are different x measures. Export PDF and
editable SVG. Do not smooth, rescale, omit negative points, or synthesize runs.
```

### ۱۰. `overlap_response_cifar100.pdf`

همان پرامپت شکل ۹، با `CIFAR-100` به‌عنوان تحلیل اصلی و بدون تغییر در مشاهدات.

### ۱۱. `tradeoff_updated_vs_final_accuracy_by_dataset_regime.pdf`

```text
Restyle the real updated-parameter-ratio versus final-accuracy trade-off plot without
changing any point. Keep dataset/regime separation and annotate only the actual Pareto
frontier points to avoid clutter. Use the fixed method colors, distinct markers with
thin black edges, light dashed grid, and concise axes. Explicitly label the x-axis as a
nominal trainable/update scope diagnostic rather than end-to-end efficiency. Export PDF
and editable SVG; never infer missing configurations.
```

### ۱۲. `tradeoff_updated_vs_worstdrop_by_dataset_regime.pdf`

همان پرامپت شکل ۱۱، با محور عمودی `Signed WorstDrop`، خط صفر و حفظ مقادیر منفی.

### ۱۳. `aaai_storage.pdf`

```text
Regenerate the resident-state accounting chart only from
results/aggregates/storage_accounting_summary.csv. Preserve EPALL and CLPU totals and
per-active-task growth exactly. Use blue and green/orange thesis colors plus hatch as a
grayscale encoding, thin black edges, short labels, and a horizontal 1×2 layout. Export
PDF and editable SVG. Do not convert resident-state accounting into a claim about peak
memory, wall time, or FLOPs.
```

### ۱۴. `mia_before_after_by_dataset_regime.pdf`

```text
Restyle the real before/after membership-inference AUC plot. Preserve all observed
means, completed-run bootstrap confidence intervals, dataset/regime panels, and the
0.5 chance line. Use paired solid-versus-hatched marks and the fixed method colors.
Keep the y-axis honest and state in the caption—not inside the artwork—that proximity
to 0.5 is a null result of this attack, not proof of deletion. Export PDF and editable
SVG; use the full thesis text width so labels remain readable.
```

### ۱۵. `forgetting_persistence.pdf`

```text
Restyle only the real persistence trajectories selected by
tools/analyze_forgetting_persistence.py. Preserve the exact first-deleted-task rule,
request offsets, observed accuracies, and chance lines. Use one row of dataset panels,
method colors plus marker/line-style redundancy, and a shared legend. Keep the x-axis
label “Subsequent deletion requests”; do not imply any learning event after deletion.
Export PDF, editable SVG, and PNG preview.
```

### ۱۶. `adapter_bottleneck_ablation.pdf`

```text
Modify only the layout of the real task-adapter bottleneck ablation. Replace the tall
3×1 figure with a horizontal 1×3 layout for final average accuracy, signed WorstDrop,
and updated parameter ratio. Use categorical widths taken directly from completed
adapter_bottleneck_ablation_v1 runs. Preserve observed means and real 95% bootstrap
confidence intervals. Draw circular markers with thin black edges and connect only
adjacent observed widths with straight segments; no interpolation, splines, dense x
grid, or implied intermediate measurements. Show a zero line for WorstDrop. Use metric
colors blue #0072B2, orange #D55E00, green #009E73. Export PDF and editable SVG, and
fail if real runs are unavailable.
```

### ۱۷. `shared_bottleneck_ablation.pdf`

```text
Keep the verified real-data horizontal 1×4 layout. Use the exact categorical widths
from completed shared_bottleneck_ablation_v1 runs and preserve final accuracy, signed
WorstDrop, updated ratio, and shared critical-overlap ratio, including real 95%
bootstrap confidence intervals. Keep circular markers with black edges, straight
segments between adjacent observed widths only, the WorstDrop zero line, and colors
#0072B2, #D55E00, #009E73, #CC79A7. Retain the footer “Descriptive, two-seed ablation;
non-monotone trends”. Never claim width 16 is optimal; it is only a compact fixed
default. Export PDF, editable SVG, and 300-DPI PNG; report exact run files and widths.
```

### ۱۸. `representative_pall_adapter_accuracy_heatmap.pdf`

```text
Restyle only the representative real PALL-Adapter accuracy heatmap selected by the
existing deterministic run-selection function. Preserve the exact task-by-request
matrix, row normalization, task labels, request labels, and forget-event columns. Use a
single perceptually ordered blue colormap, orange dashed forget-event lines and triangle
markers, readable cell geometry, and a compact horizontal layout. Export PDF and
editable SVG. Do not interpolate missing cells or select a different run merely for a
more attractive pattern.
```

---

## نه تصویر تازهٔ پیوست‌شده

### تصویر تازهٔ ۱ — Shared-bottleneck ablation

**حکم:** معتبر و قابل استفاده؛ همان پرامپت شکل ۱۷. فقط تمام‌عرض درج شود، نه کنار یک
نمودار عمودی.

### تصویر تازهٔ ۲ — Subphase drop-budget decomposition

**حکم:** از نظر ایده مناسب، ولی نسبت انقباض غلط است. از پرامپت شکل ۶ استفاده شود.

### تصویر تازهٔ ۳ — معماری افقی PALL-Adapter

**حکم:** پس از اصلاح فنی مناسب است. از پرامپت شکل ۵ استفاده شود؛ به‌طور ویژه ورودی
`B×3×H×W` و طبقه‌بند `C×512` با بلوک سطرهای کلاس وظیفه حفظ شود.

### تصاویر تازهٔ ۴، ۵، ۶ و ۷ — نمودارهای روش/دیتاست مقایسه‌ای

**حکم:** استفاده نشوند. شامل روش‌ها، دیتاست‌ها، مقادیر و علامت‌های معناداری ناموجود در
پروژه‌اند.

```text
Do not edit or cosmetically repair the supplied raster chart. Rebuild the requested
comparison from tools/plot_report_results.py and completed project run artifacts only.
First enumerate the exact datasets, methods, regimes, seed files, means, and confidence
intervals that actually exist. Omit unavailable combinations explicitly instead of
filling them. Do not invent significance brackets, methods, datasets, error bars, or
chance levels. Export PDF and editable SVG and print an audit of every source file used.
If the real matching observations are unavailable, fail with a clear message.
```

### تصویر تازهٔ ۸ — شبکهٔ هم‌پوشانی و مجموعهٔ بحرانی

**حکم:** فقط پس از بازسازی قطعی. از پرامپت شکل ۲ استفاده شود؛ مختصات شبکه در هر سه
پنل باید دقیقاً یکسان باشند.

### تصویر تازهٔ ۹ — خط زمانی یادگیری و حذف

**حکم:** مجموعه‌های فعال نسخهٔ پیوست غلط‌اند. از پرامپت شکل ۱ استفاده شود. نسخهٔ پایان‌نامه
نباید یادگیری تازه پس از آغاز حذف را نمایش دهد.

## خروجی‌های قابل‌ویرایش

SVGهای مفهومی با فرمان زیر بازتولید می‌شوند:

```bash
python3 tools/build_thesis_concept_figures.py --outdir thesis/images
```

PDF هم‌نام هر SVG برای درج در XeLaTeX ساخته می‌شود. نمودارهای تجربی از اسکریپت‌های
واقعی پروژه تولید می‌شوند و در کنار PDF، فایل SVG با متن قابل‌ویرایش ذخیره می‌کنند.
