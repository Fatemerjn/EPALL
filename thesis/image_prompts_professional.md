# دستورهای حرفه‌ای و دقیق همه شکل‌های پایان‌نامه

این سند برای ساخت دوباره شکل‌ها با ChatGPT یا ویرایش شکل قبلی تهیه شده است. هر
دستور مستقل و آماده کپی است. هدف، یک سبک «علمی روایی با جزئیات متوسط» است:
گویاتر از نسخه‌های مینیمال فعلی، ولی افقی‌تر، خلوت‌تر و دانشگاهی‌تر از
اینفوگرافیک‌های عمودی پیوست‌شده.

## قرارداد بصری قطعی

- پس‌زمینه سفید؛ نسبت افقی میان `16:6` تا `16:8`، مگر برای نمودارهای داده‌ای.
- سبک flat-vector دانشگاهی؛ بدون سه‌بعدی، درخشش، بافت، عکس، موکاپ یا کارت‌های
  تزئینی بی‌معنا.
- استفاده از آیکون خطی فقط وقتی به انتقال مفهوم کمک کند: مدل، وظیفه، حذف، سپر،
  زیرشبکه، پیراسنجه، حافظه و طبقه‌بند.
- مجاز: سایه بسیار ضعیف و یکنواخت یا تغییر ته‌رنگ ۳ تا ۵ درصد برای تفکیک لایه‌ها.
  ممنوع: شیب شدید، سایه حجیم و ظاهر پوستر تبلیغاتی.
- فارسی: `Vazirmatn` یا `XB Niloofar`؛ لاتین و ریاضی: `Times New Roman`.
- متن فارسی باید دقیقا به‌صورت text قابل‌ویرایش باشد، نه مسیر برداری و نه تصویر.
  در SVG برای متن فارسی `direction="rtl"` و تنظیم مناسب bidi استفاده شود.
- پالت: آبی `#0072B2` مسیر یادگیری/ساختار، سبز `#009E73` نگه‌داری/حفاظت، نارنجی
  `#D55E00` حذف/هشدار، سرخابی `#CC79A7` هم‌پوشانی، بنفش `#6F3C8F` مجموعه بحرانی،
  خاکستری `#69757F` بخش منجمد، متن `#263238`.
- کادرهای گروهی با خط نازک، گوشه ۱۰ تا ۱۴ پیکسل و فاصله‌گذاری منظم. مسیر اصلی
  ضخامت ۲٫۵ تا ۳ پیکسل؛ مسیر فرعی ۱٫۵ تا ۲ پیکسل.
- عنوان کلی داخل تصویر تکرار نشود؛ کپشن زیر تصویر در LaTeX قرار می‌گیرد.
- حداقل اندازه متن پس از درج در A4 برابر ۹ پوینت باشد.
- خروجی اصلی SVG واقعی با viewBox دقیق، متن زنده و گروه‌بندی منطقی لایه‌ها باشد؛
  PDF و PNG سیصد DPI نیز صادر شود.

---

## الف) شکل‌های مفهومی — قابل ساخت یا ویرایش با GPT

### ۱) چرخه عملیاتی فراموشی انتخابی — `selective_forgetting_pipeline`

#### دستور ساخت از صفر

```text
Create a professional, medium-detail scientific vector diagram for a Persian PhD/MSc
thesis. The diagram explains the operational lifecycle of selective forgetting; it is
not the internal EPALL algorithm. Use a horizontal 1800×760 SVG canvas and a clear
three-stage narrative from right to left for Persian reading while keeping mathematical
symbols left-to-right.

Stage 1 — continual learning, blue group titled exactly «۱. یادگیری پیوسته». Show three
compact event cards connected by arrows: «یادگیری T1», «یادگیری T2», «یادگیری T3».
Each card must include a small meaningful line icon (task document entering a model), a
blue plus badge, and an active-task-state pill directly below it. The pills must be
exactly «{T1}», «{T1,T2}», and «{T1,T2,T3}». Add a small shared-model rail beneath the
cards to make clear that one model is updated sequentially, not three independent
models.

Stage 2 — deletion request, orange group titled exactly «۲. درخواست فراموشی». Show a
request document with a delete badge entering one card labelled «فراموشی T2». Directly
below it show the resulting active set exactly «{T1,T3}». Do not show a new learning
event after this request. Include a small label «مدل مشترک، همان پیراسنجه‌ها» to clarify
that the request acts on the deployed shared model.

Stage 3 — two simultaneous obligations, split into two parallel outcome lanes emerging
from the same forget-request card. The orange upper lane has a target/task icon and the
labels «رفتار وظیفه هدف سرکوب شود» and “A_u → c” where c is the task chance level.
The green lower lane has a shield over retained-task icons and the labels «دانش
وظیفه‌های باقی‌مانده حفظ شود» and “A_t(after) ≈ A_t(before),  t∈{T1,T3}”. Merge both
lanes into one final deployed-model card labelled «مدل پس از درخواست» with two status
chips: «T2 حذف‌شده» in orange and «T1,T3 فعال» in green.

Add a very small gray footer: «نمای شماتیک؛ یادگیری تازه پس از حذف نمایش داده نشده
است.» Use meaningful line icons, numbered group bands, clear routing arrows, subtle
tinted fills, and generous whitespace. Avoid a bare four-box timeline; the viewer must
understand the request, the shared model, the target obligation, the retention
obligation, and the final active state without reading the caption.

Use Persian text as editable SVG text with Vazirmatn/XB Niloofar and direction=rtl.
Use Times New Roman for T1, T2, T3 and equations. Palette: blue #0072B2, green #009E73,
orange #D55E00, magenta #CC79A7, purple #6F3C8F, neutral #69757F, ink #263238. White
background, flat vector, thin consistent strokes, no strong gradients, no 3D, no
photorealism. Do not add a title inside the artwork. Deliver editable SVG, PDF, and a
300-DPI PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Edit the attached selective-forgetting figure, but do not merely recolor or translate
it. Recompose it into a professional horizontal 1800×760 SVG with medium detail. Keep
the scientifically correct sequence Learn T1 → Learn T2 → Learn T3 → Forget T2 and the
exact active sets {T1}, {T1,T2}, {T1,T2,T3}, {T1,T3}. Replace the current over-simple
four-card timeline with three visually grouped stages: «۱. یادگیری پیوسته», «۲. درخواست
فراموشی», and «۳. ارزیابی دو هدف هم‌زمان». From Forget T2 create two clearly separate
lanes: orange «رفتار وظیفه هدف سرکوب شود، A_u→c» and green «دانش وظیفه‌های باقی‌مانده
حفظ شود، A_t(after)≈A_t(before)». Merge them into «مدل پس از درخواست» with chips «T2
حذف‌شده» and «T1,T3 فعال». Add meaningful line icons and a visible shared-model rail,
but keep the figure academic and uncluttered. Do not show any learning after deletion
and do not turn this lifecycle figure into the detailed EPALL algorithm. Use editable
Persian SVG text, the thesis palette, subtle tinted fills, no strong gradients or 3D,
and export SVG/PDF/300-DPI PNG.
```

### ۲) هم‌پوشانی پیراسنجه‌ها و زیرمجموعه بحرانی — `parameter_overlap_concept`

#### دستور ساخت از صفر

```text
Create a professional horizontal three-panel scientific SVG (1800×720) explaining how
structural overlap becomes a protected critical subset. Use the exact same 6×14
parameter grid and identical cell coordinates in all three panels. Add thin arrows
between panels so the transformation is read as a pipeline, not three unrelated grids.

Panel (الف), title «هم‌پوشانی ساختاری»: show forgotten-task mask M_f in orange,
retained-task union M_r in blue, and their true intersection S_share=M_f∩M_r in
magenta. Include a compact legend and a short callout «مختصات مشترک؛ محل بالقوه
تداخل». Do not use a Venn diagram.

Panel (ب), title «حساسیت نگه‌داری»: keep the identical grid; fade all non-overlap cells
to 20% opacity. Color only S_share cells from light to dark purple according to
q_i=|∇_i L_retain|. Add a five-step color key «حساسیت کم» to «حساسیت زیاد» and a small
retained-buffer icon with the note «محاسبه فقط با داده وظایف باقی‌مانده».

Panel (ج), title «انتخاب مجموعه بحرانی»: again keep the identical grid and hatch
exactly the top 20% of S_share cells as S_crit=Top-ρ(S_share;q_i), ρ=0.2. Use a purple
outline and hatch for critical cells so the panel remains readable in grayscale. Add a
shield callout «مختصات بحرانی در گام حذف محافظت می‌شوند» and the exact relation
S_crit⊂S_share.

Use panel labels (الف)، (ب)، (ج), Persian editable text, Times New Roman equations,
white background, subtle panel bands, consistent cell spacing, and the fixed thesis
palette. No invented grid movement, no additional cells, no decorative heatmap values,
no empirical claim. Deliver editable SVG/PDF/300-DPI PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Edit the attached three-grid figure into a more explanatory but still horizontal
thesis-quality SVG. Preserve every cell coordinate across all panels. Add a visible
left-to-right transformation flow, Persian titles «هم‌پوشانی ساختاری»، «حساسیت
نگه‌داری»، «انتخاب مجموعه بحرانی», a clear M_f/M_r/S_share legend, a q_i sensitivity
scale, and a shield callout for S_crit. Keep exactly ρ=0.2 and hatch only cells that are
already in S_share. Do not redraw the grids with different coordinates, do not convert
it to a Venn diagram, and do not imply measured experimental data. Use editable Persian
text and export SVG/PDF/PNG.
```

### ۳) نقشه مفهومی ادبیات — `literature_taxonomy`

#### دستور ساخت از صفر

```text
Create a polished horizontal conceptual map (1800×700 SVG) showing how this thesis sits
between continual learning and machine unlearning. Use two subtle horizontal lanes
rather than five isolated empty cards. The upper lane is «حفظ دانش» and contains
«یادگیری پیوسته» with replay/regularization icons and «جداسازی پیراسنجه‌ها» with task
subnetwork icons. The lower lane is «حذف هدفمند» and contains «یادگیری‌زدایی ماشین»
with data/model removal icons and «فراموشی آگاه از وظیفه — PALL» with a target-task
delete icon. At the right, merge both lanes into a highlighted green-purple thesis
position card: «ترمیم آگاه از هم‌پوشانی — EPALL و PALL-Adapter».

Inside the final card show three concise contributions as icon chips: «تشخیص هم‌پوشانی»،
«حفاظت رتبه‌بندی‌شده»، «کاهش دامنه به‌روزرسانی». Draw one conceptual tension axis
under the lanes from «حفظ رفتار آموخته‌شده» to «حذف انتخابی با حفاظت از دانش مشترک».
Add a small disclaimer «رده‌بندی مفهومی؛ نه ترتیب تاریخی و نه نتیجه تجربی».

Use Persian editable labels, restrained academic icons, clear merge arrows, no citation
claims, no ranking of prior work, and no visual suggestion that all fields form a
strict chronological sequence. White background, fixed thesis palette, SVG/PDF/PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Edit the attached literature taxonomy. Replace the simple row of five boxes with two
meaningful lanes, «حفظ دانش» and «حذف هدفمند», which merge into the highlighted thesis
position «ترمیم آگاه از هم‌پوشانی — EPALL و PALL-Adapter». Retain the concepts continual
learning, parameter isolation, machine unlearning, and task-aware forgetting, but use
icons and short mechanism tags to explain their role. Add the tension axis and the
disclaimer that this is a conceptual taxonomy, not history or evidence. Use Persian
editable SVG text and a professional horizontal layout.
```

### ۴) سازوکار EPALL — `EPALL_mechanism_compact`

#### دستور ساخت از صفر

```text
Create a medium-detail, thesis-quality three-panel SVG (1900×780) explaining EPALL as a
causal narrative. Persian labels are primary; variables remain Latin/math.

Panel (الف) «چرا هم‌پوشانی خطرناک است؟»: draw a small neural/subnetwork graph with one
orange forgotten path and one blue retained path. Highlight their shared coordinates
in magenta. Show a naive reset symbol on one shared coordinate and a small before/after
retained-output gauge drifting in orange. Add the exact conclusion «بازنشانی بی‌قید
مختصات مشترک، رفتار وظایف باقی‌مانده را منحرف می‌کند.»

Panel (ب) «کدام مختصات باید محافظت شوند؟»: show a three-step funnel with meaningful
mini-visuals, not only text boxes: (1) intersect masks, S_share=M_f∩M_r; (2) rank shared
coordinates with q_i=|∇_i L_retain| using retained buffers only; (3) select
S_crit=Top-ρ(S_share;q_i). Use magenta-to-purple ranking dots and a shield over the
selected subset.

Panel (ج) «درخواست فراموشی محافظت‌شده»: show five numbered operational steps connected
vertically inside one compact request container: «حذف بافر هدف و بازنشسته‌کردن M_f»،
«تشکیل M_r و ذخیره لنگرهای نگه‌داری»، «بازنشانی هدف وابسته به شاخه»، «ترمیم فقط با
وظایف باقی‌مانده و لنگرها»، «بازسازی نقاب فعال و حذف حالت موقت». Add separate orange
forget and green repair rails along the steps. End with a neutral badge «حذف تجربی؛ نه
گواهی‌شده».

Make the causal relation clear without overfilling the panels. Use Persian editable SVG
text, fixed palette, meaningful network/mask/shield/anchor icons, no claims of certified
deletion, no extra algorithmic step, SVG/PDF/PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Edit the attached EPALL figure without changing its algorithm. Keep three panels but
replace text-only cards with medium-detail mini-visuals: shared neural coordinates and
retained drift in (الف), mask intersection→gradient ranking→protected top-ρ funnel in
(ب), and the exact five request steps with orange/green process rails in (ج). Translate
descriptive text to Persian while preserving S_share, q_i, S_crit, M_f and M_r in math
notation. Add «حذف تجربی؛ نه گواهی‌شده». Do not add certified-removal claims or invent
operations. Deliver editable horizontal SVG.
```

### ۵) معماری PALL-Adapter — `pall_adapter_architecture`

#### دستور ساخت از صفر

```text
Create a technically exact, professional horizontal system-architecture SVG
(2000×820). The main inference path runs left to right: input batch
x∈R^(B×3×H×W) → frozen ResNet feature extractor θ_base → g(x)∈R^512 → trainable shared
adapter φ_s → one active task adapter φ_t selected from φ_1…φ_T → shared classifier
W_cls∈R^(C×512) with the task class-row block C_t highlighted → prediction ŷ_t.

Use recognizable but restrained visuals: stacked image tensor for input, a gray locked
backbone, a blue shared bottleneck module, a green task-adapter bank with inactive
adapters faded and φ_t active, and an orange classifier matrix with multiple class rows
and C_t highlighted. The classifier must not be drawn as one row per task or as T×K.

Add a top zoomed inset connected to both φ_s and φ_t, labelled «ساختار یکسان لایه‌ی تطبیق
باقیمانده»: z → W_down:512→r → ReLU → W_up:r→512 → plus skip connection. Include
«W_down: Kaiming initialization», «W_up=0 ⇒ identity at initialization», d→r→d,
d=512, and «r=16 پیش‌فرض ثابت و فشرده؛ نه مقدار بهینه».

Add a bottom dashed request lane labelled forget(u) with exact arrows to affected
components: «به‌روزرسانی با نقاب نرم φ_s»، «بازنشانی φ_u»، «پاک‌سازی بلوک سطرهای
C_u». Place a gray lock note under θ_base: «θ_base بدون تغییر». Make clear that storage
grows by one small task adapter per task, not one backbone per task.

Use Persian editable labels for explanations and Times New Roman for variables. White
background, fixed palette, consistent arrow routing, no decorative perspective, no
incorrect tensor dimensions, SVG/PDF/300-DPI PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Technically correct and professionally restyle the attached PALL-Adapter architecture.
Preserve the horizontal inference path and add meaningful component visuals instead of
plain boxes. Correct input to B×3×H×W, feature dimension to 512, classifier to
C×512 with a highlighted class-row block C_t, and show one active φ_t among φ_1…φ_T.
Keep the residual bottleneck inset and the exact forget(u) operations. State that r=16
is a compact fixed default, not optimal. Translate explanatory text to editable Persian
but keep mathematical symbols Latin. Do not depict a T×K task-row classifier and do not
show the backbone as trainable.
```

### ۶) تجزیه بودجه افت در زیرمرحله نقاب نرم — `softmask_drop_decomposition`

#### دستور ساخت از صفر

```text
Create a rigorous but visually explanatory horizontal two-panel schematic SVG
(1800×760). Use a shared baseline and a common vertical conceptual axis labelled
«سهم در ریسک افزایش افت وظایف باقی‌مانده — واحد دلخواه». State visibly that the figure
is schematic, not measured data.

Panel (الف) «بدون محدودیت»: one stacked bar with exactly three segments. Bottom purple
hatched segment H⊂S_crit with m_i=1; middle magenta segment S_crit\H with m_i=1; top
orange segment F°=S_forget\S_active with m_i=1. Add a dashed total level Δ_t^unc.

Panel (ب) «نقاب نرم»: preserve the top F° segment unchanged. Contract only the
S_crit\H segment to exactly (1−p) of its height in panel (الف), and show the contraction
with a dimension bracket labelled «ضریب انقباض ×(1−p)». Show H as an empty outlined
zero-update slot at the bottom with m_i=0. Add dashed total level Δ_t^soft. Use thin
guide lines between corresponding segments so the viewer sees what changed and what
did not.

Add three compact explanatory chips: orange «خارج از مجموعه فعال: بدون تغییر»،
magenta «مختصات غیرمحافظت‌شده: مقیاس 1−p»، purple «مختصات بحرانی H: به‌روزرسانی
صفر». Footer exactly: «فقط شماتیک زیرمرحله؛ کرانی برای WorstDrop آغاز تا پایان نیست.»

Do not draw an empirical y-axis scale, do not claim a measured bound, and do not change
the F° height. Persian editable SVG text, Times New Roman equations, fixed thesis
palette, SVG/PDF/PNG.
```

#### دستور ویرایش تصویر قبلی

```text
Edit the attached decomposition figure while preserving its mathematical meaning.
Keep two adjacent bars and the same baseline. Make the unchanged F° segment visually
identical in both panels, scale only S_crit\H by exactly 1−p, and render H as a zero
update outlined slot. Add correspondence guides, a true dimension bracket, Persian
explanatory chips, and the footer that this is only a subphase schematic and not an
end-to-end WorstDrop bound. Do not turn it into a measured bar chart or alter segment
semantics. Deliver editable SVG.
```

### ۷) شکل `bound_verification`

```text
Do not generate or edit this figure. The compared quantities have different units and
uncalibrated constants, so a visually persuasive diagram would be scientifically
misleading. Preserve the caveat in prose and remove the figure reference. Return a
clear message that no replacement image should be produced.
```

---

## ب) نمودارهای تجربی — دستور برای ChatGPT/Codex کدنویس، نه مدل تصویر

> در این بخش هر دستور باید همراه پروژه اجرا شود. دادن نمودار به مدل تصویر برای
> «زیباترکردن» مجاز نیست، چون ممکن است ارتفاع میله، خطا یا عدد تغییر کند.

### ۸) دقت نهایی اصلی — `main_metrics_final_accuracy_1x3`

```text
Act as a coding agent inside this repository. Do not use image generation. Rebuild the
real-data final-average-accuracy figure from the canonical thesis CSV and matching
completed metrics.json files only. Produce a horizontal 1×3 layout in the fixed order
CIFAR-10, CIFAR-100, TinyImageNet. Preserve every canonical method/regime row, table
mean, and real 95% bootstrap CI. Group rows visually as «از صفر»، «پیش‌آموزش‌دیده»،
«افراز استاندارد» using subtle horizontal bands; use readable method labels, black CI
whiskers, the fixed method colors, and no ranking decorations. Use Persian axis label
«دقت نهایی میانگین» and concise Persian group labels while method/dataset names remain
Latin. Minimum final A4 text size 9 pt. Export PDF, editable SVG with live text, and
300-DPI PNG. Print an audit of source files. Fail if any plotted row lacks completed-run
samples; never fabricate or infer a CI.
```

### ۹) بیشترین افت اصلی — `main_metrics_worstdrop_1x3`

```text
Act as a coding agent; do not use image generation. Rebuild the signed WorstDrop main
figure only from canonical real runs. Use a horizontal 1×3 layout ordered CIFAR-10,
CIFAR-100, TinyImageNet, with the same method/regime rows and grouping as the accuracy
figure. Preserve every mean and 95% bootstrap CI, keep negative observations, and draw
a clear vertical zero line in every panel. Persian axis label: «بیشترین افت علامت‌دار».
Do not clip at zero, convert signs, rank methods, invent missing methods, or add
significance marks. Export audited PDF/SVG/300-DPI PNG and fail on missing samples.
```

### ۱۰) دقت وظیفه حذف‌شده — `main_metrics_au_1x3`

```text
Act as a coding agent; do not use image generation. Rebuild the real forgotten-task
accuracy A_u figure in a horizontal 1×3 layout for CIFAR-10, CIFAR-100, TinyImageNet.
Preserve canonical rows, real means, and completed-run bootstrap CIs. Draw the correct
dataset-specific chance line in every panel and label it «سطح تصادفی». Persian axis
label: «دقت وظیفه حذف‌شده A_u». Do not visually imply that lower is always better;
the meaningful quantity is proximity to the task chance level. Export audited
PDF/editable SVG/300-DPI PNG; never fabricate missing samples.
```

### ۱۱) پاسخ به هم‌پوشانی CIFAR-10 — `overlap_response_cifar10`

```text
Modify the plotting code, not the raster image. Rebuild the pooled CIFAR-10 overlap
response from every matching completed run. Preserve every observation, signed
WorstDrop, the zero line, the OLS fit, and HC3 95% confidence band. Use circular points
with thin black edges; distinguish methods with the fixed thesis colors plus marker and
line-style redundancy. Use a clean horizontal analytical layout with a small Persian
callout explaining that full-network mask IoU and adapter critical/shared ratio are
different x quantities and must not be merged semantically. No smoothing, point
deletion, clipping, or invented runs. Export PDF/SVG/PNG and print exact run paths.
```

### ۱۲) پاسخ به هم‌پوشانی CIFAR-100 — `overlap_response_cifar100`

```text
Modify the plotting code, not the image. Rebuild the pooled CIFAR-100 overlap-response
analysis from matching completed runs only. Preserve all points, signed WorstDrop,
zero line, OLS coefficient, HC3 confidence band, and method identity. Use the same
visual grammar as the CIFAR-10 companion so both figures read as a matched pair. Add
the same metric-compatibility warning. Do not smooth, rescale, or select attractive
subsets. Export audited PDF, editable SVG, and PNG.
```

### ۱۳) بده‌بستان دامنه به‌روزرسانی و دقت — `tradeoff_updated_vs_final_accuracy_by_dataset_regime`

```text
As a coding agent, rebuild this trade-off plot only from real aggregate rows and
matching run artifacts. Preserve every point and dataset/regime panel. x-axis must be
labelled «نسبت اسمی دامنه به‌روزرسانی» and y-axis «دقت نهایی میانگین». Use method
colors plus distinct markers with black edges. Annotate only points that are truly on
the computed Pareto frontier, using collision-free labels. Include a small note that
the x quantity is not wall time, FLOPs, peak memory, or the number of coordinates that
actually changed. No interpolated frontier curve and no missing configuration filling.
Export audited PDF/SVG/PNG.
```

### ۱۴) بده‌بستان دامنه به‌روزرسانی و بیشترین افت — `tradeoff_updated_vs_worstdrop_by_dataset_regime`

```text
Rebuild from real rows only. Keep the exact dataset/regime panels, observed update
ratios, signed WorstDrop values, negative values, and zero reference line. x-axis:
«نسبت اسمی دامنه به‌روزرسانی»; y-axis: «بیشترین افت علامت‌دار». Compute and label
only the real non-dominated points; do not draw a smoothed frontier or treat the x-axis
as an end-to-end efficiency metric. Match the colors/markers of the accuracy trade-off
figure. Export audited PDF/editable SVG/PNG.
```

### ۱۵) محاسبه حالت مقیم — `aaai_storage`

```text
Rebuild only from results/aggregates/storage_accounting_summary.csv. Make a professional
horizontal 1×2 accounting figure: left panel shows exact resident-state totals for the
matched sequence; right panel shows exact growth per active task. Use stacked or grouped
bars only when their components exactly sum to the stored total. Add hatches for
grayscale, thin black edges, direct numeric labels, and a compact Persian explanatory
note distinguishing resident state from peak GPU memory, wall time, and FLOPs. Preserve
all EPALL/CLPU values exactly. Export PDF/editable SVG/PNG and print the CSV rows used.
```

### ۱۶) MIA پیش/پس از حذف — `mia_before_after_by_dataset_regime`

```text
Rebuild the membership-inference AUC chart from completed real runs only. Preserve all
before/after means, paired identity where available, bootstrap CIs, dataset/regime
panels, and the 0.5 chance line. Use matched before/after encodings (solid versus hatch
or connected paired markers) plus method colors. Persian labels: «پیش از درخواست»،
«پس از درخواست»، «AUC حمله عضویت»، «سطح تصادفی ۰٫۵». Keep the y scale honest. Do not
write “privacy proven” or “certified deletion”; proximity to 0.5 is only a null result
for this attack. Export audited PDF/SVG/PNG.
```

### ۱۷) ماندگاری فراموشی — `forgetting_persistence`

```text
Run tools/analyze_forgetting_persistence.py and restyle only its deterministic real
trajectories. Preserve the exact first-deleted-task selection rule, request offsets,
observed accuracies, and dataset chance levels. Use one horizontal row of dataset
panels, method colors plus marker/line redundancy, and a shared legend. Persian x-axis:
«درخواست‌های حذف بعدی»; y-axis: «دقت وظیفه حذف‌شده». Clearly mark request offset 0 as
the deletion point. Do not show or imply fresh learning after deletion and do not
interpolate missing request offsets. Export audited PDF/SVG/PNG and the source CSV.
```

### ۱۸) جاروب گلوگاه لایه‌ی تطبیق وظیفه — `adapter_bottleneck_ablation`

```text
Modify only plotting code and use completed adapter_bottleneck_ablation_v1 runs. Build
a horizontal 1×3 figure with categorical observed widths in their true order. Panels:
«دقت نهایی»، «بیشترین افت علامت‌دار»، «نسبت پیراسنجه‌های به‌روزشده». Preserve observed
means and real 95% bootstrap CIs. Use circular observations with thin black edges and
connect only adjacent observed widths using straight segments. Add the WorstDrop zero
line. No spline, dense numeric x scale, implied intermediate measurement, or optimality
claim. Footer: «مطالعه توصیفی دو بذری؛ روندها الزاما یکنوا نیستند». Export audited
PDF/editable SVG/300-DPI PNG; fail if matching real runs are absent.
```

### ۱۹) جاروب گلوگاه مشترک — `shared_bottleneck_ablation`

```text
Modify only plotting code and read completed shared_bottleneck_ablation_v1 runs. Use a
professional horizontal 1×4 layout, approximately 14.0×3.8 inches, with exact
categorical widths 4, 8, 16, 32 only if those widths are confirmed by real configs.
Panel order: «دقت نهایی»، «بیشترین افت علامت‌دار»، «نسبت پیراسنجه‌های به‌روزشده»،
«نسبت هم‌پوشانی بحرانی/مشترک». Preserve every observed mean and real two-seed 95%
bootstrap CI. Use colors #0072B2, #D55E00, #009E73, #CC79A7, circular markers with
black edges, and straight segments only between adjacent observed categories. Include
the WorstDrop zero line and shared x label «عرض گلوگاه مشترک». Footer exactly:
«مطالعه توصیفی دو بذری؛ روندهای غیر یکنوا». Never claim width 16 is optimal; state
only that it is a compact fixed default. Export PDF/SVG/300-DPI PNG and print every
config.json and metrics.json path used. Fail rather than fabricate.
```

### ۲۰) نقشه حرارتی دقت لایه‌ی تطبیق — `representative_pall_adapter_accuracy_heatmap`

```text
Use the existing deterministic run-selection function and rebuild the exact real
task-by-request accuracy matrix. Preserve all task rows, request columns, values,
missing cells, row meaning, and forget-event locations. Use a perceptually ordered
blue colormap, thin white cell boundaries, readable Persian row/column descriptors,
orange dashed vertical lines and triangle markers at actual forget events, and a
single honest colorbar. Add a small visual key distinguishing «درخواست یادگیری» and
«درخواست حذف» only if those event types exist in the selected run. Do not interpolate
missing cells, reorder requests, normalize rows unless the original analysis does, or
choose a more attractive run. Export audited PDF/editable SVG/PNG and report the exact
selected run path.
```

---

## ج) دستور مستقل برای هر یک از ۹ تصویر پیوست‌شده

### تصویر پیوست ۱ — نمودار shared-bottleneck

```text
Use the attached chart only as a visual reference; rebuild it from the repository's
completed shared_bottleneck_ablation_v1 configs and metrics. Preserve exact widths,
means, signed WorstDrop, and real two-seed bootstrap intervals. Keep a horizontal 1×4
layout, larger Persian axis labels, black-edged circular points, straight adjacent
segments, zero line, shared x label «عرض گلوگاه مشترک», and the four thesis colors.
Footer: «مطالعه توصیفی دو بذری؛ روندهای غیر یکنوا». Do not claim width 16 is optimal.
Export audited PDF/SVG/300-DPI PNG and list all run files. Do not image-edit the raster.
```

### تصویر پیوست ۲ — تجزیه بودجه افت

```text
Edit the attached conceptual figure into a horizontal thesis-quality SVG. Preserve the
meaning of the three unconstrained segments. In the soft-mask panel, keep F° unchanged,
scale only S_crit\H to exactly 1−p of its original height, and show H as an outlined
zero-update slot with m_i=0. Add correspondence guides, a true contraction bracket,
Persian labels, and three explanatory chips for unchanged, contracted, and protected
coordinates. Footer: «فقط شماتیک زیرمرحله؛ کرانی برای WorstDrop آغاز تا پایان نیست.»
Do not imply measured data or an end-to-end theorem. Editable SVG/PDF/PNG.
```

### تصویر پیوست ۳ — معماری PALL-Adapter

```text
Edit the attached architecture while retaining its useful professional detail. Correct
and preserve input x∈R^(B×3×H×W), frozen θ_base, g(x)∈R^512, shared φ_s, one active
φ_t among φ_1…φ_T, classifier W_cls∈R^(C×512) with class-row block C_t, and ŷ_t. Keep
the residual bottleneck inset with 512→r→512 and skip connection. Add Persian
explanations and the exact forget(u) effects: soft-masked φ_s update, reset φ_u, clear
C_u rows, θ_base unchanged. Say r=16 is a compact fixed default, not optimal. Do not
draw a T×K task-row classifier. Deliver an editable horizontal SVG.
```

### تصویر پیوست ۴ — نمودار ۳×۳ با CIFAR-100/Tiny-ImageNet/ImageNet-R

```text
Do not cosmetically edit or reuse the attached raster because its datasets, method
names, values, error bars, chance levels, and significance brackets are not verified by
this repository. Act as a coding agent. First enumerate real datasets, regimes,
methods, seed files, means, CIs, and any valid statistical tests available in the
project. Rebuild a comparison only for combinations supported by completed runs. Omit
ImageNet-R or any method that lacks real matching observations. Never reproduce the
asterisk brackets unless a documented paired test is computed from matching seeds.
Use the professional thesis chart theme and export audited PDF/SVG/PNG. If the requested
comparison is unavailable, stop with a clear message instead of inventing values.
```

### تصویر پیوست ۵ — دو نمودار میله‌ای افقی روش‌ها

```text
Do not edit the attached raster or preserve its displayed numbers. Verify every method
and value against canonical project aggregates and completed runs. If a real matched
comparison exists, rebuild two horizontal panels only: final average accuracy and
signed WorstDrop, with real bootstrap CIs, zero line for WorstDrop, honest method order,
and no decorative ranking. Use Persian panel labels and Latin method names. If any
method/value in the reference is absent, omit it and report the omission. Export
audited PDF/editable SVG/PNG; never infer or visually copy unverified values.
```

### تصویر پیوست ۶ — شبکه ۳×۳ Split-CIFAR-10/100/CUB-200

```text
Treat the attached image as an unverified design reference only. Search the repository
for completed real runs of each shown dataset and method. Do not show CUB-200 or any
method unless matching configs and per-seed metrics exist. Rebuild the supported
comparison as separate metric rows for final accuracy, signed WorstDrop, and A_u with
real CIs and correct dataset-specific chance lines. Avoid dense 3×3 compression if
labels become smaller than 9 pt; prefer three full-width horizontal figures. Export an
audit and PDF/SVG/PNG. Fail instead of filling missing cells.
```

### تصویر پیوست ۷ — شبکه ۳×۳ CIFAR-10/CIFAR-100/Split-CIFAR

```text
Do not image-edit the attached chart. Audit the exact meaning of CIFAR-10, CIFAR-100,
and “Split-CIFAR” in this repository before plotting. Rebuild only real, protocol-
matched comparisons, separating from-scratch, pretrained-frozen, and standard-split
regimes rather than merging them. Preserve real means/CIs, signed WorstDrop, and
dataset-specific A_u chance lines. Use readable full-width metric rows instead of one
overcrowded 3×3 page. Do not copy the reference numbers or chance levels. Export
audited PDF/editable SVG/PNG and list all omitted unsupported combinations.
```

### تصویر پیوست ۸ — شبکه هم‌پوشانی و مجموعه بحرانی

```text
Edit the attached three-panel grid into a professional scientific SVG while keeping
the same grid coordinates across every panel. Panel «هم‌پوشانی ساختاری» must show
M_f, M_r and S_share=M_f∩M_r. Panel «حساسیت نگه‌داری» must fade non-overlap and rank
only S_share using q_i=|∇_iL_retain|. Panel «مجموعه بحرانی» must hatch exactly the
top ρ=0.2 subset and enforce S_crit⊂S_share. Add transformation arrows, a clear legend,
sensitivity scale, shield callout and Persian editable labels. Do not move cells,
invent values, or treat the grid as empirical data. Export SVG/PDF/PNG.
```

### تصویر پیوست ۹ — خط زمانی یادگیری و حذف

```text
Use the attached timeline only as a composition reference. Rebuild the scientifically
correct lifecycle: learn T1 → learn T2 → learn T3 → forget T2, with exact active sets
{T1}, {T1,T2}, {T1,T2,T3}, {T1,T3}. Do not show learning T4/T5 or any new learning
after deletion because the reported persistence experiment does not test it. Enrich the
diagram with a shared-model rail and two post-request outcome lanes: A_u→chance for T2
and retained accuracy near its pre-request level for T1,T3. Use Persian stage labels,
meaningful line icons, medium detail, horizontal SVG, and a footer that no
post-deletion learning is implied.
```

## نکته اجرایی برای GPT

برای شکل مفهومی، تصویر قبلی را همراه «دستور ویرایش» بارگذاری کنید. اگر نتیجه هنوز
شبیه پوستر یا تصویر رستری بود، در پیام بعدی فقط بنویسید:

```text
Return the actual SVG source code, not a rendered preview. Keep every Persian label as
editable <text>, group each semantic stage with <g id="...">, and do not rasterize any
icon or text.
```

برای نمودار تجربی، کل پوشه پروژه یا دسترسی Codex لازم است؛ تصویر به‌تنهایی منبع
معتبر داده نیست.
