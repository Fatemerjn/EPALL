# پرامپت‌های پیشنهادی برای بازطراحی شکل‌های مفهومی پایان‌نامه

این پرامپت‌ها برای ساخت تصویرهای افقی و کم‌ارتفاع طراحی شده‌اند. بهتر است خروجی‌ها نسبت تصویر `16:7` یا `16:6`، پس‌زمینه‌ی سفید، سبک برداری/آکادمیک، و بدون متن طولانی داشته باشند تا در پایان‌نامه خوانا بمانند.

## 1. جریان فراموشی انتخابی آگاه از هم‌پوشانی

```text
Create a clean academic vector-style diagram for a PhD thesis, horizontal 16:7 aspect ratio, white background, minimal color palette. Show a continual learning model receiving a sequence of tasks on the left, then a highlighted "forget request" entering an overlap-aware unlearning module in the center. Inside the center module, show three compact stages: 1) estimate parameter importance with gradients, 2) identify the critical overlap S_crit between S_forget and S_active, 3) reset task-specific parameters and softly protect shared parameters. On the right, show two outcomes: the forgotten task fades out, retained tasks remain stable. Use simple geometric blocks, thin arrows, subtle blue/green for retained knowledge and red/orange for forgetting. Avoid decorative 3D, avoid dense text, use only short math labels: S_forget, S_active, S_crit, reset, protect. Make the image suitable for insertion into a LaTeX thesis at full text width.
```

## 2. هم‌پوشانی پارامتری و ناحیه‌ی بحرانی

```text
Design a professional horizontal 16:6 vector diagram on a white background explaining parameter overlap in selective unlearning. Show a large rectangular parameter space as a faint grid of small weights. Overlay two smooth translucent regions: S_forget in warm red/orange and S_active in cool green/blue. Their intersection S_crit should be highlighted with a crisp purple/indigo outline and a subtle glow. Add a small legend with only three labels: S_forget, S_active, S_crit. Add thin arrows indicating that updates are free in S_forget minus S_active, frozen/protected in S_crit, and unchanged elsewhere. Keep it clean, mathematical, and thesis-ready; no cartoons, no icons, no long sentences.
```

## 3. معماری افقی PALL-Adapter

```text
Create a horizontal 16:7 architecture diagram for "PALL-Adapter" in an academic machine learning thesis. Use a left-to-right pipeline: input image batch -> frozen ResNet-18 backbone theta_base -> shared bottleneck adapter phi_s -> task-specific adapter phi_t -> classifier head C_t -> task prediction. Draw the frozen backbone as a long muted gray block with a lock symbol, the shared adapter as a blue bottleneck block, task adapters as three small green parallel blocks, and the classifier as a compact final block. Show that forgetting task u resets only phi_u and C_u while phi_s receives a soft-masked update. Use short labels only: theta_base frozen, phi_s shared, phi_t task adapter, reset phi_u, soft mask. White background, thin arrows, no 3D, no decorative gradients, enough empty space for readability in print.
```

## 4. تجزیه‌ی افت با ماسک نرم

```text
Generate a clean horizontal 16:6 explanatory figure comparing unconstrained unlearning and soft-masked unlearning. Split the image into two side-by-side panels. Left panel: "unconstrained" update arrows hit both task-specific and critical shared parameters, causing a large red WorstDrop bar. Right panel: "soft mask" update arrows are full strength on forget-only parameters, reduced by (1-p) on S_crit, and zero on protected parameters H, causing a smaller blue WorstDrop bar. Use simple bars and arrows, with labels F-only, S_crit, H, m=1, m=1-p, m=0, WorstDrop. Academic vector style, white background, high contrast, no long prose, no decorative effects.
```

## 5. تصویر خلاصه برای فصل نتایج

```text
Create a thesis-ready horizontal 16:7 visual summary of experimental findings for overlap-aware selective forgetting. Use three compact panels: (1) final accuracy comparison, (2) WorstDrop reduction, (3) updated parameter ratio. Do not invent numeric values; leave bars abstract with placeholders or use neutral unlabeled bars. Emphasize the qualitative message: PALL-Modified improves retention, PALL-Adapter reduces updated parameters, and pretraining improves adapter accuracy. Use a restrained academic palette: blue for PALL-Original, orange for PALL-Modified, green for PALL-Adapter. White background, print-friendly, vector-like, minimal labels only.
```
