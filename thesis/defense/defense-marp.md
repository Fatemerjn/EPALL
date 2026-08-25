---
marp: true
theme: default
paginate: true
html: true
math: katex
footer: 'دانشگاه صنعتی شریف · دفاع پایان‌نامه کارشناسی ارشد · شهریور ۱۴۰۵'
style: |
  @font-face {
    font-family: "Sharif";
    src: url("../styles/fonts/Sharif1.3-Regular.ttf") format("truetype");
    font-weight: 400;
  }
  @font-face {
    font-family: "Sharif";
    src: url("../styles/fonts/Sharif1.3-SemiBold.ttf") format("truetype");
    font-weight: 700;
  }
  :root {
    --ink: #14213d;
    --muted: #5d6577;
    --navy: #14213d;
    --teal: #007c83;
    --teal-soft: #e8f4f3;
    --gold: #f2b134;
    --red: #a63d40;
    --paper: #fbfaf7;
  }
  section {
    direction: rtl;
    text-align: right;
    font-family: "Sharif", "Vazirmatn", "Tahoma", sans-serif;
    font-size: 26px;
    line-height: 1.55;
    color: var(--ink);
    background: var(--paper);
    padding: 64px 74px 58px;
    justify-content: flex-start;
  }
  section::after {
    font-size: 15px;
    color: #778092;
  }
  header, footer {
    font-size: 14px;
    color: #6b7280;
  }
  h1 {
    color: var(--navy);
    font-size: 43px;
    line-height: 1.25;
    margin: 0 0 24px;
    border-right: 8px solid var(--gold);
    padding-right: 18px;
  }
  h2 {
    color: var(--teal);
    font-size: 31px;
    margin: 0 0 14px;
  }
  p { margin: 10px 0; }
  ul, ol { margin: 8px 0 0; padding-right: 1.15em; }
  li { margin: 7px 0; }
  strong { color: var(--teal); }
  .ltr { direction: ltr; unicode-bidi: isolate; display: inline-block; }
  .en { direction: ltr; unicode-bidi: isolate; display: inline-block; font-family: Arial, sans-serif; }
  .muted { color: var(--muted); }
  .small { font-size: 20px; line-height: 1.5; }
  .tiny { font-size: 16px; line-height: 1.45; }
  .accent { color: var(--red); font-weight: 700; }
  .takeaway {
    border-right: 7px solid var(--teal);
    padding: 12px 22px;
    background: var(--teal-soft);
    font-size: 29px;
    line-height: 1.55;
  }
  .note {
    border-right: 5px solid var(--gold);
    padding: 8px 16px;
    background: #fff6dc;
    font-size: 21px;
  }
  .cols { display: flex; gap: 34px; align-items: center; }
  .cols.top { align-items: flex-start; }
  .col { flex: 1; min-width: 0; }
  .wide { flex: 1.3; }
  .narrow { flex: .7; }
  .metrics { display: flex; gap: 18px; margin-top: 20px; }
  .metric {
    flex: 1;
    border-top: 5px solid var(--teal);
    background: white;
    padding: 15px 17px;
    box-shadow: 0 3px 14px rgba(20, 33, 61, .08);
  }
  .metric b {
    display: block;
    direction: ltr;
    unicode-bidi: isolate;
    text-align: center;
    font-family: Arial, sans-serif;
    font-size: 32px;
    color: var(--navy);
  }
  .metric span { display: block; text-align: center; font-size: 18px; color: var(--muted); }
  .flow { display: flex; direction: rtl; align-items: stretch; gap: 12px; margin-top: 20px; }
  .flow > div {
    flex: 1;
    background: white;
    border-bottom: 5px solid var(--teal);
    padding: 15px 13px;
    text-align: center;
    font-size: 21px;
  }
  .flow .arrow { flex: 0 0 auto; background: transparent; border: 0; padding: 24px 0; color: var(--gold); font-size: 34px; }
  .figure { display: block; margin: 6px auto 0; object-fit: contain; }
  .caption { text-align: center; font-size: 16px; color: var(--muted); margin-top: 3px; }
  table { width: 100%; border-collapse: collapse; font-size: 20px; background: white; }
  th { background: var(--navy); color: white; padding: 10px; }
  td { padding: 10px; border-bottom: 1px solid #d7dbe2; text-align: center; }
  tr.highlight td { background: var(--teal-soft); font-weight: 700; }
  section.title {
    background: linear-gradient(130deg, #0d1b35 0%, #16385d 64%, #007c83 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 84px 96px;
  }
  section.title h1 {
    color: white;
    border-right-color: var(--gold);
    font-size: 54px;
    max-width: 1050px;
  }
  section.title h2 { color: #c9eeeb; font-size: 29px; }
  section.title p { color: #edf4f7; }
  section.title strong { color: #ffd166; }
  section.compact { padding-top: 42px; padding-bottom: 46px; place-content: start !important; }
  section.compact h1 { font-size: 38px; margin-bottom: 14px; }
  section.compact .metric { padding: 10px 13px; }
  section.compact .metric b { font-size: 28px; }
  section.compact .metric span { font-size: 16px; }
  section.section-break {
    background: var(--navy);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.section-break h1 { color: white; font-size: 52px; }
  section.section-break .takeaway { color: white; background: rgba(255,255,255,.08); }
  section.final {
    background: linear-gradient(135deg, #14213d 0%, #007c83 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.final h1 { color: white; font-size: 49px; }
  section.final strong { color: #ffd166; }
  section.final .takeaway { color: var(--ink); background: rgba(255,255,255,.94); }
  section.final h2 { color: #ffd166; }
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# مدیریت فراموشی در یادگیری پیوسته با کمک روش‌های یادگیری ژرف

## محافظت آگاه از هم‌پوشانی برای فراموشی انتخابی

**فاطمه رائیجیان**
استاد راهنما: دکتر حمید بیگی

دانشکده مهندسی کامپیوتر · دانشگاه صنعتی شریف
شهریور ۱۴۰۵

<!--
یادداشت ارائه: مسئله‌ی محوری این دفاع، حذف یک وظیفه از مدل پیوسته بدون تخریب دانش وظایف دیگر است. مسیر اصلی پایان‌نامه EPALL است و مسیر آداپتر به‌عنوان یک نتیجه‌ی اکتشافی و ممیزی‌شده ارائه می‌شود.
-->

---

# پیام اصلی پایان‌نامه

<div class="takeaway">
آسیب جانبی فراموشی در یک مدل مشترک <strong>تصادفی نیست</strong>؛ در ناحیه‌ی هم‌پوشان پارامترها متمرکز است و با رتبه‌بندی حساسیت می‌توان آن را مهار کرد.
</div>

<div class="metrics">
  <div class="metric"><b>0.4863 / 0.0959</b><span>دقت وظیفه‌ی هدف؛ نزدیک سطح تصادفی</span></div>
  <div class="metric"><b>0.0027 / 0.0056</b><span>بیشترین افت روی CIFAR-10 / 100</span></div>
  <div class="metric"><b>1.6–2.2×</b><span>حالت مقیم کمتر از CLPU</span></div>
</div>

<p class="note">نتیجه‌ی دفاع‌پذیر: <span class="en">EPALL</span> یک راه‌حل مدل‌مشترک نزدیک به جداسازی کامل است؛ نه ادعای حذف دقیق یا گواهی‌شده.</p>

<!--
یادداشت ارائه: همان ابتدا دامنه‌ی ادعا را روشن کنید. EPALL سرکوب رفتاری را با حفظ وظایف نگه‌داری‌شده بهبود می‌دهد، اما حذف دقیق یا تضمین حریم خصوصی را اثبات نمی‌کند.
-->

---

<!-- _class: compact -->

# مسئله: حذف یک وظیفه بدون فراموشی بقیه

<div class="flow">
  <div><strong>یادگیری پیوسته</strong><br>انباشت دانش در طول زمان</div>
  <div class="arrow">←</div>
  <div><strong>درخواست حذف</strong><br>سرکوب وظیفه‌ی هدف</div>
  <div class="arrow">←</div>
  <div><strong>ترمیم مدل</strong><br>حفظ وظایف فعال</div>
</div>

<div class="cols top" style="margin-top:28px">
  <div class="col">
    <h2>هدف فراموشی</h2>
    <p>دقت وظیفه‌ی حذف‌شده باید به سطح تصادفی برسد.</p>
  </div>
  <div class="col">
    <h2>هدف نگه‌داری</h2>
    <p>دقت وظایف باقی‌مانده نباید در اثر درخواست حذف افت کند.</p>
  </div>
</div>

<p class="takeaway">تنش اصلی: همان پارامترهایی که هدف را به یاد دارند، ممکن است برای وظایف دیگر نیز ضروری باشند.</p>

<!--
یادداشت ارائه: واحد حذف در این پایان‌نامه یک وظیفه‌ی کامل در سناریوی Task-Incremental است و شناسه‌ی وظیفه هنگام استنتاج در دسترس است.
-->

---

# منشأ آسیب جانبی: هم‌پوشانی پارامتری

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/parameter_overlap_concept.svg" style="width:100%">
  </div>
  <div class="col narrow">
    <p><strong>پارامترهای انحصاری هدف</strong><br>قابل بازنشانی با خطر کمتر</p>
    <p><strong>پارامترهای مشترک</strong><br>ویرایش آن‌ها می‌تواند به وظایف فعال آسیب بزند</p>
    <p class="note"><span class="en">PALL</span> اشتراک را ساختاری می‌کند، اما مختصه‌های مشترک را بر اساس حساسیت نگه‌داری تفکیک نمی‌کند.</p>
  </div>
</div>

<!--
یادداشت ارائه: تصویر، هم‌پوشانی ماسک زیرشبکه‌ی هدف با اجتماع زیرشبکه‌های فعال را نشان می‌دهد. فرض اصلی این است که آسیب درون ناحیه‌ی مشترک نیز یکنواخت توزیع نشده است.
-->

---

<!-- _class: compact -->

# پرسش‌های پژوهش، یک زنجیره‌ی منطقی می‌سازند

1. آیا حفاظت از مختصه‌های مشترکِ حساس، هم‌زمان سرکوب و نگه‌داری را ممکن می‌کند؟
2. روش مدل‌مشترک تا چه حد به مرجع جداسازی کامل نزدیک می‌شود؟
3. آیا با رشد هم‌پوشانی، سود حفاظت بیشتر می‌شود؟
4. کدام مؤلفه‌ی روش واقعا منشأ بهبود است؟
5. آیا سازوکار مشابه در مسیر کارآمد پارامتری نیز اثرگذار است؟

<p class="takeaway">مشارکت اصلی فقط «یک روش جدید» نیست؛ طراحی روش با تحلیل جفتی، محک هم‌پوشانی، ممیزی مؤلفه‌ای و حسابداری هزینه همراه شده است.</p>

<!--
یادداشت ارائه: این اسلاید نقشه‌ی راه دفاع است. پاسخ‌ها به‌ترتیب: بله؛ نزدیک به CLPU؛ شواهد مثبت اما مشروط؛ رتبه‌بندی حساسیت؛ و برای مسیر آداپتر، خیر.
-->

---

<!-- _class: compact -->

# EPALL: محافظت انتخابی از ناحیه‌ی حساس

<img class="figure" src="../images/EPALL_mechanism_compact.svg" style="width:94%; max-height:365px">

<div class="flow">
  <div>محاسبه‌ی اشتراک<br><span class="ltr">M<sub>f</sub> ∩ M<sub>r</sub></span></div>
  <div class="arrow">←</div>
  <div>رتبه‌بندی با گرادیان<br>وظایف نگه‌داری‌شده</div>
  <div class="arrow">←</div>
  <div>انتخاب بالای بودجه<br><span class="ltr">Top-ρ</span></div>
  <div class="arrow">←</div>
  <div>ترمیم با لنگر<br><span class="ltr">ℓ₂</span></div>
</div>

<!--
یادداشت ارائه: منطق بازنشانی شاخه‌محور PALL حفظ می‌شود. نوآوری EPALL در تعیین مجموعه‌ی حساس و مهار همان مجموعه هنگام ترمیم است.
-->

---

# صورت‌بندی: بودجه‌ی حفاظت را روی حساس‌ترین مختصه‌ها خرج می‌کنیم

<div class="cols top">
  <div class="col">
    <h2>۱. ناحیه‌ی مشترک</h2>
    <p class="takeaway"><span class="ltr">S<sub>share</sub> = M<sub>f</sub> ∩ M<sub>r</sub></span></p>
    <h2>۲. حساسیت نگه‌داری</h2>
    <p class="takeaway"><span class="ltr">I<sub>i</sub> = |∂L<sub>retain</sub> / ∂θ<sub>i</sub>|</span></p>
  </div>
  <div class="col">
    <h2>۳. انتخاب دقیق بودجه</h2>
    <p>فقط بالاترین سهم <span class="ltr">ρ</span> از مختصه‌های مشترک محافظت می‌شود.</p>
    <h2>۴. ترمیم مهارشده</h2>
    <p>لنگر <span class="ltr">ℓ₂</span> جابه‌جایی مختصه‌های حساس را محدود می‌کند؛ بقیه‌ی مسیر آزادتر ترمیم می‌شود.</p>
  </div>
</div>

<p class="note">ایده‌ی کلیدی، بزرگ‌کردن بودجه نیست؛ <strong>بهبود کیفیت انتخاب با بودجه‌ی ثابت</strong> است.</p>

<!--
یادداشت ارائه: در صورت پرسش درباره‌ی لنگر توضیح دهید که لنگر یک اهرم نگه‌داری است، نه اهرم حذف. بازنشانی ناحیه‌ی انحصاری هدف عامل اصلی سرکوب است.
-->

---

# تحلیل نظری، آگاهانه دامنه‌ی محدودی دارد

<div class="cols top">
  <div class="col">
    <h2>آنچه نشان داده می‌شود</h2>
    <ul>
      <li>یک کران بالای مرتبه‌اول برای افت در زیرفاز مشترک</li>
      <li>کران با افزایش شدت حفاظت <span class="ltr">p</span> ناافزاینده است</li>
      <li>حساسیت‌ها در ناحیه‌ی هم‌پوشان یکنواخت نیستند</li>
    </ul>
  </div>
  <div class="col">
    <h2>آنچه نتیجه نمی‌شود</h2>
    <ul>
      <li>مقایسه‌ی قطعی افت واقعی دو روش</li>
      <li>تضمین انتها‌به‌انتها برای کل درخواست حذف</li>
      <li>حذف دقیق، گواهی‌شده یا تضمین حریم خصوصی</li>
    </ul>
  </div>
</div>

<p class="takeaway">نظریه یک راهنمای طراحی است؛ اعتبار ادعای روش از آزمایش‌های هم‌تراز و کنترل‌های مؤلفه‌ای می‌آید.</p>

<!--
یادداشت ارائه: این مرزبندی مهم است. از مقایسه‌ی دو کران بالا نمی‌توان رابطه‌ای میان دو افت واقعی نتیجه گرفت و کران به نقطه‌ی پایان کل درخواست تعمیم ندارد.
-->

---

<!-- _class: compact -->

# ارزیابی، سه نوع شاهد مکمل دارد

<div class="metrics">
  <div class="metric"><b>11</b><span>روش مقایسه‌شده</span></div>
  <div class="metric"><b>8</b><span>بذر هم‌تراز در Split-CIFAR</span></div>
  <div class="metric"><b>300</b><span>اجرای محک موقعیت درخواست</span></div>
  <div class="metric"><b>3</b><span>مجموعه‌داده / محک</span></div>
</div>

<div class="cols top" style="margin-top:25px">
  <div class="col">
    <h2>محک‌ها</h2>
    <ul>
      <li><span class="en">Split-CIFAR-10</span></li>
      <li><span class="en">Split-CIFAR-100</span></li>
      <li><span class="en">TinyImageNet</span>، ۲۰ وظیفه</li>
    </ul>
  </div>
  <div class="col">
    <h2>معیارها</h2>
    <ul>
      <li>دقت نهایی وظایف فعال</li>
      <li>بیشترین افت و فراموشی متوسط</li>
      <li>دقت وظیفه‌ی هدف، زمان و حالت مقیم</li>
    </ul>
  </div>
</div>

<p class="note">آزمون‌ها جفتی‌اند و شش مقایسه‌ی اصلی با اصلاح <span class="en">Holm</span> کنترل شده‌اند.</p>

<!--
یادداشت ارائه: رژیم اصلی استاندارد Split-CIFAR با 20 ایپاک به‌ازای هر وظیفه است. دنباله‌ی درخواست درون هر بذر میان روش‌ها مشترک نگه داشته شده است.
-->

---

<!-- _class: compact -->

# هدف حذف سرکوب می‌شود؛ معیار نگه‌داری باید هم‌زمان خوانده شود

<img class="figure" src="../images/main_metrics_au_1x3.png" style="width:91%; max-height:335px">

<div class="metrics">
  <div class="metric"><b>0.4863</b><span>EPALL روی CIFAR-10 · هدف 0.5</span></div>
  <div class="metric"><b>0.0959</b><span>EPALL روی CIFAR-100 · هدف 0.1</span></div>
  <div class="metric"><b>0.8122 / 0.9469</b><span>EWC / LwF روی CIFAR-10؛ حذف ناموفق</span></div>
</div>

<p class="caption">دقت وظیفه‌ی حذف‌شده؛ خط‌چین سطح تصادفی هر محک است.</p>

<!--
یادداشت ارائه: افت صفر به‌تنهایی کافی نیست. EWC و LwF ظاهرا از وظایف دیگر محافظت می‌کنند، اما وظیفه‌ی هدف را حذف نکرده‌اند؛ بنابراین معیارهای حذف و نگه‌داری باید هم‌زمان دیده شوند.
-->

---

<!-- _class: compact -->

# EPALL دقت نگه‌داری را به مرجع جداسازی نزدیک می‌کند

<img class="figure" src="../images/main_metrics_final_accuracy_1x3.png" style="width:91%; max-height:335px">

<div class="metrics">
  <div class="metric"><b>0.9346 → 0.9433</b><span>PALL → EPALL · CIFAR-10</span></div>
  <div class="metric"><b>0.7223 → 0.7371</b><span>PALL → EPALL · CIFAR-100</span></div>
  <div class="metric"><b>0.9500 / 0.7355</b><span>مرجع CLPU · CIFAR-10 / 100</span></div>
</div>

<p class="note">نزدیکی به <span class="en">CLPU</span> به معنی هم‌ارزی آماری اثبات‌شده نیست؛ آزمون هم‌ارزی اجرا نشده است.</p>

<!--
یادداشت ارائه: روی CIFAR-10 فاصله‌ی EPALL تا CLPU کمتر از یک واحد درصد است. روی CIFAR-100 اعداد بسیار نزدیک‌اند. تعبیر درست «نزدیک‌شدن» است، نه برتری یا هم‌ارزی اثبات‌شده.
-->

---

# حفاظت حساس، بدترین آسیب جانبی را کاهش می‌دهد

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/main_metrics_worstdrop_1x3.png" style="width:100%">
  </div>
  <div class="col narrow">
    <div class="metric"><b>0.0222 → 0.0056</b><span>CIFAR-100 · کاهش 75٪</span></div>
    <div class="metric" style="margin-top:14px"><b>0.0119 → 0.0027</b><span>CIFAR-10 · کاهش 77٪</span></div>
    <p class="note">پس از اصلاح چندگانگی، نتیجه‌ی بیشترین افت فقط روی <span class="en">CIFAR-100</span> معنادار می‌ماند.</p>
  </div>
</div>

<!--
یادداشت ارائه: کاهش نسبی روی هر دو مجموعه‌داده بزرگ است، اما باید تفاوت قدرت آماری را صریح گفت. بیشترین افت CIFAR-10 پس از Holm از آستانه‌ی 0.05 عبور نمی‌کند.
-->

---

<!-- _class: compact -->

# تحلیل جفتی: چهار نتیجه از شش آزمون باقی می‌ماند

<table>
  <tr><th>مجموعه‌داده</th><th>معیار</th><th>بهبود جفتی</th><th><span class="ltr">p<sub>Holm</sub></span></th><th>نتیجه</th></tr>
  <tr class="highlight"><td><span class="en">CIFAR-100</span></td><td>دقت نهایی</td><td class="ltr">+0.0147</td><td class="ltr">≤ 0.0469</td><td>معنادار</td></tr>
  <tr class="highlight"><td><span class="en">CIFAR-100</span></td><td>فراموشی متوسط</td><td class="ltr">+0.0110</td><td class="ltr">≤ 0.0469</td><td>معنادار</td></tr>
  <tr class="highlight"><td><span class="en">CIFAR-100</span></td><td>بیشترین افت</td><td class="ltr">+0.0166</td><td class="ltr">≤ 0.0469</td><td>معنادار</td></tr>
  <tr class="highlight"><td><span class="en">CIFAR-10</span></td><td>دقت نهایی</td><td class="ltr">+0.0087</td><td class="ltr">0.0469</td><td>معنادار</td></tr>
  <tr><td><span class="en">CIFAR-10</span></td><td>بیشترین افت</td><td class="ltr">+0.0092</td><td class="ltr">0.0625</td><td>نامعنادار</td></tr>
  <tr><td><span class="en">CIFAR-10</span></td><td>فراموشی متوسط</td><td class="ltr">−0.0017</td><td class="ltr">—</td><td>مختلط</td></tr>
</table>

<p class="takeaway">قوی‌ترین شاهد، بهبود پایدار روی <span class="en">CIFAR-100</span> و دقت نهایی هر دو مجموعه‌داده است.</p>

<!--
یادداشت ارائه: آزمون دقیق یک‌طرفه‌ی Wilcoxon برای دقت نهایی هر دو مجموعه‌داده p خام 0.0078 دارد. برای CIFAR-100 هر سه معیار بعد از Holm باقی می‌مانند.
-->

---

# با عقب‌افتادن درخواست، PALL آسیب‌پذیرتر می‌شود

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/aaai_overlap_response.png" style="width:93%">
  </div>
  <div class="col narrow">
    <p><strong>۳۰۰ اجرای هم‌تراز</strong></p>
    <p><span class="en">PALL</span> روی <span class="en">CIFAR-100</span>:<br><span class="ltr">0.0124 → 0.0456</span></p>
    <p><span class="en">EPALL</span>:<br><span class="ltr">0.0060 … 0.0164</span></p>
    <p class="note">محور، موقعیت درخواست است؛ فقط یک شاخص جانشین برای هم‌پوشانی مورد انتظار، نه کنترل علّی آن.</p>
  </div>
</div>

<!--
یادداشت ارائه: درجه‌ها تعداد وظایف آموزش‌دیده پس از وظیفه‌ی هدف را کد می‌کنند. با تغییر درجه، هویت و سن وظیفه نیز تغییر می‌کند؛ پس این روند توصیفی است.
-->

---

<!-- _class: compact -->

# جاروب مستقیم‌تر: سود حفاظت با هم‌پوشانی رشد می‌کند

<div class="cols">
  <div class="col">
    <img class="figure" src="../images/overlap_response_cifar10.svg" style="width:100%">
    <p class="caption"><span class="en">CIFAR-10</span> · <span class="ltr">ρ<sub>s</sub>=+1.00, p=0.0083</span></p>
  </div>
  <div class="col">
    <img class="figure" src="../images/overlap_response_cifar100.svg" style="width:100%">
    <p class="caption"><span class="en">CIFAR-100</span> · <span class="ltr">ρ<sub>s</sub>=+0.90, p=0.0417</span></p>
  </div>
</div>

<p class="takeaway">با ثابت‌ماندن هویت و موقعیت هدف، هم‌پوشانی بیش از ۱۰ برابر جابه‌جا شد و سود حفاظت رابطه‌ی رتبه‌ای مثبت نشان داد.</p>
<p class="note">اما پراکندگی، ظرفیت را هم تغییر می‌دهد؛ بنابراین اثر علّی خالص هم‌پوشانی هنوز جدا نشده است.</p>

<!--
یادداشت ارائه: پنج سطح پراکندگی، دو روش و پنج بذر در مجموع 100 اجرا می‌سازند. آزمون دقیق بر اساس همه‌ی 120 جایگشت ممکن پنج سطح انجام شده است.
-->

---

# ممیزی مؤلفه‌ای: تنها کیفیت انتخاب تایید می‌شود

<div class="cols top">
  <div class="col wide">
    <table>
      <tr><th>کنترل</th><th>مشاهده</th><th>تفسیر امن</th></tr>
      <tr class="highlight"><td>بودجه‌ی تصادفی</td><td>در هر دو داده بدتر</td><td>رتبه‌بندی حساسیت مؤثر است</td></tr>
      <tr><td>بدون جریمه‌ی <span class="ltr">ℓ₂</span></td><td>جدایی پایدار ندارد</td><td>سهم مستقل لنگر تایید نشد</td></tr>
      <tr><td>رتبه‌بندی بدون اشتراک</td><td>دقیقا برابر بازوی کامل</td><td>فیلتر بالادستی اضافی است</td></tr>
      <tr><td>فقط هم‌پوشانی</td><td>افت کمتر، دقت گاه پایین‌تر</td><td>یک مصالحه؛ نه غالب مطلق</td></tr>
    </table>
  </div>
  <div class="col narrow">
    <div class="metric"><b>+0.0076</b><span>زیان بیشترین افتِ بودجه‌ی تصادفی روی CIFAR-10</span></div>
    <div class="metric" style="margin-top:14px"><b>+0.0148</b><span>زیان بیشترین افتِ بودجه‌ی تصادفی روی CIFAR-100</span></div>
  </div>
</div>

<p class="takeaway">ادعای مکانیزمی نهایی باریک‌تر از توصیف روش است: <strong>رتبه‌بندی حساسیت با بودجه‌ی ثابت</strong>.</p>

<!--
یادداشت ارائه: کنترل‌ها نشان می‌دهند روش کار می‌کند، اما همه‌ی اجزای طراحی سهم مستقل قابل‌اندازه‌گیری ندارند. این محدودکردن ادعا نقطه‌ی قوت ممیزی است.
-->

---

<!-- _class: compact -->

# نزدیک‌شدن به جداسازی کامل، حافظه‌ی کمتری می‌خواهد

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/aaai_storage.svg" style="width:94%; max-height:390px">
  </div>
  <div class="col narrow">
    <div class="metric"><b>135.85 MiB</b><span>EPALL · CIFAR-10</span></div>
    <div class="metric" style="margin-top:12px"><b>212.94 MiB</b><span>CLPU · CIFAR-10</span></div>
    <div class="metric" style="margin-top:12px"><b>336.35 vs 730.78</b><span>MiB · CIFAR-100</span></div>
  </div>
</div>

<p class="note"><span class="en">EPALL</span> در زمان درخواست کندتر از <span class="en">PALL</span> است؛ مزیت آن در حالت مقیم و رشد کندتر با تعداد وظایف به‌دست می‌آید.</p>

<!--
یادداشت ارائه: حسابداری شامل پارامترهای مقیم، ماسک‌ها، پشتیبان‌های پراکنده و بازپخش است؛ حالت بهینه‌ساز و حافظه‌ی اوج را شامل نمی‌شود.
-->

---

# مسیر PALL-Adapter یک نتیجه‌ی منفیِ مفید است

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/pall_adapter_architecture.png" style="width:100%">
  </div>
  <div class="col narrow">
    <ul>
      <li>بدنه‌ی منجمد + آداپتر مشترک + آداپتر وظیفه</li>
      <li>بازوهای کامل و یکنواخت عملا یکسان‌اند</li>
      <li>ماسک پیوسته واقعا تغییر می‌کند، اما اثر انتها‌به‌انتها ندارد</li>
      <li>سرکوب مشاهده‌شده از <strong>بازنشانی مسیر هدف</strong> می‌آید</li>
    </ul>
  </div>
</div>

<p class="takeaway">برای نسخه‌ی بعدی، ماسک مبتنی بر <strong>رتبه</strong> از ماسک مبتنی بر مقدار امیدبخش‌تر است.</p>

<!--
یادداشت ارائه: این مسیر مشارکت دوم ادعاشده نیست. تحلیل نشان می‌دهد انرژی تعارض در دنباله‌ای نازک متمرکز است و ضریب خطی ماسک، مختصه‌ی نوعی را به‌قدر کافی تضعیف نمی‌کند.
-->

---

# سرکوب رفتاری با حذف اثر یکی نیست

<div class="cols">
  <div class="col wide">
    <img class="figure" src="../images/mia_before_after_by_dataset_regime.svg" style="width:100%">
  </div>
  <div class="col narrow small">
    <p><strong>AUC نزدیک ۰٫۵</strong><br>یک نتیجه‌ی تهیِ کم‌توان؛ نه شاهد حذف</p>
    <p><strong>آماره‌ی تفکیک‌پذیری S</strong><br>برای EPALL از صفر به میانگین حدود <span class="ltr">0.10</span> و بیشینه‌ی <span class="ltr">0.55</span> می‌رسد</p>
    <p class="accent">فراموشی، نمونه‌های هدف را در این حمله تفکیک‌پذیرتر کرده است.</p>
  </div>
</div>

<p class="note">دامنه: حذف در سطح وظیفه و سناریوی <span class="en">Task-Incremental</span>؛ بدون ادعای حذف رکوردی، ذخیره‌سازی، دقیق یا گواهی‌شده.</p>

<!--
یادداشت ارائه: S پارامتر حریم خصوصی تفاضلی نیست. حمله‌های مدل‌سایه و ممیزی صوری انجام نشده‌اند. دقت تصادفی هدف می‌تواند هم‌زمان با الگوی خطای متفاوت و قابل‌تفکیک رخ دهد.
-->

---

<!-- _class: final -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# جمع‌بندی: شناخت ساختار، آسیب فراموشی را مهار می‌کند

<div class="takeaway">
<strong>EPALL</strong> با محافظت از مختصه‌های مشترکِ حساس، وظیفه‌ی هدف را سرکوب و دقت وظایف فعال را به مرجع جداسازی نزدیک می‌کند—با حالت مقیم کمتر.
</div>

<ul>
  <li>شاهد اصلی: بهبود جفتی دقت و کاهش آسیب، به‌ویژه روی <span class="en">CIFAR-100</span></li>
  <li>جزء تاییدشده: رتبه‌بندی حساسیت با بودجه‌ی ثابت</li>
  <li>قید اصلی: شواهد هم‌پوشانی مثبت اما غیرعلّی؛ حذف دقیق و تضمین حریم خصوصی اثبات نشده است</li>
  <li>گام بعد: دستکاری ظرفیت‌همتراز هم‌پوشانی، ممیزی رسمی‌تر، و ماسک رتبه‌محور</li>
</ul>

## سپاسگزارم — پرسش‌ها؟

<!--
یادداشت ارائه: پاسخ نهایی به دفاع این است که روش اصلی از هم‌پوشانی به‌عنوان ساختار قابل‌اندازه‌گیری استفاده می‌کند، اما ممیزی‌ها اجازه نمی‌دهند ادعا از شواهد فراتر برود.
-->
