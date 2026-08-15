---
marp: true
theme: default
size: 16:9
paginate: true
html: true
math: katex
header: ''
footer: 'مدیریت فراموشی در یادگیری پیوسته · فاطمه رائیجیان · شهریور ۱۴۰۵'
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
    --navy: #1d2c63;
    --navy-2: #314375;
    --salmon: #c75b4d;
    --blue: #dfeaf7;
    --blue-2: #eef4fb;
    --ink: #20283d;
    --muted: #687086;
    --paper: #fffefd;
    --green: #177a6b;
    --red: #a73d45;
  }
  section {
    direction: rtl;
    text-align: right;
    font-family: "Sharif", "Vazirmatn", Tahoma, sans-serif;
    font-size: 23px;
    line-height: 1.48;
    color: var(--ink);
    background: var(--paper);
    padding: 72px 72px 54px;
    justify-content: flex-start;
  }
  header {
    top: 26px;
    right: 120px;
    left: auto;
    color: var(--navy);
    font-size: 16px;
    font-weight: 700;
  }
  footer {
    right: 72px;
    left: auto;
    bottom: 23px;
    font-size: 13px;
    color: #8a90a0;
  }
  section::after {
    left: 72px;
    right: auto;
    bottom: 22px;
    font-family: Arial, sans-serif;
    font-size: 14px;
    color: #858c9d;
  }
  section.s1::before, section.s2::before, section.s3::before,
  section.s4::before, section.s5::before, section.s6::before,
  section.backup::before {
    position: absolute;
    top: 20px;
    right: 55px;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--navy);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: Arial, sans-serif;
    font-size: 19px;
    font-weight: 700;
  }
  section.s1::before { content: "1"; }
  section.s2::before { content: "2"; }
  section.s3::before { content: "3"; }
  section.s4::before { content: "4"; }
  section.s5::before { content: "5"; }
  section.s6::before { content: "6"; }
  section.backup::before { content: "+"; background: var(--salmon); }
  h1 {
    color: var(--navy);
    font-size: 40px;
    line-height: 1.22;
    margin: 0 0 22px;
    font-weight: 700;
  }
  h1::after {
    content: "";
    display: block;
    width: 105px;
    height: 4px;
    background: var(--salmon);
    margin-top: 12px;
  }
  h2 { color: var(--navy-2); font-size: 27px; margin: 4px 0 9px; }
  p { margin: 8px 0; }
  ul, ol { margin: 6px 0; padding-right: 1.2em; }
  li { margin: 6px 0; }
  strong { color: var(--navy); }
  .ltr, .en { direction: ltr; unicode-bidi: isolate; display: inline-block; }
  .en { font-family: Arial, sans-serif; }
  .small { font-size: 19px; line-height: 1.48; }
  .tiny { font-size: 16px; line-height: 1.42; }
  .muted { color: var(--muted); }
  .salmon { color: var(--salmon); font-weight: 700; }
  .green { color: var(--green); font-weight: 700; }
  .red { color: var(--red); font-weight: 700; }
  .cols { display: flex; gap: 34px; align-items: center; }
  .cols.top { align-items: flex-start; }
  .col { flex: 1; min-width: 0; }
  .wide { flex: 1.25; }
  .narrow { flex: .75; }
  .callout {
    background: var(--blue-2);
    border-right: 6px solid var(--navy);
    padding: 14px 20px;
    font-size: 27px;
    line-height: 1.52;
  }
  .warning {
    background: #fff2ed;
    border-right: 6px solid var(--salmon);
    padding: 11px 18px;
    font-size: 20px;
  }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .card {
    background: white;
    border-top: 5px solid var(--navy);
    box-shadow: 0 4px 17px rgba(29, 44, 99, .09);
    padding: 17px 18px;
    min-height: 108px;
  }
  .card.salmon-card { border-top-color: var(--salmon); }
  .card b { display: block; color: var(--navy); font-size: 25px; margin-bottom: 4px; }
  .card span { color: var(--muted); font-size: 18px; }
  .metric-row { display: flex; gap: 16px; margin-top: 16px; }
  .metric {
    flex: 1;
    background: var(--blue-2);
    padding: 13px 14px;
    text-align: center;
  }
  .metric b {
    display: block;
    direction: ltr;
    unicode-bidi: isolate;
    font-family: Arial, sans-serif;
    color: var(--navy);
    font-size: 31px;
  }
  .metric span { display: block; font-size: 16px; color: var(--muted); }
  .steps { display: flex; direction: rtl; align-items: stretch; gap: 10px; margin-top: 18px; }
  .step {
    flex: 1;
    background: var(--blue-2);
    border-bottom: 5px solid var(--navy);
    padding: 15px 12px;
    text-align: center;
    font-size: 19px;
  }
  .step b { display: block; font-size: 22px; }
  .arrow { flex: 0 0 auto; align-self: center; color: var(--salmon); font-size: 31px; }
  .plot { display: block; object-fit: contain; margin: 2px auto; max-width: 100%; }
  .caption { text-align: center; font-size: 15px; color: var(--muted); margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; background: white; font-size: 18px; }
  th { background: var(--navy); color: white; padding: 9px; }
  td { padding: 8px 9px; border-bottom: 1px solid #dfe3ec; text-align: center; }
  tr.good td { background: #edf7f3; font-weight: 700; }
  tr.warn td { background: #fff2ed; }
  .roadmap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 22px; }
  .roadmap > div { background: var(--blue-2); padding: 18px; min-height: 112px; }
  .roadmap b {
    display: inline-flex; width: 36px; height: 36px; border-radius: 50%;
    align-items: center; justify-content: center; background: var(--navy); color: white;
    font-family: Arial, sans-serif; margin-left: 8px;
  }
  .roadmap span { display: block; margin-top: 9px; color: var(--muted); font-size: 18px; }
  section.title {
    background:
      radial-gradient(circle at 12% 18%, rgba(199,91,77,.95) 0 62px, transparent 63px),
      radial-gradient(circle at 89% 84%, rgba(91,130,183,.55) 0 115px, transparent 116px),
      linear-gradient(132deg, #17265a 0%, #243873 68%, #334b82 100%);
    color: white;
    justify-content: center;
    padding: 82px 92px;
  }
  section.title h1 { color: white; font-size: 51px; max-width: 1040px; margin-bottom: 14px; }
  section.title h1::after { background: #ed8b7d; width: 145px; }
  section.title h2 { color: #e7edf8; font-size: 29px; }
  section.title p { color: #f4f6fb; }
  section.title strong { color: #ffb3a6; }
  section.compact { padding-top: 66px; }
  section.compact h1 { font-size: 37px; margin-bottom: 15px; }
  section.compact .metric b { font-size: 28px; }
  section.final {
    background:
      radial-gradient(circle at 10% 18%, rgba(199,91,77,.9) 0 58px, transparent 59px),
      radial-gradient(circle at 90% 82%, rgba(101,141,193,.55) 0 105px, transparent 106px),
      linear-gradient(132deg, #17265a, #2b427b);
    color: white;
    justify-content: center;
  }
  section.final h1, section.final h2, section.final strong { color: white; }
  section.final h1::after { background: #ef8a7b; }
  section.final .callout { color: var(--ink); background: rgba(255,255,255,.95); font-size: 24px; }
  section.final .callout strong { color: var(--navy); }
  section.appendix-title { background: var(--navy); color: white; justify-content: center; }
  section.appendix-title h1 { color: white; font-size: 54px; }
  section.appendix-title h1::after { background: var(--salmon); }
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
شروع پیشنهادی: این پایان‌نامه درباره‌ی حذف یک وظیفه از مدل یادگیری پیوسته است؛ به‌گونه‌ای که اثر رفتاری آن سرکوب شود اما دانش وظایف باقی‌مانده کمترین آسیب را ببیند.
-->

---

# نقشه‌ی راه ارائه

<div class="roadmap">
  <div><b>1</b> مسئله و پرسش پژوهش<span>چرا حذف در مدل مشترک آسیب جانبی دارد؟</span></div>
  <div><b>2</b> پیشینه و شکاف<span>جداسازی کامل یا ویرایش مدل مشترک؟</span></div>
  <div><b>3</b> روش پیشنهادی<span>EPALL و محافظت حساس به هم‌پوشانی</span></div>
  <div><b>4</b> ارزیابی تجربی<span>مقایسه، آمار جفتی، هم‌پوشانی و هزینه</span></div>
  <div><b>5</b> ممیزی سازوکار<span>نتیجه‌ی منفی PALL-Adapter</span></div>
  <div><b>6</b> جمع‌بندی<span>محدودیت‌ها و مسیر ادامه</span></div>
</div>

<!--
نقشه‌ی راه را سریع مرور کنید. تاکید کنید که بخش چهارم هسته‌ی دفاع است و بخش پنجم یک ممیزی صادقانه از مسیر اکتشافی است.
-->

---

<!-- header: مقدمه و بیان مسئله -->
<!-- _class: s1 -->

# مسئله از یک درخواست واقعی آغاز می‌شود

<img class="plot" src="../images/selective_forgetting_pipeline.svg" style="height:325px">

<div class="callout">پس از یادگیری چند وظیفه، باید <strong>یک وظیفه‌ی مشخص</strong> حذف شود؛ بدون بازآموزی از ابتدا و بدون تخریب وظایف فعال.</div>

<p class="warning">واحد حذف در این پایان‌نامه یک وظیفه‌ی کامل در سناریوی <span class="en">Task-Incremental</span> است.</p>

<!--
مرز مسئله را روشن کنید: شناسه‌ی وظیفه در استنتاج در دسترس است و درخواست حذف در سطح وظیفه تعریف می‌شود، نه رکورد یا کلاس.
-->

---

<!-- _class: s1 -->

# موفقیت فقط با دو شرط هم‌زمان تعریف می‌شود

<div class="cols top">
  <div class="col">
    <h2>سرکوب هدف</h2>
    <div class="callout">دقت وظیفه‌ی حذف‌شده باید به سطح تصادفی نزدیک شود.</div>
  </div>
  <div class="col">
    <h2>نگه‌داری دانش</h2>
    <div class="callout">دقت وظایف فعال نباید پس از درخواست حذف افت محسوسی کند.</div>
  </div>
</div>

<div class="metric-row">
  <div class="metric"><b>A<sub>u</sub> ↓</b><span>دقت وظیفه‌ی حذف‌شده</span></div>
  <div class="metric"><b>A<sub>final</sub> ↑</b><span>دقت نهایی وظایف فعال</span></div>
  <div class="metric"><b>WorstDrop ↓</b><span>بدترین آسیب جانبی</span></div>
</div>

<p class="warning">یک روش می‌تواند افت نگه‌داری کمی داشته باشد، فقط چون اصلاً وظیفه‌ی هدف را فراموش نکرده است.</p>

<!--
این نکته را برای تفسیر نتایج نگه دارید: معیار حذف و معیار نگه‌داری باید کنار هم خوانده شوند.
-->

---

<!-- _class: s1 -->

# منشأ آسیب جانبی، اشتراک پارامترهاست

<div class="cols">
  <div class="col wide">
    <img class="plot" src="../images/parameter_overlap_concept.svg" style="height:420px">
  </div>
  <div class="col narrow">
    <h2>ناحیه‌ی انحصاری</h2>
    <p>بازنشانی آن معمولا خطر کمتری برای سایر وظایف دارد.</p>
    <h2>ناحیه‌ی مشترک</h2>
    <p>هم هدف و هم وظایف فعال به آن وابسته‌اند.</p>
    <div class="warning">اما همه‌ی مختصه‌های مشترک به یک اندازه حساس نیستند.</div>
  </div>
</div>

<!--
فرض مرکزی پایان‌نامه را بیان کنید: آسیب نه فقط در ناحیه‌ی مشترک، بلکه روی زیرمجموعه‌ای حساس از آن متمرکز است.
-->

---

<!-- _class: s1 compact -->

# شکاف پژوهش به چهار پرسش تبدیل شد

<div class="cards">
  <div class="card"><b>RQ1</b><span>آیا حفاظت حساس، سرکوب و نگه‌داری را هم‌زمان ممکن می‌کند؟</span></div>
  <div class="card"><b>RQ2</b><span>مدل مشترک تا چه حد به جداسازی کامل نزدیک می‌شود؟</span></div>
  <div class="card"><b>RQ3</b><span>آیا با افزایش هم‌پوشانی، سود حفاظت رشد می‌کند؟</span></div>
  <div class="card salmon-card"><b>RQ4</b><span>کدام مؤلفه واقعا منشأ بهبود مشاهده‌شده است؟</span></div>
  <div class="card salmon-card"><b>هزینه</b><span>دقت بهتر با چه حافظه و زمان درخواستی به‌دست می‌آید؟</span></div>
  <div class="card salmon-card"><b>دامنه‌ی ادعا</b><span>چه چیزی حذف رفتاری است و چه چیزی هنوز اثبات نشده؟</span></div>
</div>

<!--
این پرسش‌ها ترتیب بخش نتایج را نیز تعیین می‌کنند. پاسخ کوتاه RQ4 این است: رتبه‌بندی حساسیت، نه همه‌ی اجزای اولیه‌ی طراحی.
-->

---

<!-- header: پیشینه و شکاف پژوهش -->
<!-- _class: s2 -->

# راه‌حل‌های موجود میان دو سر طیف قرار می‌گیرند

<div class="cols top">
  <div class="col">
    <h2>ویرایش مدل مشترک</h2>
    <ul>
      <li>حالت مقیم کمتر</li>
      <li>اشتراک دانش بیشتر</li>
      <li class="red">خطر آسیب جانبی</li>
    </ul>
  </div>
  <div class="col">
    <h2>جداسازی کامل</h2>
    <ul>
      <li>حذف ساختاری ساده‌تر</li>
      <li>تداخل کمتر میان وظایف</li>
      <li class="red">رشد حافظه با تعداد وظایف</li>
    </ul>
  </div>
</div>

<div class="callout">هدف این پایان‌نامه: نزدیک‌شدن به کیفیت نگه‌داریِ جداسازی کامل، بدون پرداخت کل هزینه‌ی آن.</div>

<!--
CLPU را به‌عنوان مرجع جداسازی کامل معرفی کنید، نه لزوماً رقیبی با همان ظرفیت. روش اصلی در سوی مدل مشترک قرار دارد.
-->

---

<!-- _class: s2 -->

# PALL ساختار را می‌بیند، اما حساسیت را نه

<div class="steps">
  <div class="step"><b>زیرشبکه‌ی وظیفه</b>ماسک دودویی برای هر وظیفه</div>
  <div class="arrow">←</div>
  <div class="step"><b>بازنشانی هدف</b>حذف ناحیه‌ی انحصاری</div>
  <div class="arrow">←</div>
  <div class="step"><b>ترمیم</b>به‌روزرسانی ناحیه‌ی مشترک</div>
</div>

<div class="cols top" style="margin-top:28px">
  <div class="col"><h2>دستاورد PALL</h2><p>مرز وظایف را در یک مدل مشترک صریح می‌کند.</p></div>
  <div class="col"><h2>مسئله‌ی باز</h2><p>در ترمیم، مختصه‌های مشترک را با ارزش یکسان می‌بیند.</p></div>
</div>

<p class="warning">شکاف: بودجه‌ی محافظت باید به مختصه‌هایی برسد که برای وظایف نگه‌داری‌شده حساس‌ترند.</p>

<!--
EPALL جایگزین کامل PALL نیست؛ منطق شاخه‌محور آن را حفظ می‌کند و مرحله‌ی انتخاب مختصه‌ی حساس را به آن اضافه می‌کند.
-->

---

<!-- header: روش پیشنهادی EPALL -->
<!-- _class: s3 -->

# EPALL در یک جمله

<div class="callout">در ناحیه‌ی مشترک، مختصه‌ها را با گرادیان وظایف فعال رتبه‌بندی می‌کنیم و بودجه‌ی حفاظت را فقط روی حساس‌ترین‌ها می‌گذاریم.</div>

<div class="metric-row" style="margin-top:30px">
  <div class="metric"><b>Overlap</b><span>کاندیدهای مشترک را محدود می‌کند</span></div>
  <div class="metric"><b>Rank</b><span>حساسیت نگه‌داری را مرتب می‌کند</span></div>
  <div class="metric"><b>Anchor</b><span>حرکت زیرمجموعه‌ی حساس را مهار می‌کند</span></div>
</div>

<p class="warning">ادعای تجربی نهایی از توصیف روش باریک‌تر است: قوی‌ترین شاهد به سود <strong>رتبه‌بندی حساسیت با بودجه‌ی ثابت</strong> است.</p>

<!--
از همان ابتدا میان «اجزای طراحی» و «اجزای تاییدشده با ممیزی» تمایز بگذارید؛ این زمینه‌ی اسلاید ابلیشن است.
-->

---

<!-- _class: s3 compact -->

# سازوکار روش پیشنهادی

<img class="plot" src="../images/EPALL_mechanism_compact.svg" style="height:315px">

<!--
شکل را از راست به چپ توضیح دهید. ورودی انتخاب، گرادیان بازپخش وظایف فعال است و خروجی، ماسک مختصه‌های حساس است.
-->

---

<!-- _class: s3 -->

# صورت‌بندی: بودجه مهم است، کیفیت انتخاب مهم‌تر

<div class="cols top">
  <div class="col">
    <h2>فضای کاندید</h2>
    <div class="callout"><span class="ltr">S<sub>share</sub> = M<sub>f</sub> ∩ M<sub>r</sub></span></div>
    <h2>امتیاز حساسیت</h2>
    <div class="callout"><span class="ltr">I<sub>i</sub> = |∂L<sub>retain</sub>/∂θ<sub>i</sub>|</span></div>
  </div>
  <div class="col">
    <h2>انتخاب</h2>
    <p>بالاترین سهم <span class="ltr">ρ</span> از مختصه‌های کاندید، با اندازه‌ی دقیق بودجه.</p>
    <h2>هدف ترمیم</h2>
    <p><span class="ltr">L = L<sub>repair</sub> + λ ‖(θ−θ<sup>old</sup>) ⊙ S<sub>crit</sub>‖²</span></p>
  </div>
</div>

<div class="warning">لنگر اهرم نگه‌داری است؛ سرکوب اصلی از بازنشانی ناحیه‌ی انحصاری هدف می‌آید.</div>

<!--
اگر درباره‌ی تساوی بودجه پرسیده شد: کنترل تصادفی دقیقا همان تعداد مختصه را از همان پشتیبان کاندید انتخاب می‌کند.
-->

---

<!-- _class: s3 compact -->

# یک درخواست حذف در چهار فاز اجرا می‌شود

<div class="steps" style="margin-top:45px">
  <div class="step"><b>۱. تشخیص</b>هدف، وظایف فعال و ناحیه‌ی مشترک</div>
  <div class="arrow">←</div>
  <div class="step"><b>۲. انتخاب</b>گرادیان نگه‌داری و <span class="en">Top-ρ</span></div>
  <div class="arrow">←</div>
  <div class="step"><b>۳. حذف</b>بازنشانی شاخه‌ی انحصاری هدف</div>
  <div class="arrow">←</div>
  <div class="step"><b>۴. ترمیم</b>بازپخش محدود با حفاظت انتخابی</div>
</div>

<div class="cols top" style="margin-top:38px">
  <div class="col"><h2>داده‌ی در دسترس</h2><p>حافظه‌ی بازپخش ۵۰۰ نمونه، با سهم ثابت برای هر وظیفه.</p></div>
  <div class="col"><h2>نکته‌ی کنترلی</h2><p>پس از حذف، بودجه‌ی آزادشده میان وظایف دیگر بازتوزیع نمی‌شود.</p></div>
</div>

<!--
ثابت‌بودن سهم بازپخش، یک عامل مخدوش‌کننده را حذف می‌کند: بهبود نباید صرفا از افزایش حافظه‌ی وظایف باقی‌مانده بیاید.
-->

---

<!-- _class: s3 -->

# دامنه‌ی تحلیل نظری عمداً محدود است

<div class="cols top">
  <div class="col">
    <h2 class="green">آنچه نشان می‌دهیم</h2>
    <ul>
      <li>کران بالای مرتبه‌اول برای افت در زیرفاز مشترک</li>
      <li>ناافزایندگی کران با شدت حفاظت</li>
      <li>راهنمای طراحی برای محدودکردن جابه‌جایی حساس</li>
    </ul>
  </div>
  <div class="col">
    <h2 class="red">آنچه نتیجه نمی‌شود</h2>
    <ul>
      <li>رابطه‌ی قطعی میان افت واقعی دو روش</li>
      <li>تضمین انتها‌به‌انتها برای کل درخواست</li>
      <li>حذف دقیق، گواهی‌شده یا تضمین حریم خصوصی</li>
    </ul>
  </div>
</div>

<div class="callout">نظریه راهنمای طراحی است؛ اعتبار ادعای روش از آزمایش‌های جفتی و کنترل‌های مؤلفه‌ای می‌آید.</div>

<!--
مقایسه‌ی دو کران بالا، ترتیب افت واقعی را ثابت نمی‌کند. این محدودیت را صریح نگه دارید تا ادعا از شواهد جلو نزند.
-->

---

<!-- _class: s3 -->

# پیش‌بینی آزمون‌پذیر روش

<div class="callout">اگر آسیب از مختصه‌های مشترکِ حساس بیاید، با افزایش هم‌پوشانی باید سود حفاظت انتخابی نیز بیشتر شود.</div>

<div class="cols top" style="margin-top:28px">
  <div class="col">
    <h2>پیش‌بینی ۱</h2>
    <p>EPALL باید از PALL دقت نهایی بیشتر و بیشترین افت کمتر داشته باشد.</p>
  </div>
  <div class="col">
    <h2>پیش‌بینی ۲</h2>
    <p>در بودجه‌ی برابر، انتخاب حساس باید از انتخاب تصادفی بهتر باشد.</p>
  </div>
  <div class="col">
    <h2>پیش‌بینی ۳</h2>
    <p>سود حفاظت باید با هم‌پوشانی اندازه‌گیری‌شده رابطه‌ی مثبت نشان دهد.</p>
  </div>
</div>

<p class="warning">این سه پیش‌بینی، طراحی بخش ارزیابی را تعیین می‌کنند.</p>

<!--
این اسلاید پل میان روش و آزمایش است. بعد از آن وارد نتایج شوید و هر پیش‌بینی را جدا پاسخ دهید.
-->

---

<!-- header: ارزیابی تجربی -->
<!-- _class: s4 compact -->

# پروتکل ارزیابی برای مقایسه‌ی هم‌تراز طراحی شد

<div class="metric-row">
  <div class="metric"><b>11</b><span>روش</span></div>
  <div class="metric"><b>8</b><span>بذر اصلی</span></div>
  <div class="metric"><b>300</b><span>اجرای محک موقعیت</span></div>
  <div class="metric"><b>100</b><span>اجرای جاروب هم‌پوشانی</span></div>
</div>

<div class="cols top" style="margin-top:24px">
  <div class="col">
    <h2>محک‌ها</h2>
    <p><span class="en">Split-CIFAR-10</span> · <span class="en">Split-CIFAR-100</span> · <span class="en">TinyImageNet</span></p>
  </div>
  <div class="col">
    <h2>مرجع‌ها</h2>
    <p>روش‌های یادگیری پیوسته، یادگیری‌زدایی، جداسازی و PEFT</p>
  </div>
</div>

<div class="warning">دنباله‌ی درخواست درون هر بذر میان روش‌ها مشترک است؛ شش آزمون اصلی با اصلاح <span class="en">Holm</span> کنترل می‌شوند.</div>

<!--
رژیم اصلی Split-CIFAR هشت بذر دارد. مطالعات دو بذری فقط توصیفی‌اند و در محدودیت‌ها ذکر می‌شوند.
-->

---

<!-- _class: s4 -->

# چهار معیار، چهار سؤال متفاوت را پاسخ می‌دهند

<div class="cards">
  <div class="card"><b><span class="ltr">A<sub>u</sub></span></b><span>آیا وظیفه‌ی هدف رفتاری سرکوب شده است؟</span></div>
  <div class="card"><b><span class="ltr">A<sub>final</sub></span></b><span>پس از درخواست، چه مقدار دانش فعال باقی است؟</span></div>
  <div class="card"><b><span class="en">WorstDrop</span></b><span>آسیب‌پذیرترین وظیفه چه‌قدر افت کرده است؟</span></div>
  <div class="card salmon-card"><b><span class="ltr">F<sub>avg</sub></span></b><span>فراموشی متوسط وظایف نگه‌داری‌شده چقدر است؟</span></div>
  <div class="card salmon-card"><b>حالت مقیم</b><span>مدل، ماسک، پشتیبان و بازپخش چقدر حافظه می‌خواهند؟</span></div>
  <div class="card salmon-card"><b>حریم خصوصی</b><span>آیا رفتار پس از حذف هنوز عضویت را تفکیک‌پذیر می‌کند؟</span></div>
</div>

<!--
تاکید کنید هیچ معیار منفردی به‌تنهایی کافی نیست؛ مخصوصاً دقت تصادفی هدف به معنای حذف اثر نیست.
-->

---

<!-- _class: s4 compact -->

# نتیجه ۱ — وظیفه‌ی هدف تا سطح تصادفی سرکوب می‌شود

<img class="plot" src="../images/main_metrics_au_1x3.png" style="height:345px">

<div class="metric-row">
  <div class="metric"><b>0.4863</b><span>EPALL · CIFAR-10 · شانس 0.5</span></div>
  <div class="metric"><b>0.0959</b><span>EPALL · CIFAR-100 · شانس 0.1</span></div>
  <div class="metric"><b>0.8122 / 0.9469</b><span>EWC / LwF روی CIFAR-10</span></div>
</div>

<p class="caption">دقت وظیفه‌ی حذف‌شده؛ خط‌چین سطح تصادفی است.</p>

<!--
EWC و LwF افت کمی روی وظایف دیگر دارند اما هدف را فراموش نکرده‌اند؛ بنابراین حذف و نگه‌داری را هم‌زمان تفسیر کنید.
-->

---

<!-- _class: s4 compact -->

# نتیجه ۲ — دقت نگه‌داری به مرجع جداسازی نزدیک می‌شود

<img class="plot" src="../images/main_metrics_final_accuracy_1x3.png" style="height:345px">

<div class="metric-row">
  <div class="metric"><b>0.9346 → 0.9433</b><span>PALL → EPALL · CIFAR-10</span></div>
  <div class="metric"><b>0.7223 → 0.7371</b><span>PALL → EPALL · CIFAR-100</span></div>
  <div class="metric"><b>0.9500 / 0.7355</b><span>CLPU · CIFAR-10 / 100</span></div>
</div>

<p class="warning">«نزدیک» به معنای هم‌ارزی آماری اثبات‌شده نیست؛ آزمون هم‌ارزی اجرا نشده است.</p>

<!--
روی CIFAR-100 میانگین EPALL اندکی بالاتر از CLPU است، اما تعبیر درست نزدیکی در پراکندگی بذرهاست، نه برتری قطعی.
-->

---

<!-- _class: s4 -->

# نتیجه ۳ — بدترین آسیب جانبی به‌طور محسوسی کم می‌شود

<div class="cols">
  <div class="col wide">
    <img class="plot" src="../images/main_metrics_worstdrop_1x3.png" style="height:405px">
  </div>
  <div class="col narrow">
    <div class="metric"><b>0.0222 → 0.0056</b><span>CIFAR-100 · کاهش 75٪</span></div>
    <div class="metric" style="margin-top:15px"><b>0.0119 → 0.0027</b><span>CIFAR-10 · کاهش 77٪</span></div>
    <div class="warning" style="margin-top:16px">بعد از Holm، نتیجه‌ی WorstDrop فقط روی CIFAR-100 معنادار می‌ماند.</div>
  </div>
</div>

<!--
کاهش نسبی هر دو داده بزرگ است، ولی قدرت آماری یکسان نیست. این تمایز را دقیق بیان کنید.
-->

---

<!-- _class: s4 compact -->

# تحلیل جفتی: چهار نتیجه از شش آزمون باقی ماند

<table>
  <tr><th>داده</th><th>معیار</th><th>بهبود جفتی</th><th><span class="ltr">p<sub>Holm</sub></span></th><th>داوری</th></tr>
  <tr class="good"><td>CIFAR-100</td><td>دقت نهایی</td><td class="ltr">+0.0147</td><td class="ltr">≤0.0469</td><td>معنادار</td></tr>
  <tr class="good"><td>CIFAR-100</td><td>فراموشی متوسط</td><td class="ltr">+0.0110</td><td class="ltr">≤0.0469</td><td>معنادار</td></tr>
  <tr class="good"><td>CIFAR-100</td><td>بیشترین افت</td><td class="ltr">+0.0166</td><td class="ltr">≤0.0469</td><td>معنادار</td></tr>
  <tr class="good"><td>CIFAR-10</td><td>دقت نهایی</td><td class="ltr">+0.0087</td><td class="ltr">0.0469</td><td>معنادار</td></tr>
  <tr class="warn"><td>CIFAR-10</td><td>بیشترین افت</td><td class="ltr">+0.0092</td><td class="ltr">0.0625</td><td>نامعنادار</td></tr>
  <tr><td>CIFAR-10</td><td>فراموشی متوسط</td><td class="ltr">−0.0017</td><td>—</td><td>مختلط</td></tr>
</table>

<p class="callout">قوی‌ترین شاهد: بهبود پایدار روی CIFAR-100 و دقت نهایی هر دو مجموعه‌داده.</p>

<!--
برای دقت نهایی هر دو داده، آزمون دقیق یک‌طرفه‌ی Wilcoxon مقدار خام 0.0078 و Holm برابر 0.0469 دارد.
-->

---

<!-- _class: s4 -->

# نتیجه ۴ — با تأخیر درخواست، PALL آسیب‌پذیرتر می‌شود

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/aaai_overlap_response.png" style="height:430px"></div>
  <div class="col narrow">
    <h2>۳۰۰ اجرای هم‌تراز</h2>
    <p><strong>PALL · CIFAR-100</strong><br><span class="ltr">0.0124 → 0.0456</span></p>
    <p><strong>EPALL</strong><br><span class="ltr">0.0060 … 0.0164</span></p>
    <div class="warning">موقعیت درخواست فقط شاخص جانشین هم‌پوشانی است؛ هویت و سن وظیفه نیز تغییر می‌کند.</div>
  </div>
</div>

<!--
این شاهد توصیفی است، نه علّی. برای آزمون مستقیم‌تر به جاروب پراکندگی در دو اسلاید بعد بروید.
-->

---

<!-- _class: s4 compact -->

# جاروب مستقیم‌تر — CIFAR-10

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/overlap_response_cifar10.svg" style="height:435px"></div>
  <div class="col narrow">
    <div class="metric"><b>0.0018 → 0.0349</b><span>سود حفاظت از کمترین تا بیشترین هم‌پوشانی</span></div>
    <div class="metric" style="margin-top:16px"><b>ρ<sub>s</sub>=+1.00</b><span>همبستگی رتبه‌ای · p=0.0083</span></div>
    <p class="warning">هم‌پوشانی اندازه‌گیری‌شده 19.7 برابر جابه‌جا شد.</p>
  </div>
</div>

<!--
هویت و موقعیت هدف ثابت است و پنج سطح پراکندگی ساخته می‌شود. آزمون دقیق با همه‌ی 120 جایگشت ممکن انجام شده است.
-->

---

<!-- _class: s4 compact -->

# جاروب مستقیم‌تر — CIFAR-100

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/overlap_response_cifar100.svg" style="height:435px"></div>
  <div class="col narrow">
    <div class="metric"><b>0.0146 → 0.1002</b><span>سود حفاظت تا سطح پراکندگی 0.6</span></div>
    <div class="metric" style="margin-top:16px"><b>ρ<sub>s</sub>=+0.90</b><span>همبستگی رتبه‌ای · p=0.0417</span></div>
    <p class="warning">روند مثبت اما یکنوا نیست؛ مقدار انتهایی 0.0794 است.</p>
  </div>
</div>

<!--
شاهد به سود پیش‌بینی است، اما پراکندگی ظرفیت را هم تغییر می‌دهد. پس اثر علّی خالص هم‌پوشانی هنوز جدا نشده است.
-->

---

<!-- _class: s4 -->

# ممیزی مؤلفه‌ای، ادعای مکانیزمی را باریک کرد

<table>
  <tr><th>کنترل هم‌بودجه</th><th>مشاهده</th><th>نتیجه‌ی امن</th></tr>
  <tr class="good"><td>انتخاب تصادفی</td><td>WorstDrop به‌اندازه 0.0076 و 0.0148 بدتر</td><td>کیفیت رتبه‌بندی مؤثر است</td></tr>
  <tr><td>بدون لنگر <span class="ltr">ℓ₂</span></td><td>جدایی پایدار ندارد</td><td>سهم مستقل لنگر تایید نشد</td></tr>
  <tr><td>رتبه‌بندی بدون فیلتر اشتراک</td><td>دقیقا برابر بازوی کامل</td><td>فیلتر بالادستی اضافی است</td></tr>
  <tr><td>فقط هم‌پوشانی</td><td>افت کمتر، دقت گاهی پایین‌تر</td><td>یک مصالحه؛ نه غالب مطلق</td></tr>
</table>

<div class="callout">جزء تاییدشده: <strong>رتبه‌بندی حساسیت نگه‌داری با بودجه‌ی ثابت</strong>.</div>

<!--
کنترل‌ها ضرورت همه‌ی اجزای روش را اثبات نمی‌کنند. روش کار می‌کند، اما مکانیزم تاییدشده از توصیف اولیه ساده‌تر است.
-->

---

<!-- _class: s4 compact -->

# هزینه‌ی نزدیک‌شدن به جداسازی کامل

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/aaai_storage.svg" style="height:425px"></div>
  <div class="col narrow">
    <div class="metric"><b>135.85 vs 212.94</b><span>MiB · EPALL / CLPU · CIFAR-10</span></div>
    <div class="metric" style="margin-top:15px"><b>336.35 vs 730.78</b><span>MiB · EPALL / CLPU · CIFAR-100</span></div>
    <p class="warning">CLPU حدود 1.6 تا 2.2 برابر حالت مقیم بیشتری می‌خواهد.</p>
  </div>
</div>

<p class="caption">حسابداری: پارامتر مقیم، ماسک، پشتیبان پراکنده و بازپخش؛ نه حافظه‌ی اوج و حالت بهینه‌ساز.</p>

<!--
مزیت EPALL در حالت مقیم است؛ زمان درخواست آن نسبت به PALL بیشتر است و این نکته در محدودیت‌ها می‌آید.
-->

---

<!-- _class: s4 -->

# آزمون‌های تکمیلی، مرز تعمیم را نشان می‌دهند

<div class="cards">
  <div class="card"><b>بدنه‌ی پیش‌آموخته</b><span>PALL-Adapter به 0.9841، 0.8277 و 0.9047 روی سه محک می‌رسد.</span></div>
  <div class="card"><b>آموزش از صفر</b><span>PEFT روی بدنه‌ی منجمد تصادفی، گلوگاه بازنمایی دارد.</span></div>
  <div class="card"><b>TinyImageNet</b><span>سرکوب هدف برقرار است، اما WorstDrop بالا به محدودیت ظرفیت اشاره می‌کند.</span></div>
  <div class="card salmon-card"><b>عرض گلوگاه</b><span>رابطه یکنوا نیست؛ عرض 16 پیش‌فرض فشرده است.</span></div>
  <div class="card salmon-card"><b>پایداری حذف</b><span>بازگشت هدف اندازه‌گیری شد؛ جداسازی ساختاری بازگشت صفر دارد.</span></div>
  <div class="card salmon-card"><b>دامنه‌ی شاهد</b><span>تحلیل‌های کم‌بذر، توصیفی‌اند و ادعای اصلی بر Split-CIFAR است.</span></div>
</div>

<!--
این اسلاید را کوتاه ارائه کنید. هدف نشان‌دادن این است که کجا روش خوب تعمیم می‌یابد و کجا ظرفیت یا پروتکل محدودکننده است.
-->

---

<!-- header: ممیزی مسیر PALL-Adapter -->
<!-- _class: s5 -->

# مسیر اکتشافی: کاهش دامنه‌ی به‌روزرسانی با آداپتر

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/pall_adapter_architecture.png" style="height:420px"></div>
  <div class="col narrow">
    <ul>
      <li>بدنه‌ی منجمد</li>
      <li>آداپتر مشترک</li>
      <li>آداپتر مخصوص هر وظیفه</li>
      <li>ماسک نرم تعارض روی مسیر مشترک</li>
    </ul>
    <div class="warning">پرسش ممیزی: آیا ماسک نرم واقعا نقطه‌ی پایان را بهتر می‌کند؟</div>
  </div>
</div>

<!--
این مسیر با هدف PEFT طراحی شد، اما در پایان‌نامه به‌عنوان مشارکت دوم معرفی نمی‌شود؛ نتیجه‌ی اصلی آن ممیزی سازوکار است.
-->

---

<!-- _class: s5 -->

# نتیجه‌ی منفی: ماسک نرم اثر انتها‌به‌انتها نشان نداد

<div class="cols top">
  <div class="col">
    <h2>آنچه تغییر نکرد</h2>
    <ul>
      <li>بازوی کامل و یکنواخت عملا یکسان</li>
      <li>حذف یا حفظ به‌روزرسانی مشترک، نقطه‌ی پایان را جدا نکرد</li>
      <li>تغییر شدت ماسک، دقت و WorstDrop را جابه‌جا نکرد</li>
    </ul>
  </div>
  <div class="col">
    <h2>آنچه اثر داشت</h2>
    <ul>
      <li>بازنشانی آداپتر هدف</li>
      <li>بازنشانی برش طبقه‌بند</li>
      <li>کیفیت بدنه‌ی منجمد و پیش‌آموزش</li>
    </ul>
  </div>
</div>

<div class="callout">سرکوب مشاهده‌شده عمدتاً به <strong>بازنشانی مسیر هدف</strong> نسبت داده می‌شود، نه ماسک نرم آگاه از هم‌پوشانی.</div>

<!--
نتیجه‌ی منفی را مستقیم بگویید. این ممیزی جلوی نسبت‌دادن اثر به مؤلفه‌ای را می‌گیرد که داده از آن پشتیبانی نمی‌کند.
-->

---

<!-- _class: s5 -->

# چرا ماسک مقدارمحور ضعیف بود؟

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/softmask_drop_decomposition.svg" style="height:410px"></div>
  <div class="col narrow small">
    <p><strong>۴۰–۴۵٪</strong> مختصه‌ها تعارض غیرصفر دارند.</p>
    <p>اما انرژی تعارض در یک <strong>دنباله‌ی نازک</strong> متمرکز است.</p>
    <p>ضریب خطی پیوسته، مختصه‌ی معمولی را به‌اندازه‌ی کافی تضعیف نمی‌کند.</p>
    <div class="warning">فرضیه‌ی بعدی: ماسک رتبه‌محور یا بودجه‌ی صریح، نه مقیاس‌گذاری صرف مقدار.</div>
  </div>
</div>

<!--
ماسک واقعا با گاما تغییر می‌کند؛ مشکل خرابی پیاده‌سازی نیست. مشکل این است که تغییر ماسک به تغییر نقطه‌ی پایان تبدیل نمی‌شود.
-->

---

<!-- header: جمع‌بندی و مسیر آینده -->
<!-- _class: s6 -->

# سرکوب رفتاری، معادل حذف اثر نیست

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/mia_before_after_by_dataset_regime.svg" style="height:415px"></div>
  <div class="col narrow small">
    <p><strong>AUC نزدیک ۰٫۵</strong><br>نتیجه‌ی تهیِ کم‌توان؛ نه شاهد حذف.</p>
    <p><strong>تفکیک‌پذیری S</strong><br>برای EPALL از صفر به میانگین حدود 0.10 و بیشینه‌ی حدود 0.55 می‌رسد.</p>
    <p class="red">نمونه‌های هدف پس از فراموشی در این حمله تفکیک‌پذیرتر شده‌اند.</p>
  </div>
</div>

<p class="warning">هیچ نتیجه‌ای حذف دقیق، گواهی‌شده، ذخیره‌سازی یا حریم خصوصی رسمی را اثبات نمی‌کند.</p>

<!--
S پارامتر حریم خصوصی تفاضلی نیست. حمله‌ی مدل‌سایه و ممیزی رسمی انجام نشده و دقت تصادفی هدف تضمین حریم خصوصی نمی‌دهد.
-->

---

<!-- _class: s6 compact -->

# محدودیت‌ها، مرز اعتبار نتیجه را مشخص می‌کنند

<div class="cards">
  <div class="card"><b>سناریو</b><span>نیاز به شناسه‌ی وظیفه در استنتاج؛ حذف در سطح وظیفه</span></div>
  <div class="card"><b>نوع حذف</b><span>نه حذف رکوردی، کلاسی، دقیق، گواهی‌شده یا ذخیره‌سازی</span></div>
  <div class="card"><b>علیت هم‌پوشانی</b><span>جاروب پراکندگی، ظرفیت را نیز تغییر می‌دهد</span></div>
  <div class="card salmon-card"><b>مقیاس</b><span>WorstDrop بالا در TinyImageNet بیست‌وظیفه‌ای</span></div>
  <div class="card salmon-card"><b>هزینه</b><span>زمان درخواست EPALL از PALL بیشتر و نرمال‌نشده است</span></div>
  <div class="card salmon-card"><b>توان آماری</b><span>برخی تحلیل‌های انتقال و حساسیت فقط دو بذر دارند</span></div>
</div>

<!--
این محدودیت‌ها ضعف پنهان‌شده نیستند؛ هر کدام مستقیماً یک مسیر پژوهشی آینده را تعریف می‌کنند.
-->

---

<!-- _class: final -->
<!-- _paginate: false -->
<!-- _footer: '' -->
<!-- _header: '' -->

# پاسخ نهایی پایان‌نامه

<div class="callout"><strong>EPALL</strong> با رتبه‌بندی حساسیت در ناحیه‌ی مشترک، وظیفه‌ی هدف را سرکوب و آسیب به وظایف فعال را کاهش می‌دهد؛ با حالتی کوچک‌تر از جداسازی کامل.</div>

<div class="cols top" style="margin-top:25px">
  <div class="col">
    <h2>مشارکت تاییدشده</h2>
    <p>رتبه‌بندی حساسیت با بودجه‌ی ثابت، ارزیابی جفتی، محک هم‌پوشانی و حسابداری هزینه</p>
  </div>
  <div class="col">
    <h2>گام بعد</h2>
    <p>کنترل ظرفیت‌همتراز، ماسک رتبه‌محور، حذف رکورد/کلاس و ممیزی رسمی حریم خصوصی</p>
  </div>
</div>

## سپاسگزارم — پرسش‌ها؟

<!--
جمع‌بندی یک‌جمله‌ای: شناخت اینکه کدام پارامتر مشترک واقعاً حساس است، از حفاظت یکنواخت مؤثرتر است؛ اما حذف رفتاری را نباید با حذف گواهی‌شده یکی دانست.
-->

---

<!-- header: اسلایدهای پشتیبان -->
<!-- _class: appendix-title backup -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# پیوست دفاع

## جزئیات نظری، آماری و بازتولیدپذیری

<!--
این اسلاید شروع بخش پشتیبان است. فقط در پاسخ به پرسش داور وارد این بخش شوید.
-->

---

<!-- _class: backup -->

# پیوست ۱ — صورت دقیق دامنه‌ی کران

<div class="cols top">
  <div class="col">
    <h2>در یک زیرفاز</h2>
    <p class="callout"><span class="ltr">ΔL<sub>retain</sub> ≤ η Σ<sub>i</sub> |g<sub>r,i</sub>| · |m<sub>i</sub> g<sub>f,i</sub>| + O(η²)</span></p>
    <p>با کاهش ضریب به‌روزرسانی روی مختصه‌های حساس، کران بالای مرتبه‌اول کاهش می‌یابد.</p>
  </div>
  <div class="col">
    <h2>اما نه انتها‌به‌انتها</h2>
    <ul>
      <li>بازنشانی، طبقه‌بند و ترمیم خارج از این زیرفازند.</li>
      <li>دو کران کوچک‌تر، ترتیب افت واقعی را تعیین نمی‌کنند.</li>
      <li>لم، ادعای حذف یا حریم خصوصی تولید نمی‌کند.</li>
    </ul>
  </div>
</div>

<!--
اگر داور درباره‌ی قضیه پرسید، تاکید کنید نتیجه یک کران موضعی است و به کل الگوریتم تعمیم داده نشده است.
-->

---

<!-- _class: backup compact -->

# پیوست ۲ — جدول کامل آزمون‌های جفتی اصلی

<table>
  <tr><th>داده</th><th>معیار</th><th>میانه‌ی اختلاف</th><th>آزمون</th><th><span class="ltr">p<sub>raw</sub></span></th><th><span class="ltr">p<sub>Holm</sub></span></th></tr>
  <tr class="good"><td>CIFAR-10</td><td>دقت نهایی</td><td class="ltr">+0.0087</td><td>Wilcoxon دقیق، یک‌طرفه</td><td class="ltr">0.0078</td><td class="ltr">0.0469</td></tr>
  <tr><td>CIFAR-10</td><td>فراموشی متوسط</td><td class="ltr">−0.0017</td><td>جفتی</td><td>—</td><td>—</td></tr>
  <tr class="warn"><td>CIFAR-10</td><td>WorstDrop</td><td class="ltr">+0.0092</td><td>Wilcoxon دقیق</td><td>—</td><td class="ltr">0.0625</td></tr>
  <tr class="good"><td>CIFAR-100</td><td>دقت نهایی</td><td class="ltr">+0.0147</td><td>Wilcoxon دقیق، یک‌طرفه</td><td class="ltr">0.0078</td><td class="ltr">0.0469</td></tr>
  <tr class="good"><td>CIFAR-100</td><td>فراموشی متوسط</td><td class="ltr">+0.0110</td><td>جفتی</td><td>—</td><td class="ltr">≤0.0469</td></tr>
  <tr class="good"><td>CIFAR-100</td><td>WorstDrop</td><td class="ltr">+0.0166</td><td>جفتی</td><td>—</td><td class="ltr">≤0.0469</td></tr>
</table>

<p class="warning">تعداد بذر ۸ است؛ آزمون دقیق و گزارش اصلاح چندگانگی برای جلوگیری از تفسیر بیش‌ازحد به‌کار رفته است.</p>

<!--
اگر درباره‌ی علامت اختلاف پرسیدند، اختلاف‌ها در جهتی تعریف شده‌اند که مقدار مثبت به سود EPALL باشد.
-->

---

<!-- _class: backup -->

# پیوست ۳ — بازتولیدپذیری پروتکل اصلی

<div class="cards">
  <div class="card"><b>دنباله‌ی مشترک</b><span>درون هر بذر، ترتیب وظایف و درخواست‌ها میان روش‌ها یکسان است.</span></div>
  <div class="card"><b>بازپخش</b><span>بودجه ۵۰۰ نمونه و سهم ثابت؛ بدون بازتوزیع پس از حذف.</span></div>
  <div class="card"><b>آموزش</b><span>۲۰ ایپاک به‌ازای وظیفه در رژیم اصلی Split-CIFAR.</span></div>
  <div class="card salmon-card"><b>آمار</b><span>هشت بذر، تحلیل جفتی و اصلاح Holm برای شش آزمون اصلی.</span></div>
  <div class="card salmon-card"><b>حافظه</b><span>حالت مقیم گزارش شده؛ حافظه‌ی اوج و optimizer state خارج است.</span></div>
  <div class="card salmon-card"><b>انتقال</b><span>بدنه‌ی ImageNet و نرمال‌سازی سازگار برای آزمایش PEFT.</span></div>
</div>

<!--
پیکربندی دقیق همه‌ی روش‌ها و نگاشت گروه‌های آزمایش در پیوست پایان‌نامه ثبت شده است.
-->

---

<!-- _class: backup -->

# پیوست ۴ — تفسیر درست ممیزی حریم خصوصی

<div class="cols top">
  <div class="col">
    <h2>AUC حمله‌ی عضویت</h2>
    <p>نزدیکی به ۰٫۵ می‌تواند از حذف، ضعف حمله یا کمبود توان آماری ناشی شود.</p>
    <div class="warning">پس نتیجه‌ی تهی، شاهد مثبت حذف نیست.</div>
  </div>
  <div class="col">
    <h2>تفکیک‌پذیری S</h2>
    <p>تغییر توزیع امتیازهای عضو و غیرعضو را پس از حذف اندازه می‌گیرد.</p>
    <div class="warning">S پارامتر DP و تضمین حریم خصوصی نیست.</div>
  </div>
</div>

<p class="callout">نتیجه‌ی امن: سرکوب رفتاری برقرار است، اما حذف اثر یا عدم‌عضویت قابل‌گواهی اثبات نشده است.</p>

<!--
مسیر آینده شامل حمله‌های مدل‌سایه، ممیزی قوی‌تر و صورت‌بندی رسمی‌تر در سطح رکورد است.
-->

---

<!-- _class: backup -->

# پیوست ۵ — جزئیات نتیجه‌ی منفی PALL-Adapter

<div class="cols">
  <div class="col wide"><img class="plot" src="../images/representative_pall_adapter_accuracy_heatmap.svg" style="height:430px"></div>
  <div class="col narrow small">
    <p>ماسک پیوسته با شدت تعارض تغییر می‌کند.</p>
    <p>اما بازوی کامل، یکنواخت و بدون صعودِ مشترک نقطه‌های پایانی نزدیک دارند.</p>
    <p>بازنشانی هدف، اثر واضح و تکرارشونده دارد.</p>
    <div class="warning">پس ماسک نرم در این پیاده‌سازی و پروتکل، پشتوانه‌ی تجربی برای ادعای مکانیزمی ندارد.</div>
  </div>
</div>

<!--
نقشه‌ی حرارتی افت ستونی در زمان درخواست‌ها را نشان می‌دهد. آن را شاهد تصویری رفتار بدانید، نه انتساب علّی ماسک.
-->

---

<!-- _class: backup -->

# پیوست ۶ — اگر فقط یک اسلاید از نتایج بماند

<div class="metric-row">
  <div class="metric"><b>0.4863 / 0.0959</b><span>هدف در سطح تصادفی</span></div>
  <div class="metric"><b>0.9433 / 0.7371</b><span>دقت نهایی EPALL</span></div>
  <div class="metric"><b>0.0027 / 0.0056</b><span>WorstDrop</span></div>
</div>

<div class="cols top" style="margin-top:30px">
  <div class="col">
    <h2>شاهد مثبت</h2>
    <p>بهبود جفتی دقت هر دو داده، رابطه‌ی مثبت با هم‌پوشانی، و برتری انتخاب حساس بر بودجه‌ی تصادفی.</p>
  </div>
  <div class="col">
    <h2>قید ضروری</h2>
    <p>نه همه‌ی اجزای روش سهم مستقل نشان دادند، نه حذف دقیق و حریم خصوصی رسمی اثبات شد.</p>
  </div>
</div>

<div class="callout">پیام نهایی: <strong>ساختار هم‌پوشانی مهم است، اما کیفیت رتبه‌بندی حساسیت مهم‌تر است.</strong></div>

<!--
این اسلاید پاسخ سریع برای جمع‌بندی دوباره‌ی نتایج در بخش پرسش‌وپاسخ است.
-->
