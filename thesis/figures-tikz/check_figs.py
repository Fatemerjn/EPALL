#!/usr/bin/env python3
"""بازرس شکل‌های TikZ: سرریز متن از کادر، و همترازی پنل‌ها.

اجرا:  cd thesis && python3 figures-tikz/check_figs.py
"""
import sys, fitz

# مثبت‌های کاذبِ راستی‌آزمایی‌شده: متریک قلم کادر متن را بزرگ‌تر از واقع می‌دهد.
# هر دو با بزرگ‌نمایی ۳۶۰dpi بررسی و تأیید شدند که کاملاً داخل کادرشان‌اند.
KNOWN_OK = {
    ("parameter_overlap_concept", "\u2282"),
    ("pall_adapter_architecture", "Re\u216cU"),
}

FIGS = ["parameter_overlap_concept", "softmask_drop_decomposition",
        "selective_forgetting_pipeline", "pall_adapter_architecture",
        "EPALL_mechanism_compact"]

def boxes(page):
    """مستطیل‌های رسم‌شده‌ای که می‌توانند «کادر» باشند."""
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 18 or r.height < 10:
            continue
        # فقط مسیرهایی که واقعاً مستطیل‌اند (نه خط و منحنی)
        kinds = {it[0] for it in d["items"]}
        if not kinds <= {"re", "l", "c"}:
            continue
        # فقط کادرهای «دارای خط دور» ظرف حساب می‌شوند؛
        # پرکردنی‌های بدون کادر، پس‌زمینه‌ی تزیینی‌اند نه ظرف.
        if d.get("color") is None:
            continue
        out.append(r)
    return out

def overflow(page, tol=0.6):
    """متنی که از کوچک‌ترین کادرِ دربرگیرنده‌اش بیرون زده."""
    bs = sorted(boxes(page), key=lambda r: r.width * r.height)
    bad = []
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln["spans"]:
                t = sp["text"].strip()
                if not t:
                    continue
                tb = fitz.Rect(sp["bbox"])
                c = fitz.Point((tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2)
                # کادر میزبان: کوچک‌ترین کادری که مرکز متن را دربر دارد و
                # دست‌کم دو برابر خودِ متن است (تا با پرکردنی‌های تزیینی اشتباه نشود)
                need = (tb.width * tb.height) * 2
                host = next((r for r in bs
                             if r.contains(c) and r.width * r.height >= need), None)
                if host is None:
                    continue
                d = max(host.x0 - tb.x0, tb.x1 - host.x1,
                        host.y0 - tb.y0, tb.y1 - host.y1)
                if d > tol:
                    bad.append((t[:30], round(d, 1)))
    return bad

def panels(page, min_h=100):
    """کادرهای بزرگ = پنل‌ها. برای بررسی همترازی."""
    ps = [r for r in boxes(page) if r.height >= min_h and r.width >= 60]
    # حذف کادرهای تودرتو
    ps = [r for r in ps if not any(o != r and o.contains(r) for o in ps)]
    return sorted(ps, key=lambda r: -r.x0)

def main():
    total = 0
    for f in FIGS:
        page = fitz.open(f"figures-tikz/{f}.pdf")[0]
        bad = [(t, d) for t, d in overflow(page) if (f, t) not in KNOWN_OK]
        total += len(bad)
        print(f"\n■ {f}")
        print(f"   سرریز: {len(bad)}")
        for t, d in bad[:8]:
            print(f"      {d:>5} pt  {t!r}")
        ps = panels(page)
        if len(ps) > 1:
            tops = {round(r.y0) for r in ps}
            bots = {round(r.y1) for r in ps}
            ws   = [round(r.width) for r in ps]
            print(f"   پنل‌ها: {len(ps)} | عرض‌ها: {ws}")
            print(f"      بالا هم‌تراز: {'بله' if max(tops)-min(tops) <= 1 else f'خیر (اختلاف {max(tops)-min(tops)}pt)'}")
            print(f"      پایین هم‌تراز: {'بله' if max(bots)-min(bots) <= 1 else f'خیر (اختلاف {max(bots)-min(bots)}pt)'}")
    print(f"\n{'='*46}\nمجموع سرریز: {total}")
    return 1 if total else 0

if __name__ == "__main__":
    sys.exit(main())
