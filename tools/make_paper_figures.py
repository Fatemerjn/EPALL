#!/usr/bin/env python3
"""Generate the thesis figures from REAL experiment data.

Outputs (vector PDF) written to ``--outdir`` (use ``thesis/images``):
    - drop_decomposition.pdf : conceptual WorstDrop decomposition (visualizes the
      theorem); ``p`` is matched to the experiments' adapter_shared_protect_strength.
    - worstdrop_chart.pdf    : grouped WorstDrop bar chart per dataset/method.
    - overlap_results.pdf    : per-run critical-overlap ratio vs. final accuracy,
      with a linear regression + 95% CI band and R^2 / Pearson rho / p-value.
    - tradeoff_results.pdf   : updated-parameter ratio vs. WorstDrop scatter.

Data integrity
--------------
By default the real result CSVs are wired in (see DEFAULT_* below). Pass
``--require-real-data`` to FORBID the manuscript fallback constants: if a needed
CSV or column is missing, that figure fails with an explicit error instead of
silently drawing fabricated numbers. After running, each figure reports which CSV
it read and how many points/cells it used.

Usage
-----
    python tools/make_paper_figures.py --outdir thesis/images --require-real-data
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

# Headless-safe backend: never opens a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default real-data sources (relative to repo root). --*-csv flags override these.
DEFAULT_WORSTDROP_CSV = REPO_ROOT / "results/aggregates/server_thesis_table.csv"
DEFAULT_OVERLAP_CSVS = [
    REPO_ROOT / "results/thesis/overlap_vs_damage.csv",
    REPO_ROOT / "results/thesis/cifar10_candidate_overlap_results.csv",
    REPO_ROOT / "results/thesis/cifar100_candidate_overlap_results_e1_seed0.csv",
]


class RealDataError(RuntimeError):
    """Raised under --require-real-data when a figure cannot use real data."""


# --------------------------------------------------------------------------- #
# IEEE-compliant global style                                                  #
# --------------------------------------------------------------------------- #
IEEE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}
COL_WIDTH = 3.5

METHOD_STYLE = {
    "PALL-Original": {"color": "#2ca02c", "marker": "^", "hatch": "//"},
    "PALL-Modified": {"color": "#ff7f0e", "marker": "s", "hatch": ".."},
    "PALL-Adapter":  {"color": "#1f77b4", "marker": "o", "hatch": None},
}
DATASETS = ["CIFAR-10", "CIFAR-100", "TinyImageNet"]
METHODS = ["PALL-Original", "PALL-Modified", "PALL-Adapter"]

METHOD_ALIASES = {
    "pall_original": "PALL-Original", "pall": "PALL-Original",
    "pall_modified": "PALL-Modified", "pall_mod": "PALL-Modified",
    "pall_adapter": "PALL-Adapter", "adapter": "PALL-Adapter",
}
DATASET_ALIASES = {
    "cifar10": "CIFAR-10", "cifar-10": "CIFAR-10",
    "cifar100": "CIFAR-100", "cifar-100": "CIFAR-100",
    "tinyimagenet": "TinyImageNet", "tiny-imagenet": "TinyImageNet",
    "tiny_imagenet": "TinyImageNet",
}

# Candidate column names (case-insensitive; first match wins).
COLS_DATASET = ["dataset"]
COLS_METHOD = ["method", "algorithm", "algo"]
COLS_WORSTDROP = ["worstdrop_mean", "worstdrop"]
COLS_UPDATED = ["updated_param_ratio_mean", "updated_param_ratio", "r_updated"]
COLS_OVERLAP = [
    "overlap_s_share_crit_ratio", "overlap_shared_critical_ratio",
    "critical_overlap_ratio", "s_share_crit_ratio",
]
COLS_ACCURACY = ["final_avg_acc_mean", "final_avg_accuracy", "final_avg_acc", "a_final"]
COLS_PROTECT_STRENGTH = ["adapter_shared_protect_strength"]

# --------------------------------------------------------------------------- #
# Manuscript fallback values (Table I) -- ONLY used when NOT --require-real-data #
# --------------------------------------------------------------------------- #
PAPER_WORSTDROP = {
    "CIFAR-10":     {"PALL-Original": 0.0160, "PALL-Modified": 0.0008, "PALL-Adapter": 0.0050},
    "CIFAR-100":    {"PALL-Original": 0.0710, "PALL-Modified": 0.0140, "PALL-Adapter": 0.0050},
    "TinyImageNet": {"PALL-Original": 0.0940, "PALL-Modified": 0.0600, "PALL-Adapter": 0.0080},
}
PAPER_UPDATED = {
    "CIFAR-10":     {"PALL-Original": 0.0475, "PALL-Modified": 0.0475, "PALL-Adapter": 0.0061},
    "CIFAR-100":    {"PALL-Original": 0.0763, "PALL-Modified": 0.0763, "PALL-Adapter": 0.0123},
    "TinyImageNet": {"PALL-Original": 0.0960, "PALL-Modified": 0.0960, "PALL-Adapter": 0.0292},
}
PAPER_OVERLAP_POINTS = np.array(
    [(0.05, 0.50), (0.10, 0.53), (0.15, 0.55), (0.20, 0.53), (0.30, 0.45)]
)


# --------------------------------------------------------------------------- #
# Persian localization (optional): needs arabic_reshaper + python-bidi + a TTF; #
# otherwise labels stay English, as requested.                                  #
# --------------------------------------------------------------------------- #
def _make_localizer():
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        from matplotlib import font_manager

        font_path = REPO_ROOT / "thesis/styles/fonts/XB Niloofar.ttf"
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        font_manager.fontManager.addfont(str(font_path))
        name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = [name, "serif"]

        def L(en, fa):
            return get_display(arabic_reshaper.reshape(fa))

        return L, "fa"
    except Exception:
        return (lambda en, fa: en), "en"


# --------------------------------------------------------------------------- #
# Dependency-free statistics (no scipy)                                         #
# --------------------------------------------------------------------------- #
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t, df):
    """CDF of Student's t with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)  # = P(T > |t|)
    return 1.0 - tail if t >= 0 else tail


def _t_two_sided_p(t, df):
    """Two-sided p-value P(|T| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2)."""
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _t_ppf(q, df):
    """Inverse Student-t CDF via bisection (for the CI critical value)."""
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _linregress(xs, ys):
    """Linear regression + Pearson statistics, dependency-free.

    Returns dict with slope, intercept, r, r2, p (two-sided), s_e (residual std),
    xbar, Sxx, n. p/r are NaN-safe for degenerate inputs.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = xs.size
    out = {"n": int(n)}
    if n < 3 or np.allclose(xs.std(), 0.0) or np.allclose(ys.std(), 0.0):
        out.update({"slope": np.nan, "intercept": np.nan, "r": np.nan,
                    "r2": np.nan, "p": np.nan, "s_e": np.nan,
                    "xbar": float(xs.mean()) if n else np.nan, "Sxx": np.nan})
        return out
    slope, intercept = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    r2 = r * r
    df = n - 2
    if r2 >= 1.0:
        t, p = math.inf, 0.0
    else:
        t = r * math.sqrt(df / (1.0 - r2))
        p = _t_two_sided_p(t, df)
    yhat = slope * xs + intercept
    sse = float(((ys - yhat) ** 2).sum())
    out.update({
        "slope": float(slope), "intercept": float(intercept), "r": r, "r2": r2,
        "p": float(p), "s_e": math.sqrt(sse / df), "xbar": float(xs.mean()),
        "Sxx": float(((xs - xs.mean()) ** 2).sum()),
    })
    return out


# --------------------------------------------------------------------------- #
# CSV helpers                                                                  #
# --------------------------------------------------------------------------- #
def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_metric_by_method(csv_path, value_cols, label, require):
    """Load a {dataset: {method: value}} table from a tidy results CSV.

    Returns ``None`` to signal fallback (only when not ``require``); under
    ``require`` raises ``RealDataError`` with an explicit reason instead.
    """
    def _miss(reason):
        if require:
            raise RealDataError(f"{label}: {reason} (--require-real-data)")
        print(f"[fallback] {label}: {reason}", file=sys.stderr)
        return None

    if pd is None:
        return _miss("pandas is unavailable")
    if csv_path is None:
        return _miss("no CSV path given")
    path = Path(csv_path)
    if not path.is_file():
        return _miss(f"CSV not found: {path}")
    df = pd.read_csv(path)
    ds_col = _find_col(df, COLS_DATASET)
    me_col = _find_col(df, COLS_METHOD)
    va_col = _find_col(df, value_cols)
    if not all([ds_col, me_col, va_col]):
        return _miss(f"required columns missing in {path.name} (dataset/method/{value_cols})")
    table = {}
    for _, row in df.iterrows():
        ds = DATASET_ALIASES.get(str(row[ds_col]).strip().lower())
        me = METHOD_ALIASES.get(str(row[me_col]).strip().lower())
        if ds is None or me is None:
            continue
        try:
            val = float(row[va_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        table.setdefault(ds, {})
        if me not in table[ds] or val > table[ds][me]:
            table[ds][me] = val
    if not table:
        return _miss(f"no rows matching the PALL methods/datasets found in {path.name}")
    return table


def _resolve_table(csv_path, value_cols, fallback, label, require):
    """Return (table, n_cells, source). Under ``require`` never backfills paper values."""
    table = _load_metric_by_method(csv_path, value_cols, label, require)
    if table is None:  # only reachable when not require
        print(f"[fallback] {label}: using manuscript (Table I) values.")
        n = sum(len(v) for v in fallback.values())
        return fallback, n, "manuscript fallback"
    n = sum(len(v) for v in table.values())
    print(f"[ok] {label}: loaded {n} cells from {Path(csv_path).name}.")
    return table, n, Path(csv_path).name


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)


# --------------------------------------------------------------------------- #
# Figure 1: WorstDrop grouped bar chart                                        #
# --------------------------------------------------------------------------- #
def plot_worstdrop_chart(outdir, L, csv_path, require):
    table, _, _ = _resolve_table(csv_path, COLS_WORSTDROP, PAPER_WORSTDROP,
                                 "worstdrop_chart", require)
    x = np.arange(len(DATASETS))
    n = len(METHODS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))
    for i, method in enumerate(METHODS):
        vals = [table.get(ds, {}).get(method, np.nan) for ds in DATASETS]
        offset = (i - (n - 1) / 2) * width
        st = METHOD_STYLE[method]
        bars = ax.bar(x + offset, vals, width, label=method, color=st["color"],
                      edgecolor="black", linewidth=0.5, hatch=st["hatch"], alpha=0.95)
        ax.bar_label(bars, fmt="%.4f", padding=1.5, fontsize=5, rotation=90)
    ax.set_ylabel(L(r"WorstDrop $(\downarrow)$", r"بیشینه افت $(\downarrow)$"))
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, max(0.1, ax.get_ylim()[1] * 1.18))
    ax.yaxis.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1, handlelength=1.4)
    _style_axes(ax)
    _save(fig, outdir / "worstdrop_chart.pdf")


# --------------------------------------------------------------------------- #
# Figure 2: Overlap ratio vs. final accuracy -- linear regression + 95% CI     #
# --------------------------------------------------------------------------- #
def _load_overlap_points(csv_paths, require):
    """Return (points Nx2, source_name). Tries each candidate CSV in order and
    keeps the first that exposes BOTH a critical-overlap ratio and an accuracy
    column. Under ``require`` raises if none work."""
    tried = []
    if pd is not None:
        for path in csv_paths:
            path = Path(path)
            if not path.is_file():
                tried.append(f"{path.name} (missing)")
                continue
            df = pd.read_csv(path)
            ox = _find_col(df, COLS_OVERLAP)
            ay = _find_col(df, COLS_ACCURACY)
            if ox and ay:
                sub = df[[ox, ay]].dropna().astype(float)
                if len(sub) >= 3:
                    return sub.values, path.name
                tried.append(f"{path.name} (<3 rows)")
            else:
                tried.append(f"{path.name} (no overlap/accuracy column)")
    reason = "no candidate CSV exposed overlap+accuracy: " + ", ".join(tried)
    if require:
        raise RealDataError(f"overlap_results: {reason} (--require-real-data)")
    print(f"[fallback] overlap_results: {reason}", file=sys.stderr)
    return PAPER_OVERLAP_POINTS, "manuscript fallback"


def plot_overlap_results(outdir, L, csv_paths, require):
    points, source = _load_overlap_points(csv_paths, require)
    xs, ys = points[:, 0], points[:, 1]
    print(f"[ok] overlap_results: {len(xs)} points from {source}.")

    stats = _linregress(xs, ys)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    ax.scatter(xs, ys, color=METHOD_STYLE["PALL-Adapter"]["color"], edgecolor="black",
               linewidth=0.5, s=28, zorder=3, label=L("Runs", "اجراها"))

    if np.isfinite(stats["slope"]):
        grid = np.linspace(xs.min(), xs.max(), 200)
        yfit = stats["slope"] * grid + stats["intercept"]
        ax.plot(grid, yfit, color="#d62728", linewidth=1.3, zorder=2,
                label=L("Linear fit", "برازش خطی"))
        # 95% confidence band for the mean response.
        df = stats["n"] - 2
        tcrit = _t_ppf(0.975, df)
        se_mean = stats["s_e"] * np.sqrt(1.0 / stats["n"] + (grid - stats["xbar"]) ** 2 / stats["Sxx"])
        ci = tcrit * se_mean
        ax.fill_between(grid, yfit - ci, yfit + ci, color="#d62728", alpha=0.15,
                        linewidth=0, zorder=1, label=L("95% CI", "بازه اطمینان ۹۵٪"))
        p = stats["p"]
        p_txt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
        txt = f"$R^2 = {stats['r2']:.3f}$\n" + r"$\rho = " + f"{stats['r']:.3f}$\n" + p_txt
        ax.text(0.04, 0.04, txt, transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5, alpha=0.9))

    ax.set_xlabel(L(r"Critical Shared Overlap Ratio ($r_{\mathrm{crit}}$)",
                    "نسبت هم‌پوشانی بحرانی (r_crit)"))
    ax.set_ylabel(L("Final Average Accuracy", "دقت میانگین نهایی"))
    ax.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.6)
    _style_axes(ax)
    _save(fig, outdir / "overlap_results.pdf")


# --------------------------------------------------------------------------- #
# Figure 3: Updated-parameter ratio vs. WorstDrop scatter                      #
# --------------------------------------------------------------------------- #
def plot_tradeoff_results(outdir, L, worstdrop_csv, updated_csv, require):
    wd, _, _ = _resolve_table(worstdrop_csv, COLS_WORSTDROP, PAPER_WORSTDROP,
                              "tradeoff (WorstDrop)", require)
    rp, _, _ = _resolve_table(updated_csv, COLS_UPDATED, PAPER_UPDATED,
                              "tradeoff (R_updated)", require)
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    for method in METHODS:
        st = METHOD_STYLE[method]
        xs = [rp.get(ds, {}).get(method, np.nan) for ds in DATASETS]
        ys = [wd.get(ds, {}).get(method, np.nan) for ds in DATASETS]
        ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=34,
                   edgecolor="black", linewidth=0.5, label=method, zorder=3)
    for method in METHODS:
        for ds in DATASETS:
            x, y = rp.get(ds, {}).get(method), wd.get(ds, {}).get(method)
            if x is None or y is None:
                continue
            ax.annotate(ds.replace("CIFAR-", "C").replace("TinyImageNet", "Tiny"),
                        (x, y), textcoords="offset points", xytext=(3, 3),
                        fontsize=5, color="0.35")
    ax.set_xlabel(L(r"Updated Parameter Ratio $R_{\mathrm{updated}}$ $(\downarrow)$",
                    r"نسبت پارامتر به‌روزشده $(\downarrow)$"))
    ax.set_ylabel(L(r"WorstDrop $(\downarrow)$", r"بیشینه افت $(\downarrow)$"))
    ax.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=1.2)
    ax.annotate(L("better", "بهتر"), xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=6, color="0.4", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="<-", color="0.5", lw=0.7),
                xytext=(0.16, 0.16), textcoords="axes fraction")
    _style_axes(ax)
    _save(fig, outdir / "tradeoff_results.pdf")


# --------------------------------------------------------------------------- #
# Figure 4: Conceptual WorstDrop decomposition                                 #
# --------------------------------------------------------------------------- #
def _protect_strength_from_csv(csv_path):
    """Mean adapter_shared_protect_strength over PALL-Adapter rows, or None."""
    if pd is None or csv_path is None or not Path(csv_path).is_file():
        return None
    df = pd.read_csv(csv_path)
    me_col = _find_col(df, COLS_METHOD)
    ps_col = _find_col(df, COLS_PROTECT_STRENGTH)
    if not me_col or not ps_col:
        return None
    vals = []
    for _, row in df.iterrows():
        if METHOD_ALIASES.get(str(row[me_col]).strip().lower()) == "PALL-Adapter":
            try:
                v = float(row[ps_col])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else None


def plot_drop_decomposition(outdir, L, protect_strength, csv_path):
    # Conceptual figure: match p to the experiments' protect strength when present.
    p_csv = _protect_strength_from_csv(csv_path)
    if protect_strength is not None:
        p, p_src = float(protect_strength), "--protect-strength"
    elif p_csv is not None:
        p, p_src = p_csv, f"{Path(csv_path).name}:adapter_shared_protect_strength"
    else:
        p, p_src = 2.0 / 3.0, "default (2/3)"
    print(f"[ok] drop_decomposition: protect_strength p={p:.3f} from {p_src}.")

    f_excl, frozen, crit_unc = 0.15, 0.0, 0.75
    regions = [L("Moves, irrelevant", "متحرک، بی‌اثر") + "\n" + r"$\mathcal{F}^{\circ}\ (\leq \epsilon)$",
               L("Relevant, frozen", "مهم، منجمد") + "\n" + r"$H \cup \mathcal{N}\ (=0)$",
               L("Critical overlap", "هم‌پوشانی بحرانی") + "\n" + r"$S_{\mathrm{share\_crit}}$"]
    unc = [f_excl, frozen, crit_unc]
    soft = [f_excl, frozen, (1.0 - p) * crit_unc]
    x = np.arange(len(regions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))
    ax.bar(x - width / 2, unc, width, label=L("Unconstrained ($m_i=1$)", "بدون قید ($m_i=1$)"),
           color="0.55", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, soft, width, label=L("Soft-masked (ours)", "ماسک نرم (ما)"),
           color=METHOD_STYLE["PALL-Adapter"]["color"], edgecolor="black", linewidth=0.5)
    wd_unc, wd_soft = sum(unc), sum(soft)
    ax.axhline(wd_unc, color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(wd_soft, color=METHOD_STYLE["PALL-Adapter"]["color"], linestyle="--",
               linewidth=0.9, zorder=0)
    ax.text(-0.45, wd_unc + 0.012, L("WorstDrop (unconstr.)", "بیشینه افت (بدون قید)"),
            fontsize=5.5, color="0.4")
    ax.text(-0.45, wd_soft + 0.012, L("WorstDrop (soft)", "بیشینه افت (نرم)"),
            fontsize=5.5, color=METHOD_STYLE["PALL-Adapter"]["color"])
    ax.annotate("", xy=(2 + width / 2, soft[2] + 0.01), xytext=(2 - width / 2, unc[2] - 0.01),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3))
    ax.text(2.08, (unc[2] + soft[2]) / 2, r"$\times(1-p)$", color="#d62728",
            fontsize=8, fontweight="bold", ha="left", va="center")
    ax.set_ylabel(L(r"Contribution to WorstDrop", "سهم در بیشینه افت"))
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylim(0, 0.9)
    ax.yaxis.grid(True, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", handlelength=1.4)
    _style_axes(ax)
    _save(fig, outdir / "drop_decomposition.pdf")


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"[saved] {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Directory to write the PDF figures into (use thesis/images).")
    parser.add_argument("--worstdrop-csv", type=Path, default=DEFAULT_WORSTDROP_CSV)
    parser.add_argument("--updated-csv", type=Path, default=None,
                        help="CSV for R_updated (defaults to --worstdrop-csv).")
    parser.add_argument("--overlap-csv", type=Path, default=None,
                        help="Override the overlap-scatter CSV (default tries the real candidates).")
    parser.add_argument("--protect-strength", type=float, default=None,
                        help="Override p in the decomposition figure (default: read from CSV).")
    parser.add_argument("--require-real-data", action="store_true",
                        help="Forbid the manuscript fallback: fail a figure with an explicit "
                             "error if its CSV/columns are missing (prevents fabricated figures).")
    args = parser.parse_args(argv)

    plt.rcParams.update(IEEE_STYLE)
    L, lang = _make_localizer()
    print(f"[info] label language: {lang}"
          + ("" if lang == "fa" else " (install arabic_reshaper + python-bidi for Persian)"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    updated_csv = args.updated_csv or args.worstdrop_csv
    overlap_csvs = [args.overlap_csv] if args.overlap_csv is not None else DEFAULT_OVERLAP_CSVS

    figures = [
        ("drop_decomposition",
         lambda: plot_drop_decomposition(args.outdir, L, args.protect_strength, args.worstdrop_csv)),
        ("worstdrop_chart",
         lambda: plot_worstdrop_chart(args.outdir, L, args.worstdrop_csv, args.require_real_data)),
        ("overlap_results",
         lambda: plot_overlap_results(args.outdir, L, overlap_csvs, args.require_real_data)),
        ("tradeoff_results",
         lambda: plot_tradeoff_results(args.outdir, L, args.worstdrop_csv, updated_csv, args.require_real_data)),
    ]
    failures = []
    for name, fn in figures:
        try:
            fn()
        except RealDataError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"\n{len(failures)} figure(s) skipped (no real data): {', '.join(failures)}", file=sys.stderr)
        sys.exit(2)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
