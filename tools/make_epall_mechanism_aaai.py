#!/usr/bin/env python3
"""Build the AAAI-compliant, full-width EPALL mechanism figure.

The output is exactly 7 inches wide (the AAAI text width), uses no text below
9.2 pt.  It emits an editable SVG in a Times-compatible family, a 300-dpi PNG
for visual QA, and a PDF whose glyphs are converted to vector outlines so the
submission has no external font dependency.  The paper should include the PDF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-epall-aaai")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fontconfig-epall")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "AuthorKit27" / "Figures"
OUT_PDF = OUT_DIR / "EPALL_mechanism_aaai.pdf"
OUT_PNG = OUT_DIR / "EPALL_mechanism_aaai.png"
OUT_SVG = OUT_DIR / "EPALL_mechanism_fixed.svg"

# Okabe--Ito-inspired colors; line styles and labels also carry the meaning.
RED = "#C43C39"
BLUE = "#2563B8"
ORANGE = "#D97706"
GREEN = "#25855A"
INK = "#171717"
GRAY = "#666666"
LIGHT = "#F8FAFC"


def rounded(ax, xy, width, height, *, edge=INK, face="white", lw=0.9, radius=0.025):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=INK, lw=1.0, style="-", mutation=9):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        transform=ax.transAxes,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def panel_a(ax):
    ax.text(0.03, 0.95, "(a) Shared weights interfere", fontsize=9.8,
            fontweight="bold", va="top", transform=ax.transAxes)

    ax.text(0.20, 0.80, "forget path  f", color=RED, fontsize=9.2,
            ha="center", transform=ax.transAxes)
    ax.text(0.78, 0.80, "retain path  r", color=BLUE, fontsize=9.2,
            ha="center", transform=ax.transAxes)

    for x, c, label in ((0.20, RED, "f"), (0.78, BLUE, "r")):
        ax.add_patch(Circle((x, 0.70), 0.045, transform=ax.transAxes,
                            facecolor="white", edgecolor=c, linewidth=1.1))
        ax.text(x, 0.70, label, color=c, fontsize=9.2, ha="center", va="center",
                transform=ax.transAxes)

    # Both task paths meet at the same coordinate.
    ax.add_patch(Circle((0.49, 0.48), 0.065, transform=ax.transAxes,
                        facecolor="#FDE7C2", edgecolor=ORANGE, linewidth=1.2))
    ax.text(0.49, 0.48, "w", fontsize=10.0, fontstyle="italic", ha="center",
            va="center", transform=ax.transAxes)
    arrow(ax, (0.23, 0.67), (0.45, 0.53), color=RED, lw=1.25)
    arrow(ax, (0.75, 0.67), (0.53, 0.53), color=BLUE, lw=1.25, style="--")
    ax.text(0.49, 0.37, r"shared $w\in S_{\mathrm{share}}$", color=ORANGE,
            fontsize=9.2, ha="center", transform=ax.transAxes)

    # A target reset at the shared coordinate also perturbs the retained path.
    rounded(ax, (0.08, 0.09), 0.82, 0.17, edge=GRAY, face=LIGHT, lw=0.8)
    ax.text(0.11, 0.175, "reset", color=RED, fontsize=9.2,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.29, 0.175, "×", color=RED, fontsize=16, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    arrow(ax, (0.35, 0.175), (0.55, 0.175), color=BLUE, lw=1.2, style="--")
    ax.text(0.59, 0.175, "retained drift", color=BLUE, fontsize=9.2,
            fontweight="bold", va="center", transform=ax.transAxes)


def panel_b(ax):
    ax.text(0.03, 0.95, "(b) Rank overlap", fontsize=9.8,
            fontweight="bold", va="top", transform=ax.transAxes)

    # Structural candidate set.
    ax.add_patch(Circle((0.37, 0.66), 0.19, transform=ax.transAxes,
                        facecolor="#FFF1F0", edgecolor=RED, linewidth=1.1))
    ax.add_patch(Circle((0.59, 0.66), 0.19, transform=ax.transAxes,
                        facecolor="#EEF5FF", edgecolor=BLUE, linewidth=1.1))
    ax.text(0.24, 0.66, "forget mask\n" + r"$M_f$", color=RED, fontsize=9.2,
            ha="center", transform=ax.transAxes)
    ax.text(0.72, 0.66, "retained union\n" + r"$M_r$", color=BLUE, fontsize=9.2,
            ha="center", transform=ax.transAxes)
    ax.add_patch(Rectangle((0.42, 0.49), 0.12, 0.34, transform=ax.transAxes,
                           facecolor="#F9CF8B", edgecolor="none", alpha=0.9))
    ax.text(0.48, 0.66, r"$S_{\mathrm{share}}$", color=INK, fontsize=9.2, fontweight="bold",
            ha="center", va="center", rotation=90, transform=ax.transAxes)

    ax.text(0.50, 0.40, r"$S_{\mathrm{share}}=M_f\cap M_r$", fontsize=9.5,
            ha="center", transform=ax.transAxes)
    arrow(ax, (0.50, 0.36), (0.50, 0.29), color=INK, lw=0.9)

    # Rank candidates by retained sensitivity. Blue rings identify the selected top-rho set.
    values = [0.30, 0.48, 0.63, 0.78, 0.92]
    xs = [0.20, 0.34, 0.48, 0.62, 0.76]
    for i, (x, value) in enumerate(zip(xs, values)):
        ax.add_patch(Circle((x, 0.22), 0.035, transform=ax.transAxes,
                            facecolor=mpl.colors.to_rgba(ORANGE, value),
                            edgecolor=INK, linewidth=0.65))
        if i >= 3:
            ax.add_patch(Circle((x, 0.22), 0.050, transform=ax.transAxes,
                                facecolor="none", edgecolor=BLUE, linewidth=1.2))
    ax.text(0.50, 0.105,
            "rank by retained gradient\n" + r"protect top-$\rho$: $S_{\mathrm{share,crit}}$",
            fontsize=9.2, color=INK, ha="center", va="center",
            linespacing=1.15, transform=ax.transAxes)


def step_box(ax, y, number, text, color, face):
    rounded(ax, (0.06, y), 0.88, 0.145, edge=color, face=face, lw=0.9)
    ax.add_patch(Circle((0.105, y + 0.0725), 0.034, transform=ax.transAxes,
                        facecolor=color, edgecolor=color, linewidth=0.8))
    ax.text(0.105, y + 0.0725, str(number), color="white", fontsize=9.2,
            fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.165, y + 0.0725, text, color=INK, fontsize=9.2,
            va="center", ha="left", linespacing=1.1, transform=ax.transAxes)


def panel_c(ax):
    ax.text(0.03, 0.95, "(c) Protected request", fontsize=9.8,
            fontweight="bold", va="top", transform=ax.transAxes)

    step_box(ax, 0.745, 1, r"optional diagnostic; delete $B_f$" + "\n" + r"and retire $M_f$", RED, "#FFF4F3")
    step_box(ax, 0.555, 2, r"form $M_r$; rank overlap;" + "\n" + "cache anchor values", ORANGE, "#FFF7E8")
    step_box(ax, 0.365, 3, "branch-dependent PALL\ntarget reset", GREEN, "#F0FAF5")
    step_box(ax, 0.175, 4, "retained repair + anchor;\nrebuild active union", BLUE, "#F1F6FF")

    ax.text(0.50, 0.075, "empirical — not certified",
            fontsize=9.2, fontstyle="italic", color=GRAY, ha="center",
            transform=ax.transAxes)


def build():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "PT Serif", "DejaVu Serif"],
        "font.size": 9.2,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Keep the SVG editable in Inkscape.  The submission PDF is outlined
        # below, so it does not depend on this font being installed elsewhere.
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(7.0, 2.62), facecolor="white")
    axes = [
        fig.add_axes([0.010, 0.025, 0.310, 0.950]),
        fig.add_axes([0.345, 0.025, 0.310, 0.950]),
        fig.add_axes([0.680, 0.025, 0.310, 0.950]),
    ]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        rounded(ax, (0.0, 0.0), 1.0, 1.0, edge="#A8B0BA", face="white",
                lw=0.75, radius=0.025)

    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white", edgecolor="none")
    fig.savefig(OUT_SVG, format="svg", facecolor="white", edgecolor="none")

    # Save a vector PDF, then convert glyphs to vector outlines. This avoids
    # Type-3 and unembedded-font problems in the submission PDF.
    with tempfile.TemporaryDirectory(prefix="epall-figure-") as tmpdir:
        raw_pdf = Path(tmpdir) / "raw.pdf"
        fig.savefig(raw_pdf, format="pdf", facecolor="white", edgecolor="none")
        gs = shutil.which("gs")
        if not gs:
            raise RuntimeError("Ghostscript is required to outline figure fonts")
        subprocess.run(
            [
                gs,
                "-q",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.7",
                "-dNoOutputFonts",
                f"-sOutputFile={OUT_PDF}",
                str(raw_pdf),
            ],
            check=True,
        )
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    build()
