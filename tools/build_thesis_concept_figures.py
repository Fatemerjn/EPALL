#!/usr/bin/env python3
"""Build the editable conceptual figures used by the Persian thesis.

The SVG files are the editable sources.  PDF siblings are rendered with
``rsvg-convert`` for reliable XeLaTeX inclusion.  No experimental observation is
read, modified, or synthesized by this script; every output is a schematic.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "thesis" / "images"

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
MAGENTA = "#CC79A7"
PURPLE = "#6F3C8F"
INK = "#263238"
MID = "#69757F"
LIGHT = "#E6EBEF"
PALE = "#F5F7F9"
BLUE_PALE = "#EAF4FA"
GREEN_PALE = "#EAF7F2"
ORANGE_PALE = "#FCEFEA"
MAGENTA_PALE = "#F8EDF4"


class SVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="8.2" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>',
            '<marker id="arrow-orange" viewBox="0 0 10 10" refX="8.2" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE}"/></marker>',
            '<pattern id="critical-hatch" width="9" height="9" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)">'
            f'<rect width="9" height="9" fill="{MAGENTA_PALE}"/>'
            f'<line x1="0" y1="0" x2="0" y2="9" stroke="{PURPLE}" stroke-width="2"/>'
            "</pattern>",
            "<style>",
            "text{font-family:'Times New Roman','DejaVu Serif',serif;fill:#263238}",
            ".label{font-size:25px}.small{font-size:20px}.tiny{font-size:17px}",
            ".panel{font-size:29px;font-weight:700}.math{font-style:italic}",
            ".card{stroke-width:2;rx:14;ry:14}.link{fill:none;stroke-width:3;marker-end:url(#arrow)}",
            "</style>",
            "</defs>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x, y, w, h, fill="white", stroke=INK, sw=2, rx=14, dash=None, opacity=None):
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
            f'rx="{rx}"', f'fill="{fill}"', f'stroke="{stroke}"', f'stroke-width="{sw}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add("<rect " + " ".join(attrs) + "/>")

    def line(self, x1, y1, x2, y2, stroke=INK, sw=3, dash=None, arrow=False, arrow_id="arrow", opacity=None):
        attrs = [
            f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
            f'stroke="{stroke}"', f'stroke-width="{sw}"', 'fill="none"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if arrow:
            attrs.append(f'marker-end="url(#{arrow_id})"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity}"')
        self.add("<line " + " ".join(attrs) + "/>")

    def circle(self, cx, cy, r, fill="white", stroke=INK, sw=2, opacity=None):
        op = "" if opacity is None else f' opacity="{opacity}"'
        self.add(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{op}/>'
        )

    def text(self, x, y, value, size=25, anchor="middle", weight=None, fill=INK, italic=False, rotate=None):
        attrs = [f'x="{x}"', f'y="{y}"', f'font-size="{size}px"', f'text-anchor="{anchor}"', f'fill="{fill}"']
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if italic:
            attrs.append('font-style="italic"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x} {y})"')
        self.add(f'<text {" ".join(attrs)}>{html.escape(value)}</text>')

    def multiline(self, x, y, lines, size=24, anchor="middle", weight=None, fill=INK, gap=1.25):
        attrs = [f'x="{x}"', f'y="{y}"', f'font-size="{size}px"', f'text-anchor="{anchor}"', f'fill="{fill}"']
        if weight:
            attrs.append(f'font-weight="{weight}"')
        self.add(f'<text {" ".join(attrs)}>')
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else size * gap
            self.add(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
        self.add("</text>")

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def panel_frame(svg: SVG, x: int, y: int, w: int, h: int, label: str, title: str) -> None:
    svg.rect(x, y, w, h, fill="white", stroke=LIGHT, sw=2, rx=18)
    svg.text(x + 24, y + 38, label, size=28, anchor="start", weight="700")
    svg.text(x + 72, y + 38, title, size=25, anchor="start", weight="700")


def build_selective_forgetting_pipeline() -> str:
    svg = SVG(1700, 520, "Selective forgetting lifecycle")
    events = [
        (55, "Learn T1", BLUE_PALE, BLUE, "{T1}"),
        (360, "Learn T2", BLUE_PALE, BLUE, "{T1, T2}"),
        (665, "Learn T3", BLUE_PALE, BLUE, "{T1, T2, T3}"),
        (970, "Forget T2", ORANGE_PALE, ORANGE, "{T1, T3}"),
    ]
    for index, (x, label, fill, stroke, state) in enumerate(events):
        svg.rect(x, 118, 250, 105, fill=fill, stroke=stroke, sw=3, rx=16)
        sign = "+" if label.startswith("Learn") else "−"
        svg.circle(x + 205, 170, 25, fill="white", stroke=stroke, sw=3)
        svg.text(x + 205, 179, sign, size=35, weight="700", fill=stroke)
        svg.text(x + 105, 181, label, size=29, weight="700")
        svg.rect(x + 25, 288, 200, 64, fill=GREEN_PALE if index != 3 else PALE, stroke=GREEN if index != 3 else MID, sw=2, rx=12)
        svg.text(x + 125, 329, state, size=25, weight="700", fill=GREEN if index != 3 else MID)
        if index < len(events) - 1:
            svg.line(x + 250, 170, events[index + 1][0] - 12, 170, stroke=INK, sw=3, arrow=True)
    svg.text(565, 394, "Active task set", size=21, fill=MID)
    svg.line(1095, 112, 1095, 69, stroke=ORANGE, sw=3, arrow=True, arrow_id="arrow-orange")
    svg.text(1095, 48, "Aᵤ → chance level", size=25, weight="700", fill=ORANGE)

    svg.rect(1295, 82, 345, 125, fill=ORANGE_PALE, stroke=ORANGE, sw=3, rx=16)
    svg.text(1468, 128, "Target task", size=24, weight="700", fill=ORANGE)
    svg.text(1468, 171, "suppressed after request", size=22, fill=ORANGE)
    svg.rect(1295, 245, 345, 125, fill=GREEN_PALE, stroke=GREEN, sw=3, rx=16)
    svg.text(1468, 291, "Retained tasks", size=24, weight="700", fill=GREEN)
    svg.text(1468, 334, "remain near pre-request level", size=21, fill=GREEN)
    svg.line(1220, 170, 1282, 145, stroke=ORANGE, sw=3, arrow=True, arrow_id="arrow-orange")
    svg.line(1220, 188, 1282, 286, stroke=GREEN, sw=3, arrow=True)
    svg.text(850, 474, "Schematic lifecycle; no post-deletion learning is implied.", size=20, fill=MID)
    return svg.finish()


def _draw_mask_grid(svg: SVG, x0: int, y0: int, mode: str) -> None:
    rows, cols, size, gap = 6, 14, 30, 6
    overlap = [(0, 5), (0, 8), (1, 4), (1, 7), (2, 5), (2, 9), (3, 4), (3, 8), (4, 6), (5, 7)]
    crit = {(1, 7), (3, 4)}  # exactly 20% of the ten overlap cells
    forget_only = {(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (4, 1), (5, 2)}
    retain_only = {(0, 11), (1, 10), (1, 13), (2, 11), (3, 10), (3, 12), (4, 9), (4, 12), (5, 10), (5, 13)}
    q_order = {coord: rank for rank, coord in enumerate(overlap)}
    purple_bins = ["#EFE2EC", "#E4C8DE", "#D7A8CC", "#C881B9", "#A9519D"]
    for row in range(rows):
        for col in range(cols):
            coord = (row, col)
            fill, stroke, sw, opacity = PALE, "#D4DADE", 1.2, 1.0
            if mode == "masks":
                if coord in overlap:
                    fill, stroke = MAGENTA_PALE, MAGENTA
                elif coord in forget_only:
                    fill, stroke = ORANGE_PALE, ORANGE
                elif coord in retain_only:
                    fill, stroke = BLUE_PALE, BLUE
            elif mode == "rank":
                if coord in overlap:
                    fill = purple_bins[min(4, q_order[coord] // 2)]
                    stroke = PURPLE
                elif coord in forget_only:
                    fill, stroke, opacity = ORANGE_PALE, ORANGE, 0.38
                elif coord in retain_only:
                    fill, stroke, opacity = BLUE_PALE, BLUE, 0.38
            else:
                if coord in overlap:
                    fill, stroke = MAGENTA_PALE, "#B7A1B2"
                elif coord in forget_only:
                    fill, stroke, opacity = ORANGE_PALE, ORANGE, 0.22
                elif coord in retain_only:
                    fill, stroke, opacity = BLUE_PALE, BLUE, 0.22
                if coord in crit:
                    fill, stroke, sw, opacity = "url(#critical-hatch)", PURPLE, 3.5, 1.0
            svg.rect(x0 + col * (size + gap), y0 + row * (size + gap), size, size, fill=fill, stroke=stroke, sw=sw, rx=3, opacity=opacity)


def build_parameter_overlap_concept() -> str:
    svg = SVG(1800, 610, "Structural overlap, retention sensitivity, and critical subset")
    panels = [
        (22, "(a)", "Structural overlap", "masks"),
        (612, "(b)", "Retention sensitivity", "rank"),
        (1202, "(c)", "Critical subset", "critical"),
    ]
    for x, label, title, mode in panels:
        panel_frame(svg, x, 18, 565, 550, label, title)
        _draw_mask_grid(svg, x + 29, 112, mode)
    svg.rect(56, 414, 22, 22, fill=ORANGE_PALE, stroke=ORANGE, sw=2, rx=2)
    svg.text(90, 432, "M_f", size=21, anchor="start", italic=True, fill=ORANGE)
    svg.rect(168, 414, 22, 22, fill=BLUE_PALE, stroke=BLUE, sw=2, rx=2)
    svg.text(202, 432, "M_r", size=21, anchor="start", italic=True, fill=BLUE)
    svg.rect(280, 414, 22, 22, fill=MAGENTA_PALE, stroke=MAGENTA, sw=2, rx=2)
    svg.text(314, 432, "S_share = M_f ∩ M_r", size=21, anchor="start", italic=True, fill=MAGENTA)
    svg.multiline(304, 484, ["One fixed mask grid is reused", "in all three panels."], size=20, fill=MID)

    svg.text(895, 430, "low qᵢ", size=19, anchor="end", fill=MID)
    for index, color in enumerate(["#EFE2EC", "#E4C8DE", "#D7A8CC", "#C881B9", "#A9519D"]):
        svg.rect(910 + index * 48, 409, 38, 25, fill=color, stroke="none", sw=0, rx=2)
    svg.text(1162, 430, "high qᵢ", size=19, anchor="start", fill=PURPLE)
    svg.multiline(894, 484, ["qᵢ = |∇ᵢ L_retain| is evaluated", "only on S_share."], size=20, fill=MID)

    svg.rect(1260, 411, 28, 28, fill="url(#critical-hatch)", stroke=PURPLE, sw=3, rx=2)
    svg.text(1303, 433, "S_crit = Top-ρ(S_share; qᵢ)", size=21, anchor="start", italic=True, fill=PURPLE)
    svg.rect(1518, 475, 150, 48, fill="white", stroke=PURPLE, sw=2, rx=10)
    svg.text(1593, 507, "ρ = 0.2", size=22, weight="700", fill=PURPLE)
    svg.text(1485, 548, "S_crit ⊂ S_share by construction", size=20, fill=MID)
    return svg.finish()


def build_epall_mechanism_compact() -> str:
    svg = SVG(1800, 650, "Compact EPALL mechanism")
    panel_frame(svg, 20, 18, 555, 580, "(a)", "Shared coordinates create interference")
    panel_frame(svg, 622, 18, 555, 580, "(b)", "Select retention-sensitive overlap")
    panel_frame(svg, 1224, 18, 555, 580, "(c)", "Protected forgetting request")

    nodes = [(100, 140), (280, 140), (460, 140), (175, 280), (365, 280), (115, 430), (280, 430), (465, 430)]
    forget_edges = [(0, 3), (0, 4), (3, 5), (4, 6)]
    retained_edges = [(1, 3), (1, 4), (3, 6), (4, 7)]
    for a, b in retained_edges:
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        svg.line(x1 + 20, y1 + 16, x2 + 20, y2 - 16, stroke=BLUE, sw=3, dash="10 7", arrow=True)
    for a, b in forget_edges:
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        svg.line(x1 + 20, y1 + 16, x2 + 20, y2 - 16, stroke=ORANGE, sw=3, arrow=True, arrow_id="arrow-orange")
    for index, (x, y) in enumerate(nodes):
        overlap = index in {3, 4, 6}
        svg.circle(x + 20, y, 21, fill=MAGENTA_PALE if overlap else "white", stroke=MAGENTA if overlap else INK, sw=3 if overlap else 2)
    svg.text(120, 96, "forgotten path f", size=22, fill=ORANGE)
    svg.text(385, 96, "retained path r", size=22, fill=BLUE)
    svg.rect(66, 492, 462, 73, fill=ORANGE_PALE, stroke=ORANGE, sw=2, rx=12, dash="8 6")
    svg.text(297, 522, "Naive reset of a shared coordinate", size=21, weight="700", fill=ORANGE)
    svg.text(297, 551, "causes retained-task drift.", size=20, fill=ORANGE)

    svg.rect(670, 91, 458, 110, fill=MAGENTA_PALE, stroke=MAGENTA, sw=3, rx=15)
    svg.text(899, 132, "1  OVERLAP", size=25, weight="700", fill=MAGENTA)
    svg.text(899, 174, "S_share = M_f ∩ M_r", size=30, italic=True, fill=PURPLE)
    svg.line(899, 202, 899, 245, stroke=INK, sw=3, arrow=True)
    svg.rect(670, 260, 458, 145, fill=BLUE_PALE, stroke=BLUE, sw=3, rx=15)
    svg.text(899, 302, "2  RANK", size=25, weight="700", fill=BLUE)
    svg.text(899, 346, "qᵢ = |∇ᵢ L_retain|", size=30, italic=True, fill=BLUE)
    svg.text(899, 383, "use retained buffers only", size=20, fill=MID)
    svg.line(899, 406, 899, 449, stroke=INK, sw=3, arrow=True)
    svg.rect(670, 463, 458, 96, fill=MAGENTA_PALE, stroke=PURPLE, sw=3, rx=15)
    svg.text(899, 504, "3  PROTECT TOP-ρ", size=24, weight="700", fill=PURPLE)
    svg.text(899, 540, "S_crit = Top-ρ(S_share; qᵢ)", size=24, italic=True, fill=PURPLE)

    workflow = [
        (1263, 82, 478, 80, ORANGE_PALE, ORANGE, "Delete target buffer; retire M_f"),
        (1263, 184, 478, 80, GREEN_PALE, GREEN, "Form M_r and cache retained anchors"),
        (1263, 286, 478, 80, MAGENTA_PALE, PURPLE, "Apply branch-dependent target reset"),
        (1263, 388, 478, 80, BLUE_PALE, BLUE, "Retained-only repair with anchors"),
        (1263, 490, 478, 68, PALE, MID, "Rebuild active mask; discard temporary state"),
    ]
    for index, (x, y, w, h, fill, stroke, label) in enumerate(workflow):
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=3, rx=13)
        svg.text(x + 27, y + h / 2 + 8, str(index + 1), size=25, anchor="start", weight="700", fill=stroke)
        svg.text(x + 76, y + h / 2 + 8, label, size=21, anchor="start", fill=stroke)
        if index < len(workflow) - 1:
            svg.line(x + w / 2, y + h, x + w / 2, workflow[index + 1][1] - 8, stroke=INK, sw=2.5, arrow=True)
    svg.rect(1343, 576, 318, 45, fill="white", stroke=MID, sw=1.5, rx=8)
    svg.text(1502, 606, "Empirical — not certified removal", size=19, italic=True, fill=MID)
    return svg.finish()


def build_pall_adapter_architecture() -> str:
    svg = SVG(1800, 650, "PALL-Adapter architecture")
    svg.rect(270, 18, 1260, 196, fill="white", stroke=BLUE, sw=2.5, rx=18, dash="8 6")
    svg.text(300, 55, "Residual bottleneck used by both φ_s and φ_t", size=25, anchor="start", weight="700", fill=BLUE)
    x = 350
    svg.text(x, 123, "z ∈ R⁵¹²", size=25, anchor="start", italic=True)
    svg.line(455, 114, 535, 114, stroke=INK, sw=3, arrow=True)
    svg.rect(545, 75, 190, 78, fill=BLUE_PALE, stroke=BLUE, sw=2.5, rx=10)
    svg.multiline(640, 107, ["W_down", "512 → r"], size=21, weight="700")
    svg.line(735, 114, 805, 114, stroke=INK, sw=3, arrow=True)
    svg.rect(815, 75, 120, 78, fill=ORANGE_PALE, stroke=ORANGE, sw=2.5, rx=10)
    svg.text(875, 124, "ReLU", size=22, weight="700")
    svg.line(935, 114, 1005, 114, stroke=INK, sw=3, arrow=True)
    svg.rect(1015, 75, 190, 78, fill=BLUE_PALE, stroke=BLUE, sw=2.5, rx=10)
    svg.multiline(1110, 107, ["W_up", "r → 512"], size=21, weight="700")
    svg.line(1205, 114, 1280, 114, stroke=INK, sw=3, arrow=True)
    svg.circle(1305, 114, 20, fill="white", stroke=INK, sw=2.5)
    svg.text(1305, 124, "+", size=28, weight="700")
    svg.line(1325, 114, 1415, 114, stroke=INK, sw=3, arrow=True)
    svg.line(430, 145, 430, 184, stroke=INK, sw=2)
    svg.line(430, 184, 1305, 184, stroke=INK, sw=2)
    svg.line(1305, 184, 1305, 137, stroke=INK, sw=2, arrow=True)
    svg.text(1350, 123, "output ∈ R⁵¹²", size=20, anchor="start")
    svg.text(460, 205, "W_down: Kaiming init", size=18, anchor="start", fill=MID)
    svg.text(1000, 205, "W_up = 0 ⇒ identity at initialization", size=18, anchor="start", fill=MID)
    svg.text(1460, 72, "d → r → d", size=21, weight="700", fill=BLUE)
    svg.text(1460, 102, "d = 512", size=18, fill=MID)
    svg.text(1460, 132, "fixed default r = 16", size=17, fill=MID)

    y = 390
    blocks = [
        (40, 330, 235, 145, PALE, MID, ["Input batch", "x ∈ Rᴮˣ³ˣᴴˣᵂ"]),
        (335, 305, 300, 195, PALE, MID, ["Frozen feature extractor", "θ_base", "g(x) ∈ R⁵¹²"]),
        (695, 320, 220, 165, BLUE_PALE, BLUE, ["Shared adapter", "φ_s", "trainable"]),
        (975, 285, 245, 235, GREEN_PALE, GREEN, ["Task adapters", "φ₁ … φ_t … φ_T", "φ_t active"]),
        (1280, 300, 265, 205, ORANGE_PALE, ORANGE, ["Shared classifier", "W_cls ∈ Rᶜˣ⁵¹²", "task row-block C_t"]),
        (1600, 340, 155, 125, PALE, INK, ["Prediction", "ŷ_t"]),
    ]
    for x0, y0, w, h, fill, stroke, lines in blocks:
        svg.rect(x0, y0, w, h, fill=fill, stroke=stroke, sw=3, rx=16)
        svg.multiline(x0 + w / 2, y0 + 45, lines, size=22, weight="700" if len(lines) < 3 else None, fill=stroke, gap=1.35)
    for left, right in zip(blocks[:-1], blocks[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        svg.line(x1 + 8, y, x2 - 10, y, stroke=INK, sw=3, arrow=True)
    svg.rect(1315, 420, 195, 46, fill="white", stroke=ORANGE, sw=2, rx=6)
    svg.text(1412, 450, "C_t: class rows", size=18, weight="700", fill=ORANGE)
    svg.rect(650, 545, 975, 78, fill="white", stroke=ORANGE, sw=2.5, rx=14, dash="8 6")
    svg.text(676, 578, "forget(u):", size=23, anchor="start", weight="700", fill=ORANGE)
    svg.text(790, 578, "soft-masked update on φ_s", size=20, anchor="start", fill=ORANGE)
    svg.text(1050, 578, "reset φ_u", size=20, anchor="start", fill=ORANGE)
    svg.text(1248, 578, "reset classifier row-block C_u", size=20, anchor="start", fill=ORANGE)
    svg.text(810, 608, "θ_base unchanged", size=19, anchor="start", fill=MID)
    svg.line(805, 545, 805, 497, stroke=ORANGE, sw=2.5, dash="7 5", arrow=True, arrow_id="arrow-orange")
    svg.line(1097, 545, 1097, 528, stroke=ORANGE, sw=2.5, dash="7 5", arrow=True, arrow_id="arrow-orange")
    svg.line(1415, 545, 1415, 514, stroke=ORANGE, sw=2.5, dash="7 5", arrow=True, arrow_id="arrow-orange")
    return svg.finish()


def build_softmask_drop_decomposition() -> str:
    svg = SVG(1600, 560, "Soft-mask subphase drop-budget decomposition")
    base_y, unit = 450, 0.95
    left_x, right_x, bar_w = 330, 930, 250
    segments = [("H ⊂ S_crit", 120, PURPLE, "url(#critical-hatch)"), ("S_crit \\ H", 95, MAGENTA, MAGENTA_PALE), ("F° = S_forget \\ S_active", 110, ORANGE, ORANGE_PALE)]

    svg.text(455, 55, "(a) Unconstrained", size=28, weight="700")
    svg.text(1055, 55, "(b) Soft mask", size=28, weight="700")
    y = base_y
    for label, height, stroke, fill in segments:
        h = height * unit
        y -= h
        svg.rect(left_x, y, bar_w, h, fill=fill, stroke=stroke, sw=2.5, rx=0)
        svg.text(left_x + bar_w / 2, y + h / 2 + 8, label, size=21, weight="700", fill=stroke)
        svg.text(left_x + bar_w - 16, y + h / 2 + 35, "mᵢ = 1", size=18, anchor="end", fill=stroke)
    left_top = y

    # H is exactly zero in the soft-masked subphase and is drawn as an outlined slot.
    h_zero = segments[0][1] * unit
    svg.rect(right_x, base_y - h_zero, bar_w, h_zero, fill="white", stroke=PURPLE, sw=2.5, rx=0, dash="8 6")
    svg.text(right_x + bar_w / 2, base_y - h_zero / 2 + 8, "H: mᵢ = 0", size=21, weight="700", fill=PURPLE)
    y2 = base_y - h_zero
    crit_h = segments[1][1] * unit * 0.5
    y2 -= crit_h
    svg.rect(right_x, y2, bar_w, crit_h, fill=MAGENTA_PALE, stroke=MAGENTA, sw=2.5, rx=0)
    svg.text(right_x + bar_w / 2, y2 + crit_h / 2 + 7, "S_crit \\ H", size=20, weight="700", fill=MAGENTA)
    svg.text(right_x + bar_w - 14, y2 + crit_h - 10, "mᵢ = 1 − p", size=17, anchor="end", fill=MAGENTA)
    f_h = segments[2][1] * unit
    y2 -= f_h
    svg.rect(right_x, y2, bar_w, f_h, fill=ORANGE_PALE, stroke=ORANGE, sw=2.5, rx=0)
    svg.text(right_x + bar_w / 2, y2 + f_h / 2 + 8, "F° = S_forget \\ S_active", size=20, weight="700", fill=ORANGE)
    svg.text(right_x + bar_w - 14, y2 + f_h - 10, "mᵢ = 1", size=17, anchor="end", fill=ORANGE)
    right_top = y2

    svg.line(230, base_y, 1260, base_y, stroke=INK, sw=2.5)
    svg.line(250, left_top, 650, left_top, stroke=INK, sw=2, dash="9 6")
    svg.line(850, right_top, 1250, right_top, stroke=INK, sw=2, dash="9 6")
    svg.text(690, left_top - 10, "Δ_t^unc", size=26, italic=True)
    svg.text(1290, right_top - 10, "Δ_t^soft", size=26, italic=True)
    svg.line(835, base_y - h_zero - segments[1][1] * unit, 835, y2 + f_h, stroke=MAGENTA, sw=3)
    svg.line(815, base_y - h_zero - segments[1][1] * unit, 855, base_y - h_zero - segments[1][1] * unit, stroke=MAGENTA, sw=3)
    svg.line(815, y2 + f_h, 855, y2 + f_h, stroke=MAGENTA, sw=3)
    svg.text(800, (base_y - h_zero - segments[1][1] * unit + y2 + f_h) / 2, "×(1−p)", size=23, anchor="end", weight="700", fill=MAGENTA)
    svg.text(800, 525, "Schematic subphase only; not a bound on end-to-end WorstDrop.", size=20, fill=MID)
    return svg.finish()


def build_literature_taxonomy() -> str:
    svg = SVG(1800, 560, "Taxonomy and positioning of the thesis")
    cards = [
        (35, BLUE_PALE, BLUE, ["Continual learning", "replay · regularization"], ["retain knowledge"]),
        (375, PALE, MID, ["Parameter isolation", "task subnetworks"], ["limit interference"]),
        (715, ORANGE_PALE, ORANGE, ["Machine unlearning", "data/model removal"], ["remove influence"]),
        (1055, MAGENTA_PALE, PURPLE, ["Task-aware forgetting", "PALL"], ["delete one task"]),
        (1395, GREEN_PALE, GREEN, ["Overlap-aware repair", "EPALL · PALL-Adapter"], ["protect shared", "reduce trainable scope"]),
    ]
    for index, (x, fill, stroke, lines, badges) in enumerate(cards):
        svg.rect(x, 95, 290, 245, fill=fill, stroke=stroke, sw=3, rx=18)
        svg.text(x + 28, 135, str(index + 1), size=25, anchor="start", weight="700", fill=stroke)
        svg.multiline(x + 145, 185, lines, size=24, weight="700", fill=stroke, gap=1.45)
        for badge_index, badge in enumerate(badges):
            svg.rect(x + 42, 256 + badge_index * 39, 206, 29, fill="white", stroke=stroke, sw=1.5, rx=14)
            svg.text(x + 145, 277 + badge_index * 39, badge, size=16, fill=stroke)
        if index < len(cards) - 1:
            svg.line(x + 290, 210, cards[index + 1][0] - 12, 210, stroke=INK, sw=3, arrow=True)
    svg.line(180, 405, 1640, 405, stroke=INK, sw=2.5, arrow=True)
    svg.text(180, 443, "preserve previously learned behavior", size=20, anchor="start", fill=BLUE)
    svg.text(1640, 443, "selectively remove while protecting overlap", size=20, anchor="end", fill=GREEN)
    svg.rect(1160, 475, 570, 52, fill="white", stroke=GREEN, sw=2, rx=12)
    svg.text(1445, 509, "Position of this thesis: overlap-aware, task-level, empirical", size=20, weight="700", fill=GREEN)
    return svg.finish()


BUILDERS = {
    "selective_forgetting_pipeline": build_selective_forgetting_pipeline,
    "parameter_overlap_concept": build_parameter_overlap_concept,
    "EPALL_mechanism_compact": build_epall_mechanism_compact,
    "pall_adapter_architecture": build_pall_adapter_architecture,
    "softmask_drop_decomposition": build_softmask_drop_decomposition,
    "literature_taxonomy": build_literature_taxonomy,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--svg-only", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("rsvg-convert")
    if not args.svg_only and converter is None:
        raise RuntimeError("rsvg-convert is required to render thesis-ready PDF siblings")
    for stem, builder in BUILDERS.items():
        svg_path = args.outdir / f"{stem}.svg"
        svg_path.write_text(builder(), encoding="utf-8")
        print(f"[saved] {svg_path}")
        if not args.svg_only:
            pdf_path = args.outdir / f"{stem}.pdf"
            subprocess.run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
            print(f"[saved] {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
