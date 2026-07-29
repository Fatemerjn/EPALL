#!/usr/bin/env bash
# Rebuild the EPALL mechanism figure as a self-contained vector PDF.
#
# The paper must include PDF, not PNG.  Inkscape's "PDF + LaTeX" export is NOT
# usable here: it splits the drawing into a text-less layered PDF plus a
# .pdf_tex overlay that re-typesets every string at one uniform LaTeX font
# size, which destroys this figure's deliberate size hierarchy.  We therefore
# export a plain PDF with --export-text-to-path, so the file carries no font
# dependency at all and renders identically everywhere.
#
# Source of truth: paper/AuthorKit27/Figures/EPALL_mechanism.svg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIG_DIR="$ROOT/paper/AuthorKit27/Figures"
SVG="$FIG_DIR/EPALL_mechanism.svg"
PDF="$FIG_DIR/EPALL_mechanism.pdf"

INKSCAPE="${INKSCAPE:-/Applications/Inkscape.app/Contents/MacOS/inkscape}"
if ! command -v "$INKSCAPE" >/dev/null 2>&1; then
    INKSCAPE="$(command -v inkscape || true)"
fi
if [[ -z "$INKSCAPE" || ! -x "$INKSCAPE" ]]; then
    echo "FAIL: inkscape not found; set INKSCAPE=/path/to/inkscape" >&2
    exit 1
fi
if [[ ! -f "$SVG" ]]; then
    echo "FAIL: missing $SVG" >&2
    exit 1
fi

"$INKSCAPE" --export-type=pdf --export-text-to-path \
    --export-filename="$PDF" "$SVG"

# A self-contained export embeds no fonts, because every glyph is now a path.
if python3 - "$PDF" <<'PY'
import re, sys
data = open(sys.argv[1], "rb").read()
sys.exit(0 if not re.search(rb"/BaseFont", data) else 1)
PY
then
    echo "OK: $PDF (vector, text as paths, no font dependency)"
else
    echo "WARN: $PDF still references fonts; check --export-text-to-path" >&2
    exit 1
fi
