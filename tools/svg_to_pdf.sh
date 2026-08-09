#!/usr/bin/env bash
# Export thesis figures to PDF from their SVG source.
#
# The thesis \includegraphics every figure as .pdf, so a redrawn .svg only
# reaches the document once it has been exported. This converts every .svg in
# thesis/images/ whose .pdf is missing or older than the .svg, and leaves
# everything else alone.
#
#   bash tools/svg_to_pdf.sh              # convert what is stale
#   bash tools/svg_to_pdf.sh --all        # reconvert everything
#   bash tools/svg_to_pdf.sh a.svg b.svg  # convert just these
#
# Vector in, vector out: text stays selectable and the figure stays sharp at any
# zoom. PNG is deliberately not accepted as a source, because wrapping a raster
# image in a PDF container looks like a PDF figure while still being a bitmap.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
IMAGES="thesis/images"

if ! command -v rsvg-convert >/dev/null 2>&1; then
    echo "rsvg-convert not found. Install it with:" >&2
    echo "    brew install librsvg" >&2
    exit 1
fi

FORCE=0
declare -a TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --all) FORCE=1 ;;
        *)     TARGETS+=("$arg") ;;
    esac
done

if ((${#TARGETS[@]} == 0)); then
    while IFS= read -r file; do TARGETS+=("$file"); done < <(find "$IMAGES" -maxdepth 1 -name '*.svg' | sort)
fi

converted=0
skipped=0
failed=()

for svg in "${TARGETS[@]}"; do
    [[ -f "$svg" ]] || { echo "missing: $svg" >&2; failed+=("$svg"); continue; }
    pdf="${svg%.svg}.pdf"
    if (( ! FORCE )) && [[ -f "$pdf" && "$pdf" -nt "$svg" ]]; then
        ((skipped++))
        continue
    fi
    if rsvg-convert -f pdf -o "$pdf" "$svg" 2>/dev/null && [[ -s "$pdf" ]]; then
        printf '  %-52s -> %s\n' "$(basename "$svg")" "$(basename "$pdf")"
        ((converted++))
    else
        echo "  FAILED: $svg" >&2
        failed+=("$svg")
    fi
done

echo
echo "converted: ${converted}   already up to date: ${skipped}   failed: ${#failed[@]}"
if ((${#failed[@]})); then
    printf 'failed files: %s\n' "${failed[*]}" >&2
    exit 1
fi
echo "Now rebuild:  cd thesis && xelatex thesis.tex"
