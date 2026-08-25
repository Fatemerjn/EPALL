#!/usr/bin/env bash
# Export thesis figures to PDF from their SVG source.
#
# The thesis \includegraphics every figure as .pdf, so a redrawn .svg only
# reaches the document once it has been exported. This converts every .svg in
# thesis/images/ whose .pdf is missing or older than the .svg, and leaves
# everything else alone.
#
#   bash tools/svg_to_pdf.sh                # convert what is out of date
#   bash tools/svg_to_pdf.sh --all          # ignore PDF timestamps, reconvert all
#   bash tools/svg_to_pdf.sh --force-stale  # ALSO overwrite from an outdated SVG
#   bash tools/svg_to_pdf.sh a.svg b.svg    # convert just these
#
# --all only relaxes the "PDF is newer than SVG" skip. An SVG that is older than
# its own PNG is a superseded draft, and converting it would put stale artwork
# back into the thesis, so that guard needs the explicit --force-stale.
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
FORCE_STALE=0
declare -a TARGETS=()
for arg in "$@"; do
    case "$arg" in
        --all)         FORCE=1 ;;
        --force-stale) FORCE_STALE=1 ;;
        *)             TARGETS+=("$arg") ;;
    esac
done

if ((${#TARGETS[@]} == 0)); then
    while IFS= read -r file; do TARGETS+=("$file"); done < <(find "$IMAGES" -maxdepth 1 -name '*.svg' | sort)
fi

converted=0
skipped=0
failed=()
stale=()

for svg in "${TARGETS[@]}"; do
    [[ -f "$svg" ]] || { echo "missing: $svg" >&2; failed+=("$svg"); continue; }
    pdf="${svg%.svg}.pdf"
    # Guard: several figures were finalised by editing the PNG directly, leaving
    # the SVG behind as an older draft. Converting such an SVG would silently
    # reintroduce the superseded artwork, so refuse unless --force says otherwise.
    png="${svg%.svg}.png"
    if (( ! FORCE_STALE )) && [[ -f "$png" && "$png" -nt "$svg" ]]; then
        echo "  STALE SOURCE, skipped: $(basename "$svg") is older than its .png" >&2
        stale+=("$svg")
        continue
    fi
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
echo "converted: ${converted}   already up to date: ${skipped}   stale source: ${#stale[@]}   failed: ${#failed[@]}"
if ((${#stale[@]})); then
    echo "Stale sources were skipped; their .png is newer than the .svg."
    echo "Re-export the SVG from the current artwork, or pass --force-stale to override."
fi
if ((${#failed[@]})); then
    printf 'failed files: %s\n' "${failed[*]}" >&2
    exit 1
fi
echo "Now rebuild:  cd thesis && xelatex thesis.tex"
