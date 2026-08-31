#!/usr/bin/env bash
# ساخت همه‌ی شکل‌های TikZ پایان‌نامه.
# اجرا:  cd thesis && bash figures-tikz/build.sh [--install]
#   --install : PDF ها را در thesis/images/ کپی می‌کند تا \includegraphics پیدایشان کند
set -euo pipefail
cd "$(dirname "$0")/.."

FIGS=(parameter_overlap_concept softmask_drop_decomposition
      selective_forgetting_pipeline pall_adapter_architecture
      EPALL_mechanism_compact)

fail=0
for f in "${FIGS[@]}"; do
  if xelatex -interaction=nonstopmode -halt-on-error \
       -output-directory=figures-tikz "figures-tikz/$f.tex" >/dev/null 2>&1; then
    printf '  OK   %s\n' "$f"
  else
    printf '  FAIL %s  (جزئیات: figures-tikz/%s.log)\n' "$f" "$f"; fail=1
  fi
done
rm -f figures-tikz/*.aux
[[ $fail -eq 0 ]] && rm -f figures-tikz/*.log

if [[ "${1:-}" == "--install" ]]; then
  for f in "${FIGS[@]}"; do cp "figures-tikz/$f.pdf" "images/$f.pdf"; done
  echo "نصب شد در images/ — یادت باشد ارجاع .png را در فصل‌ها به .pdf تغییر دهی."
fi
exit $fail
