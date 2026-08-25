#!/usr/bin/env bash
# Create a reproducible, network-isolated Python environment on the Linux server.
# The wheelhouse is intentionally local-only: build it for the server's exact
# Python, Linux, CUDA, and package requirements before copying it to the server.
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
Usage: bash scripts/bootstrap_linux_offline.sh [--recreate]

Environment overrides:
  PYTHON_BIN=python3.13   Python interpreter used to create the venv.
  VENV_DIR=.venv          Destination virtual environment.
  WHEELHOUSE=wheels_linux Directory containing wheels matching requirements.txt.

This script never contacts PyPI: installation is always --no-index.
EOF
}

RECREATE=0
while (( $# )); do
    case "$1" in
        --recreate) RECREATE=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

PYTHON_BIN="${PYTHON_BIN:-python3.13}"
VENV_DIR="${VENV_DIR:-.venv}"
WHEELHOUSE="${WHEELHOUSE:-wheels_linux}"

if [[ -z "$VENV_DIR" || "$VENV_DIR" == "/" || "$VENV_DIR" == "." ]]; then
    echo "[ERROR] Refusing unsafe VENV_DIR value: ${VENV_DIR:-<empty>}" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "[ERROR] Wheelhouse not found: $WHEELHOUSE" >&2
    echo "        Build a Linux/Python-3.13 wheelhouse for requirements.txt and copy it here first." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit(
        f"[ERROR] requirements.txt is validated for Python 3.13; got {sys.version.split()[0]}"
    )
PY

if (( RECREATE )) && [[ -e "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    --requirement requirements.txt
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.9.1",
    "torchvision": "0.24.1",
    "numpy": "2.4.1",
    "pandas": "2.3.3",
    "matplotlib": "3.10.9",
}
for package, wanted in expected.items():
    installed = version(package)
    if installed != wanted:
        raise SystemExit(f"[ERROR] {package}=={installed}; expected {wanted}")
print("[OK] Offline environment matches requirements.txt")
PY
