#!/usr/bin/env bash
# Capture the actual experiment host without changing its software or drivers.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-results/aggregates/SERVER_ENVIRONMENT.txt}"
PY="${PYTHON:-python3}"
mkdir -p "$(dirname "$OUT")"

{
    echo "capture_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname=$(hostname)"
    echo
    echo "[OS / kernel]"
    uname -a
    if [[ -r /etc/os-release ]]; then
        cat /etc/os-release
    fi
    echo
    echo "[CPU / memory]"
    command -v lscpu >/dev/null 2>&1 && lscpu || true
    command -v free >/dev/null 2>&1 && free -h || true
    echo
    echo "[GPU / driver]"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi unavailable"
    echo
    echo "[Python]"
    "$PY" --version
    "$PY" -m pip freeze
} > "$OUT"

echo "Wrote $OUT"
