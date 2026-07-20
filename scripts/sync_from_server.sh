#!/usr/bin/env bash
# Pull run results and logs back from the offline university server via rsync.
# Never deletes local runs/. Pass --add-only to preserve every existing local
# file byte-for-byte while still pulling newly created run artifacts.
#
# Usage:
#   bash scripts/sync_from_server.sh                # uses defaults below
#   SERVER_HOST=1.2.3.4 SERVER_USER=me bash scripts/sync_from_server.sh
#   bash scripts/sync_from_server.sh --dry-run       # preview only, no transfer
#   bash scripts/sync_from_server.sh --add-only      # transfer only absent files
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_USER="${SERVER_USER:-fatemerjn}"
SERVER_HOST="${SERVER_HOST:-172.27.50.51}"
SERVER_PATH="${SERVER_PATH:-~/Overlap-Aware-Selective-Forgetting-/}"

RSYNC_FLAGS=(-avz --progress)
DRY_RUN=0
ADD_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --add-only) ADD_ONLY=1 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done
if (( DRY_RUN )); then
    RSYNC_FLAGS+=(--dry-run)
    echo "[INFO] Dry run -- nothing will actually be transferred."
fi
if (( ADD_ONLY )); then
    RSYNC_FLAGS+=(--ignore-existing)
    echo "[INFO] Add-only mode -- existing local files will not be overwritten."
fi

echo "[INFO] Pulling runs/ from ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
rsync "${RSYNC_FLAGS[@]}" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}runs/" \
    "runs/"

echo "[INFO] Pulling logs/ from ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
rsync "${RSYNC_FLAGS[@]}" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}logs/" \
    "logs/" 2>/dev/null || echo "[WARN] logs/ not found on server (ok if the group hasn't started yet)."

echo "[INFO] Pulling server environment manifest when available"
mkdir -p results/aggregates
rsync "${RSYNC_FLAGS[@]}" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}results/aggregates/SERVER_ENVIRONMENT.txt" \
    "results/aggregates/" 2>/dev/null \
    || echo "[WARN] SERVER_ENVIRONMENT.txt not found on server."

echo "[DONE] Results synced."
echo "[NEXT] Regenerate tables/figures locally:"
echo "    bash scripts/reproduce_all.sh"
echo "    ./.venv/bin/python tools/check_thesis_numbers.py"
