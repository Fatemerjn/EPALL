#!/usr/bin/env bash
# Pull run results and logs back from the offline university server via rsync.
# Never deletes local runs/ -- only adds/updates, so it is safe to run repeatedly
# (e.g. to check on partial progress mid-sweep).
#
# Usage:
#   bash scripts/sync_from_server.sh                # uses defaults below
#   SERVER_HOST=1.2.3.4 SERVER_USER=me bash scripts/sync_from_server.sh
#   bash scripts/sync_from_server.sh --dry-run       # preview only, no transfer
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_USER="${SERVER_USER:-fatemerjn}"
SERVER_HOST="${SERVER_HOST:-172.27.50.51}"
SERVER_PATH="${SERVER_PATH:-~/Overlap-Aware-Selective-Forgetting-/}"

RSYNC_FLAGS=(-avz --progress)
if [[ "${1:-}" == "--dry-run" ]]; then
    RSYNC_FLAGS+=(--dry-run)
    echo "[INFO] Dry run -- nothing will actually be transferred."
fi

echo "[INFO] Pulling runs/ from ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
rsync "${RSYNC_FLAGS[@]}" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}runs/" \
    "runs/"

echo "[INFO] Pulling logs/ from ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"
rsync "${RSYNC_FLAGS[@]}" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}logs/" \
    "logs/" 2>/dev/null || echo "[WARN] logs/ not found on server (ok if the group hasn't started yet)."

echo "[DONE] Results synced."
echo "[NEXT] Regenerate tables/figures locally:"
echo "    bash scripts/reproduce_all.sh"
echo "    ./.venv/bin/python tools/check_thesis_numbers.py"
