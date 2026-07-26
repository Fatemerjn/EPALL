#!/usr/bin/env bash
# Push source code (no data/runs/logs) to the offline university server via rsync.
# The server has no internet access, so `git pull` does not work there -- this
# script is the only way to deploy code changes to it.
#
# Usage:
#   bash scripts/sync_to_server.sh                # uses defaults below
#   SERVER_HOST=1.2.3.4 SERVER_USER=me bash scripts/sync_to_server.sh
#   bash scripts/sync_to_server.sh --dry-run       # preview only, no transfer
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

echo "[INFO] Syncing code to ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"

rsync "${RSYNC_FLAGS[@]}" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude ".venv/" \
    --exclude "data/" \
    --exclude "runs/" \
    --exclude "runs_archive/" \
    --exclude "logs/" \
    --exclude "graphify-out/" \
    --exclude ".git/" \
    main.py data.py feature_cache.py privacy_metrics.py reference_metrics.py \
    methods models tools schedules docs requirements.txt \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}"

echo "[DONE] Code synced. On the server, no git pull is needed/possible (offline)."
echo "[NEXT] ssh in, then:"
echo "    cd ${SERVER_PATH}"
echo "    source .venv/bin/activate"
echo "    tmux new -s <name>"
echo "    bash tools/run_server_experiments.sh <group> 2>&1 | tee logs/<group>_console.log"
