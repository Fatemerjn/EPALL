#!/usr/bin/env bash
# Push source code (no data/runs/logs) to the offline university server via rsync.
# The server has no internet access, so `git pull` does not work there -- this
# script is the only way to deploy code changes to it.
#
# Usage:
#   SERVER_USER=me SERVER_HOST=host.example bash scripts/sync_to_server.sh
#   SERVER_USER=me SERVER_HOST=host.example bash scripts/sync_to_server.sh --dry-run
set -euo pipefail

cd "$(dirname "$0")/.."

# The training server is an internal machine, so its address is not baked into
# the repository.  Set SERVER_USER and SERVER_HOST in your environment, e.g.
#   export SERVER_USER=your-user SERVER_HOST=your.server.example
SERVER_USER="${SERVER_USER:?set SERVER_USER to your account on the training server}"
SERVER_HOST="${SERVER_HOST:?set SERVER_HOST to the training server address}"
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
