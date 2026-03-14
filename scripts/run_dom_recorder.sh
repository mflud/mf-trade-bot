#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_dom_recorder.sh  —  Runs dom_client.py --record and auto-restarts on exit.
#
# Usage:
#   ./scripts/run_dom_recorder.sh &          # background, logs to logs/dom_recorder.log
#   nohup ./scripts/run_dom_recorder.sh &    # survives terminal close
#
# Stop it:
#   kill $(cat logs/dom_recorder.pid)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_DIR/logs/dom_recorder.log"
PID_FILE="$REPO_DIR/logs/dom_recorder.pid"

# Conda env name
CONDA_ENV="topstep-project"

# dom_client arguments — edit as needed
DOM_ARGS="--record --record-interval 5"

# Restart back-off: wait this many seconds between restarts (doubles after each
# consecutive fast failure, resets after a stable run of STABLE_SECS seconds).
MIN_BACKOFF=5
MAX_BACKOFF=300
STABLE_SECS=120

# ── Helpers ───────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG_FILE"
}

# ── Setup ─────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"
echo $$ > "$PID_FILE"
log "Wrapper started (PID=$$)  env=$CONDA_ENV  args: $DOM_ARGS"
log "Logs → $LOG_FILE   PID file → $PID_FILE"

# Activate conda (works even when not in an interactive shell)
# shellcheck source=/dev/null
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Restart loop ──────────────────────────────────────────────────────────────

backoff=$MIN_BACKOFF
attempt=0

while true; do
    attempt=$((attempt + 1))
    start_ts=$(date +%s)
    log "── Attempt $attempt ─────────────────────────────────────────────"

    # Run; capture exit code without triggering set -e
    python src/dom_client.py $DOM_ARGS >> "$LOG_FILE" 2>&1 || true

    exit_code=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    log "Process exited (code=$exit_code) after ${elapsed}s"

    # If it ran stably for a while, reset back-off
    if [ "$elapsed" -ge "$STABLE_SECS" ]; then
        backoff=$MIN_BACKOFF
        log "Ran >${STABLE_SECS}s — resetting back-off to ${backoff}s"
    else
        log "Fast failure — waiting ${backoff}s before restart …"
        sleep "$backoff"
        backoff=$(( backoff * 2 ))
        if [ "$backoff" -gt "$MAX_BACKOFF" ]; then
            backoff=$MAX_BACKOFF
        fi
    fi
done
