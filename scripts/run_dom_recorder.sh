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

set -uo pipefail

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

# Locate conda — launchd doesn't load shell profiles so we search common paths
CONDA_BASE=""
for candidate in \
    "$HOME/anaconda3" \
    "$HOME/miniconda3" \
    "$HOME/opt/anaconda3" \
    "$HOME/opt/miniconda3" \
    "/opt/anaconda3" \
    "/opt/miniconda3" \
    "/usr/local/anaconda3" \
    "/usr/local/miniconda3"; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    log "ERROR: could not locate conda installation — tried common paths"
    exit 1
fi

log "Using conda at $CONDA_BASE"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Restart loop ──────────────────────────────────────────────────────────────

backoff=$MIN_BACKOFF
attempt=0

wait_for_session() {
    # Poll every 60s rather than sleeping 30m at a stretch — a flat 30m sleep
    # doesn't resume counting until the Mac has been awake for the full 30m,
    # so a laptop nap near the session open can silently push the actual
    # start out by however long it slept. Polling short caps that delay at
    # ~60s. Log throttled to once per ~30m so this doesn't spam the log file.
    local last_log_ts=0
    while true; do
        dow=$(TZ=America/New_York date +%u)       # 1=Mon…7=Sun
        hhmm=$(TZ=America/New_York date +%H%M)   # e.g. 0830
        hhmm=$((10#$hhmm))                        # force base-10 (avoid octal interpretation)
        if [[ $dow -le 5 ]] && [[ $hhmm -ge 830 ]] && [[ $hhmm -lt 1700 ]]; then
            return
        fi
        now_ts=$(date +%s)
        if (( now_ts - last_log_ts >= 1800 )); then
            log "Outside 08:30–17:00 ET (ET=$(TZ=America/New_York date +%H:%M) dow=$dow) — polling every 60s"
            last_log_ts=$now_ts
        fi
        sleep 60
    done
}

while true; do
    wait_for_session
    attempt=$((attempt + 1))
    start_ts=$(date +%s)
    log "── Attempt $attempt ─────────────────────────────────────────────"

    # Run; capture exit code without triggering set -e
    python -u src/dom_client.py $DOM_ARGS >> "$LOG_FILE" 2>&1 || true

    exit_code=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    log "Process exited (code=$exit_code) after ${elapsed}s"

    # Keep log from growing unbounded — trim to last 5000 lines on each restart
    if [ -f "$LOG_FILE" ]; then
        tail -5000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
    fi

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
