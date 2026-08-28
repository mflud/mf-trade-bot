#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_bar_collector.sh  —  Runs bar_collector.py with auto-restart on exit.
#
# Managed by launchd via com.mf-trade-bot.bar-collector.plist.
# launchd provides outer KeepAlive; this wrapper adds exponential back-off so
# rapid crash loops don't hammer the API.
# ──────────────────────────────────────────────────────────────────────────────

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_DIR/logs/bar_collector.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

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
log "Wrapper started (PID=$$)"

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

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

    "$PYTHON" -u src/bar_collector.py >> "$LOG_FILE" 2>&1 || true

    exit_code=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    log "Process exited (code=$exit_code) after ${elapsed}s"

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
